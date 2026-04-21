"""
SIPT Pro v3 — Génération, Entraînement & Sauvegarde
Isolation Forest + LSTM Autoencoder · Détection hybride
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION RÉSEAU
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = ["NORD", "SUD", "EST", "OUEST", "CENTRE"]

CLIENT_PROFILES = {
    "RESIDENTIEL": {
        "base_range": (3, 18), "weekend_factor": 1.25,
        "noise_std": 0.12, "seasonal": True,
        "description": "Foyer domestique – consommation modérée, sensible aux week-ends",
    },
    "INDUSTRIEL": {
        "base_range": (80, 250), "weekend_factor": 0.60,
        "noise_std": 0.04, "seasonal": False,
        "description": "Site industriel – forte consommation, chute le weekend",
    },
    "COMMERCIAL": {
        "base_range": (25, 90), "weekend_factor": 0.45,
        "noise_std": 0.09, "seasonal": True,
        "description": "Commerce/bureau – pic en semaine, quasi-nul le weekend",
    },
    "AGRICOLE": {
        "base_range": (8, 40), "weekend_factor": 1.05,
        "noise_std": 0.18, "seasonal": True,
        "description": "Exploitation agricole – consommation saisonnière",
    },
}

PROFILE_WEIGHTS = [0.55, 0.15, 0.20, 0.10]

FEATURE_NAMES = [
    "Conso moyenne (kWh/j)",
    "Volatilité (écart-type)",
    "Tendance linéaire (slope)",
    "Variation journalière moy",
    "Ratio 1ère/2ème période",
    "Amplitude max-min",
    "Coefficient de variation",
    "Jours très faible conso (<0.5 kWh)",
    "Ratio Weekend/Weekday",
    "Volatilité Weekend vs Weekday",
    "Autocorrélation temporelle (lag-1)",
    "Instabilité directionnelle",
]

WINDOW_SIZE = 30  # fenêtre commune IF + LSTM


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DATASET  (seed aléatoire → données différentes à chaque run)
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n_clients: int = 300, days: int = 90, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    records, client_meta = [], []
    profile_names = list(CLIENT_PROFILES.keys())

    for c in range(n_clients):
        profile_name = np.random.choice(profile_names, p=PROFILE_WEIGHTS)
        profile      = CLIENT_PROFILES[profile_name]
        region       = np.random.choice(REGIONS)
        base         = np.random.uniform(*profile["base_range"])
        is_fraudster = np.random.random() < 0.05
        fraud_mode   = np.random.choice(["bypass", "tamper", "inject"]) if is_fraudster else None
        client_id    = f"{profile_name[:3]}_{region[:3]}_{c:04d}"

        client_meta.append({
            "client_id":    client_id,
            "region":       region,
            "profile":      profile_name,
            "is_fraudster": is_fraudster,
            "fraud_mode":   fraud_mode if is_fraudster else "—",
        })

        for d in range(days):
            seasonal = 1.0
            if profile["seasonal"]:
                seasonal = 1.35 if (d // 30) % 6 <= 2 else 0.80
            dow   = d % 7
            wk    = profile["weekend_factor"] if dow >= 5 else 1.0
            noise = np.random.normal(1, profile["noise_std"])
            cons  = base * seasonal * wk * noise

            if is_fraudster and np.random.random() < 0.30:
                if fraud_mode == "bypass":
                    cons *= np.random.uniform(0.05, 0.30)
                elif fraud_mode == "tamper":
                    cons *= np.random.uniform(0.35, 0.65)
                else:
                    cons -= np.random.uniform(base * 0.5, base * 0.85)

            records.append({
                "timestamp":       pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "client_id":       client_id,
                "region":          region,
                "profile":         profile_name,
                "consumption_kwh": max(0.0, round(cons, 2)),
            })

    return pd.DataFrame(records), pd.DataFrame(client_meta)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (fenêtre glissante — pour Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(s: np.ndarray) -> np.ndarray:
    s        = np.array(s, dtype=float)
    n        = len(s)
    wk_vals  = s[np.arange(n) % 7 >= 5]
    wd_vals  = s[np.arange(n) % 7 <  5]
    diff_s   = np.diff(s)
    mean_s   = np.mean(s)   if n > 0 else 0.0
    std_s    = np.std(s)    if n > 1 else 0.0
    trend    = np.polyfit(np.arange(n), s, 1)[0] if n >= 2 else 0.0
    h1       = np.mean(s[:n // 2]) if n >= 2 else mean_s
    h2       = np.mean(s[n // 2:]) if n >= 2 else mean_s
    mean_wk  = np.mean(wk_vals) if len(wk_vals) > 0 else mean_s
    mean_wd  = np.mean(wd_vals) if len(wd_vals) > 0 else mean_s
    std_wk   = np.std(wk_vals)  if len(wk_vals) > 1 else std_s
    std_wd   = np.std(wd_vals)  if len(wd_vals) > 1 else std_s
    if n > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0:
        autocorr = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    else:
        autocorr = 0.0
    return np.array([
        mean_s,
        std_s,
        trend,
        np.mean(diff_s)  if n > 1 else 0.0,
        h1 / h2          if h2 > 0 else 1.0,
        np.max(s) - np.min(s),
        std_s / mean_s   if mean_s > 0 else 0.0,
        float(np.sum(s < 0.5)),
        mean_wk / mean_wd if mean_wd > 0 else 1.0,
        std_wk  / std_wd  if std_wd  > 0 else 1.0,
        autocorr,
        float(np.sum(diff_s[1:] * diff_s[:-1] < 0)) if n > 2 else 0.0,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LSTM AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_autoencoder(window_size: int):
    """
    Autoencoder LSTM pour la détection d'anomalies temporelles.

    Architecture :
      Encoder : LSTM(64) → LSTM(32) → vecteur latent
      Decoder : RepeatVector → LSTM(32) → LSTM(64) → Dense(1)

    Principe : entraîné UNIQUEMENT sur des clients normaux.
    Un pattern de consommation frauduleux sera mal reconstruit
    → MSE de reconstruction élevée → signal d'anomalie.

    Cette approche est complémentaire à l'Isolation Forest :
      - IF   détecte les outliers statistiques sur features agrégées.
      - LSTM détecte les anomalies de séquence temporelle fine.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout,
    )
    inp = Input(shape=(window_size, 1), name="seq_input")

    # Encoder
    x       = LSTM(64, activation="tanh", return_sequences=True,  name="enc1")(inp)
    x       = Dropout(0.1)(x)
    encoded = LSTM(32, activation="tanh", return_sequences=False, name="enc2")(x)

    # Decoder
    x   = RepeatVector(window_size, name="bottleneck")(encoded)
    x   = LSTM(32, activation="tanh", return_sequences=True,  name="dec1")(x)
    x   = Dropout(0.1)(x)
    x   = LSTM(64, activation="tanh", return_sequences=True,  name="dec2")(x)
    out = TimeDistributed(Dense(1), name="reconstruction")(x)

    model = Model(inp, out, name="LSTM_Autoencoder_SIPT")
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_lstm_sequences(df: pd.DataFrame, meta: pd.DataFrame, window_size: int):
    """
    Construit les séquences d'entraînement LSTM à partir des clients normaux.
    Normalisation z-score par client pour homogénéiser les échelles
    (RESIDENTIEL ~10 kWh vs INDUSTRIEL ~150 kWh sinon le modèle apprend l'échelle et non le pattern).
    """
    normal_cids = meta[meta["is_fraudster"] == False]["client_id"].tolist()
    sequences   = []

    for cid in normal_cids:
        vals = (
            df[df["client_id"] == cid]
            .sort_values("timestamp")["consumption_kwh"]
            .values.astype(float)
        )
        std_c = vals.std()
        if std_c < 1e-6:
            continue
        vals_norm = (vals - vals.mean()) / std_c
        for i in range(len(vals_norm) - window_size + 1):
            sequences.append(vals_norm[i : i + window_size])

    X = np.array(sequences, dtype=np.float32)[..., np.newaxis]  # (N, W, 1)
    return X


def compute_lstm_threshold(model, X_normal: np.ndarray, percentile: float = 95.0) -> float:
    """Seuil LSTM = Xe percentile des MSE sur données normales d'entraînement."""
    recons = model.predict(X_normal, batch_size=512, verbose=0)
    errors = np.mean((X_normal[:, :, 0] - recons[:, :, 0]) ** 2, axis=1)
    return float(np.percentile(errors, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  SIPT Pro v3 — Pipeline Isolation Forest + LSTM Autoencoder")
    print("=" * 65)

    # ── 1. Génération ─────────────────────────────────────────────────────
    print("\n[1/6] Génération du dataset (seed aléatoire)...")
    seed = int(time.time() * 1000) % 99999
    df, meta = generate_dataset(n_clients=300, days=90, seed=seed)
    df.to_csv("electricity_fraud_dataset.csv", index=False)
    meta.to_csv("client_metadata.csv", index=False)
    joblib.dump(seed, "dataset_seed.pkl")

    n_fraud = meta["is_fraudster"].sum()
    print(f"  ✓ {len(df):,} obs | {df['client_id'].nunique()} clients | seed={seed}")
    print(f"  ✓ {n_fraud} fraudeurs ({n_fraud / len(meta):.1%})")
    for p in CLIENT_PROFILES:
        cnt = (meta["profile"] == p).sum()
        print(f"     · {p:<12} : {cnt:>3} clients")

    # ── 2. Features IF ────────────────────────────────────────────────────
    print(f"\n[2/6] Extraction features IF (fenêtre {WINDOW_SIZE}j)...")
    features_list, client_ids_feat = [], []
    for cid, grp in df.groupby("client_id"):
        vals = grp.sort_values("timestamp")["consumption_kwh"].values
        if len(vals) >= WINDOW_SIZE:
            features_list.append(extract_features(vals[-WINDOW_SIZE:]))
            client_ids_feat.append(cid)
    X = np.array(features_list)
    print(f"  ✓ {len(X)} vecteurs ({X.shape[1]} features)")

    # ── 3. Isolation Forest ───────────────────────────────────────────────
    print("\n[3/6] Isolation Forest...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso      = IsolationForest(
        n_estimators=200, contamination=0.05,
        max_features=1.0, bootstrap=False,
        random_state=None, n_jobs=-1,
    )
    iso.fit(X_scaled)
    scores_if = iso.decision_function(X_scaled)
    thresh_if = np.percentile(scores_if, 5)
    print(f"  ✓ Seuil IF : {thresh_if:.4f} | {np.sum(scores_if < thresh_if)} anomalies")

    # ── 4. Sauvegarde IF ──────────────────────────────────────────────────
    print("\n[4/6] Sauvegarde artefacts IF...")
    joblib.dump(iso,            "isolation_forest.pkl")
    joblib.dump(scaler,         "scaler.pkl")
    joblib.dump(FEATURE_NAMES,  "feature_names.pkl")
    joblib.dump(X_scaled[:100], "background_samples.pkl")
    feat_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
    feat_df.insert(0, "client_id", client_ids_feat)
    feat_df["score_IF"] = scores_if
    feat_df["label_IF"] = iso.predict(X_scaled)
    feat_df.to_csv("features_debug.csv", index=False)
    print("  ✓ isolation_forest.pkl | scaler.pkl | features_debug.csv")

    # ── 5. LSTM Autoencoder ───────────────────────────────────────────────
    print(f"\n[5/6] LSTM Autoencoder (fenêtre {WINDOW_SIZE}j)...")
    lstm_ok = False
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X_lstm = prepare_lstm_sequences(df, meta, WINDOW_SIZE)
        print(f"  ✓ {len(X_lstm)} séquences normales")

        lstm_model = build_lstm_autoencoder(WINDOW_SIZE)
        callbacks  = [
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
        ]
        history = lstm_model.fit(
            X_lstm, X_lstm,
            epochs=60, batch_size=128,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1, shuffle=True,
        )
        val_loss = min(history.history["val_loss"])
        print(f"  ✓ Val loss finale : {val_loss:.6f}")

        lstm_thresh = compute_lstm_threshold(lstm_model, X_lstm, percentile=95.0)
        print(f"  ✓ Seuil LSTM (95e pct) : {lstm_thresh:.6f}")

        lstm_model.save("lstm_autoencoder.keras")
        joblib.dump(lstm_thresh, "lstm_threshold.pkl")
        print("  ✓ lstm_autoencoder.keras | lstm_threshold.pkl")
        lstm_ok = True

    except ImportError:
        print("  ⚠  TensorFlow absent → pip install tensorflow")
    except Exception as e:
        print(f"  ⚠  LSTM échoué : {e}")

    # ── 6. Résumé ─────────────────────────────────────────────────────────
    print("\n[6/6] Rapport final.")
    print("=" * 65)
    print(f"  IF   : ✓")
    print(f"  LSTM : {'✓' if lstm_ok else '✗  (lancer après pip install tensorflow)'}")
    print(f"  Seed : {seed}  (nouveau à chaque run)")
    print("  → streamlit run app.py")
    print("=" * 65)
