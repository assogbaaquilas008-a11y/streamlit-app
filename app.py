"""
SIPT Pro v3 — Interface Surveillance Réseau Électrique
Détection hybride Isolation Forest + LSTM · Registre persistant
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SIPT Pro v3 — Surveillance Réseau",
    layout="wide", page_icon="⚡",
    initial_sidebar_state="expanded",
)

REGISTRY_FILE    = "fraud_registry.json"   # persistance sur disque
CONFIRM_THRESHOLD = 2                       # nb détections pour "fraudeur avéré"
WINDOW_SIZE      = 30
HISTORY_DAYS     = 60   # jours pré-chargés silencieusement
SIM_DAYS         = 7    # jours streamés = "la semaine de surveillance"

# ══════════════════════════════════════════════════════════════════════════════
# CSS SCADA
# ══════════════════════════════════════════════════════════════════════════════
SCADA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{background-color:#080d1a!important;color:#c8d8f0!important;font-family:'Inter',sans-serif}
.stApp{background:#080d1a}
.scada-header{background:linear-gradient(90deg,#060b18 0%,#0d1a35 50%,#060b18 100%);border-bottom:1px solid #1a3a6e;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.scada-title{font-family:'Share Tech Mono',monospace;font-size:22px;color:#00d4ff;letter-spacing:3px;text-transform:uppercase}
.scada-subtitle{font-size:11px;color:#4a7aaa;letter-spacing:2px}
.scada-clock{font-family:'Share Tech Mono',monospace;font-size:18px;color:#00ff9d;letter-spacing:2px}
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:20px}
.kpi-card{background:#0d1a35;border:1px solid #1a3a6e;border-radius:8px;padding:14px 10px;text-align:center;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.kpi-ok::before{background:#00ff9d}.kpi-warn::before{background:#ffaa00}.kpi-alert::before{background:#ff3b3b}.kpi-info::before{background:#00d4ff}.kpi-purple::before{background:#a855f7}.kpi-lstm::before{background:#f97316}
.kpi-num{font-family:'Share Tech Mono',monospace;font-size:28px;font-weight:700;margin-bottom:4px}
.kpi-lbl{font-size:10px;color:#4a7aaa;text-transform:uppercase;letter-spacing:1.5px}
.kpi-ok .kpi-num{color:#00ff9d}.kpi-warn .kpi-num{color:#ffaa00}.kpi-alert .kpi-num{color:#ff3b3b}.kpi-info .kpi-num{color:#00d4ff}.kpi-purple .kpi-num{color:#a855f7}.kpi-lstm .kpi-num{color:#f97316}
.region-card{background:#0d1a35;border:1px solid #1a3a6e;border-radius:10px;padding:16px;position:relative}
.region-card.has-alert{border-color:#ff3b3b}.region-card.has-warning{border-color:#ffaa00}
.region-name{font-family:'Share Tech Mono',monospace;font-size:13px;color:#00d4ff;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase}
.region-stat{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;border-bottom:1px solid #1a3a6e;color:#8aa8cc}
.region-stat:last-child{border-bottom:none}
.region-stat strong{color:#c8d8f0;font-family:'Share Tech Mono',monospace}
.alert-ticker{background:#0d0812;border:1px solid #4a1a2e;border-radius:6px;padding:7px 12px;font-family:'Share Tech Mono',monospace;font-size:11px;color:#ff3b3b;margin-bottom:6px;display:flex;align-items:center;gap:8px}
.ticker-badge{background:#ff3b3b;color:#fff;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700}
.ticker-badge-lstm{background:#f97316;color:#fff;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700}
.ticker-badge-both{background:linear-gradient(90deg,#ff3b3b,#f97316);color:#fff;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:700}
.ticker-meta{color:#8aa8cc;font-size:10px}
.client-row{display:flex;align-items:center;justify-content:space-between;padding:6px 10px;border-radius:5px;margin-bottom:3px;font-size:11px;font-family:'Share Tech Mono',monospace;border:1px solid transparent}
.client-row.normal{background:#0a1628;border-color:#1a3a6e;color:#6e9abf}
.client-row.warning{background:#1a1200;border-color:#4a3a00;color:#ffaa00}
.client-row.alert{background:#1a0808;border-color:#6e1010;color:#ff6b6b}
.client-row.alert_high{background:#1a0510;border-color:#9e1030;color:#ff3b3b}

/* Fraud cards — registre confirmé */
.fraud-card-confirmed{background:#0f0818;border:1px solid #6a1a3e;border-left:4px solid #ff3b3b;border-radius:8px;padding:14px 18px;margin-bottom:12px}
.fraud-card-suspect{background:#0f100a;border:1px solid #3a4a1e;border-left:4px solid #ffaa00;border-radius:8px;padding:12px 16px;margin-bottom:8px}
.fraud-id{font-family:'Share Tech Mono',monospace;font-size:14px;color:#ff3b3b;font-weight:700}
.suspect-id{font-family:'Share Tech Mono',monospace;font-size:13px;color:#ffaa00;font-weight:700}
.fraud-meta{font-size:11px;color:#8aa8cc;margin:3px 0 8px 0}
.detection-badge{display:inline-block;background:#2a0820;border:1px solid #6a1030;border-radius:12px;padding:2px 10px;font-family:'Share Tech Mono',monospace;font-size:11px;color:#ff6b6b;margin-right:6px}
.detection-badge-lstm{background:#1a0810;border-color:#7a3010;color:#f97316}
.confirmed-badge{display:inline-block;background:#3a0a20;border:1px solid #aa2040;border-radius:4px;padding:2px 8px;font-size:10px;color:#ff3b3b;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.shap-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #2a1020;font-size:11px}
.shap-row:last-child{border-bottom:none}
.shap-feat{color:#c8a0d0}.shap-val-neg{color:#ff6b6b;font-family:'Share Tech Mono',monospace;font-weight:700}.shap-val-pos{color:#00ff9d;font-family:'Share Tech Mono',monospace;font-weight:700}
.shap-bar{width:60px;height:5px;background:#2a1020;border-radius:3px;overflow:hidden}
.shap-fill-neg{height:100%;background:#ff3b3b;border-radius:3px}.shap-fill-pos{height:100%;background:#00ff9d;border-radius:3px}
.section-header{font-family:'Share Tech Mono',monospace;font-size:12px;color:#00d4ff;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid #1a3a6e;padding-bottom:5px;margin:14px 0 10px 0}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px}
.dot-ok{background:#00ff9d;box-shadow:0 0 5px #00ff9d}.dot-warn{background:#ffaa00;box-shadow:0 0 5px #ffaa00}.dot-alert{background:#ff3b3b;box-shadow:0 0 7px #ff3b3b;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.stTabs [data-baseweb="tab-list"]{background:#0d1a35!important;border-bottom:1px solid #1a3a6e!important;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#4a7aaa!important;font-family:'Share Tech Mono',monospace!important;font-size:11px!important;letter-spacing:1px!important;border-radius:6px 6px 0 0!important;padding:7px 14px!important}
.stTabs [aria-selected="true"]{background:#1a3a6e!important;color:#00d4ff!important;border-bottom:2px solid #00d4ff!important}
.stButton > button{background:#0d2244!important;border:1px solid #00d4ff!important;color:#00d4ff!important;font-family:'Share Tech Mono',monospace!important;letter-spacing:1px!important;border-radius:6px!important;font-size:11px!important}
.stButton > button:hover{background:#00d4ff!important;color:#080d1a!important}
[data-testid="stSidebar"]{background:#060b18!important;border-right:1px solid #1a3a6e!important}
[data-testid="stSidebar"] *{color:#c8d8f0!important}
.stProgress > div > div{background-color:#00d4ff!important}
[data-testid="stDataFrame"]{background:#0d1a35!important;border:1px solid #1a3a6e!important}
.week-badge{display:inline-block;background:#0d1a35;border:1px solid #1a4a6e;border-radius:16px;padding:3px 12px;font-family:'Share Tech Mono',monospace;font-size:11px;color:#00d4ff;margin-bottom:10px}
</style>
"""
st.markdown(SCADA_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRE PERSISTANT  (JSON sur disque)
# ══════════════════════════════════════════════════════════════════════════════
def load_registry() -> dict:
    """Charge le registre depuis le disque. Vide si absent."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_registry(registry: dict) -> None:
    """Sauvegarde le registre sur disque (atomique via fichier temp)."""
    tmp = REGISTRY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, REGISTRY_FILE)


def registry_add_detection(
    registry: dict, client_id: str, region: str, profile: str,
    date_str: str, score_if: float, score_lstm: float | None,
    risk_pct: float, narrative: str, shap_reasons: list,
    detected_by: str,  # "IF", "LSTM", "BOTH"
) -> dict:
    """
    Ajoute ou met à jour une entrée dans le registre.
    Le registre est indexé par client_id.
    """
    entry = registry.get(client_id, {
        "client_id":       client_id,
        "region":          region,
        "profile":         profile,
        "detection_count": 0,
        "lstm_count":      0,
        "if_count":        0,
        "both_count":      0,
        "max_risk_pct":    0.0,
        "min_score_if":    9999.0,
        "max_lstm_mse":    0.0,
        "first_seen":      date_str,
        "last_seen":       date_str,
        "is_confirmed":    False,
        "shap_reasons":    [],
        "detections":      [],
    })

    entry["detection_count"] += 1
    entry["last_seen"]        = date_str
    entry["max_risk_pct"]     = max(entry["max_risk_pct"], risk_pct)
    entry["min_score_if"]     = min(entry["min_score_if"], score_if)
    if score_lstm is not None:
        entry["max_lstm_mse"] = max(entry["max_lstm_mse"], score_lstm)

    if detected_by == "IF":
        entry["if_count"] += 1
    elif detected_by == "LSTM":
        entry["lstm_count"] += 1
    elif detected_by == "BOTH":
        entry["both_count"] += 1
        entry["if_count"]   += 1
        entry["lstm_count"] += 1

    entry["is_confirmed"] = entry["detection_count"] >= CONFIRM_THRESHOLD

    # Garde le SHAP du pire score IF
    if score_if < entry.get("min_score_if", 0) or not entry["shap_reasons"]:
        entry["shap_reasons"] = shap_reasons

    entry["detections"].append({
        "date":        date_str,
        "score_if":    round(score_if, 4),
        "score_lstm":  round(score_lstm, 6) if score_lstm is not None else None,
        "risk_pct":    round(risk_pct, 1),
        "narrative":   narrative,
        "detected_by": detected_by,
    })

    registry[client_id] = entry
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "surveillance_active": False,
    "client_windows":      {},
    "client_scores":       {},
    "client_lstm_mse":     {},
    "client_detect_by":    {},
    "current_week":        0,
    "sim_day":             0,
    "shap_ready":          False,
    "lstm_ready":          False,
    "registry":            None,   # chargé depuis disque au premier run
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Chargement registre persistant (une seule fois par session)
if st.session_state.registry is None:
    st.session_state.registry = load_registry()


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLES & DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    iso        = joblib.load("isolation_forest.pkl")
    scaler     = joblib.load("scaler.pkl")
    feat_names = joblib.load("feature_names.pkl")
    return iso, scaler, feat_names


@st.cache_data
def load_data():
    df   = pd.read_csv("electricity_fraud_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    meta = pd.read_csv("client_metadata.csv")
    return df, meta


@st.cache_resource
def init_shap(_model):
    try:
        import shap
        bg         = joblib.load("background_samples.pkl")
        explainer  = shap.TreeExplainer(_model, bg)
        return explainer
    except Exception:
        return None


@st.cache_resource
def load_lstm():
    """Charge le LSTM si disponible. Retourne (model, threshold) ou (None, None)."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        model  = tf.keras.models.load_model("lstm_autoencoder.keras")
        thresh = joblib.load("lstm_threshold.pkl")
        # Warm-up pour éviter le lag du premier predict
        dummy = np.zeros((1, WINDOW_SIZE, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
        return model, thresh
    except Exception:
        return None, None


try:
    iso_forest, scaler, feature_names = load_models()
    df, meta = load_data()
    explainer = init_shap(iso_forest)
    lstm_model, lstm_threshold = load_lstm()
    st.session_state.shap_ready = explainer is not None
    st.session_state.lstm_ready = lstm_model is not None
    data_loaded = True
    REGIONS  = sorted(df["region"].unique())
    PROFILES = sorted(df["profile"].unique())
    clients  = sorted(df["client_id"].unique())
except Exception as e:
    data_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS MÉTIER
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(s):
    s = np.array(s, dtype=float); n = len(s)
    wk_vals = s[np.arange(n) % 7 >= 5]; wd_vals = s[np.arange(n) % 7 < 5]
    diff_s  = np.diff(s)
    mean_s  = np.mean(s); std_s = np.std(s)
    trend   = np.polyfit(np.arange(n), s, 1)[0] if n >= 2 else 0.0
    h1 = np.mean(s[:n//2]); h2 = np.mean(s[n//2:])
    mean_wk = np.mean(wk_vals) if len(wk_vals) > 0 else mean_s
    mean_wd = np.mean(wd_vals) if len(wd_vals) > 0 else mean_s
    std_wk  = np.std(wk_vals)  if len(wk_vals) > 1 else std_s
    std_wd  = np.std(wd_vals)  if len(wd_vals) > 1 else std_s
    autocorr = float(np.corrcoef(s[:-1], s[1:])[0,1]) if n>2 and np.std(s[:-1])>0 and np.std(s[1:])>0 else 0.0
    return np.array([
        mean_s, std_s, trend,
        np.mean(diff_s) if n>1 else 0.0,
        h1/h2 if h2>0 else 1.0,
        np.max(s)-np.min(s), std_s/mean_s if mean_s>0 else 0.0,
        float(np.sum(s < 0.5)),
        mean_wk/mean_wd if mean_wd>0 else 1.0,
        std_wk/std_wd   if std_wd>0  else 1.0,
        autocorr,
        float(np.sum(diff_s[1:]*diff_s[:-1] < 0)) if n>2 else 0.0,
    ])


def lstm_score(window_vals: np.ndarray) -> tuple[float, bool]:
    """
    Score LSTM : MSE de reconstruction après normalisation z-score de la fenêtre.
    Retourne (mse, is_anomaly).
    """
    if not st.session_state.lstm_ready or lstm_model is None:
        return 0.0, False
    mean_w = window_vals.mean()
    std_w  = window_vals.std()
    if std_w < 1e-6:
        return 0.0, False
    norm_w = ((window_vals - mean_w) / std_w).astype(np.float32)
    inp    = norm_w.reshape(1, -1, 1)
    recon  = lstm_model.predict(inp, verbose=0)[0, :, 0]
    mse    = float(np.mean((norm_w - recon) ** 2))
    return mse, mse > lstm_threshold


def batch_lstm_score(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Score LSTM vectorisé pour un batch de fenêtres (N, W).
    Beaucoup plus rapide que N appels individuels.
    """
    if not st.session_state.lstm_ready or lstm_model is None:
        return np.zeros(len(windows)), np.zeros(len(windows), dtype=bool)
    normed = []
    for w in windows:
        std_w = w.std()
        normed.append((w - w.mean()) / std_w if std_w > 1e-6 else w * 0)
    X = np.array(normed, dtype=np.float32)[..., np.newaxis]  # (N, W, 1)
    recons = lstm_model.predict(X, batch_size=64, verbose=0)
    mses   = np.mean((X[:, :, 0] - recons[:, :, 0]) ** 2, axis=1)
    return mses, mses > lstm_threshold


def combined_level(if_score, if_thresh, lstm_is_anom):
    """
    Niveau combiné IF + LSTM :
      BOTH  → alert_high  (les deux détectent)
      IF    → alert
      LSTM  → warning
      None  → normal / warning zone
    """
    if_anom  = if_score < if_thresh
    if if_anom and lstm_is_anom:
        return "alert_high", "ALERTE DOUBLE", "BOTH"
    elif if_anom:
        return "alert",      "ALERTE IF",     "IF"
    elif lstm_is_anom:
        return "warning",    "SUSPECT LSTM",  "LSTM"
    elif if_score < if_thresh + 0.08:
        return "warning",    "SURVEILLANCE",  "—"
    else:
        return "normal",     "NORMAL",        "—"


def risk_pct(if_score, if_thresh, lstm_mse):
    """Pourcentage de risque combiné (IF 60%, LSTM 40%)."""
    if_risk   = max(0, min(1, (if_thresh - if_score) / 0.4 + 0.5))
    lstm_risk = 0.0
    if st.session_state.lstm_ready and lstm_threshold and lstm_threshold > 0:
        lstm_risk = min(1.0, lstm_mse / (lstm_threshold * 3.0))
    return round((0.6 * if_risk + 0.4 * lstm_risk) * 100, 1)


def shap_explain(feat_s, top_n=5):
    if not st.session_state.shap_ready or explainer is None:
        return []
    try:
        import shap
        sv = explainer.shap_values(feat_s)
        if isinstance(sv, list): sv = sv[1]
        sv = sv[0] if len(sv.shape) > 1 else sv
        abs_sv = np.abs(sv)
        idx    = np.argsort(abs_sv)[::-1][:top_n]
        return [(feature_names[i], float(sv[i]), float(abs_sv[i])) for i in idx]
    except Exception:
        return []


def shap_html(reasons):
    if not reasons:
        return '<p style="color:#4a7aaa;font-size:11px">SHAP non disponible</p>'
    html    = ""
    max_abs = max(abs(v) for _, v, _ in reasons) or 1
    for name, val, abs_v in reasons:
        pct      = round(abs_v / max_abs * 100)
        cls_val  = "shap-val-neg" if val < 0 else "shap-val-pos"
        cls_fill = "shap-fill-neg" if val < 0 else "shap-fill-pos"
        direction = "▼ Anomalement bas" if val < 0 else "▲ Anomalement élevé"
        html += (
            f'<div class="shap-row">'
            f'<span class="shap-feat">{name}</span>'
            f'<div class="shap-bar"><div class="{cls_fill}" style="width:{pct}%"></div></div>'
            f'<span class="{cls_val}">{val:+.3f}</span>'
            f'</div>'
            f'<div style="font-size:9px;color:#6a6a8a;padding:1px 0 3px 0">{direction}</div>'
        )
    return html


def build_narrative(reasons, if_score, lstm_mse, detected_by):
    parts = []
    if reasons:
        for name, val, _ in reasons[:2]:
            parts.append(f"«{name}» {'anorm. basse' if val < 0 else 'anorm. élevée'}")
    base = f"[{detected_by}] Score IF={if_score:.4f}"
    if lstm_mse:
        base += f" / MSE LSTM={lstm_mse:.4f}"
    return base + (f" — {' ; '.join(parts)}" if parts else ".")


def meta_of(cid):
    row = meta[meta["client_id"] == cid]
    return row.iloc[0] if len(row) > 0 else None


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
lstm_indicator = (
    '<span style="font-family:Share Tech Mono,monospace;font-size:10px;'
    'background:#1a0810;border:1px solid #f97316;color:#f97316;'
    'padding:2px 8px;border-radius:4px;margin-left:10px">LSTM ✓</span>'
    if st.session_state.lstm_ready else ""
)
st.markdown(f"""
<div class="scada-header">
  <div>
    <div class="scada-title">⚡ SIPT PRO v3 — RÉSEAU ÉLECTRIQUE NATIONAL{lstm_indicator}</div>
    <div class="scada-subtitle">Système Intelligent de Protection &amp; Télésurveillance · IF + LSTM Hybride</div>
  </div>
  <div class="scada-clock">{now_str}</div>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("⚠ Fichiers modèles introuvables. Exécutez `python train.py` d'abord.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:13px;'
        'color:#00d4ff;letter-spacing:2px;margin-bottom:14px">▶ PARAMÈTRES</div>',
        unsafe_allow_html=True,
    )
    threshold = st.slider("Seuil détection IF", -0.30, 0.10, -0.08, 0.005)
    speed     = st.slider("Vitesse simulation (s/jour)", 0.01, 2.0, 0.15, 0.01)
    win_size  = st.slider("Fenêtre analyse (jours)", 15, 60, 30, 5)

    st.divider()
    st.markdown("**Région surveillée**")
    region_mode = st.radio("", ["Toutes", "Choisir"], key="rmode",
                           label_visibility="collapsed")
    if region_mode == "Choisir":
        sel_region  = st.selectbox("", REGIONS, label_visibility="collapsed")
        sel_clients = sorted(df[df["region"] == sel_region]["client_id"].unique())
    else:
        sel_region  = None
        sel_clients = clients

    sel_profiles = st.multiselect("Profil", PROFILES, default=PROFILES)
    if sel_profiles:
        valid = meta[meta["profile"].isin(sel_profiles)]["client_id"].tolist()
        sel_clients = [c for c in sel_clients if c in valid]

    st.markdown(f"**{CONFIRM_THRESHOLD}+ détections = fraudeur avéré**",
                help="Modifiable dans le code source (CONFIRM_THRESHOLD)")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Vider session", use_container_width=True):
            st.session_state.client_windows    = {}
            st.session_state.client_scores     = {}
            st.session_state.client_lstm_mse   = {}
            st.session_state.client_detect_by  = {}
            st.rerun()
    with col_b:
        if st.button("💣 Reset registre", use_container_width=True):
            st.session_state.registry = {}
            save_registry({})
            st.rerun()

    registry = st.session_state.registry
    confirmed = [k for k, v in registry.items() if v.get("is_confirmed")]
    st.markdown(
        f'<div style="font-size:10px;color:#4a7aaa;text-align:center;margin-top:8px">'
        f'SHAP {"🟢" if st.session_state.shap_ready else "🔴"} | '
        f'LSTM {"🟠" if st.session_state.lstm_ready else "🔴"} | '
        f'Registre: {len(registry)} clients ({len(confirmed)} avérés)'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# KPI GLOBAUX
# ══════════════════════════════════════════════════════════════════════════════
registry = st.session_state.registry
confirmed_list   = [v for v in registry.values() if v.get("is_confirmed")]
suspect_list     = [v for v in registry.values() if not v.get("is_confirmed")]
both_count_total = sum(v.get("both_count", 0) for v in registry.values())
n_monitored      = len(sel_clients)
active_alerts    = sum(
    1 for c in sel_clients
    if st.session_state.client_scores.get(c, 1) < threshold
)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-info">
    <div class="kpi-num">{n_monitored}</div>
    <div class="kpi-lbl">Surveillés</div>
  </div>
  <div class="kpi-card kpi-alert">
    <div class="kpi-num">{len(confirmed_list)}</div>
    <div class="kpi-lbl">Fraudeurs avérés</div>
  </div>
  <div class="kpi-card kpi-warn">
    <div class="kpi-num">{len(suspect_list)}</div>
    <div class="kpi-lbl">Suspects</div>
  </div>
  <div class="kpi-card kpi-lstm">
    <div class="kpi-num">{both_count_total}</div>
    <div class="kpi-lbl">Détect. IF+LSTM</div>
  </div>
  <div class="kpi-card kpi-ok">
    <div class="kpi-num">{active_alerts}</div>
    <div class="kpi-lbl">Alertes actives</div>
  </div>
  <div class="kpi-card kpi-purple">
    <div class="kpi-num">{len(registry)}</div>
    <div class="kpi-lbl">Dans registre</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_reseau, tab_surv, tab_confirmed, tab_suspects, tab_analytics, tab_info = st.tabs([
    "🗺 VUE RÉSEAU",
    "📡 SURVEILLANCE",
    "🚨 FRAUDEURS AVÉRÉS",
    "⚠ SUSPECTS",
    "📊 ANALYTICS",
    "🧠 MODÈLES",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VUE RÉSEAU
# ══════════════════════════════════════════════════════════════════════════════
with tab_reseau:
    st.markdown('<div class="section-header">◈ ÉTAT DU RÉSEAU PAR RÉGION</div>',
                unsafe_allow_html=True)
    reg_cols = st.columns(len(REGIONS))
    for i, reg in enumerate(REGIONS):
        reg_meta    = meta[meta["region"] == reg]
        n_clients_r = reg_meta["client_id"].nunique()
        avg_conso   = df[df["region"] == reg]["consumption_kwh"].mean()
        reg_reg     = {k: v for k, v in registry.items() if v.get("region") == reg}
        n_confirmed = sum(1 for v in reg_reg.values() if v.get("is_confirmed"))
        n_suspect   = sum(1 for v in reg_reg.values() if not v.get("is_confirmed"))
        active_reg  = sum(
            1 for c in sel_clients
            if not meta[meta.client_id == c].empty
            and meta.loc[meta.client_id == c, "region"].values[0] == reg
            and st.session_state.client_scores.get(c, 1) < threshold
        )
        card_cls = "has-alert" if active_reg > 0 or n_confirmed > 0 else (
                   "has-warning" if n_suspect > 0 else "")
        dot_cls  = "dot-alert" if active_reg > 0 else (
                   "dot-warn" if n_suspect > 0 else "dot-ok")
        profiles_dist = reg_meta["profile"].value_counts()
        profile_tags  = " ".join(
            f'<span style="font-size:9px;background:#1a3a6e;padding:1px 4px;'
            f'border-radius:3px;color:#8aa8cc">{p[:3]}</span>'
            for p in profiles_dist.index[:3]
        )
        with reg_cols[i]:
            col_av = '#ff3b3b' if n_confirmed>0 else '#00ff9d'
            col_su = '#ffaa00' if n_suspect>0 else '#00ff9d'
            col_ac = '#ff3b3b' if active_reg>0 else '#00ff9d'
            st.markdown(
                f'<div class="region-card {card_cls}">'
                f'<div class="region-name"><span class="dot {dot_cls}"></span>{reg}</div>'
                f'<div class="region-stat"><span>Clients</span><strong>{n_clients_r}</strong></div>'
                f'<div class="region-stat"><span>Conso moy.</span><strong>{avg_conso:.1f} kWh</strong></div>'
                f'<div class="region-stat"><span>Avérés</span><strong style="color:{col_av}">{n_confirmed}</strong></div>'
                f'<div class="region-stat"><span>Suspects</span><strong style="color:{col_su}">{n_suspect}</strong></div>'
                f'<div class="region-stat"><span>Actifs</span><strong style="color:{col_ac}">{active_reg}</strong></div>'
                f'<div style="margin-top:6px">{profile_tags}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">◈ CONSOMMATION PAR RÉGION & PROFIL</div>',
                unsafe_allow_html=True)
    agg     = df.groupby(["region", "profile"])["consumption_kwh"].mean().reset_index()
    fig_reg = px.bar(agg, x="region", y="consumption_kwh", color="profile",
                     barmode="group",
                     color_discrete_sequence=["#00d4ff","#ff3b3b","#00ff9d","#a855f7"],
                     template="plotly_dark",
                     labels={"consumption_kwh":"kWh/j","region":"Région","profile":"Profil"})
    fig_reg.update_layout(paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
                          font_color="#c8d8f0", height=300,
                          margin=dict(l=0,r=0,t=20,b=0),
                          legend=dict(bgcolor="#0d1a35", bordercolor="#1a3a6e"))
    st.plotly_chart(fig_reg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVEILLANCE TEMPS RÉEL (semaine glissante)
# ══════════════════════════════════════════════════════════════════════════════
with tab_surv:
    st.markdown('<div class="section-header">◈ SURVEILLANCE — SEMAINE GLISSANTE</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c_info = st.columns([1, 1, 1, 4])
    with c1:
        start_btn = st.button("▶ DÉMARRER", type="primary", use_container_width=True)
    with c2:
        stop_btn  = st.button("■ ARRÊTER",  use_container_width=True)
    with c3:
        reset_btn = st.button("↺ RESET",    use_container_width=True)
    with c_info:
        status_color = "#ff3b3b" if st.session_state.surveillance_active else "#00ff9d"
        status_lbl   = "EN COURS" if st.session_state.surveillance_active else "EN ATTENTE"
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
            f'color:{status_color};padding:8px 0">'
            f'{status_lbl} | {sel_region or "TOUTES RÉGIONS"} | '
            f'{len(sel_clients)} CLIENTS | '
            f'{"LSTM+IF" if st.session_state.lstm_ready else "IF seul"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if stop_btn:
        st.session_state.surveillance_active = False
        st.rerun()
    if reset_btn:
        st.session_state.client_windows   = {}
        st.session_state.client_scores    = {}
        st.session_state.client_lstm_mse  = {}
        st.session_state.client_detect_by = {}
        st.rerun()
    if start_btn:
        st.session_state.surveillance_active = True
        st.session_state.client_windows   = {}
        st.session_state.client_scores    = {}
        st.session_state.client_lstm_mse  = {}
        st.session_state.client_detect_by = {}
        st.rerun()

    # ── Placeholders UI ───────────────────────────────────────────────────
    feed_ph  = st.empty()
    prog_ph  = st.empty()
    info_ph  = st.empty()

    # Vue par région
    display_regs  = [sel_region] if sel_region else REGIONS
    region_ph     = {}

    for reg in display_regs:
        rc = [c for c in sel_clients
              if not meta[meta.client_id == c].empty
              and meta.loc[meta.client_id == c, "region"].values[0] == reg]
        if not rc:
            continue
        n_al = sum(1 for c in rc
                   if st.session_state.client_scores.get(c, 1) < threshold)
        exp_lbl = f"{'🔴' if n_al > 0 else '🟢'} RÉGION {reg}  ·  {len(rc)} clients  ·  {n_al} alerte(s)"
        with st.expander(exp_lbl, expanded=(n_al > 0)):
            rows_html = ""
            for cid in rc:
                sc    = st.session_state.client_scores.get(cid)
                lmse  = st.session_state.client_lstm_mse.get(cid, 0)
                dby   = st.session_state.client_detect_by.get(cid, "—")
                if sc is None:
                    lvl, lbl = "normal", "EN ATTENTE"; sc_str = "—"
                else:
                    lvl, lbl, _ = combined_level(sc, threshold,
                                                 lmse > (lstm_threshold or 9999))
                    sc_str = f"{sc:.4f}"
                prof = meta.loc[meta.client_id == cid, "profile"].values[0] \
                       if not meta[meta.client_id == cid].empty else "?"
                lstm_tag = (
                    f'<span style="font-size:9px;color:#f97316"> LSTM:{lmse:.3f}</span>'
                    if st.session_state.lstm_ready and lmse > 0 else ""
                )
                rows_html += (
                    f'<div class="client-row {lvl}">'
                    f'<span>{cid}</span>'
                    f'<span style="color:#4a7aaa;font-size:9px">{prof[:3]}</span>'
                    f'<span>{sc_str}{lstm_tag}</span>'
                    f'<span style="font-size:10px">[{dby}] {lbl}</span>'
                    f'</div>'
                )
            region_ph[reg] = st.empty()
            region_ph[reg].markdown(rows_html, unsafe_allow_html=True)

    # ── BOUCLE DE SURVEILLANCE ────────────────────────────────────────────
    if st.session_state.surveillance_active and sel_clients:
        client_data = {
            cid: df[df["client_id"] == cid].sort_values("timestamp")
            for cid in sel_clients
        }
        max_days  = min(max(len(v) for v in client_data.values()),
                        HISTORY_DAYS + SIM_DAYS)
        prog      = prog_ph.progress(0)

        # Pré-charger l'historique silencieusement (jours 0 → HISTORY_DAYS-1)
        for cid in sel_clients:
            cdf = client_data[cid]
            hist_vals = cdf["consumption_kwh"].values[:HISTORY_DAYS]
            st.session_state.client_windows[cid] = list(hist_vals)

        info_ph.markdown(
            '<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
            'color:#4a7aaa">⚙ Historique pré-chargé — démarrage semaine de surveillance...</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.3)

        # Streamer les SIM_DAYS jours de la semaine de surveillance
        for t_offset in range(SIM_DAYS):
            if not st.session_state.surveillance_active:
                break
            t = HISTORY_DAYS + t_offset
            date_label = f"Jour {t+1}"

            # ── Batch : récupérer fenêtres de tous les clients ────────────
            ready_clients = []
            windows_batch = []
            feats_batch   = []

            for cid in sel_clients:
                cdf = client_data[cid]
                if t >= len(cdf):
                    continue
                new_val = cdf.iloc[t]["consumption_kwh"]
                st.session_state.client_windows[cid].append(new_val)
                w = np.array(st.session_state.client_windows[cid][-win_size:])
                if len(w) < win_size:
                    continue
                ready_clients.append(cid)
                windows_batch.append(w)
                feats_batch.append(extract_features(w))

            if not ready_clients:
                continue

            # ── Batch IF scores ───────────────────────────────────────────
            X_batch  = np.array(feats_batch)
            Xs_batch = scaler.transform(X_batch)
            if_scores = iso_forest.decision_function(Xs_batch)

            # ── Batch LSTM scores ─────────────────────────────────────────
            if st.session_state.lstm_ready:
                lstm_mses, lstm_anoms = batch_lstm_score(np.array(windows_batch))
            else:
                lstm_mses  = np.zeros(len(ready_clients))
                lstm_anoms = np.zeros(len(ready_clients), dtype=bool)

            # ── Traitement par client ─────────────────────────────────────
            new_alerts = []
            for idx, cid in enumerate(ready_clients):
                ifs   = float(if_scores[idx])
                lmse  = float(lstm_mses[idx])
                lanom = bool(lstm_anoms[idx])

                st.session_state.client_scores[cid]   = ifs
                st.session_state.client_lstm_mse[cid] = lmse

                lvl, lbl, dby = combined_level(ifs, threshold, lanom)
                st.session_state.client_detect_by[cid] = dby

                if lvl in ("alert", "alert_high"):
                    cdf   = client_data[cid]
                    row   = cdf.iloc[t] if t < len(cdf) else cdf.iloc[-1]
                    jour  = row["timestamp"].strftime("%d/%m/%Y")
                    m     = meta_of(cid)
                    reg_c = m["region"]  if m is not None else "?"
                    pro_c = m["profile"] if m is not None else "?"

                    # Une seule entrée par client par jour
                    prev = registry.get(cid, {})
                    if prev.get("last_seen") != jour or prev.get("detection_count", 0) == 0:
                        reasons   = shap_explain(Xs_batch[[idx]], top_n=5)
                        narrative = build_narrative(reasons, ifs, lmse, dby)
                        rpc       = risk_pct(ifs, threshold, lmse)

                        st.session_state.registry = registry_add_detection(
                            st.session_state.registry,
                            cid, reg_c, pro_c, jour,
                            ifs, lmse if st.session_state.lstm_ready else None,
                            rpc, narrative, reasons, dby,
                        )
                        registry = st.session_state.registry
                        save_registry(registry)  # persistance immédiate

                        new_alerts.append({
                            "cid": cid, "region": reg_c, "score": ifs,
                            "lstm_mse": lmse, "dby": dby, "jour": jour,
                            "confirmed": registry[cid].get("is_confirmed"),
                            "count": registry[cid]["detection_count"],
                        })

            # ── Live feed (dernières alertes) ─────────────────────────────
            if new_alerts:
                feed_html = ""
                for a in reversed(new_alerts[-5:]):
                    badge_cls = (
                        "ticker-badge-both" if a["dby"] == "BOTH" else
                        "ticker-badge-lstm" if a["dby"] == "LSTM" else
                        "ticker-badge"
                    )
                    conf_tag = (
                        ' <span style="color:#ff3b3b;font-weight:700">[AVÉRÉ]</span>'
                        if a["confirmed"] else ""
                    )
                    feed_html += (
                        f'<div class="alert-ticker">'
                        f'<span class="{badge_cls}">{a["dby"]}</span>'
                        f'<span>{a["cid"]} | {a["region"]}{conf_tag}</span>'
                        f'<span class="ticker-meta">'
                        f'IF={a["score"]:.4f}'
                        + (f' | LSTM_MSE={a["lstm_mse"]:.4f}' if st.session_state.lstm_ready else '')
                        + f' | {a["jour"]} | #{a["count"]}'
                        + f'</span></div>'
                    )
                feed_ph.markdown(feed_html, unsafe_allow_html=True)

            # ── Refresh expanders ─────────────────────────────────────────
            for reg in display_regs:
                if reg not in region_ph:
                    continue
                rc = [c for c in sel_clients
                      if not meta[meta.client_id == c].empty
                      and meta.loc[meta.client_id == c, "region"].values[0] == reg]
                rows_html = ""
                for cid in rc:
                    sc   = st.session_state.client_scores.get(cid)
                    lmse = st.session_state.client_lstm_mse.get(cid, 0)
                    dby  = st.session_state.client_detect_by.get(cid, "—")
                    if sc is None:
                        lvl, lbl = "normal", "EN ATTENTE"; sc_str = "—"
                    else:
                        lvl, lbl, _ = combined_level(sc, threshold,
                                                     lmse > (lstm_threshold or 9999))
                        sc_str = f"{sc:.4f}"
                    prof = meta.loc[meta.client_id == cid, "profile"].values[0] \
                           if not meta[meta.client_id == cid].empty else "?"
                    lstm_tag = (
                        f'<span style="font-size:9px;color:#f97316"> {lmse:.3f}</span>'
                        if st.session_state.lstm_ready and lmse > 0 else ""
                    )
                    rows_html += (
                        f'<div class="client-row {lvl}">'
                        f'<span>{cid}</span>'
                        f'<span style="color:#4a7aaa;font-size:9px">{prof[:3]}</span>'
                        f'<span>{sc_str}{lstm_tag}</span>'
                        f'<span style="font-size:9px">[{dby}] {lbl}</span>'
                        f'</div>'
                    )
                region_ph[reg].markdown(rows_html, unsafe_allow_html=True)

            # ── Progress ──────────────────────────────────────────────────
            n_alerts_now = sum(
                1 for c in sel_clients
                if st.session_state.client_scores.get(c, 1) < threshold
            )
            prog.progress((t_offset + 1) / SIM_DAYS)
            info_ph.markdown(
                f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
                f'color:#4a7aaa">SEMAINE J+{t_offset+1}/{SIM_DAYS} | '
                f'🔴 {n_alerts_now} ALERTES ACTIVES | '
                f'📋 {len(registry)} dans registre</div>',
                unsafe_allow_html=True,
            )
            time.sleep(speed)

        st.session_state.surveillance_active = False
        st.success(
            f"✅ Semaine de surveillance terminée — "
            f"{len([v for v in registry.values() if v.get('is_confirmed')])} "
            f"fraudeurs avérés confirmés dans le registre persistant."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FRAUDEURS AVÉRÉS  (registre persistant groupé par client)
# ══════════════════════════════════════════════════════════════════════════════
with tab_confirmed:
    registry = st.session_state.registry
    confirmed = sorted(
        [v for v in registry.values() if v.get("is_confirmed")],
        key=lambda x: -x.get("detection_count", 0),
    )

    st.markdown(
        f'<div class="section-header">◈ FRAUDEURS AVÉRÉS '
        f'({len(confirmed)} clients · ≥{CONFIRM_THRESHOLD} détections)</div>',
        unsafe_allow_html=True,
    )

    if not confirmed:
        st.markdown(
            '<div style="color:#4a7aaa;font-family:Share Tech Mono,monospace;'
            'text-align:center;padding:40px">AUCUN FRAUDEUR AVÉRÉ — '
            f'Un client doit être détecté ≥{CONFIRM_THRESHOLD} fois pour apparaître ici.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Filtres ───────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        with f1:
            f_regions  = st.multiselect("Région", REGIONS,
                                        key="cf_reg", placeholder="Toutes")
        with f2:
            f_profiles = st.multiselect("Profil", PROFILES,
                                        key="cf_prof", placeholder="Tous")
        with f3:
            sort_by = st.selectbox("Trier par",
                                   ["Nb détections ↓", "Risque max ↓", "Dernière détection", "Client ID"],
                                   key="cf_sort")

        filt = confirmed
        if f_regions:
            filt = [x for x in filt if x.get("region") in f_regions]
        if f_profiles:
            filt = [x for x in filt if x.get("profile") in f_profiles]
        if sort_by == "Risque max ↓":
            filt = sorted(filt, key=lambda x: -x.get("max_risk_pct", 0))
        elif sort_by == "Dernière détection":
            filt = sorted(filt, key=lambda x: x.get("last_seen", ""), reverse=True)
        elif sort_by == "Client ID":
            filt = sorted(filt, key=lambda x: x.get("client_id", ""))

        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
            f'color:#4a7aaa;margin-bottom:12px">'
            f'{len(filt)} fraudeurs affichés</div>',
            unsafe_allow_html=True,
        )

        for entry in filt:
            cid        = entry["client_id"]
            n_det      = entry["detection_count"]
            n_both     = entry.get("both_count", 0)
            n_lstm     = entry.get("lstm_count", 0)
            max_risk   = entry.get("max_risk_pct", 0)
            min_score  = entry.get("min_score_if", 0)
            max_lstm   = entry.get("max_lstm_mse", 0)
            first_seen = entry.get("first_seen", "?")
            last_seen  = entry.get("last_seen", "?")
            reasons    = entry.get("shap_reasons", [])
            shap_block = shap_html(reasons)
            risk_color = "#ff3b3b" if max_risk > 70 else ("#ffaa00" if max_risk > 40 else "#00ff9d")

            # Timeline des détections (compacte)
            det_timeline = ""
            for d in entry.get("detections", [])[-5:]:
                dbadge_col = (
                    "#f97316" if d.get("detected_by") == "BOTH" else
                    "#ffaa00" if d.get("detected_by") == "LSTM" else "#ff3b3b"
                )
                det_timeline += (
                    f'<span style="font-size:9px;background:#1a0820;border:1px solid {dbadge_col};'
                    f'color:{dbadge_col};border-radius:3px;padding:1px 5px;margin-right:4px">'
                    f'{d["date"]} [{d.get("detected_by","?")}]</span>'
                )

            lstm_section = ""
            if st.session_state.lstm_ready and max_lstm > 0:
                lstm_section = (
                    f'<div style="font-size:10px;color:#f97316;margin-top:4px">'
                    f'🟠 MSE LSTM max : <span style="font-family:Share Tech Mono,monospace">{max_lstm:.4f}</span>'
                    f'&nbsp;|&nbsp; Détections doubles (IF+LSTM) : <b>{n_both}</b>'
                    f'</div>'
                )

            badge_lstm = f'<span class="detection-badge detection-badge-lstm">{n_lstm} LSTM</span>' if n_lstm > 0 else ''
            badge_both = f'<span class="detection-badge" style="background:#2a0840;border-color:#9a2060;color:#f97316">{n_both} doubles</span>' if n_both > 0 else ''
            st.markdown(
                f'<div class="fraud-card-confirmed">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div>'
                f'<div class="fraud-id">⚠ {cid} <span class="confirmed-badge" style="margin-left:8px">AVÉRÉ</span></div>'
                f'<div class="fraud-meta">Région : <b>{entry.get("region","?")}</b> &nbsp;|&nbsp; '
                f'Profil : <b>{entry.get("profile","?")}</b> &nbsp;|&nbsp; '
                f'1ère det. : {first_seen} &nbsp;|&nbsp; Dernière : {last_seen}</div>'
                f'<div style="margin:4px 0"><span class="detection-badge">{n_det} détections IF</span>'
                f'{badge_lstm}{badge_both}</div>'
                f'<div style="margin-top:6px">{det_timeline}</div>'
                f'{lstm_section}'
                f'</div>'
                f'<div style="text-align:right">'
                f'<div style="font-family:Share Tech Mono,monospace;font-size:24px;color:{risk_color}">{max_risk:.0f}%</div>'
                f'<div style="font-size:9px;color:#4a7aaa">RISQUE MAX</div>'
                f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#8aa8cc;margin-top:4px">IF min : {min_score:.4f}</div>'
                f'</div>'
                f'</div>'
                f'<div style="margin-top:12px">'
                f'<div style="font-size:9px;color:#a855f7;letter-spacing:1px;margin-bottom:5px">🧠 SHAP — PIRE DÉTECTION</div>'
                f'{shap_block}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Tableau récap
        st.markdown('<div class="section-header">◈ TABLEAU RÉCAPITULATIF</div>',
                    unsafe_allow_html=True)
        recap_rows = [
            {
                "Client":        v["client_id"],
                "Région":        v.get("region", "?"),
                "Profil":        v.get("profile", "?"),
                "Détections":    v["detection_count"],
                "Double IF+LSTM":v.get("both_count", 0),
                "Risque max %":  v.get("max_risk_pct", 0),
                "Score IF min":  v.get("min_score_if", 0),
                "Dernière det.": v.get("last_seen", "?"),
            }
            for v in filt
        ]
        if recap_rows:
            rdf = pd.DataFrame(recap_rows)
            st.dataframe(
                rdf, use_container_width=True, hide_index=True,
                column_config={
                    "Risque max %": st.column_config.ProgressColumn(min_value=0, max_value=100),
                    "Score IF min": st.column_config.NumberColumn(format="%.4f"),
                },
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SUSPECTS  (< CONFIRM_THRESHOLD détections)
# ══════════════════════════════════════════════════════════════════════════════
with tab_suspects:
    registry = st.session_state.registry
    suspects = sorted(
        [v for v in registry.values() if not v.get("is_confirmed")],
        key=lambda x: -x.get("max_risk_pct", 0),
    )

    st.markdown(
        f'<div class="section-header">◈ SUSPECTS ACTIFS '
        f'({len(suspects)} clients · < {CONFIRM_THRESHOLD} détections)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:11px;color:#4a7aaa;margin-bottom:12px">'
        f'Ces clients ont déclenché au moins une alerte mais pas encore '
        f'le seuil de confirmation ({CONFIRM_THRESHOLD} détections). Ils restent sous surveillance.</div>',
        unsafe_allow_html=True,
    )

    if not suspects:
        st.markdown(
            '<div style="color:#4a7aaa;font-family:Share Tech Mono,monospace;'
            'text-align:center;padding:30px">AUCUN SUSPECT ENREGISTRÉ</div>',
            unsafe_allow_html=True,
        )
    else:
        for entry in suspects[:30]:
            cid      = entry["client_id"]
            n_det    = entry["detection_count"]
            max_risk = entry.get("max_risk_pct", 0)
            reasons  = entry.get("shap_reasons", [])
            risk_col = "#ffaa00" if max_risk > 40 else "#8aa8cc"
            det_str  = " ".join(
                f'<span style="font-size:9px;background:#1a1800;border:1px solid #4a3a00;'
                f'color:#ffaa00;border-radius:3px;padding:1px 5px;margin-right:3px">'
                f'{d["date"]}</span>'
                for d in entry.get("detections", [])[-3:]
            )
            st.markdown(
                f'<div class="fraud-card-suspect">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div>'
                f'<div class="suspect-id">⚡ {cid}</div>'
                f'<div class="fraud-meta">{entry.get("region","?")} | {entry.get("profile","?")} | 1ère det. {entry.get("first_seen","?")}</div>'
                f'<div>{det_str}</div>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<div style="font-family:Share Tech Mono,monospace;font-size:18px;color:{risk_col}">{max_risk:.0f}%</div>'
                f'<div style="font-size:9px;color:#4a7aaa">{n_det}/{CONFIRM_THRESHOLD} det.</div>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown('<div class="section-header">◈ ANALYTICS</div>', unsafe_allow_html=True)

    registry = st.session_state.registry
    if not registry:
        st.info("Aucune donnée. Lancez une surveillance.")
    else:
        all_entries = list(registry.values())
        ch1, ch2 = st.columns(2)

        with ch1:
            # Top fraudeurs par nb détections
            top_data = sorted(all_entries, key=lambda x: -x.get("detection_count", 0))[:15]
            fig_top = go.Figure(go.Bar(
                x=[v["detection_count"] for v in top_data],
                y=[v["client_id"] for v in top_data],
                orientation="h",
                marker_color=["#ff3b3b" if v.get("is_confirmed") else "#ffaa00"
                              for v in top_data],
                text=[f"{'✓' if v.get('is_confirmed') else '?'}" for v in top_data],
            ))
            fig_top.update_layout(
                title="Top clients — nb détections (rouge=avéré)",
                template="plotly_dark", paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
                font_color="#c8d8f0", height=350, margin=dict(l=0,r=0,t=30,b=0),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with ch2:
            # Distribution scores IF
            all_scores = [v.get("min_score_if", 0) for v in all_entries if v.get("min_score_if", 9999) < 9]
            if all_scores:
                fig_sc = go.Figure(go.Histogram(
                    x=all_scores, nbinsx=25, marker_color="#00d4ff", opacity=0.8,
                ))
                fig_sc.add_vline(x=threshold, line_color="#ff3b3b", line_dash="dash",
                                 annotation_text=f"Seuil {threshold:.3f}",
                                 annotation_font_color="#ff3b3b")
                fig_sc.update_layout(
                    title="Distribution scores IF (pires par client)",
                    template="plotly_dark", paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
                    font_color="#c8d8f0", height=350, margin=dict(l=0,r=0,t=30,b=0),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        ch3, ch4 = st.columns(2)
        with ch3:
            reg_alert = pd.Series([v.get("region","?") for v in all_entries]).value_counts().reset_index()
            reg_alert.columns = ["region","count"]
            fig_ra = px.bar(reg_alert, x="region", y="count", color="count",
                            color_continuous_scale="Reds", template="plotly_dark",
                            title="Clients dans registre par région")
            fig_ra.update_layout(paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
                                 font_color="#c8d8f0", height=280, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_ra, use_container_width=True)

        with ch4:
            # Répartition IF vs LSTM vs BOTH
            if_only  = sum(1 for v in all_entries if v.get("if_count",0)>0 and v.get("both_count",0)==0)
            lstm_only= sum(1 for v in all_entries if v.get("lstm_count",0)>0 and v.get("both_count",0)==0)
            both_c   = sum(1 for v in all_entries if v.get("both_count",0)>0)
            fig_pie  = go.Figure(go.Pie(
                labels=["IF seul", "LSTM seul", "IF+LSTM"],
                values=[if_only, lstm_only, both_c],
                marker_colors=["#00d4ff", "#f97316", "#ff3b3b"],
                hole=0.4,
            ))
            fig_pie.update_layout(
                title="Méthode de détection",
                template="plotly_dark", paper_bgcolor="#0d1a35",
                font_color="#c8d8f0", height=280, margin=dict(l=0,r=0,t=30,b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Analyse client individuel
    st.markdown('<div class="section-header">◈ PROFIL CLIENT INDIVIDUEL</div>',
                unsafe_allow_html=True)
    sel_client = st.selectbox("Sélectionner un client", clients, key="ind_client")
    if sel_client:
        c_df  = df[df["client_id"] == sel_client].sort_values("timestamp")
        c_reg = registry.get(sel_client, {})
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Profil",      meta_of(sel_client)["profile"] if meta_of(sel_client) is not None else "?")
        m2.metric("Région",      meta_of(sel_client)["region"]  if meta_of(sel_client) is not None else "?")
        m3.metric("Conso moy.", f"{c_df['consumption_kwh'].mean():.1f} kWh")
        m4.metric("Dans registre", f"{'✓ Avéré' if c_reg.get('is_confirmed') else ('⚡ Suspect' if c_reg else 'Non')}")
        fig_c = go.Figure(go.Scatter(
            x=c_df["timestamp"], y=c_df["consumption_kwh"],
            mode="lines", line=dict(color="#00d4ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ))
        fig_c.add_hline(y=0.5, line_color="#ff3b3b", line_dash="dot",
                        annotation_text="Seuil faible conso")
        fig_c.update_layout(
            title=f"Consommation — {sel_client}",
            template="plotly_dark", paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
            font_color="#c8d8f0", height=300, margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODÈLES (pédagogique)
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown('<div class="section-header">◈ ARCHITECTURE DÉTECTION HYBRIDE</div>',
                unsafe_allow_html=True)

    lstm_status_html = (
        f'<span style="color:#f97316;font-weight:700">ACTIF</span> — '
        f'Seuil MSE : <code style="color:#f97316">{lstm_threshold:.4f}</code>'
        if st.session_state.lstm_ready and lstm_threshold
        else '<span style="color:#ff3b3b">INACTIF</span> (exécuter train.py avec TensorFlow)'
    )

    _rf  = REGISTRY_FILE
    _win = WINDOW_SIZE
    _thr = f"{threshold:.3f}"
    _ct  = CONFIRM_THRESHOLD
    st.markdown(
        f'<div style="background:#0d1a35;border:1px solid #1a3a6e;border-radius:10px;'
        f'padding:20px;font-size:13px;color:#c8d8f0;margin-bottom:16px">'

        f'<h4 style="color:#00d4ff;font-family:Share Tech Mono;letter-spacing:2px">ISOLATION FOREST (IF)</h4>'
        f'<p>Détecte les outliers dans l\'espace des 12 features statistiques. '
        f'Score ∈ [-0.5, 0.5] : plus négatif = plus anormal. '
        f'Seuil actuel : <b style="color:#ff3b3b;font-family:Share Tech Mono">{_thr}</b></p>'

        f'<h4 style="color:#f97316;font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">'
        f'LSTM AUTOENCODER — {lstm_status_html}</h4>'
        f'<p>Entraîné uniquement sur des clients normaux. Reconstruction d\'une fenêtre '
        f'de {_win} jours (z-score par client). '
        f'Pattern frauduleux → <b>MSE élevée</b> = anomalie temporelle.</p>'
        f'<pre style="background:#080d1a;border:1px solid #1a3a6e;padding:10px;'
        f'font-size:11px;color:#a855f7;border-radius:6px">'
        f'Input (30,1) -&gt; LSTM(64) -&gt; LSTM(32) -&gt; [latent 32D]\n'
        f'                                    |\n'
        f'Output(30,1) &lt;- Dense(1) &lt;- LSTM(64) &lt;- LSTM(32) &lt;- RepeatVector</pre>'

        f'<h4 style="color:#00ff9d;font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">SCORE HYBRIDE</h4>'
        f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
        f'<tr style="background:#1a3a6e;color:#00d4ff;font-family:Share Tech Mono">'
        f'<th style="padding:7px">Combinaison</th><th style="padding:7px">Niveau</th><th style="padding:7px">Risque %</th></tr>'
        f'<tr style="background:#0a1628"><td style="padding:6px">IF + LSTM anomalie</td>'
        f'<td style="padding:6px;color:#ff3b3b">ALERTE DOUBLE</td><td style="padding:6px">60% IF + 40% LSTM</td></tr>'
        f'<tr style="background:#0d1a35"><td style="padding:6px">IF seul</td>'
        f'<td style="padding:6px;color:#ff6b6b">ALERTE IF</td><td style="padding:6px">Basé IF</td></tr>'
        f'<tr style="background:#0a1628"><td style="padding:6px">LSTM seul</td>'
        f'<td style="padding:6px;color:#ffaa00">SUSPECT LSTM</td><td style="padding:6px">Basé LSTM</td></tr>'
        f'<tr style="background:#0d1a35"><td style="padding:6px">Zone grise IF</td>'
        f'<td style="padding:6px;color:#ffaa00">SURVEILLANCE</td><td style="padding:6px">Faible</td></tr>'
        f'</table>'

        f'<h4 style="color:#a855f7;font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">REGISTRE PERSISTANT</h4>'
        f'<p>Fichier <code style="color:#a855f7">{_rf}</code> sur disque. '
        f'Un client devient <b style="color:#ff3b3b">AVÉRÉ</b> à {_ct} détections. '
        f'Le registre survit aux redémarrages.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Table features
    st.markdown('<div class="section-header">◈ FEATURES ISOLATION FOREST</div>',
                unsafe_allow_html=True)
    feat_info = [
        ("Conso moyenne",            "mean(fenêtre)",            "Très faible vs profil"),
        ("Volatilité (std)",          "std(fenêtre)",             "Anormalement élevée ou nulle"),
        ("Tendance linéaire",         "polyfit deg.1",            "Chute brutale persistante"),
        ("Variation journalière moy", "mean(diff)",               "Sauts brusques répétés"),
        ("Ratio 1ère/2ème période",   "mean[:15] / mean[15:]",    "Chute soudaine en 2ème moitié"),
        ("Amplitude max-min",         "max - min",                "Très faible = plateau anormal"),
        ("Coefficient de variation",  "std / mean",               "CV élevé = erratique"),
        ("Jours très faible conso",   "count(< 0.5 kWh)",         "Beaucoup de jours quasi-nuls"),
        ("Ratio Weekend/Weekday",     "mean_wk / mean_wd",        "Pas de variation typique"),
        ("Volatilité Wk/Wd",          "std_wk / std_wd",          "Asymétrie anormale"),
        ("Autocorrélation lag-1",     "corr(s[:-1], s[1:])",      "Faible = comportement aléatoire"),
        ("Instabilité directionnelle","count(changements dir.)",   "Élevé = oscillations suspectes"),
    ]
    table_html = """
    <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead><tr style="background:#1a3a6e;color:#00d4ff;font-family:Share Tech Mono">
      <th style="padding:7px;text-align:left">Feature</th>
      <th style="padding:7px;text-align:left">Calcul</th>
      <th style="padding:7px;text-align:left">Signal fraude si...</th>
    </tr></thead><tbody>"""
    for i,(name,calc,sig) in enumerate(feat_info):
        bg = "#0a1628" if i%2==0 else "#0d1a35"
        table_html += (
            f'<tr style="background:{bg};color:#c8d8f0">'
            f'<td style="padding:5px 8px;color:#a855f7;font-family:Share Tech Mono">{name}</td>'
            f'<td style="padding:5px 8px;color:#8aa8cc;font-family:monospace">{calc}</td>'
            f'<td style="padding:5px 8px;color:#ffaa00">{sig}</td></tr>'
        )
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
