from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataset import transform_for_inference
from model import Autoencoder

st.set_page_config(page_title="Hybrid Fusion NIDS Demo", layout="wide")
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    mn = float(scores.min())
    mx = float(scores.max())
    denom = (mx - mn) if (mx - mn) > 1e-12 else 1.0
    return np.clip((scores - mn) / denom, 0.0, 1.0)

def apply_fixed_normalization(x, score_min, score_max):
    x = np.asarray(x, dtype=float)
    denom = (score_max - score_min) if (score_max - score_min) > 1e-12 else 1.0
    return np.clip((x - score_min) / denom, 0.0, 1.0)

st.title("Hybrid Fusion Network Intrusion Detection Demo")
st.caption("This demo combines supervised Random Forest classification with fixed-normalized Dense Autoencoder anomaly scoring.")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
default_path = ROOT / "data" / "UNSW_NB15_testing-set.csv"
use_default = st.checkbox("Use bundled UNSW-NB15 testing file", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_default and default_path.exists():
    df = pd.read_csv(default_path)
else:
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

needed = [
    MODELS_DIR / "preprocessor.joblib",
    MODELS_DIR / "hybrid_random_forest.joblib",
    MODELS_DIR / "hybrid_autoencoder.pt",
    MODELS_DIR / "hybrid_fusion_config.json",
]
if not all(p.exists() for p in needed):
    st.warning("Hybrid artifacts are missing. Run `python train.py` first.")
    st.stop()

preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
X = transform_for_inference(df, preprocessor)

rf = joblib.load(MODELS_DIR / "hybrid_random_forest.joblib")
cfg = json.loads((MODELS_DIR / "hybrid_fusion_config.json").read_text())
rf_prob = rf.predict_proba(X)[:, 1]

ae = Autoencoder(input_dim=X.shape[1])
ae.load_state_dict(torch.load(MODELS_DIR / "hybrid_autoencoder.pt", map_location="cpu"))
ae.eval()
with torch.no_grad():
    x_t = torch.tensor(X, dtype=torch.float32)
    recon = ae(x_t)
    ae_err = ((recon - x_t) ** 2).mean(dim=1).cpu().numpy()

anomaly_score = apply_fixed_normalization(np.log1p(ae_err), cfg["score_min"], cfg["score_max"])
fusion = cfg["alpha"] * normalize_scores(rf_prob) + (1.0 - cfg["alpha"]) * anomaly_score
pred = (fusion >= cfg["threshold"]).astype(int)

result_df = df.copy()
result_df["supervised_score"] = rf_prob
result_df["anomaly_score"] = anomaly_score
result_df["fusion_score"] = fusion
result_df["prediction"] = pred

st.subheader("Scored Output")
st.dataframe(result_df.head(100), use_container_width=True)
st.download_button(
    "Download scored predictions",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="hybrid_fusion_predictions.csv",
    mime="text/csv",
)

metrics_file = MODELS_DIR / "hybrid_metrics.json"
if metrics_file.exists():
    st.subheader("Hybrid model metrics")
    st.json(json.loads(metrics_file.read_text()))
