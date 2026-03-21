from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch

from dataset import transform_for_inference
from model import Autoencoder

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

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--model", choices=["random_forest", "logistic_regression", "hybrid_fusion"], default="hybrid_fusion")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    X = transform_for_inference(df, preprocessor)
    out_df = df.copy()

    if args.model in ["random_forest", "logistic_regression"]:
        model = joblib.load(MODELS_DIR / f"{args.model}.joblib")
        if hasattr(model, "predict_proba"):
            out_df["score"] = model.predict_proba(X)[:, 1]
            out_df["prediction"] = (out_df["score"] >= 0.5).astype(int)
        else:
            out_df["prediction"] = model.predict(X)
    else:
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
        out_df["supervised_score"] = rf_prob
        out_df["anomaly_score"] = anomaly_score
        out_df["fusion_score"] = cfg["alpha"] * normalize_scores(rf_prob) + (1.0 - cfg["alpha"]) * anomaly_score
        out_df["prediction"] = (out_df["fusion_score"] >= cfg["threshold"]).astype(int)

    out_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
