from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from dataset import load_data, prepare_binary_data
from model import CNN1D, Autoencoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def summarize(y_true, y_pred, y_score=None):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            payload["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return payload

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

def find_best_threshold_with_recall(y_true, scores, min_recall=0.40):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if rec >= min_recall and f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    if best_f1 < 0:
        for t in thresholds:
            pred = (scores >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(t)
    return best_t, best_f1

def tune_alpha_and_threshold(y_val, rf_val_prob, ae_val_score, min_recall=0.40):
    alphas = np.linspace(0.1, 0.9, 9)
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"alpha": 0.7, "threshold": 0.5, "f1": -1.0}
    for alpha in alphas:
        fusion = alpha * normalize_scores(rf_val_prob) + (1.0 - alpha) * ae_val_score
        for t in thresholds:
            pred = (fusion >= t).astype(int)
            rec = recall_score(y_val, pred, zero_division=0)
            f1 = f1_score(y_val, pred, zero_division=0)
            if rec >= min_recall and f1 > best["f1"]:
                best = {"alpha": float(alpha), "threshold": float(t), "f1": float(f1)}
    if best["f1"] < 0:
        for alpha in alphas:
            fusion = alpha * normalize_scores(rf_val_prob) + (1.0 - alpha) * ae_val_score
            for t in thresholds:
                pred = (fusion >= t).astype(int)
                f1 = f1_score(y_val, pred, zero_division=0)
                if f1 > best["f1"]:
                    best = {"alpha": float(alpha), "threshold": float(t), "f1": float(f1)}
    return best

def train_ml(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000, random_state=17)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    rf = RandomForestClassifier(n_estimators=200, random_state=17, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    return {
        "logistic_regression": summarize(y_test, lr_pred, lr_prob),
        "random_forest": summarize(y_test, rf_pred, rf_prob),
    }

def train_cnn(X_train, y_train, X_test, y_test):
    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=512, shuffle=True)
    model = CNN1D(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test_t)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

    torch.save(model.state_dict(), MODELS_DIR / "cnn_1d.pt")
    return {"cnn_1d": summarize(y_test, pred, prob)}

def train_autoencoder(X_train, y_train, X_test, y_test):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=17
    )
    normal_tr = X_tr[y_tr == 0]
    x_normal_t = torch.tensor(normal_tr, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x_normal_t, x_normal_t), batch_size=512, shuffle=True)

    model = Autoencoder(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(30):
        for xb, target in train_loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_recon = model(val_tensor)
        val_err = ((val_recon - val_tensor) ** 2).mean(dim=1).cpu().numpy()
        val_err_log = np.log1p(val_err)
        score_min = float(val_err_log.min())
        score_max = float(val_err_log.max())
        val_score = apply_fixed_normalization(val_err_log, score_min, score_max)
        threshold, best_val_f1 = find_best_threshold_with_recall(y_val, val_score, min_recall=0.40)

        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_recon = model(test_tensor)
        test_err = ((test_recon - test_tensor) ** 2).mean(dim=1).cpu().numpy()
        test_err_log = np.log1p(test_err)
        test_score = apply_fixed_normalization(test_err_log, score_min, score_max)
        pred = (test_score >= threshold).astype(int)

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    cfg = {
        "threshold": float(threshold),
        "score_min": score_min,
        "score_max": score_max,
        "best_val_f1": float(best_val_f1)
    }
    (MODELS_DIR / "autoencoder_config.json").write_text(json.dumps(cfg, indent=2))
    payload = summarize(y_test, pred, test_score)
    payload.update(cfg)
    return {"autoencoder": payload}

def hybrid_fusion_train(X_train, y_train, X_test, y_test):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=17
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=17, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    normal_tr = X_tr[y_tr == 0]
    ae = Autoencoder(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(normal_tr, dtype=torch.float32), torch.tensor(normal_tr, dtype=torch.float32)),
        batch_size=512,
        shuffle=True,
    )

    ae.train()
    for _ in range(30):
        for xb, target in loader:
            optimizer.zero_grad()
            recon = ae(xb)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        rf_val_prob = rf.predict_proba(X_val)[:, 1]
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        ae_val_recon = ae(val_tensor)
        ae_val_err = ((ae_val_recon - val_tensor) ** 2).mean(dim=1).cpu().numpy()
        ae_val_log = np.log1p(ae_val_err)
        score_min = float(ae_val_log.min())
        score_max = float(ae_val_log.max())
        ae_val_score = apply_fixed_normalization(ae_val_log, score_min, score_max)
        best = tune_alpha_and_threshold(y_val, rf_val_prob, ae_val_score, min_recall=0.40)

        rf_test_prob = rf.predict_proba(X_test)[:, 1]
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        ae_test_recon = ae(test_tensor)
        ae_test_err = ((ae_test_recon - test_tensor) ** 2).mean(dim=1).cpu().numpy()
        ae_test_log = np.log1p(ae_test_err)
        ae_test_score = apply_fixed_normalization(ae_test_log, score_min, score_max)

    fusion_score = best["alpha"] * normalize_scores(rf_test_prob) + (1.0 - best["alpha"]) * ae_test_score
    fusion_pred = (fusion_score >= best["threshold"]).astype(int)

    joblib.dump(rf, MODELS_DIR / "hybrid_random_forest.joblib")
    torch.save(ae.state_dict(), MODELS_DIR / "hybrid_autoencoder.pt")
    best.update({"score_min": score_min, "score_max": score_max})
    (MODELS_DIR / "hybrid_fusion_config.json").write_text(json.dumps(best, indent=2))

    return {
        "hybrid_fusion": summarize(y_test, fusion_pred, fusion_score),
        "fusion_config": best,
    }

def main():
    train_df, test_df = load_data(DATA_DIR / "UNSW_NB15_training-set.csv", DATA_DIR / "UNSW_NB15_testing-set.csv")
    X_train, y_train, X_test, y_test, preprocessor, meta = prepare_binary_data(train_df, test_df)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    ml_metrics = {"metadata": meta, **train_ml(X_train, y_train, X_test, y_test)}
    (MODELS_DIR / "ml_metrics.json").write_text(json.dumps(ml_metrics, indent=2))

    cnn_metrics = {"metadata": meta, **train_cnn(X_train, y_train, X_test, y_test)}
    (MODELS_DIR / "cnn_metrics.json").write_text(json.dumps(cnn_metrics, indent=2))

    ae_metrics = {"metadata": meta, **train_autoencoder(X_train, y_train, X_test, y_test)}
    (MODELS_DIR / "autoencoder_metrics.json").write_text(json.dumps(ae_metrics, indent=2))

    hybrid_metrics = {"metadata": meta, **hybrid_fusion_train(X_train, y_train, X_test, y_test)}
    (MODELS_DIR / "hybrid_metrics.json").write_text(json.dumps(hybrid_metrics, indent=2))

    print("Training complete. Metrics saved in models/")

if __name__ == "__main__":
    main()
