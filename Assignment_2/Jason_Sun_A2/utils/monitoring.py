import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

try:
    from utils.training import TRAIN_CUTOFF
except Exception:
    TRAIN_CUTOFF = "2024-01-01"

PSI_BINS = 10


def _ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    pos = np.sort(proba[y_true == 1]); neg = np.sort(proba[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    eps = 1e-6
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected) + eps
    act_pct = np.histogram(actual,   bins=edges)[0] / len(actual)   + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _baseline_scores(gold_predictions_dir: str) -> np.ndarray:
    cut = pd.to_datetime(TRAIN_CUTOFF); scores = []
    for f in sorted(glob.glob(os.path.join(gold_predictions_dir, "gold_predictions_*.parquet"))):
        df = pd.read_parquet(f, columns=["snapshot_date", "predicted_proba"])
        if pd.to_datetime(df["snapshot_date"].iloc[0]) <= cut:
            scores.append(df["predicted_proba"].to_numpy())
    return np.concatenate(scores) if scores else np.array([])


def _baseline_features(gold_feature_dir: str, numeric_cols: list) -> pd.DataFrame:
    cut = pd.to_datetime(TRAIN_CUTOFF); frames = []
    for f in sorted(glob.glob(os.path.join(gold_feature_dir, "gold_feature_store_*.parquet"))):
        df = pd.read_parquet(f)
        if pd.to_datetime(df["snapshot_date"].iloc[0]) <= cut:
            cols = [c for c in numeric_cols if c in df.columns]
            frames.append(df[cols])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=numeric_cols)


def _feature_csi(baseline_df: pd.DataFrame, current_df: pd.DataFrame, numeric_cols: list):
    csis = {}
    for c in numeric_cols:
        if c not in baseline_df.columns or c not in current_df.columns:
            continue
        b = baseline_df[c].dropna().to_numpy(); a = current_df[c].dropna().to_numpy()
        if len(b) < 50 or len(a) < 10:
            continue
        v = _psi(b, a)
        if v == v:
            csis[c] = v
    if not csis:
        return float("nan"), float("nan"), float("nan"), None
    top = max(csis, key=csis.get)
    vals = np.array(list(csis.values()))
    return float(vals.max()), float(vals.mean()), float(np.median(vals)), top


def _load_labels(gold_label_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(gold_label_dir, "*")))
    if not files:
        return pd.DataFrame(columns=["loan_id", "label"])
    labels = pd.read_parquet(gold_label_dir)
    return labels[["loan_id", "label"]].drop_duplicates("loan_id")


def monitor(snapshot_date_str: str, gold_predictions_dir: str, gold_label_dir: str,
            gold_monitoring_dir: str, gold_feature_dir: str = None,
            model_bank_dir: str = None):
    date_suffix = snapshot_date_str.replace("-", "_")
    pred_path = os.path.join(gold_predictions_dir, f"gold_predictions_{date_suffix}.parquet")
    if not os.path.exists(pred_path):
        print(f"[monitoring] no predictions for {snapshot_date_str} - skipping")
        return None

    preds = pd.read_parquet(pred_path)
    labels = _load_labels(gold_label_dir)
    merged = preds.merge(labels, on="loan_id", how="left")
    has_label = merged["label"].notna()
    n_labelled = int(has_label.sum())

    row = {
        "snapshot_date":        snapshot_date_str,
        "model_run_id":         preds["model_run_id"].iloc[0],
        "model_name":           preds["model_name"].iloc[0],
        "n_scored":             int(len(preds)),
        "n_labelled":           n_labelled,
        "pred_positive_rate":   float(preds["predicted_label"].mean()),
        "mean_predicted_proba": float(preds["predicted_proba"].mean()),
    }

    if n_labelled > 0:
        y = merged.loc[has_label, "label"].astype(int).to_numpy()
        p = merged.loc[has_label, "predicted_proba"].to_numpy()
        yhat = merged.loc[has_label, "predicted_label"].astype(int).to_numpy()
        both = len(np.unique(y)) == 2
        row.update({
            "actual_default_rate": float(np.mean(y)),
            "auc":               float(roc_auc_score(y, p)) if both else float("nan"),
            "average_precision": float(average_precision_score(y, p)) if both else float("nan"),
            "ks":                _ks_statistic(y, p) if both else float("nan"),
            "precision":         float(precision_score(y, yhat, zero_division=0)),
            "recall":            float(recall_score(y, yhat, zero_division=0)),
            "f1":                float(f1_score(y, yhat, zero_division=0)),
        })
    else:
        for k in ["actual_default_rate", "auc", "average_precision", "ks",
                  "precision", "recall", "f1"]:
            row[k] = float("nan")

    baseline = _baseline_scores(gold_predictions_dir)
    row["psi_score"] = _psi(baseline, preds["predicted_proba"].to_numpy())

    csi_max = csi_mean = csi_med = float("nan"); csi_top = None
    if gold_feature_dir and model_bank_dir:
        man_path = os.path.join(model_bank_dir, "latest_manifest.json")
        feat_path = os.path.join(gold_feature_dir, f"gold_feature_store_{date_suffix}.parquet")
        if os.path.exists(man_path) and os.path.exists(feat_path):
            num_cols = json.load(open(man_path)).get("numeric_columns", [])
            if num_cols:
                cur = pd.read_parquet(feat_path)
                base = _baseline_features(gold_feature_dir, num_cols)
                if len(base):
                    csi_max, csi_mean, csi_med, csi_top = _feature_csi(base, cur, num_cols)
    row["csi_max"] = csi_max
    row["csi_mean"] = csi_mean
    row["csi_median"] = csi_med
    row["csi_top_feature"] = csi_top

    row["monitoring_ts"] = datetime.utcnow().isoformat(timespec="seconds")

    os.makedirs(gold_monitoring_dir, exist_ok=True)
    out_path = os.path.join(gold_monitoring_dir, f"gold_monitoring_{date_suffix}.parquet")
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    auc_str = f"{row['auc']:.4f}" if row["auc"] == row["auc"] else "n/a"
    csi_str = f"{csi_max:.3f}" if csi_max == csi_max else "n/a"
    print(f"[monitoring] {snapshot_date_str}: n={row['n_scored']} labelled={n_labelled} "
          f"auc={auc_str} psi={row['psi_score']:.3f} csi_max={csi_str} -> {out_path}")
    return row


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-date", required=True)
    p.add_argument("--gold-predictions-dir", default="datamart/gold/predictions")
    p.add_argument("--gold-label-dir",       default="datamart/gold/label_store")
    p.add_argument("--gold-monitoring-dir",  default="datamart/gold/monitoring")
    p.add_argument("--gold-feature-dir",     default="datamart/gold/feature_store")
    p.add_argument("--model-bank-dir",       default="model_bank")
    args = p.parse_args()
    monitor(args.snapshot_date, args.gold_predictions_dir, args.gold_label_dir,
            args.gold_monitoring_dir, args.gold_feature_dir, args.model_bank_dir)
