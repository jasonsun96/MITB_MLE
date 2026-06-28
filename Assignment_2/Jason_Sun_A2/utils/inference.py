"""
Inference stage of the A2 ML pipeline.

Loads the best model from the model_bank (via latest_manifest.json), scores the
gold feature_store rows for a given snapshot_date, and writes per-loan
predictions to a gold predictions table partitioned by snapshot_date.

The model is the single source of truth for the feature schema: we read the
raw feature columns the manifest recorded and let the model's own sklearn
Pipeline do all imputation / scaling / encoding. Nothing about the input data
is re-fit at inference time, so scoring is deterministic and leakage-free.
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd

DEFAULT_THRESHOLD = 0.5


def _load_manifest(model_bank_dir: str) -> dict:
    path = os.path.join(model_bank_dir, "latest_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no latest_manifest.json in {model_bank_dir} - train a model first")
    with open(path) as f:
        return json.load(f)


def _load_best_model(manifest: dict, model_bank_dir: str):
    run_dir = manifest.get("run_dir") or os.path.join(
        model_bank_dir, manifest["training_run_id"])
    artefact = os.path.join(run_dir, manifest["best_model_artefact"])
    if not os.path.exists(artefact):
        raise FileNotFoundError(f"best model artefact missing: {artefact}")
    print(f"[inference] loading best model '{manifest['best_model']}' from {artefact}")
    return joblib.load(artefact)


def infer(snapshot_date_str: str, gold_feature_dir: str, model_bank_dir: str,
          gold_predictions_dir: str, threshold: float = DEFAULT_THRESHOLD):
    """
    Score one snapshot's feature store with the current best model.

    Args:
        snapshot_date_str     "YYYY-MM-DD"
        gold_feature_dir      datamart/gold/feature_store
        model_bank_dir        model_bank
        gold_predictions_dir  datamart/gold/predictions  (written here)
        threshold             prob cut for the hard 0/1 prediction
    """
    date_suffix = snapshot_date_str.replace("-", "_")
    feat_path = os.path.join(gold_feature_dir, f"gold_feature_store_{date_suffix}.parquet")
    if not os.path.exists(feat_path):
        print(f"[inference] no feature_store partition for {snapshot_date_str} - skipping")
        return None

    manifest = _load_manifest(model_bank_dir)
    model = _load_best_model(manifest, model_bank_dir)
    feature_cols = manifest["feature_columns"]

    features = pd.read_parquet(feat_path)
    if len(features) == 0:
        print(f"[inference] empty feature_store for {snapshot_date_str} - skipping")
        return None

    # Reindex to the exact training feature schema. Any column the model expects
    # but that is absent in this partition becomes NaN and is imputed by the
    # model's own pipeline; any extra columns are dropped.
    X = features.reindex(columns=feature_cols)

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "loan_id":       features["loan_id"].values,
        "Customer_ID":   features["Customer_ID"].values,
        "snapshot_date": pd.to_datetime(features["snapshot_date"]).dt.date.astype(str).values,
        "model_run_id":  manifest["training_run_id"],
        "model_name":    manifest["best_model"],
        "predicted_proba": proba,
        "predicted_label": (proba >= threshold).astype(int),
        "threshold":     threshold,
        "inference_ts":  datetime.utcnow().isoformat(timespec="seconds"),
    })

    os.makedirs(gold_predictions_dir, exist_ok=True)
    out_path = os.path.join(gold_predictions_dir, f"gold_predictions_{date_suffix}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"[inference] {snapshot_date_str}: scored {len(out):,} loans "
          f"(mean proba={proba.mean():.4f}) -> {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-date", required=True)
    p.add_argument("--gold-feature-dir",     default="datamart/gold/feature_store")
    p.add_argument("--model-bank-dir",       default="model_bank")
    p.add_argument("--gold-predictions-dir", default="datamart/gold/predictions")
    args = p.parse_args()
    infer(args.snapshot_date, args.gold_feature_dir, args.model_bank_dir,
          args.gold_predictions_dir)
