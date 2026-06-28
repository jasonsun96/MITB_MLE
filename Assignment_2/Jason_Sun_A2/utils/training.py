"""
Training stage of the A2 ML pipeline.

Reads the gold ml_training_set produced by the datamart pipeline, does a
chronological train / val / test / oot split, trains four loan-default model
candidates (logreg / decision tree / random forest / xgboost), picks the best
by validation ROC-AUC, and writes all four artefacts plus a manifest to the
model_bank.

Design choices baked in here:
  - Chronological 4-way split by feature snapshot_date (NO random split - that
    would leak future information into past training rows: temporal leakage).
      train      -> fit the models
      validation -> model selection (choose best by AUC)
      test       -> held-out, in-period sanity check (never seen in fit/select)
      oot        -> out-of-time, the strictest generalisation check
  - Preprocessing is fit on TRAIN ONLY and lives inside each model's sklearn
    Pipeline, so val / test / oot / production rows are only *transformed*
    (avoids train-test contamination).
  - Mixed feature types handled with a ColumnTransformer:
        numeric      -> median impute (+ StandardScaler for logreg only)
        categorical  -> most-frequent impute + one-hot (handle_unknown=ignore)
  - Class imbalance handled via class_weight='balanced' (sklearn) and
    scale_pos_weight (xgboost).
  - All four models saved into the bank; the manifest records which is "best".
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Columns that are identifiers / metadata / target, never features.
NON_FEATURE_COLS = {
    "loan_id", "Customer_ID", "snapshot_date", "loan_start_date",
    "label", "label_def",
    "feature_snapshot_date", "label_snapshot_date",
}

# Chronological 4-way split, 70 / 10 / 10 / 10 by row percentile of
# snapshot_date (matured data is monthly, so exact percentages can only be hit
# at the row level). Rows are sorted by snapshot_date, then sliced:
#   train = earliest 70%   val = next 10%   test = next 10%   oot = latest 10%
# This keeps the ordering strictly chronological (earliest loans train the model,
# the most recent are the out-of-time holdout) with no random shuffling.
SPLIT_FRACS = (0.70, 0.80, 0.90)   # cumulative cut points for train/val/test

# Approximate period boundaries (used only for the PSI baseline window and the
# monitoring chart shading - NOT for the split itself, which is by row above).
TRAIN_CUTOFF = "2024-01-01"
VAL_CUTOFF   = "2024-03-01"
TEST_CUTOFF  = "2024-05-01"


def _load_ml_training_set(gold_ml_training_dir: str) -> pd.DataFrame:
    path = os.path.join(gold_ml_training_dir, "ml_training_set.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ml_training_set parquet not found at {path}")
    df = pd.read_parquet(path)
    print(f"[training] loaded ml_training_set: {len(df):,} rows, {df.shape[1]} columns")
    return df


def _chrono_split(df: pd.DataFrame):
    """Chronological 70/10/10/10 train/val/test/oot split by snapshot_date order."""
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values("snapshot_date", kind="mergesort").reset_index(drop=True)
    n = len(df)
    i70, i80, i90 = (int(round(n * f)) for f in SPLIT_FRACS)
    train, val, test, oot = df.iloc[:i70], df.iloc[i70:i80], df.iloc[i80:i90], df.iloc[i90:]
    print(f"[training] split sizes: train={len(train):,}  val={len(val):,}  "
          f"test={len(test):,}  oot={len(oot):,}  "
          f"({len(train)/n:.0%}/{len(val)/n:.0%}/{len(test)/n:.0%}/{len(oot)/n:.0%})")
    for name, part in (("train", train), ("val", val), ("test", test), ("oot", oot)):
        if len(part):
            print(f"[training] {name:5s} range: {part['snapshot_date'].min().date()} -> {part['snapshot_date'].max().date()}")
    return train, val, test, oot


def _feature_lists(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    cat_cols = [c for c in feature_cols
                if (df[c].dtype == object or str(df[c].dtype).startswith("string"))]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"[training] {len(feature_cols)} feature cols -> {len(num_cols)} numeric, {len(cat_cols)} categorical")
    if cat_cols:
        print(f"[training] categorical: {cat_cols}")
    return feature_cols, num_cols, cat_cols


def _xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    y = df["label"].astype(int).to_numpy()
    return X, y


def _make_preprocessor(num_cols, cat_cols, scale_numeric: bool):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe",    OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )


def _build_models(num_cols, cat_cols, scale_pos_weight: float):
    return {
        "logreg": Pipeline([
            ("pre", _make_preprocessor(num_cols, cat_cols, scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "decision_tree": Pipeline([
            ("pre", _make_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            ("model", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=50,
                class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("pre", _make_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("pre", _make_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight, tree_method="hist",
                eval_metric="auc", random_state=42, n_jobs=-1)),
        ]),
    }


def _evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
    }


def train_models(gold_ml_training_dir: str, model_bank_dir: str) -> dict:
    """End-to-end training entrypoint. Returns the manifest dict that was written."""
    df = _load_ml_training_set(gold_ml_training_dir)

    train_df, val_df, test_df, oot_df = _chrono_split(df)
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("empty train or val split - check the cutoffs")

    feature_cols, num_cols, cat_cols = _feature_lists(df)

    X_train, y_train = _xy(train_df, feature_cols)
    X_val,   y_val   = _xy(val_df,   feature_cols)
    X_test,  y_test  = (_xy(test_df, feature_cols) if len(test_df) else (None, None))
    X_oot,   y_oot   = (_xy(oot_df,  feature_cols) if len(oot_df)  else (None, None))

    pos_rate = float(np.mean(y_train))
    print(f"[training] positive (default) rate in train: {pos_rate:.4f}")
    scale_pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)

    models = _build_models(num_cols, cat_cols, scale_pos_weight)

    val_metrics, test_metrics, oot_metrics = {}, {}, {}
    for name, pipe in models.items():
        print(f"[training] fitting {name}...")
        pipe.fit(X_train, y_train)
        val_metrics[name] = _evaluate(pipe, X_val, y_val)
        if X_test is not None and len(X_test):
            test_metrics[name] = _evaluate(pipe, X_test, y_test)
        if X_oot is not None and len(X_oot):
            oot_metrics[name] = _evaluate(pipe, X_oot, y_oot)
        print(f"           val auc={val_metrics[name]['auc']:.4f}"
              + (f"  test auc={test_metrics[name]['auc']:.4f}" if name in test_metrics else "")
              + (f"  oot auc={oot_metrics[name]['auc']:.4f}" if name in oot_metrics else ""))

    # model selection on validation only
    best_name = max(val_metrics, key=lambda n: val_metrics[n]["auc"])
    print(f"[training] best model by val auc: {best_name} ({val_metrics[best_name]['auc']:.4f})")

    training_date = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(model_bank_dir, training_date)
    os.makedirs(run_dir, exist_ok=True)
    for name, pipe in models.items():
        joblib.dump(pipe, os.path.join(run_dir, f"{name}.joblib"))
    print(f"[training] wrote 4 model artefacts to {run_dir}")

    manifest = {
        "training_run_id":     training_date,
        "training_data_path":  os.path.join(gold_ml_training_dir, "ml_training_set.parquet"),
        "feature_columns":     feature_cols,
        "numeric_columns":     num_cols,
        "categorical_columns": cat_cols,
        "split": {
            "method": "chronological row-percentile 70/10/10/10",
            "fractions": {"train": 0.70, "val": 0.10, "test": 0.10, "oot": 0.10},
            "train_rows":   int(len(train_df)),
            "val_rows":     int(len(val_df)),
            "test_rows":    int(len(test_df)),
            "oot_rows":     int(len(oot_df)),
            "train_range":  [str(train_df["snapshot_date"].min().date()), str(train_df["snapshot_date"].max().date())],
            "val_range":    [str(val_df["snapshot_date"].min().date()),   str(val_df["snapshot_date"].max().date())],
            "test_range":   [str(test_df["snapshot_date"].min().date()),  str(test_df["snapshot_date"].max().date())],
            "oot_range":    [str(oot_df["snapshot_date"].min().date()),   str(oot_df["snapshot_date"].max().date())],
            "positive_rate_train": pos_rate,
        },
        "models": list(models.keys()),
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "oot_metrics":  oot_metrics,
        "best_model":  best_name,
        "best_model_artefact": f"{best_name}.joblib",
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(model_bank_dir, "latest_manifest.json"), "w") as f:
        json.dump({**manifest, "run_dir": run_dir}, f, indent=2, default=str)

    print(f"[training] manifest written. best={best_name} val_auc={val_metrics[best_name]['auc']:.4f}")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gold-ml-training-dir", default="datamart/gold/ml_training_set")
    p.add_argument("--model-bank-dir",       default="model_bank")
    args = p.parse_args()
    train_models(args.gold_ml_training_dir, args.model_bank_dir)
