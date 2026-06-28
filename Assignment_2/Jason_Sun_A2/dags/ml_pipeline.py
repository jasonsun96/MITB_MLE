"""
A2 end-to-end ML pipeline DAG (loan default prediction).

Per monthly snapshot (driven by Airflow's logical date {{ ds }} + catchup backfill):

  bronze_ingest -> silver_process -> gold_label_store
                                  -> gold_feature_store
                                  -> gold_ml_training_set
                                  -> train_gate --(train month)--> train_model
                                  |                \--(other)----> skip_train
                                  -> inference -> monitor

Model lifecycle (governance baked into the flow):
  * The model is trained ONCE, at TRAIN_AS_OF (the last snapshot, when the
    full matured dataset is available; labels mature at MOB 6), then back-tests.
  * At that training run we also BACK-TEST the model across every historical
    snapshot, populating predictions + monitoring for the whole timeline so the
    performance/stability series can be visualised.
  * From then on, each monthly run scores that month forward and appends a
    fresh monitoring row.

Backfill the whole thing with:
  airflow dags backfill loan_default_ml_pipeline -s 2023-01-01 -e 2024-12-01
or just unpause the DAG (catchup=True will replay the range).
"""
from datetime import datetime

from airflow import DAG
# Airflow 3.x moved the core operators into the "standard" provider; fall back
# to the 2.x import path so the DAG works on either version.
try:
    from airflow.providers.standard.operators.python import (
        PythonOperator, BranchPythonOperator)
    from airflow.providers.standard.operators.empty import EmptyOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator

# ---------------------------------------------------------------------------
# paths (inside the Airflow container; mounted from the repo by docker-compose)
# ---------------------------------------------------------------------------
BASE        = "/opt/airflow"
DATA_DIR    = f"{BASE}/data"
DM          = f"{BASE}/datamart"
BRONZE_DIR  = f"{DM}/bronze"
SILVER_DIR  = f"{DM}/silver"
GOLD_LABEL  = f"{DM}/gold/label_store"
GOLD_FEAT   = f"{DM}/gold/feature_store"
GOLD_MLTS   = f"{DM}/gold/ml_training_set"
GOLD_PRED   = f"{DM}/gold/predictions"
GOLD_MON    = f"{DM}/gold/monitoring"
MODEL_BANK  = f"{BASE}/model_bank"

# label definition (Lab 2): default = 30+ DPD at month-on-book 6
DPD, MOB = 30, 6

# the snapshot at which we train (train<=2023-09 + val<=2023-12 labels matured)
TRAIN_AS_OF = "2024-12-01"   # last snapshot: trains on the full matured dataset

SOURCES = [
    ("loan_daily",  f"{DATA_DIR}/lms_loan_daily.csv"),
    ("attributes",  f"{DATA_DIR}/features_attributes.csv"),
    ("financials",  f"{DATA_DIR}/features_financials.csv"),
    ("clickstream", f"{DATA_DIR}/feature_clickstream.csv"),
]


def _spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder
            .appName("a2_ml_pipeline")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.driver.memory", "2g")
            .getOrCreate())


def _ensure_dirs():
    import os
    for d in [BRONZE_DIR, f"{SILVER_DIR}/loan_daily", f"{SILVER_DIR}/attributes",
              f"{SILVER_DIR}/financials", f"{SILVER_DIR}/clickstream",
              GOLD_LABEL, GOLD_FEAT, GOLD_MLTS, GOLD_PRED, GOLD_MON, MODEL_BANK]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# task callables
# ---------------------------------------------------------------------------
def run_bronze(ds, **_):
    import sys; sys.path.insert(0, BASE)
    from utils.data_processing_bronze_table import process_bronze_table
    _ensure_dirs()
    spark = _spark()
    try:
        for name, path in SOURCES:
            process_bronze_table(name, path, ds, BRONZE_DIR, spark)
    finally:
        spark.stop()


def run_silver(ds, **_):
    import sys; sys.path.insert(0, BASE)
    from utils.data_processing_silver_table import (
        process_silver_loan_daily, process_silver_attributes,
        process_silver_financials, process_silver_clickstream)
    _ensure_dirs()
    spark = _spark()
    try:
        process_silver_loan_daily(ds, BRONZE_DIR, f"{SILVER_DIR}/loan_daily", spark)
        process_silver_attributes(ds, BRONZE_DIR, f"{SILVER_DIR}/attributes", spark)
        process_silver_financials(ds, BRONZE_DIR, f"{SILVER_DIR}/financials", spark)
        process_silver_clickstream(ds, BRONZE_DIR, f"{SILVER_DIR}/clickstream", spark)
    finally:
        spark.stop()


def run_gold_label(ds, **_):
    import sys; sys.path.insert(0, BASE)
    from utils.data_processing_gold_table import process_labels_gold_table
    _ensure_dirs()
    spark = _spark()
    try:
        process_labels_gold_table(ds, f"{SILVER_DIR}/loan_daily", GOLD_LABEL,
                                  spark, DPD, MOB)
    finally:
        spark.stop()


def run_gold_feature(ds, **_):
    import sys; sys.path.insert(0, BASE)
    from utils.data_processing_gold_table import process_features_gold_table
    _ensure_dirs()
    spark = _spark()
    try:
        process_features_gold_table(ds, SILVER_DIR, GOLD_FEAT, spark)
    finally:
        spark.stop()


def run_gold_training_set(ds, **_):
    import sys; sys.path.insert(0, BASE)
    from utils.data_processing_gold_table import process_ml_training_set
    _ensure_dirs()
    spark = _spark()
    try:
        process_ml_training_set(GOLD_LABEL, GOLD_FEAT, GOLD_MLTS, spark)
    finally:
        spark.stop()


def train_gate(ds, **_):
    """Branch: train only on the designated training snapshot."""
    return "train_model" if ds == TRAIN_AS_OF else "skip_train"


def run_train(ds, **_):
    """Train the model bank, then back-test across every historical snapshot."""
    import sys, os, glob; sys.path.insert(0, BASE)
    from utils.training import train_models
    from utils.inference import infer
    from utils.monitoring import monitor
    _ensure_dirs()

    train_models(GOLD_MLTS, MODEL_BANK)

    # back-test the freshly trained model over the full timeline so the
    # monitoring series exists end-to-end the moment training completes.
    parts = sorted(glob.glob(os.path.join(GOLD_FEAT, "gold_feature_store_*.parquet")))
    for p in parts:
        snap = os.path.basename(p).replace("gold_feature_store_", "").replace(".parquet", "").replace("_", "-")
        infer(snap, GOLD_FEAT, MODEL_BANK, GOLD_PRED)
        monitor(snap, GOLD_PRED, GOLD_LABEL, GOLD_MON, GOLD_FEAT, MODEL_BANK)


def run_inference(ds, **_):
    import sys, os; sys.path.insert(0, BASE)
    from utils.inference import infer
    _ensure_dirs()
    if not os.path.exists(os.path.join(MODEL_BANK, "latest_manifest.json")):
        print(f"[inference] no model in bank yet at {ds} - skipping")
        return
    infer(ds, GOLD_FEAT, MODEL_BANK, GOLD_PRED)


def run_monitor(ds, **_):
    import sys, os; sys.path.insert(0, BASE)
    from utils.monitoring import monitor
    _ensure_dirs()
    if not os.path.exists(os.path.join(GOLD_PRED, f"gold_predictions_{ds.replace('-', '_')}.parquet")):
        print(f"[monitoring] no predictions for {ds} - skipping")
        return
    monitor(ds, GOLD_PRED, GOLD_LABEL, GOLD_MON, GOLD_FEAT, MODEL_BANK)


def run_visualize(ds, **_):
    import sys, os, glob; sys.path.insert(0, BASE)
    from utils.visualization import make_monitoring_charts
    _ensure_dirs()
    if not glob.glob(os.path.join(GOLD_MON, "gold_monitoring_*.parquet")):
        print(f"[visualisation] no monitoring partitions yet at {ds} - skipping")
        return
    make_monitoring_charts(GOLD_MON)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="loan_default_ml_pipeline",
    description="A2: end-to-end loan default ML pipeline (datamart + train + infer + monitor)",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    schedule="@monthly",
    catchup=True,
    max_active_runs=1,
    default_args={"retries": 0, "depends_on_past": False},
    tags=["a2", "loan_default"],
) as dag:

    ingest_bronze = PythonOperator(task_id="ingest_bronze", python_callable=run_bronze)
    process_silver = PythonOperator(task_id="process_silver", python_callable=run_silver)
    gold_label = PythonOperator(task_id="gold_label_store", python_callable=run_gold_label)
    gold_feature = PythonOperator(task_id="gold_feature_store", python_callable=run_gold_feature)
    gold_training_set = PythonOperator(task_id="gold_ml_training_set", python_callable=run_gold_training_set)

    train_branch = BranchPythonOperator(task_id="train_gate", python_callable=train_gate)
    train_model = PythonOperator(task_id="train_model", python_callable=run_train)
    skip_train = EmptyOperator(task_id="skip_train")

    inference = PythonOperator(
        task_id="inference", python_callable=run_inference,
        trigger_rule="none_failed_min_one_success")
    monitor_task = PythonOperator(task_id="monitor", python_callable=run_monitor)
    visualize = PythonOperator(task_id="visualize", python_callable=run_visualize)

    ingest_bronze >> process_silver >> [gold_label, gold_feature] >> gold_training_set
    gold_training_set >> train_branch >> [train_model, skip_train] >> inference >> monitor_task >> visualize
