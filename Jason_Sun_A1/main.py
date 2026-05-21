

import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# =============================================================================
# Configuration
# =============================================================================

SOURCES = {
    "loan_daily":  "data/lms_loan_daily.csv",
    "clickstream": "data/feature_clickstream.csv",
    "attributes":  "data/features_attributes.csv",
    "financials":  "data/features_financials.csv",
}


START_DATE = "2023-01-01"
END_DATE   = "2024-12-01"

# Label definition (from Lab 2): default = 30+ days past due at 6 months on book
LABEL_DPD = 30
LABEL_MOB = 6


# =============================================================================
# Helpers
# =============================================================================

def generate_first_of_month_dates(start_date_str, end_date_str):
    """Return a list of first-of-month dates (YYYY-MM-DD) between start and end."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end   = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


def banner(title):
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    # --- Spark session ---
    spark = pyspark.sql.SparkSession.builder \
        .appName("assignment_1") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(START_DATE, END_DATE)
    print(f"Snapshot dates to process: {dates[0]} → {dates[-1]} ({len(dates)} months)")

    # --- Set up datamart directory structure ---
    datamart = "datamart"
    bronze_dirs = {s: os.path.join(datamart, "bronze", s) for s in SOURCES}
    silver_dirs = {s: os.path.join(datamart, "silver", s) for s in SOURCES}
    gold_label_dir       = os.path.join(datamart, "gold", "label_store")
    gold_feature_dir     = os.path.join(datamart, "gold", "feature_store")
    gold_ml_training_dir = os.path.join(datamart, "gold", "ml_training_set")
    silver_root          = os.path.join(datamart, "silver")

    for d in list(bronze_dirs.values()) + list(silver_dirs.values()) + [
        gold_label_dir, gold_feature_dir, gold_ml_training_dir
    ]:
        os.makedirs(d, exist_ok=True)

    # --- BRONZE: ingest each source per snapshot date ---
    banner("BRONZE — ingest all 4 sources per snapshot")
    for date_str in dates:
        for source, csv_path in SOURCES.items():
            utils.data_processing_bronze_table.process_bronze_table(
                source, csv_path, date_str, bronze_dirs[source], spark
            )

    # --- SILVER: clean each source per snapshot date ---
    banner("SILVER — clean per source")
    for date_str in dates:
        utils.data_processing_silver_table.process_silver_loan_daily(
            date_str, bronze_dirs["loan_daily"], silver_dirs["loan_daily"], spark)
        utils.data_processing_silver_table.process_silver_attributes(
            date_str, bronze_dirs["attributes"], silver_dirs["attributes"], spark)
        utils.data_processing_silver_table.process_silver_financials(
            date_str, bronze_dirs["financials"], silver_dirs["financials"], spark)
        utils.data_processing_silver_table.process_silver_clickstream(
            date_str, bronze_dirs["clickstream"], silver_dirs["clickstream"], spark)

    # --- GOLD label store: same as Lab 2 ---
    banner("GOLD — label store (default = 30dpd_6mob)")
    for date_str in dates:
        utils.data_processing_gold_table.process_labels_gold_table(
            date_str, silver_dirs["loan_daily"], gold_label_dir, spark,
            dpd=LABEL_DPD, mob=LABEL_MOB,
        )

    # --- GOLD feature store: joined features with clickstream leakage filter ---
    banner("GOLD — feature store (joined + temporal-safe)")
    for date_str in dates:
        utils.data_processing_gold_table.process_features_gold_table(
            date_str, silver_root, gold_feature_dir, spark
        )

    banner("GOLD — ML training set (inner join of feature_store + label_store)")
    utils.data_processing_gold_table.process_ml_training_set(
        gold_label_dir, gold_feature_dir, gold_ml_training_dir, spark
    )

    banner("DONE — sanity check")

    label_files = sorted(glob.glob(os.path.join(gold_label_dir, "*")))
    feature_files = sorted(glob.glob(os.path.join(gold_feature_dir, "*")))

    if label_files:
        labels = spark.read.parquet(*label_files)
        print(f"Label store    : {labels.count():,} rows across {len(label_files)} partitions")
        print("  Label distribution:")
        labels.groupBy("label").count().orderBy("label").show()

    if feature_files:
        features = spark.read.parquet(*feature_files)
        print(f"Feature store  : {features.count():,} rows across {len(feature_files)} partitions")

    if label_files and feature_files:
        # Join on loan_id so the pipeline stays correct if a customer has multiple loans
        joined = features.join(
            labels.select("loan_id", "label"), "loan_id", "inner"
        )
        print(f"Joinable rows  : {joined.count():,} loans with both features and a matured label")

    training_files = sorted(glob.glob(os.path.join(gold_ml_training_dir, "*")))
    if training_files:
        training = spark.read.parquet(*training_files)
        print(f"ML training set: {training.count():,} rows (inner-joined features + labels) "
              f"with {len(training.columns)} columns")


if __name__ == "__main__":
    main()
