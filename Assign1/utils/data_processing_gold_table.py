"""
Gold layer processing — two functions:

1. process_labels_gold_table  (from Lab 2, structurally unchanged):
   - Reads silver/loan_daily
   - Filters to mob=6
   - Labels default = (dpd >= 30)
   - Writes gold/label_store

2. process_features_gold_table  (new for this assignment):
   - For each snapshot_date, finds customers whose loan_start_date == snapshot_date
   - Joins their attributes + financials + aggregated clickstream features
   - APPLIES TEMPORAL LEAKAGE FILTER on clickstream: only snapshots strictly
     before each customer's loan_start_date contribute to the aggregation
   - Writes gold/feature_store partitioned by snapshot_date
"""

import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# =============================================================================
# LABEL STORE (from Lab 2)
# =============================================================================

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory,
                              gold_label_store_directory, spark, dpd, mob):
    """
    Reads silver/loan_daily parquet for the snapshot, filters to mob, derives
    the default label (dpd >= threshold), writes to gold/label_store.

    Same logic as Lab 2.
    """
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    df = spark.read.parquet(filepath)
    print(f"[gold labels] loaded from {filepath} row count: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    out_name = "gold_label_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_label_store_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[gold labels] saved to: {out_path} ({df.count()} rows)")

    return df


# =============================================================================
# FEATURE STORE — the new bit
# =============================================================================

def process_features_gold_table(snapshot_date_str, silver_root_dir,
                                gold_feature_store_directory, spark):
    """
    Build the gold feature store for one snapshot date.

    For each customer whose loan_start_date == snapshot_date:
      - Pull their attributes and financials snapshot (taken at loan start)
      - Aggregate their clickstream history from STRICTLY BEFORE their
        loan_start_date (avoids temporal leakage)
      - Join all three into one feature row per customer

    Args:
        snapshot_date_str            "YYYY-MM-DD"
        silver_root_dir              Path to datamart/silver (contains per-source subdirs)
        gold_feature_store_directory Where to write gold_feature_store_YYYY_MM_DD.parquet
        spark                        SparkSession
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_suffix = snapshot_date_str.replace("-", "_")

    # -----------------------------------------------------------------------
    # 1. Identify customers whose loan started on this snapshot date.
    #    We pull from silver/loan_daily at this snapshot, where installment_num==0
    #    means "row at loan_start" (= the day the loan begins).
    # -----------------------------------------------------------------------
    loan_daily_path = os.path.join(silver_root_dir, "loan_daily",
                                    f"silver_loan_daily_{date_suffix}.parquet")
    try:
        loan_daily = spark.read.parquet(loan_daily_path)
    except Exception as e:
        print(f"[gold features] no loan_daily silver for {snapshot_date_str} — skipping")
        return None

    customers_today = loan_daily.filter(col("installment_num") == 0) \
                                .select("Customer_ID", "loan_start_date") \
                                .distinct()

    if customers_today.count() == 0:
        # No new loans started this snapshot — write empty partition for completeness
        print(f"[gold features] no new loan starts on {snapshot_date_str}; writing empty partition")
        # We still need a placeholder so reads with wildcard don't crash on missing dates.
        # Use the LMS schema for an empty placeholder, then early return.
        empty = customers_today
        out_name = f"gold_feature_store_{date_suffix}.parquet"
        out_path = os.path.join(gold_feature_store_directory, out_name)
        empty.write.mode("overwrite").parquet(out_path)
        return empty

    # -----------------------------------------------------------------------
    # 2. Attributes silver at this snapshot (one row per customer at loan_start)
    # -----------------------------------------------------------------------
    attr_path = os.path.join(silver_root_dir, "attributes", f"silver_attributes_{date_suffix}.parquet")
    attributes = spark.read.parquet(attr_path).drop("snapshot_date")

    # -----------------------------------------------------------------------
    # 3. Financials silver at this snapshot
    # -----------------------------------------------------------------------
    fin_path = os.path.join(silver_root_dir, "financials", f"silver_financials_{date_suffix}.parquet")
    financials = spark.read.parquet(fin_path).drop("snapshot_date")

    # -----------------------------------------------------------------------
    # 4. Clickstream: load ALL clickstream silvers, then for each customer
    #    filter to snapshots strictly before their loan_start_date.
    #    This is THE critical temporal leakage filter.
    # -----------------------------------------------------------------------
    clickstream_glob = os.path.join(silver_root_dir, "clickstream", "silver_clickstream_*.parquet")
    clickstream_files = sorted(glob.glob(clickstream_glob))

    if clickstream_files:
        clickstream_all = spark.read.parquet(*clickstream_files)

        # Bring in each customer's loan_start_date for the filter
        ck_joined = clickstream_all.join(customers_today, "Customer_ID", "inner")

        # TEMPORAL LEAKAGE FILTER: only pre-loan snapshots
        ck_pre_loan = ck_joined.filter(col("snapshot_date") < col("loan_start_date"))

        # Aggregate per customer: mean of each fe_X + count of pre-loan snaps
        agg_exprs = [F.mean(f"fe_{i}").alias(f"clickstream_mean_fe_{i}") for i in range(1, 21)]
        agg_exprs.append(F.count("*").alias("clickstream_n_pre_loan_snaps"))
        ck_agg = ck_pre_loan.groupBy("Customer_ID").agg(*agg_exprs)
    else:
        # No clickstream silvers exist yet — empty agg
        ck_agg = customers_today.select("Customer_ID").limit(0)

    # -----------------------------------------------------------------------
    # 5. Join everything into one feature row per customer
    # -----------------------------------------------------------------------
    features = customers_today \
        .join(attributes,  "Customer_ID", "left") \
        .join(financials,  "Customer_ID", "left") \
        .join(ck_agg,      "Customer_ID", "left")

    # Stamp the snapshot_date for traceability (= loan_start_date for this batch)
    features = features.withColumn("snapshot_date", F.lit(snapshot_date).cast(DateType()))

    # Fill clickstream_n_pre_loan_snaps with 0 for customers with no pre-loan history
    # (so the model can distinguish "no history" from "missing").
    features = features.fillna(0, subset=["clickstream_n_pre_loan_snaps"])

    print(f"[gold features] {snapshot_date_str} customer count: {features.count()}")

    out_name = f"gold_feature_store_{date_suffix}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    features.write.mode("overwrite").parquet(out_path)
    print(f"[gold features] saved to: {out_path}")

    return features
