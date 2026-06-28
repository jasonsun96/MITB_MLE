import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory,
                              gold_label_store_directory, spark, dpd, mob):
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    if not os.path.exists(filepath):
        print(f"[gold labels] no silver/loan_daily for {snapshot_date_str} - skipping")
        return None
    df = spark.read.parquet(filepath)
    print(f"[gold labels] loaded from {filepath} row count: {df.count()}")

    df = df.filter(col("mob") == mob)
    if df.count() == 0:
        print(f"[gold labels] no mob={mob} rows on {snapshot_date_str} - skipping write")
        return None

    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    out_name = "gold_label_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    out_path = os.path.join(gold_label_store_directory, out_name)
    df.write.mode("overwrite").parquet(out_path)
    print(f"[gold labels] saved to: {out_path} ({df.count()} rows)")

    return df


def process_features_gold_table(snapshot_date_str, silver_root_dir,
                                gold_feature_store_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_suffix = snapshot_date_str.replace("-", "_")

    loan_daily_path = os.path.join(silver_root_dir, "loan_daily",
                                    f"silver_loan_daily_{date_suffix}.parquet")
    if not os.path.exists(loan_daily_path):
        print(f"[gold features] no silver/loan_daily for {snapshot_date_str} - skipping")
        return None
    loan_daily = spark.read.parquet(loan_daily_path)

    customers_today = loan_daily.filter(col("installment_num") == 0) \
                                .select("Customer_ID", "loan_id", "loan_start_date") \
                                .distinct()

    if customers_today.count() == 0:
        print(f"[gold features] no new loan starts on {snapshot_date_str} - skipping write")
        return None

    attr_path = os.path.join(silver_root_dir, "attributes", f"silver_attributes_{date_suffix}.parquet")
    if os.path.exists(attr_path):
        attributes = spark.read.parquet(attr_path).drop("snapshot_date")
    else:
        attributes = customers_today.select("Customer_ID").limit(0)
        print(f"[gold features] no attributes silver for {snapshot_date_str} - left-joining empty")

    fin_path = os.path.join(silver_root_dir, "financials", f"silver_financials_{date_suffix}.parquet")
    if os.path.exists(fin_path):
        financials = spark.read.parquet(fin_path).drop("snapshot_date")
    else:
        financials = customers_today.select("Customer_ID").limit(0)
        print(f"[gold features] no financials silver for {snapshot_date_str} - left-joining empty")

    clickstream_glob = os.path.join(silver_root_dir, "clickstream", "silver_clickstream_*.parquet")
    clickstream_files = sorted(glob.glob(clickstream_glob))

    if clickstream_files:
        clickstream_all = spark.read.parquet(*clickstream_files)
        ck_joined = clickstream_all.join(customers_today, "Customer_ID", "inner")
        ck_pre_loan = ck_joined.filter(col("snapshot_date") < col("loan_start_date"))
        agg_exprs = [F.mean(f"fe_{i}").alias(f"clickstream_mean_fe_{i}") for i in range(1, 21)]
        agg_exprs.append(F.count("*").alias("clickstream_n_pre_loan_snaps"))
        ck_agg = ck_pre_loan.groupBy("Customer_ID").agg(*agg_exprs)
    else:
        ck_agg = customers_today.select("Customer_ID").limit(0)

    features = customers_today \
        .join(attributes,  "Customer_ID", "left") \
        .join(financials,  "Customer_ID", "left") \
        .join(ck_agg,      "Customer_ID", "left")

    LOAN_TYPES = [
        "Payday Loan", "Credit-Builder Loan", "Not Specified",
        "Home Equity Loan", "Student Loan", "Mortgage Loan",
        "Personal Loan", "Debt Consolidation Loan", "Auto Loan",
    ]
    for lt in LOAN_TYPES:
        col_name = "has_" + lt.lower().replace(" ", "_").replace("-", "_")
        features = features.withColumn(
            col_name,
            F.coalesce(
                F.array_contains(col("loan_types_array"), lt),
                F.lit(False)
            ).cast(IntegerType())
        )
    features = features.withColumn(
        "n_loan_types",
        F.coalesce(F.size(col("loan_types_array")), F.lit(0))
    )
    features = features.drop("loan_types_array")

    features = features.withColumn("snapshot_date", F.lit(snapshot_date).cast(DateType()))

    features = features.fillna(0, subset=["clickstream_n_pre_loan_snaps"])

    print(f"[gold features] {snapshot_date_str} customer count: {features.count()}")

    out_name = f"gold_feature_store_{date_suffix}.parquet"
    out_path = os.path.join(gold_feature_store_directory, out_name)
    features.write.mode("overwrite").parquet(out_path)
    print(f"[gold features] saved to: {out_path}")

    return features


def process_ml_training_set(gold_label_store_directory,
                            gold_feature_store_directory,
                            gold_ml_training_set_directory,
                            spark):
    label_files = sorted(glob.glob(os.path.join(gold_label_store_directory, "*")))
    if not label_files:
        print("[gold ml_training_set] no label partitions found - skipping")
        return None
    labels = spark.read.parquet(*label_files)

    feature_files = sorted(glob.glob(os.path.join(gold_feature_store_directory, "*")))
    if not feature_files:
        print("[gold ml_training_set] no feature partitions found - skipping")
        return None
    features = spark.read.parquet(*feature_files)

    print(f"[gold ml_training_set] features available: {features.count()} rows  "
          f"labels available: {labels.count()} rows")

    labels_slim = labels.select("loan_id", "label", "label_def")

    training_set = features.join(labels_slim, "loan_id", "inner")

    print(f"[gold ml_training_set] training-ready rows after inner join: {training_set.count()}")

    out_path = os.path.join(gold_ml_training_set_directory, "ml_training_set.parquet")
    training_set.write.mode("overwrite").parquet(out_path)
    print(f"[gold ml_training_set] saved to: {out_path}")

    return training_set
