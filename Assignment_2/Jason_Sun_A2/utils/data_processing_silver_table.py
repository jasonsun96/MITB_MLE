import os
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


SENTINELS = {
    "Occupation":            "_______",
    "SSN":                   "#F%$D@*&8",
    "Credit_Mix":            "_",
    "Payment_of_Min_Amount": "NM",
    "Payment_Behaviour":     "!@9#%8",
}


FINANCIALS_NUMERIC_CAPS = {
    "Annual_Income":           (1_000,  1_000_000, FloatType()),
    "Monthly_Inhand_Salary":   (0,      100_000,   FloatType()),
    "Num_Bank_Accounts":       (0,      20,        IntegerType()),
    "Num_Credit_Card":         (0,      15,        IntegerType()),
    "Interest_Rate":           (0,      50,        IntegerType()),
    "Num_of_Loan":             (0,      15,        IntegerType()),
    "Delay_from_due_date":     (-10,    100,       IntegerType()),
    "Num_of_Delayed_Payment":  (0,      50,        IntegerType()),
    "Changed_Credit_Limit":    (-50,    50,        FloatType()),
    "Num_Credit_Inquiries":    (0,      50,        IntegerType()),
    "Outstanding_Debt":        (0,      10_000,    FloatType()),
    "Total_EMI_per_month":     (0,      50_000,    FloatType()),
    "Amount_invested_monthly": (0,      50_000,    FloatType()),
    "Monthly_Balance":         (-10_000, 50_000,   FloatType()),
}


def _read_bronze(source_name, snapshot_date_str, bronze_dir, spark):
    partition_name = f"bronze_{source_name}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(bronze_dir, partition_name)
    if not os.path.exists(filepath):
        print(f"[silver {source_name}] no bronze file for {snapshot_date_str} - skipping")
        return None
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver {source_name}] loaded from {filepath} row count: {df.count()}")
    return df


def _write_silver(df, source_name, snapshot_date_str, silver_dir):
    partition_name = f"silver_{source_name}_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = os.path.join(silver_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver {source_name}] saved to: {filepath}")


def process_silver_loan_daily(snapshot_date_str, bronze_dir, silver_dir, spark):
    df = _read_bronze("loan_daily", snapshot_date_str, bronze_dir, spark)
    if df is None or df.count() == 0:
        return None

    column_type_map = {
        "loan_id":         StringType(),
        "Customer_ID":     StringType(),
        "loan_start_date": DateType(),
        "tenure":          IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt":        FloatType(),
        "due_amt":         FloatType(),
        "paid_amt":        FloatType(),
        "overdue_amt":     FloatType(),
        "balance":         FloatType(),
        "snapshot_date":   DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
         .cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date")))
         .otherwise(0)
         .cast(IntegerType())
    )

    _write_silver(df, "loan_daily", snapshot_date_str, silver_dir)
    return df


def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    df = _read_bronze("attributes", snapshot_date_str, bronze_dir, spark)
    if df is None or df.count() == 0:
        return None

    df = df.drop("Name", "SSN")

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_", "").cast(IntegerType()))
    df = df.withColumn(
        "Age",
        F.when(col("Age").between(18, 100), col("Age")).otherwise(None)
    )

    df = df.withColumn(
        "Occupation",
        F.when(col("Occupation") == SENTINELS["Occupation"], None)
         .otherwise(col("Occupation"))
    )

    _write_silver(df, "attributes", snapshot_date_str, silver_dir)
    return df


def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    df = _read_bronze("financials", snapshot_date_str, bronze_dir, spark)
    if df is None or df.count() == 0:
        return None

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    for c, (lo, hi, spark_type) in FINANCIALS_NUMERIC_CAPS.items():
        df = df.withColumn(c, F.regexp_replace(col(c), "_", "").cast(spark_type))
        df = df.withColumn(
            c,
            F.when(col(c).between(lo, hi), col(c)).otherwise(None)
        )

    for c in ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]:
        df = df.withColumn(c, F.when(col(c) == SENTINELS[c], None).otherwise(col(c)))

    df = df.withColumn(
        "credit_history_months",
        F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType()) * 12
        + F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())
    )
    df = df.drop("Credit_History_Age")

    df = df.withColumn("loan_types_array", F.split(col("Type_of_Loan"), ", "))
    df = df.withColumn(
        "loan_types_array",
        F.expr("transform(loan_types_array, x -> regexp_replace(x, '^and ', ''))")
    )
    df = df.drop("Type_of_Loan")

    _write_silver(df, "financials", snapshot_date_str, silver_dir)
    return df


def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    df = _read_bronze("clickstream", snapshot_date_str, bronze_dir, spark)
    if df is None or df.count() == 0:
        return None

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))

    _write_silver(df, "clickstream", snapshot_date_str, silver_dir)
    return df
