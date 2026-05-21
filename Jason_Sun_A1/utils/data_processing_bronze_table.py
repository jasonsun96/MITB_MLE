"""
Bronze layer processing — generic, source-aware.

A single generic function handles all 4 sources (lms, attributes, financials,
clickstream). Each call filters one source CSV to one snapshot date and
writes a partitioned CSV file. Bronze is "ingest as-is, just slice by date."
"""

import os
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_table(source_name, csv_path, snapshot_date_str, bronze_dir, spark):
    """
    Generic bronze processor.

    Reads the full source CSV, filters to one snapshot_date, writes a
    partitioned CSV file in bronze_dir.

    Args:
        source_name        Logical source name (e.g. "lms", "clickstream").
                           Used in the output filename: bronze_<source_name>_YYYY_MM_DD.csv
        csv_path           Path to raw source CSV (e.g. "data/lms_loan_daily.csv")
        snapshot_date_str  Snapshot date in "YYYY-MM-DD" format
        bronze_dir         Directory to write the bronze partition to
        spark              SparkSession

    Returns:
        The filtered Spark DataFrame.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Load raw CSV and filter to this snapshot date
    df = spark.read.csv(csv_path, header=True, inferSchema=True) \
                   .filter(col("snapshot_date") == snapshot_date)
    n = df.count()

    # Skip writing if no rows — production-style: only real data on disk
    if n == 0:
        print(f"[bronze {source_name}] {snapshot_date_str} no rows — skipping write")
        return df

    print(f"[bronze {source_name}] {snapshot_date_str} row count: {n}")
    partition_name = f"bronze_{source_name}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(bronze_dir, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze {source_name}] saved to: {filepath}")

    return df
