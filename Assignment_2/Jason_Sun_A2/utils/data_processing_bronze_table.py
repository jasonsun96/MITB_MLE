import os
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_table(source_name, csv_path, snapshot_date_str, bronze_dir, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    df = spark.read.csv(csv_path, header=True, inferSchema=True) \
                   .filter(col("snapshot_date") == snapshot_date)
    n = df.count()

    if n == 0:
        print(f"[bronze {source_name}] {snapshot_date_str} no rows - skipping write")
        return df

    print(f"[bronze {source_name}] {snapshot_date_str} row count: {n}")
    partition_name = f"bronze_{source_name}_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = os.path.join(bronze_dir, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze {source_name}] saved to: {filepath}")

    return df
