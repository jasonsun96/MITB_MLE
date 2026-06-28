# Apache Airflow 3.2.1 image with Java for PySpark + the ML deps the pipeline
# tasks need (scikit-learn, xgboost, pandas, pyspark, matplotlib, etc).
# Tasks run as PythonOperators inside the Airflow workers, so these libraries
# must live in the Airflow image itself.
FROM apache/airflow:3.2.1-python3.12

# --- system: Java for PySpark ---
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jdk-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Java 17 restricts reflective access that Spark + Arrow need. These --add-opens
# flags unblock them; JAVA_TOOL_OPTIONS is read by every JVM in the container.
ENV JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED"

# --- python ML deps (as the airflow user, which the image requires) ---
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
