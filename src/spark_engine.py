"""Small, local-first Spark helpers for schema-agnostic tabular clustering."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


def create_spark(app_name: str = "FinanceBankingClustering") -> SparkSession:
    """Start a modest local Spark session without requiring a cluster."""
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def read_dataset(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read CSV (header and inferred types) or Parquet by file suffix."""
    file_path = str(path)
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        # Finance datasets commonly use commas or semicolons. Infer a delimiter
        # from the header/sample so a valid CSV is not mistaken for one wide column.
        with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as source:
            sample = source.read(8192)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
        return spark.read.option("header", True).option("inferSchema", True).option("mode", "PERMISSIVE").option("sep", delimiter).csv(file_path)
    if suffix == ".parquet":
        return spark.read.parquet(file_path)
    raise ValueError("Supported formats are CSV and Parquet.")


def inspect_dataset(df: DataFrame) -> dict:
    """Return schema-driven quality information; no column names are assumed."""
    total = df.count()
    missing = df.select([
        F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]).first().asDict() if df.columns else {}
    numeric = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    categorical = [f.name for f in df.schema.fields if f.name not in numeric]
    return {"rows": total, "columns": len(df.columns), "missing": {k: int(v or 0) for k, v in missing.items()},
            "duplicates": total - df.dropDuplicates().count(), "numeric": numeric, "categorical": categorical}


def cluster_dataset(df: DataFrame, features: Iterable[str], k: int = 3, seed: int = 42) -> tuple[DataFrame, float]:
    """Impute, encode, scale, cluster with MLlib K-Means, and score silhouette."""
    from pyspark.ml import Pipeline
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    chosen = list(features)
    if not chosen:
        raise ValueError("Select at least one feature.")
    if len(df.take(2)) < 2:
        raise ValueError("At least two records are needed to cluster.")
    if k < 2 or k > df.count():
        raise ValueError("Choose a cluster count from 2 through the number of records.")
    source = df.withColumn("__row_id", F.monotonically_increasing_id())
    selected = source.select("__row_id", *chosen)
    numeric = [f.name for f in selected.schema.fields if isinstance(f.dataType, NumericType)]
    categorical = [c for c in chosen if c not in numeric]
    stages = []
    numeric_filled = []
    if numeric:
        stages.append(Imputer(inputCols=numeric, outputCols=[f"__imputed_{i}" for i in range(len(numeric))], strategy="median"))
        numeric_filled = [f"__imputed_{i}" for i in range(len(numeric))]
    encoded = []
    for i, col in enumerate(categorical):
        ix, oh = f"__index_{i}", f"__category_{i}"
        stages.extend([StringIndexer(inputCol=col, outputCol=ix, handleInvalid="keep"), OneHotEncoder(inputCol=ix, outputCol=oh, handleInvalid="keep")])
        encoded.append(oh)
    if not numeric_filled and categorical:
        # Encoded vector features are valid assembler inputs.
        pass
    assembler = VectorAssembler(inputCols=numeric_filled + encoded, outputCol="__raw_features", handleInvalid="keep")
    stages.append(assembler)
    stages.append(StandardScaler(inputCol="__raw_features", outputCol="features", withMean=False, withStd=True))
    stages.append(KMeans(featuresCol="features", predictionCol="prediction", k=k, seed=seed, maxIter=30))
    model = Pipeline(stages=stages).fit(selected)
    predictions = model.transform(selected)
    score = float(ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette").evaluate(predictions))
    # Bring through source columns for useful previews and exports.
    return source.join(predictions.select("__row_id", "prediction"), "__row_id", "inner").drop("__row_id"), score
