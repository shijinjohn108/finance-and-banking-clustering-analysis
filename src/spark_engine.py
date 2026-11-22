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


def _quoted_column(name: str):
    return F.col("`" + name.replace("`", "``") + "`")


def _build_preprocessing(
    frame: DataFrame,
    chosen: list[str],
    prefix: str,
    numeric_imputation: str,
    missing_policy: str,
    outlier_treatment: str,
    scaling: str,
):
    """Prepare the shared impute/encode/cap/scale steps used by K-Means and PCA."""
    from pyspark.ml.feature import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    if numeric_imputation not in {"median", "mean"}:
        raise ValueError("Numeric missing values must use mean or median imputation.")
    if missing_policy not in {"keep", "drop"}:
        raise ValueError("Missing row policy must be keep or drop.")
    if outlier_treatment not in {"keep", "cap"}:
        raise ValueError("Outlier treatment must be keep or cap.")
    if scaling not in {"standard", "minmax", "none"}:
        raise ValueError("Scaling must be standard, minmax, or none.")

    selected = frame
    numeric = [field.name for field in selected.schema.fields if field.name in chosen and isinstance(field.dataType, NumericType)]
    categorical = [name for name in chosen if name not in numeric]

    # Treat blank categorical cells as missing, just like nulls.
    for name in categorical:
        column = _quoted_column(name)
        selected = selected.withColumn(
            name,
            F.when(F.trim(column.cast("string")) == "", F.lit(None).cast("string")).otherwise(column.cast("string")),
        )

    if missing_policy == "drop":
        has_missing = F.lit(False)
        for name in chosen:
            column = _quoted_column(name)
            has_missing = has_missing | column.isNull() | (F.trim(column.cast("string")) == "")
        selected = selected.filter(~has_missing)

    if outlier_treatment == "cap" and numeric:
        bounds = selected.approxQuantile(numeric, [0.01, 0.99], 0.01)
        for name, limits in zip(numeric, bounds):
            if len(limits) == 2:
                column = _quoted_column(name)
                selected = selected.withColumn(
                    name,
                    F.when(column < F.lit(limits[0]), F.lit(limits[0]))
                    .when(column > F.lit(limits[1]), F.lit(limits[1]))
                    .otherwise(column),
                )

    stages = []
    numeric_filled = []
    if numeric:
        numeric_filled = [f"{prefix}_imputed_{i}" for i in range(len(numeric))]
        stages.append(Imputer(inputCols=numeric, outputCols=numeric_filled, strategy=numeric_imputation))

    encoded = []
    for i, name in enumerate(categorical):
        indexed = f"{prefix}_index_{i}"
        encoded_name = f"{prefix}_category_{i}"
        stages.extend([
            StringIndexer(inputCol=name, outputCol=indexed, handleInvalid="keep"),
            OneHotEncoder(inputCol=indexed, outputCol=encoded_name, handleInvalid="keep"),
        ])
        encoded.append(encoded_name)

    raw_features = f"{prefix}_raw"
    feature_column = f"{prefix}_features"
    stages.append(VectorAssembler(inputCols=numeric_filled + encoded, outputCol=raw_features, handleInvalid="keep"))
    if scaling == "standard":
        stages.append(StandardScaler(inputCol=raw_features, outputCol=feature_column, withMean=False, withStd=True))
    elif scaling == "minmax":
        stages.append(MinMaxScaler(inputCol=raw_features, outputCol=feature_column))
    else:
        feature_column = raw_features
    return selected, stages, feature_column


def cluster_dataset(
    df: DataFrame,
    features: Iterable[str],
    k: int = 3,
    seed: int = 42,
    numeric_imputation: str = "median",
    missing_policy: str = "keep",
    outlier_treatment: str = "keep",
    scaling: str = "standard",
) -> tuple[DataFrame, float]:
    """Impute, encode, optionally cap outliers, scale, cluster, and score."""
    from pyspark.ml import Pipeline
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator

    chosen = list(features)
    if not chosen:
        raise ValueError("Select at least one feature.")
    if len(df.take(2)) < 2:
        raise ValueError("At least two records are needed to cluster.")

    row_id = "__finsight_row_id"
    while row_id in df.columns:
        row_id += "_"
    source = df.withColumn(row_id, F.monotonically_increasing_id())
    selected = source.select(_quoted_column(row_id), *[_quoted_column(name) for name in chosen])
    prefix = "__finsight_cluster"
    while any(name.startswith(prefix) for name in selected.columns):
        prefix += "_"
    selected, stages, feature_column = _build_preprocessing(
        selected, chosen, prefix, numeric_imputation, missing_policy, outlier_treatment, scaling
    )
    usable_rows = selected.count()
    if k < 2 or k > usable_rows:
        raise ValueError(f"Choose a cluster count from 2 through the {usable_rows} rows that remain after preprocessing.")

    prediction_column = f"{prefix}_prediction"
    stages.append(KMeans(featuresCol=feature_column, predictionCol=prediction_column, k=k, seed=seed, maxIter=30))
    model = Pipeline(stages=stages).fit(selected)
    predictions = model.transform(selected)
    score = float(ClusteringEvaluator(featuresCol=feature_column, predictionCol=prediction_column, metricName="silhouette").evaluate(predictions))
    assigned = predictions.select(_quoted_column(row_id), _quoted_column(prediction_column).alias("prediction"))
    return source.join(assigned, row_id, "inner").drop(row_id), score


def project_cluster_map(
    df: DataFrame,
    features: Iterable[str],
    max_points: int = 5000,
    seed: int = 42,
    numeric_imputation: str = "median",
    missing_policy: str = "keep",
    outlier_treatment: str = "keep",
    scaling: str = "standard",
):
    """Return a sampled 2D PCA view of already-clustered rows.

    The PCA axes are a visualization of the chosen feature space; the original
    ``prediction`` labels are retained so this never refits or changes K-Means.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import PCA
    from pyspark.ml.functions import vector_to_array

    chosen = list(features)
    if not chosen:
        raise ValueError("Select clustering features before building the map.")
    if "prediction" not in df.columns:
        raise ValueError("Run K-Means before building the cluster map.")
    if max_points < 2:
        raise ValueError("The map needs at least two points.")

    total_rows = df.count()
    if total_rows < 2:
        raise ValueError("At least two clustered records are needed for a map.")
    if total_rows > max_points:
        sample = df.sample(withReplacement=False, fraction=min(1.0, max_points / total_rows), seed=seed).limit(max_points)
    else:
        sample = df

    sampled = sample.select(*[_quoted_column(name) for name in chosen], _quoted_column("prediction"))
    prefix = "__finsight_map"
    while any(name.startswith(prefix) for name in sampled.columns):
        prefix += "_"
    sampled, stages, feature_column = _build_preprocessing(
        sampled, chosen, prefix, numeric_imputation, missing_policy, outlier_treatment, scaling
    )
    prepared = Pipeline(stages=stages).fit(sampled).transform(sampled)
    first_vector = prepared.select(feature_column).first()[0]
    if first_vector is None or first_vector.size < 2:
        raise ValueError("This feature selection creates fewer than two dimensions. Choose features that expand to at least two numeric or categorical dimensions for a 2D map.")

    pca_features = f"{prefix}_pca"
    pca_model = PCA(k=2, inputCol=feature_column, outputCol=pca_features).fit(prepared)
    projected = pca_model.transform(prepared)
    coordinates = vector_to_array(F.col(pca_features))
    return projected.select(
        F.col("prediction").cast("int").alias("Cluster"),
        coordinates[0].alias("PC 1"),
        coordinates[1].alias("PC 2"),
    ).toPandas()
