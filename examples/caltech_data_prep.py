# Databricks notebook source
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit
from pyspark.sql.types import StructType, StructField, BinaryType, IntegerType, LongType

from torchvision.datasets import CIFAR10, CIFAR100, Caltech256

# COMMAND ----------

spark_write_path = "/tmp/msh/datasets/caltech256_duplicated_x10"
train_read_path = "/tmp/msh/datasets/caltech256_duplicated_x10"
temp_path = "/tmp/"

# COMMAND ----------

if locals().get("spark") is None:
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", "25G")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )
else:
    train_read_path = f"/dbfs{train_read_path}"
    # temp_path = f"/dbfs{temp_path}"


# COMMAND ----------


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def prepare_caltech_data(iter_count: int = 1):
    dataset = Caltech256(temp_path, download=True)
    df = None
    for i in range(iter_count):
        _data = [
            (
                read_bytes(
                    os.path.join(
                        dataset.root,
                        "256_ObjectCategories",
                        dataset.categories[dataset.y[index]],
                        f"{dataset.y[index] + 1:03d}_{dataset.index[index]:04d}.jpg",
                    )
                ),
                dataset.y[index],
            )
            for index in range(len(dataset))
        ]
        _df = spark.createDataFrame(
            _data,
            StructType(
                [StructField("image", BinaryType()), StructField("label", LongType())]
            ),
        )
        if df is not None:
            df = df.union(_df)
        else:
            df = _df

    return df


def split_spark_df(df):
    print(f"Full count = {df.count()}")
    fractions = (
        df.select("label")
        .distinct()
        .withColumn("fraction", lit(0.05))
        .rdd.collectAsMap()
    )
    test_df = df.stat.sampleBy("label", fractions)
    test_df.groupBy("label").count().orderBy("label").show()
    print(f"Test count = {test_df.count()}")
    train_df = df
    print(f"Train count = {train_df.count()}")
    return train_df, test_df


def store_as_delta(df, name):
    df = df.withColumn("new_column", lit("ABC"))
    w = Window().partitionBy("new_column").orderBy(lit("A"))
    df.withColumn("id", row_number().over(w)).drop("new_column").write.format(
        "delta"
    ).mode("overwrite").save(name)


full_df = prepare_caltech_data(iter_count=10)
train_df, test_df = split_spark_df(full_df)
#print(train_df.groupby("label").count().count())
print(f"Train count = {train_df.count()}")
#print(test_df.groupby("label").count().count())
print(f"Test count = {test_df.count()}")
store_as_delta(train_df, f"{spark_write_path}_train.delta")
store_as_delta(test_df, f"{spark_write_path}_test.delta")

# COMMAND ----------

f"{spark_write_path}_train.delta"

# COMMAND ----------

# MAGIC %sql optimize delta.`/tmp/msh/datasets/caltech256_duplicated_x10_train.delta` zorder by id

# COMMAND ----------

# MAGIC %sql select count(1) from delta.`/tmp/msh/datasets/caltech256_duplicated_x10_train.delta`

# COMMAND ----------

# MAGIC %sql select count(1) from delta.`/tmp/msh/datasets/caltech256_duplicated_x10_test.delta`

# COMMAND ----------


