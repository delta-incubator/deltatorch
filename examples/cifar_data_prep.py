# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit
from pyspark.sql.types import StructType, StructField, BinaryType, IntegerType, LongType

from torchvision.datasets import CIFAR10, CIFAR100

# COMMAND ----------

spark_write_path = "/tmp/msh/datasets/cifar"
train_read_path = "/tmp/msh/datasets/cifar"

# COMMAND ----------

if locals().get("spark") is None:
    spark = (
        SparkSession.builder.master("local[*]")
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


# COMMAND ----------


def prepare_cifar_data(is_train: bool = True, iter_count: int = 1):
    dataset = CIFAR10(".", train=is_train, download=True)
    df = None
    for i in range(iter_count):
        _df = spark.createDataFrame(
            zip(list(map(lambda x: x.tobytes(), dataset.data)), dataset.targets),
            StructType(
                [StructField("image", BinaryType()), StructField("label", LongType())]
            ),
        )
        if df is not None:
            df = df.union(_df)
        else:
            df = _df

    df = df.withColumn("new_column", lit("ABC"))
    w = Window().partitionBy("new_column").orderBy(lit("A"))
    df.withColumn("id", row_number().over(w)).drop("new_column").write.format(
        "delta"
    ).mode("overwrite").save(
        f"{spark_write_path}_{'train' if is_train else 'test'}.delta"
    )
    spark.read.format("delta").load(
        f"{spark_write_path}_{'train' if is_train else 'test'}.delta"
    ).select("id", "label").show(100, False)


prepare_cifar_data(True)
prepare_cifar_data(False)
