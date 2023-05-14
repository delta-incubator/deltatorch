# Databricks notebook source
from datasets import load_dataset
ds = load_dataset("huggan/pokemon", split="train")
ds.to_parquet("/dbfs/tmp/msh/huggan_pokemon.parquet")

# COMMAND ----------

from pyspark.sql.functions import col, explode
sdf = spark.read.parquet("/tmp/msh/huggan_pokemon.parquet").select(col("image.bytes"))
display(sdf)

# COMMAND ----------

from PIL import Image
from io import BytesIO

Image.open(BytesIO(sdf.take(1)[0]["bytes"]))

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number, rand

def store_as_delta(df, name):
  w = Window().orderBy(rand())
  df.withColumn("id", row_number().over(w)).write.format(
        "delta"
  ).mode("overwrite").save(name)

store_as_delta(sdf, "/tmp/msh/huggan_pokemon.delta")

# COMMAND ----------

# MAGIC %sql optimize delta.`/tmp/msh/huggan_pokemon.delta` zorder by id;

# COMMAND ----------

# MAGIC %sql select * from delta.`/tmp/msh/huggan_pokemon.delta`

# COMMAND ----------

# MAGIC %sql select count(1) from delta.`/tmp/msh/huggan_pokemon.delta`

# COMMAND ----------


