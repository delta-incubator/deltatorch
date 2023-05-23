# Databricks notebook source
from datasets import load_dataset

ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
ds.to_parquet("/dbfs/tmp/msh/lambdalabs_pokemon_blip_captions.parquet")

# COMMAND ----------

from pyspark.sql.functions import col, explode

sdf = spark.read.parquet("/tmp/msh/lambdalabs_pokemon_blip_captions.parquet").select(
    col("image.bytes").alias("image"), col("text")
)
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
    df.withColumn("id", row_number().over(w)).write.format("delta").mode(
        "overwrite"
    ).save(name)


store_as_delta(sdf, "/tmp/msh/lambdalabs_pokemon_blip_captions.delta")

# COMMAND ----------

# MAGIC %sql optimize delta.`/tmp/msh/lambdalabs_pokemon_blip_captions.delta` zorder by id;

# COMMAND ----------

# MAGIC %sql select * from delta.`/tmp/msh/lambdalabs_pokemon_blip_captions.delta`

# COMMAND ----------

# MAGIC %sql select count(1) from delta.`/tmp/msh/lambdalabs_pokemon_blip_captions.delta`

# COMMAND ----------
