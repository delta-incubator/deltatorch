# Databricks notebook source
# MAGIC %pip install kaggle
# MAGIC %pip install xmltodict

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists cv_datasets
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC USE cv_datasets

# COMMAND ----------

# DBTITLE 1,Setup credentials for kaggle access
import os

os.environ[
    "kaggle_username"
] = "YOUR KAGGLE USERNAME HERE"  # replace with your own credential here temporarily or set up a secret scope with your credential

os.environ[
    "kaggle_key"
] = "YOUR KAGGLE KEY HERE"  # replace with your own credential here temporarily or set up a secret scope with your credential

# COMMAND ----------

# DBTITLE 1,Download imagenet data from kaggle
# MAGIC %sh
# MAGIC cd /local_disk0/
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle competitions download -c imagenet-object-localization-challenge
# MAGIC unzip -o imagenet-object-localization-challenge.zip

# COMMAND ----------

# DBTITLE 1,Parallel copy of data from local disk to dbfs
# SuperFastPython.com
# copy files from one directory to another concurrently with threads in batch
import glob
from pathlib import Path
from os import makedirs
from os import listdir
from os.path import join
from shutil import copy
from concurrent.futures import ThreadPoolExecutor


# copy files from source to destination
def copy_files(src_paths, dest_dir, file_pattern):
    # process all file paths
    for src_path in src_paths:
        # copy source file to dest file
        dest_path = copy(src_path, dest_dir)
        # report progress
        print(f".copied {src_path} to {dest_path}")


# copy files from src to dest
def copy_parallel(src="tmp", dest="tmp2", file_extension=".xml"):
    # create the destination directory if needed
    makedirs(dest, exist_ok=True)
    # create full paths for all files we wish to copy
    files = list(Path(src).rglob(file_extension))
    print("Total files to copy {}".format(len(files)))
    # determine chunksize
    n_workers = 100
    chunksize = round(len(files) / n_workers)
    # create the process pool
    with ThreadPoolExecutor(n_workers) as exe:
        # split the copy operations into chunks
        for i in range(0, len(files), chunksize):
            # select a chunk of filenames
            filenames = files[i : (i + chunksize)]
            # submit the batch copy task
            _ = exe.submit(copy_files, filenames, dest)
    print("Done")


# COMMAND ----------

# DBTITLE 1,Copy annotations
# Example
source_dir = ""
target_dir = ""

copy_parallel(src=source_dir, dest=target_dir, file_pattern="*.xml")

# COMMAND ----------

# DBTITLE 1,Copy images
# Example
source_dir = ""
target_dir = ""

copy_parallel(src=source_dir, dest=target_dir, file_pattern="*.JPEG")

# COMMAND ----------

df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.JPEG")
    .option("recursiveFileLookup", "true")
    .load(target_dir)
)

# COMMAND ----------

from pyspark.sql import functions as f
import pandas as pd
import json
import xmltodict
from pathlib import Path
from typing import Dict


def get_xml_file_content_as_dict(xml_file_path):
    xml_file = Path(xml_file_path)
    if xml_file.exists():
        with open(xml_file_path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
    else:
        data_dict = {}
    return data_dict


@f.pandas_udf("string")
def extract_annotation_udf(img_path_col: pd.Series) -> pd.Series:
    def xml_to_json_annotation(img_path):
        xml_file_path = (
            img_path.replace("dbfs:", "/dbfs")
            .replace("Data", "Annotations")
            .replace(".JPEG", ".xml")
        )
        print(xml_file_path)
        data_dict = get_xml_file_content_as_dict(xml_file_path)
        json_data = json.dumps(data_dict)
        return json_data

    return img_path_col.map(lambda img_path: xml_to_json_annotation(img_path))


@f.pandas_udf("string")
def extract_object_udf(annotation_col: pd.Series) -> pd.Series:
    def extract_object(json_str):
        objects = json.loads(json_str).get("annotation").get("object")
        if isinstance(objects, Dict):
            return objects.get("name")
        else:
            return objects[0].get("name")

    return annotation_col.map(lambda json_str: extract_object(json_str))


# COMMAND ----------

spark_write_path = "/tmp/udhay/cv_datasets/imagenet"

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import rand, row_number, lit


def store_as_delta(df, path):
    df1 = df.rdd.zipWithIndex().toDF()
    df2 = df1.select(col("_1.*"), col("_2").alias("id"))
    df2.write.format("delta").mode("overwrite").save(path)


# COMMAND ----------

from pyspark.sql.functions import split, element_at, from_json

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

train_df = df.filter(~df.path.contains("val") & ~df.path.contains("test"))
train_df = train_df.withColumn("annotation", extract_annotation_udf("path"))
train_df = train_df.withColumn(
    "object_id", split(element_at(split(train_df.path, "/"), -1), "_").getItem(0)
)
store_as_delta(train_df, f"{spark_write_path}_train.delta")

# COMMAND ----------

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

val_df = df.filter(df.path.contains("val"))
val_df = val_df.withColumn("annotation", extract_annotation_udf("path"))
val_df = val_df.withColumn("object_id", extract_object_udf("annotation"))
store_as_delta(val_df, f"{spark_write_path}_val.delta")

# COMMAND ----------

# MAGIC %sql optimize delta.`/tmp/udhay/cv_datasets/imagenet_train.delta` zorder by id

# COMMAND ----------

# MAGIC %sql optimize delta.`/tmp/udhay/cv_datasets/imagenet_val.delta` zorder by id
