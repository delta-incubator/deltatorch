# Databricks notebook source
# MAGIC %md
# MAGIC This example shows how to use deltatorch  with torch distributor to train on imagenet data with pytorch lightning.

# COMMAND ----------

# MAGIC %pip install pytorch-lightning git+https://github.com/delta-incubator/deltatorch

# COMMAND ----------

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from pyspark.sql.functions import col
from PIL import Image
import numpy as np
import io
import os
import torchmetrics.functional as FM
import logging
from math import ceil
import mlflow
from deltatorch import create_pytorch_dataloader, FieldSpec


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the dataset from delta table
# MAGIC
# MAGIC The train and validation dataset are stored in seperate delta tables. The content column which contains image in binary format and object_id column will be used to train the resnet model.

# COMMAND ----------

train_delta_path = "/dbfs/tmp/udhay/cv_datasets/imagenet_train.delta"
val_delta_path = "/dbfs/tmp/udhay/cv_datasets/imagenet_val.delta"

# COMMAND ----------

train_df = spark.read.format("delta").load(train_delta_path.replace("/dbfs", ""))
unique_object_ids = train_df.select("object_id").distinct().collect()
object_id_to_class_mapping = {
    unique_object_ids[idx].object_id: idx for idx in range(len(unique_object_ids))
}

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()["current_user()"]

experiment_path = f"/Users/{username}/imagenet-training"

# This is needed for later in the notebook
db_host = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .extraContext()
    .apply("api_url")
)
db_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# Manually create the experiment so that you can get the ID and can send it to the worker nodes for scaling
experiment = mlflow.set_experiment(experiment_path)

log_path = f"/dbfs/Users/{username}/imagenet_training_logger"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the model
# MAGIC
# MAGIC The following  uses resnet50 from torchvision and encapsulates it into [pl.LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

# COMMAND ----------

import time


class ImageNetClassificationModel(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float = 0.9, rank: int = 0):
        super().__init__()
        self.learn_rate = learning_rate
        self.momentum = momentum
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.state = {"epochs": 0}
        self.rank = rank

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)

        return optimizer

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        X, y = batch["content"], batch["object_id"]
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def on_train_epoch_start(self):
        print(f"Epoch {self.state['epochs']} started at {time.time()} seconds")
        self.state["epochs"] += 1

    def validation_step(self, batch, batch_idx):
        X, y = batch["content"], batch["object_id"]
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = FM.accuracy(pred, y, task="multiclass", num_classes=1000)

        # Roll validation up to epoch level
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return {"loss": loss, "acc": acc}


# COMMAND ----------


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 16,
        num_workers: int = 1,
        feature_column: str = "content",
        label_column: str = "object_id",
        id_column: str = "id",
        object_id_to_class_mapping: dict = object_id_to_class_mapping,
    ):
        super().__init__()
        self.save_hyperparameters()

    def feature_transform(self, image):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform(image)

    def target_transform(self, object_id):
        return self.hparams.object_id_to_class_mapping[object_id]

    def dataloader(self, path: str):
        return create_pytorch_dataloader(
            path,
            id_field=self.hparams.id_column,
            fields=[
                FieldSpec(
                    self.hparams.feature_column,
                    load_image_using_pil=True,
                    transform=self.feature_transform,
                ),
                FieldSpec(self.hparams.label_column, transform=self.target_transform),
            ],
            num_workers=self.hparams.num_workers,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            timeout=30,
            queue_size=1000,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_path)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_path)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the training function
# MAGIC
# MAGIC The TorchDistributor API has support for single node multi-GPU training as well as multi-node training. The following `pl_train` function takes the parameters `num_tasks` and `num_proc_per_task`.
# MAGIC
# MAGIC For additional clarity:
# MAGIC - `num_tasks` (which sets `pl.Trainer(num_nodes=num_tasks, **kwargs)`) is the number of **Spark Tasks** you want for distributed training.
# MAGIC - `num_proc_per_task` (which sets `pl.Trainer(devices=num_proc_per_task, **kwargs)`) is the number of devices/GPUs you want per **Spark task** for distributed training.
# MAGIC
# MAGIC If you are running single node multi-GPU training on the driver, set `num_tasks` to 1 and `num_proc_per_task` to the number of GPUs that you want to use on the driver.
# MAGIC
# MAGIC If you are running multi-node training, set `num_tasks` to the number of spark tasks you want to use and `num_proc_per_task` to the value of `spark.task.resource.gpu.amount` (which is usually 1).
# MAGIC
# MAGIC Therefore, the total number of GPUs used is `num_tasks * num_proc_per_task`

# COMMAND ----------

BATCH_SIZE = 212
MAX_EPOCHS = 2
num_workers = 4


def main_training_loop(num_tasks, num_proc_per_task):
    ############################
    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    # NCCL P2P can cause issues with incorrect peer settings, so let's turn this off to scale for now
    os.environ["NCCL_P2P_DISABLE"] = "1"

    mlf_logger = pl.loggers.MLFlowLogger(experiment_name=experiment_path)

    if num_tasks > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=num_proc_per_task,
        num_nodes=num_tasks,
        max_epochs=MAX_EPOCHS,
        default_root_dir=log_path,
        logger=mlf_logger,
    )

    print(f"strategy is {trainer.strategy}")
    print(f"Global Rank: {trainer.global_rank}")
    print(f"Local Rank: {trainer.local_rank}")

    print(f"World Size: {trainer.world_size}")

    datamodule = ImageNetDataModule(
        train_path=train_delta_path,
        val_path=val_delta_path,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    model = ImageNetClassificationModel(learning_rate=1e-5)

    trainer.fit(model, datamodule)

    return model, trainer.checkpoint_callback.best_model_path


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the model locally with 1 GPU
# MAGIC
# MAGIC Note that `nnodes` = 1 and `nproc_per_node` = 1.

# COMMAND ----------

# NUM_TASKS = 1
# NUM_PROC_PER_TASK = 1
# model, ckpt_path = main_training_loop(NUM_TASKS, NUM_PROC_PER_TASK)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-node setup

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor

# NUM_TASKS = 1
# NUM_GPUS_PER_WORKER = 1
# NUM_PROC_PER_TASK = NUM_GPUS_PER_WORKER
# NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK
# (model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=True, use_gpu=True).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-node setup
# MAGIC
# MAGIC For the distributor API, you want to set `num_processes` to the total amount of GPUs that you plan on using. For multi-node, this will be equal to `num_spark_tasks * num_gpus_per_spark_task`. Additionally, note that `num_gpus_per_spark_task` usually equals 1 unless you configure that value specifically.
# MAGIC
# MAGIC Note that multi-node (with `num_proc` GPUs) setup involves setting `trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=num_proc, **kwargs)`

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

NUM_NODES = 4
NUM_GPUS_PER_WORKER = 4
NUM_TASKS = NUM_NODES * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1
NUM_PROC = int(NUM_TASKS / NUM_PROC_PER_TASK)
model, ckpt_path = TorchDistributor(
    num_processes=NUM_PROC, local_mode=False, use_gpu=True
).run(main_training_loop, NUM_TASKS, NUM_PROC_PER_TASK)
