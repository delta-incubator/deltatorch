# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

from torchvision import transforms, models

from torchdelta.deltadataset import DeltaIterableDataset

# COMMAND ----------

spark_write_path = "dbfs:/tmp/msh/datasets/caltech256"
train_read_path = "/dbfs/tmp/msh/datasets/caltech256"

# train_read_path = "/tmp/datasets/caltech256"

# if locals().get("spark") is not None:
#    train_read_path = f"/dbfs{train_read_path}"

# COMMAND ----------

from pyspark.sql.functions import col
from petastorm.spark import SparkDatasetConverter, make_spark_converter

train_df = spark.read.load(
    "dbfs:/tmp/msh/datasets/caltech256_duplicated_train.delta"
).select(col("image"), col("label"))
test_df = spark.read.load(
    "dbfs:/tmp/msh/datasets/caltech256_duplicated_test.delta"
).select(col("image"), col("label"))

num_classes = train_df.select("label").distinct().count()
num_train_rows = train_df.count()
num_test_rows = test_df.count()

# COMMAND ----------

from petastorm.spark import make_spark_converter, SparkDatasetConverter

spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache"
)

# COMMAND ----------

import numpy as np
from PIL import Image
import io
from petastorm.pytorch import DataLoader
from petastorm import make_batch_reader
from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter


class DeltaDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path

        # self.train_df = train_df
        # self.test_df = test_df

        # self.train_conv =  make_spark_converter(self.train_df)
        # self.test_conv =  make_spark_converter(self.test_df)

        self.num_classes = 257

    def dataloader(self, path, batch_size):
        # def get_petastorm_urls(df):
        #    paths = df.inputFiles()
        #    return [path.replace('dbfs:/', 'file:///dbfs/') for path in paths]

        def preprocess(img):
            image = Image.open(io.BytesIO(img)).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            return transform(image)

        def transform_rows(batch):
            batch["image"] = batch["image"].map(lambda x: preprocess(x).numpy())
            # batch = batch.drop(labels=["id"], axis=1)
            return batch

        def get_transform_spec():
            return TransformSpec(
                transform_rows,
                edit_fields=[("image", np.float32, (3, 224, 224), False)],
                selected_fields=["image", "label"],
            )

        # return self.train_conv.make_torch_dataloader(
        #    transform_spec=get_transform_spec(), num_epochs=1, batch_size=batch_size
        # )
        reader = make_batch_reader(
            path, num_epochs=None, transform_spec=get_transform_spec()
        )
        return DataLoader(reader, batch_size=batch_size)

    def train_dataloader(self):
        return self.dataloader(self.train_path, 256)

    def val_dataloader(self):
        return self.dataloader(self.test_path, 64)

    def test_dataloader(self):
        return self.dataloader(self.test_path, 64)


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(1000, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

    def forward(self, x):

        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    dm = DeltaDataModule(
        train_path=f"file://{train_read_path}_train.parquet",
        test_path=f"file://{train_read_path}_test.parquet",
    )

    model = LitModel(dm.num_classes)

    # early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    # checkpoint_callback = pl.callbacks.ModelCheckpoint()

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1,
        limit_train_batches=int(num_train_rows / 256),
        # auto_scale_batch_size=True,
        # reload_dataloaders_every_n_epochs=1
        # gpus=1,
        # callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(dataloaders=dm.test_dataloader())

# COMMAND ----------
