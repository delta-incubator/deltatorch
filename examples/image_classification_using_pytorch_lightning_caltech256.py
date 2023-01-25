# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

from functools import partial

import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms, models
from torchvision.datasets import CIFAR10

from torchdelta import DeltaDataPipe
from torchdelta.deltadataset import DeltaIterableDataset

# COMMAND ----------

spark_write_path = "/tmp/msh/datasets/caltech256"
train_read_path = "/tmp/msh/datasets/caltech256"
if locals().get("spark") is not None:
    train_read_path = f"/dbfs{train_read_path}"
# COMMAND ----------


class DeltaDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.num_classes = 256

    def dataloader(self, path: str, shuffle=False, batch_size=32, num_workers=0):
        dataset = DeltaIterableDataset(
            path,
            src_field="image",
            target_field="label",
            id_field="id",
            use_fixed_rank=False,
            transform=self.transform,
            load_pil=True,
            num_workers=num_workers if num_workers > 0 else 2,
            shuffle=True
            # fixed_rank=3,
            # num_ranks=4,
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )

    def train_dataloader(self):
        return self.dataloader(
            f"{train_read_path}_train.delta",
            shuffle=False,
            batch_size=128,
            num_workers=8,
        )

    def val_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")

    def test_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv3 = nn.Conv2d(32, 64, 3, 1)
        # self.conv4 = nn.Conv2d(64, 64, 3, 1)
        #
        # self.pool1 = torch.nn.MaxPool2d(2)
        # self.pool2 = torch.nn.MaxPool2d(2)
        #
        # n_sizes = self._get_conv_output(input_shape)
        #
        self.fc1 = nn.Linear(1000, num_classes)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=256)

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

    # returns the size of the output tensor going into Linear layer from the conv block.
    # def _get_conv_output(self, shape):
    #     batch_size = 1
    #     input = torch.autograd.Variable(torch.rand(batch_size, *shape))
    #
    #     output_feat = self._forward_features(input)
    #     n_size = output_feat.data.view(batch_size, -1).size(1)
    #     return n_size

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc1(x), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
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

    dm = DeltaDataModule()

    model = LitModel((3, 32, 32), dm.num_classes)

    # Initialize wandb logger

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=5,
        auto_scale_batch_size=True,
        # reload_dataloaders_every_n_epochs=1
        # gpus=0,
        # callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(dataloaders=dm.test_dataloader())
