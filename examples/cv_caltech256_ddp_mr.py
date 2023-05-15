# Databricks notebook source
# MAGIC %pip install torch==2.0.0 pytorch-lightning==2.0.0

# COMMAND ----------

# MAGIC %sh cd .. && pip install  .

# COMMAND ----------

import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics import Accuracy

from torchvision import transforms, models

from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

# COMMAND ----------

train_path = "/dbfs/tmp/msh/datasets/caltech256_duplicated_x10_train.delta"
test_path = "/dbfs/tmp/msh/datasets/caltech256_duplicated_x10_test.delta"

# COMMAND ----------

class DeltaDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
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

        self.num_classes = 257

    def dataloader(self, path: str, batch_size=32):
        return create_pytorch_dataloader(
            path,
            id_field="id",
            fields=[
                FieldSpec("image", load_image_using_pil=True, transform=self.transform),
                FieldSpec("label"),
            ],
            shuffle=True,
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(
            train_path,
            batch_size=384,
        )

    def val_dataloader(self):
        return self.dataloader(test_path)

    def test_dataloader(self):
        return self.dataloader(test_path)


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(1000, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = models.mobilenet_v3_large()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# COMMAND ----------

def train_distributed(max_epochs: int = 1, strategy: str = "auto"):
    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    torch.set_float32_matmul_precision("medium")

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        default_root_dir="/dbfs/tmp/trainer_logs",
        max_epochs=max_epochs, callbacks=[pl.callbacks.EarlyStopping(monitor="train_loss"), pl.callbacks.ModelCheckpoint()]
    )

    print(f"Global Rank: {trainer.global_rank}")
    print(f"Local Rank: {trainer.local_rank}")

    print(f"World Size: {trainer.world_size}")

    dm = DeltaDataModule()

    model = LitModel(dm.num_classes)

    trainer.fit(model, dm)

    print("Training done!")

    trainer.test(dataloaders=dm.test_dataloader())
    print("Test done!")

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

distributed = TorchDistributor(num_processes=8, local_mode=True, use_gpu=True)
distributed.run(train_distributed, 1, "ddp")

# COMMAND ----------

distributed = TorchDistributor(num_processes=8, local_mode=True, use_gpu=True)
distributed.run(train_distributed, 10, "ddp")

# COMMAND ----------


