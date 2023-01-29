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

# spark_write_path = "/tmp/msh/datasets/caltech256"
train_read_path = "/dbfs/tmp/msh/datasets/caltech256"

# train_read_path = "/tmp/datasets/caltech256"

# if locals().get("spark") is not None:
#    train_read_path = f"/dbfs{train_read_path}"

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

    def dataloader(self, path: str, shuffle=False, batch_size=32, num_workers=0):
        from torchvision.datasets import CIFAR10, CIFAR100, Caltech256

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
            #Caltech256(root='/tmp', download=True, transform=self.transform ),
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    def train_dataloader(self):
        return self.dataloader(
            f"{train_read_path}_train.delta",
            shuffle=True,
            batch_size=384,
            num_workers=2,
        )

    def val_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")

    def test_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")


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
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

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

    torch.set_float32_matmul_precision("medium")

    dm = DeltaDataModule()

    model = LitModel(dm.num_classes)

    # early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    # checkpoint_callback = pl.callbacks.ModelCheckpoint()

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=5,
        # auto_scale_batch_size=True,
        # reload_dataloaders_every_n_epochs=1
        # gpus=1,
        # callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(dataloaders=dm.test_dataloader())
