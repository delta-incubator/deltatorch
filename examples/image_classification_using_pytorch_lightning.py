# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

from functools import partial

import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, BinaryType, IntegerType, LongType
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

from torchdelta import DeltaDataPipe

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


def prepare_cifar_data(is_train: bool = True):
    dataset = CIFAR10(".", train=is_train, download=True)
    spark.createDataFrame(
        zip(list(map(lambda x: x.tobytes(), dataset.data)), dataset.targets),
        StructType(
            [StructField("image", BinaryType()), StructField("label", LongType())]
        ),
    ).withColumn("id", monotonically_increasing_id()).write.format("delta").mode(
        "overwrite"
    ).save(
        f"{spark_write_path}_{'train' if is_train else 'test'}.delta"
    )


prepare_cifar_data(True)
prepare_cifar_data(False)

# COMMAND ----------


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        def transform_fn(bytes, transform):
            img = Image.fromarray(
                np.frombuffer(bytes, dtype=np.uint8).reshape((32, 32, 3))
            )

            if transform is not None:
                img = transform(img)
            return img

        self.transform_fn = partial(transform_fn, transform=self.transform)

        self.num_classes = 10

    def dataloader(self, path: str, shuffle=False):
        pipe = DeltaDataPipe(
            path,
            fields=["image", "label"],
            id_field="id",
            use_fixed_rank=False,
            #fixed_rank=3,
            #num_ranks=4,
        )
        pipe = pipe.map(self.transform_fn, input_col="image", output_col="image").map(
            lambda x: (x["image"], x["label"])
        )
        return DataLoader(pipe, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self.dataloader(f"{train_read_path}_train.delta", shuffle=True)

    def val_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")

    def test_dataloader(self):
        return self.dataloader(f"{train_read_path}_test.delta")


"""## 📱 Callbacks

A callback is a self-contained program that can be reused across projects. PyTorch Lightning comes with few [built-in callbacks](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks) which are regularly used. 
Learn more about callbacks in PyTorch Lightning [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html).

### Built-in Callbacks

In this tutorial, we will use [Early Stopping](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping) and [Model Checkpoint](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) built-in callbacks. They can be passed to the `Trainer`.

### Custom Callbacks
If you are familiar with Custom Keras callback, the ability to do the same in your PyTorch pipeline is just a cherry on the cake.

Since we are performing image classification, the ability to visualize the model's predictions on some samples of images can be helpful. This in the form of a callback can help debug the model at an early stage.
"""

"""## 🎺 LightningModule - Define the System

The LightningModule defines a system and not a model. Here a system groups all the research code into a single class to make it self-contained. `LightningModule` organizes your PyTorch code into 5 sections:
- Computations (`__init__`).
- Train loop (`training_step`)
- Validation loop (`validation_step`)
- Test loop (`test_step`)
- Optimizers (`configure_optimizers`)

One can thus build a dataset agnostic model that can be easily shared. Let's build a system for Cifar-10 classification.
"""


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

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


"""## 🚋 Train and Evaluate

Now that we have organized our data pipeline using `DataModule` and model architecture+training loop using `LightningModule`, the PyTorch Lightning `Trainer` automates everything else for us. 

The Trainer automates:
- Epoch and batch iteration
- Calling of `optimizer.step()`, `backward`, `zero_grad()`
- Calling of `.eval()`, enabling/disabling grads
- Saving and loading weights
- Weights and Biases logging
- Multi-GPU training support
- TPU support
- 16-bit training support
"""

dm = CIFAR10DataModule(batch_size=32)


# Samples required by the custom ImagePredictionLogger callback to log image predictions.
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape

model = LitModel((3, 32, 32), dm.num_classes)

# Initialize wandb logger

# Initialize Callbacks
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# Initialize a trainer
trainer = pl.Trainer(
    max_epochs=2,
    gpus=0,
    callbacks=[early_stop_callback, checkpoint_callback],
)

# Train the model ⚡🚅⚡
trainer.fit(model, dm)

# Evaluate the model on the held-out test set ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

"""## Final Thoughts
I come from the TensorFlow/Keras ecosystem and find PyTorch a bit overwhelming even though it's an elegant framework. Just my personal experience though. While exploring PyTorch Lightning, I realized that almost all of the reasons that kept me away from PyTorch is taken care of. Here's a quick summary of my excitement:
- Then: Conventional PyTorch model definition used to be all over the place. With the model in some `model.py` script and the training loop in the `train.py `file. It was a lot of looking back and forth to understand the pipeline. 
- Now: The `LightningModule` acts as a system where the model is defined along with the `training_step`, `validation_step`, etc. Now it's modular and shareable.
- Then: The best part about TensorFlow/Keras is the input data pipeline. Their dataset catalog is rich and growing. PyTorch's data pipeline used to be the biggest pain point. In normal PyTorch code, the data download/cleaning/preparation is usually scattered across many files. 
- Now: The DataModule organizes the data pipeline into one shareable and reusable class. It's simply a collection of a `train_dataloader`, `val_dataloader`(s), `test_dataloader`(s) along with the matching transforms and data processing/downloads steps required.
- Then: With Keras, one can call `model.fit` to train the model and `model.predict` to run inference on. `model.evaluate` offered a good old simple evaluation on the test data. This is not the case with PyTorch. One will usually find separate `train.py` and `test.py` files. 
- Now: With the `LightningModule` in place, the `Trainer` automates everything. One needs to just call `trainer.fit` and `trainer.test` to train and evaluate the model.
- Then: TensorFlow loves TPU, PyTorch...well! 
- Now: With PyTorch Lightning, it's so easy to train the same model with multiple GPUs and even on TPU. Wow!
- Then: I am a big fan of Callbacks and prefer writing custom callbacks. Something as trivial as Early Stopping used to be a point of discussion with conventional PyTorch. 
- Now: With PyTorch Lightning using Early Stopping and Model Checkpointing is a piece of cake. I can even write custom callbacks.

## 🎨 Conclusion and Resources

I hope you find this report helpful. I will encourage to play with the code and train an image classifier with a dataset of your choice. 

Here are some resources to learn more about PyTorch Lightning:
- [Step-by-step walk-through](https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html) - This is one of the official tutorials. Their documentation is really well written and I highly encourage it as a good learning resource.
- [Use Pytorch Lightning with Weights & Biases](https://wandb.me/lightning) - This is a quick colab that you can run through to learn more about how to use W&B with PyTorch Lightning.
"""
