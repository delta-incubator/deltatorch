# Databricks notebook source
# MAGIC %md # Training Image Classification on Caltech 256 using native PyTorch
# COMMAND ----------

# MAGIC %sh cd ../.. && pip install  .

# COMMAND ----------

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision import transforms, models

from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

# COMMAND ----------

train_path = "/dbfs/tmp/msh/datasets/caltech256_duplicated_x10_train.delta"
test_path = "/dbfs/tmp/msh/datasets/caltech256_duplicated_x10_test.delta"

# COMMAND ----------


def create_dataloader(path: str, batch_size=32) -> DataLoader:
    _transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # _num_classes = 257
    return create_pytorch_dataloader(
        path,
        id_field="id",
        fields=[
            FieldSpec("image", load_image_using_pil=True, transform=_transforms),
            FieldSpec("label"),
        ],
        shuffle=True,
        batch_size=batch_size,
    )


def train_epoch(
    rank: int, model: nn.Module, data_loader: DataLoader, optimizer: Optimizer
):
    model.train()
    for batch_idx, batch in enumerate(data_loader):
        image = batch["image"]
        label = batch["label"]
        image, label = image.to(rank), label.to(rank)
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(image), len(data_loader.dataset),
        #                100. * batch_idx / len(data_loader), loss.data.item()))


def test_epoch(
    rank: int,
    model: nn.Module,
    data_loader: DataLoader,
):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, batch in enumerate(data_loader):
        image = batch["image"]
        label = batch["label"]
        image, label = image.to(rank), label.to(rank)
        image, label = Variable(image), Variable(label)
        output = model(image)
        test_loss += F.nll_loss(
            output, label, size_average=False
        ).data.item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )


# COMMAND ----------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_classes = 257
        self.fc1 = nn.Linear(1000, self.num_classes)
        self.model = models.mobilenet_v3_large()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


# COMMAND ----------


def train(lr: int = 0.0001, epochs: int = 1, backend: str = "nccl"):
    torch.distributed.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    train_loader = create_dataloader(train_path, batch_size=64)
    val_loader = create_dataloader(test_path, batch_size=64)
    model = DistributedDataParallel(
        Net().to(local_rank), device_ids=[local_rank], output_device=local_rank
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train_epoch(local_rank, model, train_loader, optimizer)
        test_epoch(local_rank, model, val_loader)


# train()
# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor

distributed = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True)
distributed.run(train)


# COMMAND ----------
