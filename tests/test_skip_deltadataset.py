import time
from typing import Tuple

from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import pandas as pd
from torchtext.datasets import AG_NEWS

import pytest

from deltatorch import DeltaIterableDataset
from deltatorch import create_pytorch_dataloader
from deltatorch.id_based_deltadataset import IDBasedDeltaDataset
from deltatorch.skip_batch_deltadataset import SkipReadDeltaDataset


@pytest.fixture
def delta_table_info(tmpdir) -> Tuple[str, int]:
    train_iter = AG_NEWS(split="train")
    train_list = list(train_iter)
    train_len = len(train_list)
    classes, texts = list(zip(*train_list))
    df = pd.DataFrame(
        columns=["class", "text"], data={"class": list(classes), "text": texts}
    )
    df["id"] = range(len(df))
    _delta_path = str(tmpdir / "ag_news.delta")
    write_deltalake(_delta_path, df)
    return _delta_path, train_len


def test_simple_read(delta_table_info):
    delta_path, train_len = delta_table_info
    dt = DeltaTable(delta_path)
    files = dt.files_by_partitions([])
    print(files)
    print(dt.metadata())
    print(dt.protocol())
    print(dt.version())
    print(dt.to_pandas().count()[0])
    print(dt.history(1))
    dataset = SkipReadDeltaDataset(
        delta_path,
        length=train_len,
        src_field="text",
        target_field="class",
        use_fixed_rank=False,
        shuffle=False
    )
    #assert len(dataset) == train_len
    val = next(iter(dataset))
    assert len(val) == 2
    i=0
    for item in dataset:
        i+=1
    print(i)


    del dataset


def test_sharded_read(delta_table_info):
    delta_path, train_len = delta_table_info

    dataset = IDBasedDeltaDataset(
        delta_path,
        length=train_len,
        id_field="id",
        src_field="text",
        target_field="class",
        use_fixed_rank=True,
        rank=2,
        num_ranks=4,
    )
    ds_len = len(dataset)
    expected_shard_len = int(train_len / 4)
    assert ds_len == expected_shard_len

    it = iter(dataset)
    for _ in range(expected_shard_len):
        item = next(it)
        assert item is not None
    del dataset


def test_pt_dataloader(delta_table_info):
    delta_path, train_len = delta_table_info

    dl = create_pytorch_dataloader(
        delta_path,
        length=train_len,
        id_field="id",
        src_field="text",
        target_field="class",
        use_fixed_rank=True,
        rank=2,
        num_ranks=4,
    )

    expected_shard_len = int(train_len / 4)

    it = iter(dl)
    for _ in range(expected_shard_len):
        item = next(it)
        assert item is not None
    del dl
