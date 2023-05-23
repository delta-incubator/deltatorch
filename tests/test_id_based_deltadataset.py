from typing import Tuple

from deltalake.writer import write_deltalake
import pandas as pd
from torchtext.datasets import AG_NEWS

import pytest

from deltatorch import create_pytorch_dataloader, FieldSpec
from deltatorch.id_based_deltadataset import IDBasedDeltaDataset


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
    dataset = IDBasedDeltaDataset(
        delta_path,
        id_field="id",
        fields=[
            FieldSpec(
                "text",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
            FieldSpec(
                "class",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
        ],
        use_fixed_rank=False,
    )
    assert len(dataset) == train_len
    val = next(iter(dataset))
    assert len(val) == 2
    it = iter(dataset)
    for _ in range(train_len):
        item = next(it)
        assert item is not None
    del dataset


def test_sharded_read(delta_table_info):
    delta_path, train_len = delta_table_info
    dataset = IDBasedDeltaDataset(
        delta_path,
        id_field="id",
        fields=[
            FieldSpec(
                "text",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
            FieldSpec(
                "class",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
        ],
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
        id_field="id",
        fields=[
            FieldSpec(
                "text",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
            FieldSpec(
                "class",
                decode_numpy_and_apply_shape=None,
                load_image_using_pil=False,
                transform=None,
            ),
        ],
        use_fixed_rank=True,
        rank=2,
        num_ranks=4,
    )
    expected_shard_len = int(train_len / 4)
    assert len(dl) == expected_shard_len

    it = iter(dl)
    for _ in range(expected_shard_len):
        item = next(it)
        assert item is not None
        assert item["class"] is not None
        assert item["text"] is not None
    del dl
