from typing import Tuple

from deltalake.writer import write_deltalake
import pandas as pd
from torchtext.datasets import AG_NEWS

import pytest

from deltatorch import create_pytorch_dataloader, FieldSpec
from deltatorch.id_based_deltadataset import IDBasedDeltaDataset


def create_delta_table(tmpdir, num_rows=-1) -> Tuple[str, int]:
    train_iter = AG_NEWS(split="train")
    train_list = list(train_iter)
    if num_rows > 0:
        train_list = train_list[:num_rows]
    train_len = len(train_list)
    classes, texts = list(zip(*train_list))
    df = pd.DataFrame(
        columns=["class", "text"], data={"class": list(classes), "text": texts}
    )
    df["id"] = range(len(df))
    _delta_path = str(tmpdir / "ag_news.delta")
    write_deltalake(_delta_path, df)
    return _delta_path, train_len


def test_simple_read(tmpdir):
    delta_path, train_len = create_delta_table(tmpdir)
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

    it = iter(dataset)
    for _ in range(train_len):
        item = next(it)
        assert item is not None
    del dataset


def test_sharded_read(tmpdir):
    delta_path, train_len = create_delta_table(tmpdir)
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


def test_pt_dataloader(tmpdir):
    delta_path, train_len = create_delta_table(tmpdir)

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


def test_read_different_length(tmpdir):
    delta_path, train_len = create_delta_table(tmpdir, num_rows=789)

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
        num_ranks=2,
        rank=1,
        shuffle=False,
    )
    assert len(dataset) == int(train_len / 2)
    i = 0
    for _ in dataset:
        i += 1
    print(i)

    del dataset
