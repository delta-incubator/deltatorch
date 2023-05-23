from typing import Tuple

from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import pandas as pd
from torchtext.datasets import AG_NEWS

import pytest

from deltatorch import create_pytorch_dataloader, FieldSpec
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

    dataset = SkipReadDeltaDataset(
        delta_path,
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
        shuffle=False,
    )
    assert len(dataset) == train_len
    val = next(iter(dataset))
    assert len(val) == 2
    i = 0
    for _ in dataset:
        i += 1
    print(i)

    del dataset
