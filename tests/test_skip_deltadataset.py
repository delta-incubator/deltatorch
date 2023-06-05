import pathlib
from typing import Tuple

import pytest
from deltalake.writer import write_deltalake
import pandas as pd


from deltatorch import FieldSpec
from deltatorch.skip_batch_deltadataset import SkipReadDeltaDataset


def create_delta_table(tmpdir, num_rows=-1) -> Tuple[str, int]:
    df = pd.read_parquet(str(pathlib.Path.cwd() / "tests" / "data" / "ag_news.parquet"))
    df = df[:num_rows]
    df["id"] = range(len(df))
    _delta_path = str(tmpdir / "ag_news.delta")
    write_deltalake(_delta_path, df)
    return _delta_path, len(df)


@pytest.mark.skip
def test_simple_read(tmpdir):
    delta_path, train_len = create_delta_table(tmpdir)

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
