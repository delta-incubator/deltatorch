from deltalake.writer import write_deltalake
import pandas as pd
from torchtext.datasets import AG_NEWS

import pytest

from torchdelta.deltapipe import DeltaDataPipe


@pytest.fixture
def delta_path(tmpdir) -> str:
    train_iter = AG_NEWS(split="train")
    train_list = list(train_iter)
    classes, texts = list(zip(*train_list))
    df = pd.DataFrame(
        columns=["class", "text"], data={"class": list(classes), "text": texts}
    )
    df["id"] = range(len(df))
    _delta_path = str(tmpdir / "ag_news.delta")
    write_deltalake(_delta_path, df)
    return _delta_path


def test_data_loader(delta_path):
    num_ranks = 16
    pipe = DeltaDataPipe(
        delta_path, fields=["class", "text"], id_field="id", use_fixed_rank=False
    )
    pipe_with_rank = DeltaDataPipe(
        delta_path,
        fields=["class", "text"],
        id_field="id",
        use_fixed_rank=True,
        fixed_rank=3,
        num_ranks=num_ranks,
    )
    assert len(pipe) == len(pipe_with_rank) * num_ranks


def test_batch_read(delta_path):
    num_ranks = 16
    batch_size = 512

    pipe = DeltaDataPipe(
        delta_path,
        fields=["class", "text"],
        id_field="id",
        use_fixed_rank=True,
        fixed_rank=3,
        num_ranks=num_ranks,
        batch_size=batch_size,
    )
    assert len(next(iter(pipe))) == batch_size
