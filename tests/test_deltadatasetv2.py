from typing import Tuple

from deltalake.writer import write_deltalake
import pandas as pd
import numpy as np
from torchtext.datasets import AG_NEWS

import pytest

from deltatorch import DeltaIterableDataset, DeltaIterableDataset2
from deltatorch import create_pytorch_dataloader

from deltalake import DeltaTable
from pyarrow.dataset import FileSystemDataset


@pytest.fixture
def simple_delta_table_info(tmpdir) -> str:
    DATA_ROWS = 100
    MAX_ROW_PER_FILE = 20
    df = pd.DataFrame({"id": list(range(DATA_ROWS)), "rvalue": [np.random.rand() for i in range(DATA_ROWS)]})
    _path = f"{tmpdir}/random_values.delta"
    # 5 part delta table
    write_deltalake(_path, df, max_rows_per_file=MAX_ROW_PER_FILE, max_rows_per_group=MAX_ROW_PER_FILE, min_rows_per_group=0)
    return _path


def test_read_dataset2(simple_delta_table_info):
    delta_path = simple_delta_table_info

    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 1

    ds = DeltaIterableDataset2(
        path=delta_path,
        fields=["id", "rvalue"],
        batch_size=BATCH_SIZE,
        batches_per_epoch=BATCHES_PER_EPOCH,
        pad_batches=False,
        deterministic_file_order=True
    )

    last_record_id = None
    for i, rec in enumerate(ds.to_iterator()):
        # print(rec)
        last_record_id = rec["id"]
        # print(last_record_id)

    # last record index should be total records - 1
    assert i == ((BATCH_SIZE * BATCHES_PER_EPOCH) - 1)


def test_read_dataset2_workers_base(simple_delta_table_info):
    delta_path = simple_delta_table_info

    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 1

    ds = DeltaIterableDataset2(
        path=delta_path,
        fields=["id"],
        batch_size=BATCH_SIZE,
        batches_per_epoch=BATCHES_PER_EPOCH,
        pad_batches=False,
        deterministic_file_order=True
    )

    # dt = DeltaTable(delta_path)
    # pa_ds = dt.to_pyarrow_dataset()
    # dt_files = pa_ds.files
    # dt_files.sort()

    last_record_id = None
    for i, rec in enumerate(ds.to_iterator(rank=0, size=2)):
        last_record_id = rec["id"]
    print(last_record_id)

    # last record index should be total records - 1
    assert i == ((BATCH_SIZE * BATCHES_PER_EPOCH) - 1)

    last_record_id = None
    for i, rec in enumerate(ds.to_iterator(rank=1, size=2)):
        last_record_id = rec["id"]
    print(last_record_id)

    # py_dataset = FileSystemDataset.from_paths(paths=dt_files[1:2],
    #                                           schema=pa_ds.schema,
    #                                           format=pa_ds.format,
    #                                           filesystem=pa_ds.filesystem)
    # for batch in py_dataset.scanner(columns=["id"]).to_batches():
    #     for rec in batch.to_pylist():
    #         print(rec)

    # last record index should be total records - 1
    assert i == ((BATCH_SIZE * BATCHES_PER_EPOCH) - 1)


def test_read_dataset2_workers_multi(simple_delta_table_info):
    delta_path = simple_delta_table_info

    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 3

    ds = DeltaIterableDataset2(
        path=delta_path,
        fields=["id"],
        batch_size=BATCH_SIZE,
        batches_per_epoch=BATCHES_PER_EPOCH,
        pad_batches=False,
        deterministic_file_order=True
    )

    # dt = DeltaTable(delta_path)
    # pa_ds = dt.to_pyarrow_dataset()
    # dt_files = pa_ds.files
    # dt_files.sort()

    t0_ids = []
    for i, rec in enumerate(ds.to_iterator(rank=0, size=2)):
        t0_ids.append(rec["id"])

    # last record index should be total records - 1
    assert i == ((BATCH_SIZE * BATCHES_PER_EPOCH) - 1)

    t1_ids = []
    for i, rec in enumerate(ds.to_iterator(rank=1, size=2)):
        t1_ids.append(rec["id"])
    assert i == ((BATCH_SIZE * BATCHES_PER_EPOCH) - 1)

    # assert that record sets were non-overlapping over the number epochs and simulated threads
    assert len(set(t0_ids).intersection(set(t1_ids))) == 0
