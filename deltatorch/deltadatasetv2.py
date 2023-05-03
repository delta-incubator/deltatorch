from time import sleep

from deltalake import DeltaTable
from torch.utils.data import get_worker_info, IterableDataset
from typing import Optional, List

from pyarrow.dataset import FileSystemDataset


def make_batch_slice_iterator(
    batch_iterator, offset, size, batch_size, batches_per_epoch
):
    """
    batch_iterator: batching defined by source
    offset: where to start in batch stream
    size: size of distributed resources
    batch_size: expected size of each batch
    batches_per_epoch: early termination boundary
    """
    slice_start = offset * batch_size
    M = batch_size
    batch_count = 0

    stream_offset_start = 0
    for batch in batch_iterator:
        N = batch.num_rows
        stream_offset_end = stream_offset_start + N

        if slice_start >= stream_offset_end:
            stream_offset_start = stream_offset_end
            continue

        while slice_start < stream_offset_end:
            assert slice_start >= stream_offset_start
            offset = slice_start - stream_offset_start
            length = min(N - offset, M)

            # print(offset, length, stream_offset_start, stream_offset_end)
            for rec in batch.slice(offset, length).to_pylist():
                yield rec

            slice_start += length
            M -= length
            if M <= 0:
                # update batch indexing
                slice_start += (size - 1) * batch_size
                M = batch_size
                batch_count += 1

        stream_offset_start = stream_offset_end
        if batch_count >= batches_per_epoch:
            break


class DeltaIterableDataset2(IterableDataset):
    def __init__(
        self,
        path: str,
        fields: List[str],
        version: Optional[int] = None,
        batch_size: int = 64,
        batches_per_epoch: int = 128,
        pad_batches: bool = False,
        deterministic_file_order: bool = False,
    ):
        super(DeltaIterableDataset2).__init__()
        self.path = path
        self.fields = fields

        self.version = version
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.pad_batches = pad_batches
        self.deterministic_file_order = deterministic_file_order

    def to_iterator(self, rank=None, size=None):
        offset = rank
        if offset is None:
            offset = 0

        cycle_length = size
        if cycle_length is None:
            cycle_length = 1

        delta_table = DeltaTable(self.path, self.version)
        py_dataset = delta_table.to_pyarrow_dataset()

        if self.deterministic_file_order:
            # How deterministic should the iterator be?
            raw_files = py_dataset.files
            raw_files.sort()

            py_dataset = FileSystemDataset.from_paths(
                paths=raw_files,
                schema=py_dataset.schema,
                format=py_dataset.format,
                filesystem=py_dataset.filesystem,
            )

        scanner = py_dataset.scanner(batch_size=self.batch_size, columns=self.fields)

        # NOTE: Neither Pyarrow Scanner to_batch or to_reader create batches that span files (last batch of a file may not be full)
        return make_batch_slice_iterator(
            scanner.to_batches(),
            offset=offset,
            size=cycle_length,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
        )

    def __iter__(self):
        worker_info = get_worker_info()
        rank = None
        size = None
        if worker_info is not None:
            # single process
            rank = worker_info.id
            size = worker_info.num_workers

        return worker_info.dataset.to_iterator(rank, size)
