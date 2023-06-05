import logging
import math
import random
from typing import List

import pyarrow.compute as pc
from deltalake import DeltaTable
from torch.utils.data import get_worker_info

from deltatorch import DeltaIterableDataset
from deltatorch.deltadataset import FieldSpec

logger = logging.getLogger(__name__)


class IDBasedDeltaDataset(DeltaIterableDataset):
    def __init__(
        self,
        path: str,
        id_field: str,
        fields: List[FieldSpec],
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        num_workers: int = 2,
        shuffle: bool = False,
        batch_size: int = 32,
        drop_last: bool = False,
    ):
        super().__init__(
            path,
            fields,
            use_fixed_rank,
            rank,
            num_ranks,
            num_workers,
            shuffle,
            batch_size,
            drop_last,
        )
        self.id_field = id_field

    def calc_chunk_boundaries_for_current_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.start, self.end
        else:
            # per_worker_data_count = int(
            #     math.ceil((self.end - self.start) / float(worker_info.num_workers))
            # )
            # worker_id = worker_info.id
            iter_start, iter_end = DeltaIterableDataset.calc_boundaries(
                self.start, self.end, worker_info.id, worker_info.num_workers
            )
            ##iter_start = self.start + worker_id * per_worker_data_count
            # iter_end = min(iter_start + per_worker_data_count, self.end)
        return iter_start, iter_end

    def process_data(self):
        _filter = None
        iter_start, iter_end = self.calc_chunk_boundaries_for_current_worker()
        if iter_end > 0 and iter_start >= 0:
            _filter = (pc.field(self.id_field) >= pc.scalar(iter_start)) & (
                pc.field(self.id_field) < pc.scalar(iter_end)
            )
        delta_table = DeltaTable(self.path)
        scanner = delta_table.to_pyarrow_dataset().scanner(
            columns=self.arrow_fields, filter=_filter
        )
        for rb in scanner.to_reader():
            num_rows = rb.num_rows
            indexes = list(range(num_rows))
            if self.shuffle:
                random.shuffle(indexes)
            for i in indexes:
                item = rb.slice(offset=i, length=1).to_pylist()[0]
                item = DeltaIterableDataset.decode_and_transform_record(
                    item, self.field_specs
                )
                yield item
