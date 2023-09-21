import io
import logging
import math
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch.distributed
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import IterableDataset
import evalidate

logger = logging.getLogger(__name__)


@dataclass
class FieldSpec(ABC):
    name: str
    decode_numpy_and_apply_shape: Optional[Tuple[int, int]] = None
    load_image_using_pil: bool = False
    transform: Optional[Callable] = None
    target_name: Optional[str] = None


class DeltaIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        fields: List[FieldSpec],
        partition_filter: Optional[List[Tuple[str, str, Any]]] = None,
        version: Optional[int] = None,
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        num_workers: int = 2,
        shuffle: bool = False,
        batch_size: int = 32,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        self.version = version
        self.partition_filter = partition_filter
        self.field_specs = fields
        self.arrow_fields = {field.name: ds.field(field.name) for field in fields}
        self.use_fixed_rank = use_fixed_rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.path = path
        self.batch_size = batch_size
        self.init_boundaries(path)

    @abstractmethod
    def process_data(self):
        pass

    def init_boundaries(self, path, init_start_end: bool = True):
        self.start = 0
        self.end = self.count()
        logger.debug(f"Dataset for path {path}. Count:{self.end}")

        if self.use_fixed_rank:
            if init_start_end:
                self.start, self.end = self.calc_boundaries(
                    self.start, self.end, self.rank, self.num_ranks
                )
                logger.debug(
                    f"Using fixed rank.  Current rank: {self.rank} "
                    f"World size: {self.num_ranks}"
                )
                logger.debug(f"Start: {self.start} End: {self.end}")
        elif torch.distributed.is_initialized():
            self.num_ranks = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            logger.debug(
                f"Detected DDP process. "
                f"Current rank: {self.rank} World size: {self.num_ranks}"
            )
            if init_start_end:
                self.start, self.end = self.calc_boundaries(
                    self.start, self.end, self.rank, self.num_ranks
                )
                logger.debug(
                    f"This rank will use the following "
                    f"set of rows: {self.start}-{self.end}"
                )
        else:
            self.num_ranks = 1
            self.rank = 1

    @staticmethod
    def calc_boundaries(start, end, rank, num_ranks):
        per_worker_data_count = int(math.ceil((end - start) / float(num_ranks)))
        new_start = start + rank * per_worker_data_count
        new_end = min(start + (rank + 1) * per_worker_data_count, end)
        return new_start, new_end

    @staticmethod
    def decode_and_transform_record(
        item: Dict[str, Any], field_specs: List[FieldSpec]
    ) -> Dict[str, Any]:
        for spec in field_specs:
            _target_name = spec.target_name if spec.target_name else spec.name
            _item = item[spec.name]
            if spec.decode_numpy_and_apply_shape:
                _item = np.frombuffer(_item, dtype=np.uint8).reshape(
                    spec.decode_numpy_and_apply_shape
                )
            if spec.load_image_using_pil:
                _item = Image.open(io.BytesIO(_item))
            if spec.transform:
                _item = spec.transform(_item)
            item[_target_name] = _item
        return item

    def count(self):
        _delta_table = self.create_delta_table()

        if self.partition_filter:
            return self.count_with_partition_filters(_delta_table)
        else:
            _add_actions = _delta_table.get_add_actions().to_pandas()
            return _add_actions["num_records"].sum()

    def count_with_partition_filters(self, _delta_table):
        _cnt = 0
        _add_actions_list = _delta_table.get_add_actions().to_pylist()
        _partition_exprs = " and ".join(["".join(e) for e in self.partition_filter])
        _partition_exprs = _partition_exprs.replace("=", "==")
        _partition_exprs = _partition_exprs.replace("true", "True")
        _partition_exprs = _partition_exprs.replace("false", "False")
        for item in _add_actions_list:
            if evalidate.Expr(_partition_exprs).eval(item["partition_values"]):
                _cnt += item["num_records"]
        return _cnt

    def create_delta_table(self):
        return DeltaTable(self.path, version=self.version)

    def __iter__(self):
        return self.process_data()

    def __len__(self):
        # if self.drop_last:
        #     per_machine_length = int(self.end - self.start)
        #     per_worker_length = int(per_machine_length / self.num_workers)
        #     number_of_batches_per_worker = per_worker_length // self.batch_size
        #     batch_size_adjusted_per_machine_length = (
        #         number_of_batches_per_worker * self.batch_size * self.num_workers
        #     )
        #     return batch_size_adjusted_per_machine_length
        # else:
        return int(self.end - self.start)
