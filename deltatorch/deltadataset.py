import io
import logging
import math
import threading
from abc import abstractmethod, ABC
from dataclasses import dataclass
from queue import Queue
from queue import Empty

import numpy as np
import torch.distributed
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import IterableDataset
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import Optional, Callable, List, Tuple, Dict, Any

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
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        num_workers: int = 2,
        shuffle: bool = False,
        timeout: int = 15,
        queue_size: int = 25000,
    ) -> None:
        super().__init__()
        self.path = path
        self.field_specs = fields
        self.arrow_fields = {field.name: ds.field(field.name) for field in fields}
        self.use_fixed_rank = use_fixed_rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.timeout = timeout
        self.path = path
        self.queue_size = queue_size

        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.delta_table = None
        self.scanner = None
        self.workers = []

        self.loaded = False
        self.boundaries_set = False

    @abstractmethod
    def init_loading(self, path):
        pass

    def init_boundaries(self, path, init_start_end: bool = True):
        self.start = 0
        self.end = self.count()
        logger.debug(f"Dataset for path {path}. Count:{self.end}")

        if self.use_fixed_rank:
            if init_start_end:
                new_start = self.rank * self.end / self.num_ranks
                new_end = (self.rank + 1) * self.end / self.num_ranks
                self.start = new_start
                self.end = new_end
                logger.debug(
                    f"Using fixed rank.  Current rank: {self.rank} World size: {self.num_ranks}"
                )
                logger.debug(f"Start: {self.start} End: {self.end}")
        elif torch.distributed.is_initialized():
            self.num_ranks = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            if init_start_end:
                logger.debug(
                    f"Detected DDP process. Current rank: {self.rank} World size: {self.num_ranks}"
                )
                new_start = self.rank * self.end / self.num_ranks
                new_end = (self.rank + 1) * self.end / self.num_ranks
                logger.debug(
                    "This rank will use the following set of rows: {self.start}-{self.end}"
                )
                self.start = new_start
                self.end = new_end
        else:
            self.num_ranks = 1
            self.rank = 1
        self.boundaries_set = True

    @abstractmethod
    def stop(self):
        pass

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
        _delta_table = DeltaTable(self.path)
        _add_actions = _delta_table.get_add_actions().to_pandas()
        return _add_actions["num_records"].sum()

    def __iter__(self):
        if self.loaded:
            self.stop()
        self.init_loading(self.path)
        i = 0
        while True:
            try:
                item = self.queue.get(block=True, timeout=self.timeout)
                yield item
                i += 1
                # if i >= self.end:
                #    return
            except Empty:
                print("empty ", i)
                return

    def __len__(self):
        if not self.boundaries_set:
            self.init_boundaries(self.path)
        return int(self.end - self.start)
