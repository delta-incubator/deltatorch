import logging
import math
import threading
from abc import abstractmethod
from queue import Queue
from queue import Empty

from deltalake import DeltaTable
from torch.utils.data import get_worker_info, IterableDataset
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class DeltaIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        length: int,
        src_field: str,
        target_field: str,
        apply_src_numpy_shape=None,
        load_pil: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
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
        self.fields = {
            src_field: ds.field(src_field),
            target_field: ds.field(target_field),
        }  # {field_name: ds.field(field_name) for field_name in fields}
        self.src_field = src_field
        self.target_field = target_field
        self.use_fixed_rank = use_fixed_rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_workers = num_workers
        self.apply_src_numpy_shape = apply_src_numpy_shape
        self.load_pil = load_pil
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.timeout = timeout
        self.length = length
        self.path = path
        self.queue_size = queue_size

        self.loaded = False

        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.delta_table = None
        self.scanner = None
        self.workers = []

    @abstractmethod
    def init_loading(self, length, path):
        pass

    @abstractmethod
    def stop(self):
        pass

    def _get_active_spark_session(self):
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            return None
        try:
            return SparkSession.getActiveSession()
        except Exception:
            return None

    def count(self):
        spark = self._get_active_spark_session()
        if spark is not None:
            logger.debug("Using spark to determine length..")
            _dbfs_path = self.path.replace("/dbfs/", "dbfs:/")
            return spark.read.format("delta").load(_dbfs_path).count()
        else:
            _cnt = 0
            for rb in self.init_scanner().to_reader():
                _cnt += rb.num_rows
            return _cnt

    def init_scanner_and_apply_rank(self):
        _info = get_worker_info()
        if self.use_fixed_rank:
            return self.init_scanner(
                use_rank=True, rank=self.rank, num_ranks=self.num_ranks
            )
        elif _info is not None:
            return self.init_scanner(
                use_rank=True, rank=_info.id, num_ranks=_info.num_workers
            )
        else:
            return self.init_scanner(use_rank=False)

    def init_scanner(self, use_rank: bool = False, rank: int = -1, num_ranks: int = -1):
        if self.delta_table is None:
            self.delta_table = DeltaTable(self.path)
            _filter = None
            if use_rank:
                per_worker = int(math.ceil((self.end - self.start) / float(num_ranks)))
                iter_start = self.start + rank * per_worker
                iter_end = min(iter_start + per_worker, self.end)
                _filter = (pc.field(self.id_field) >= pc.scalar(iter_start)) & (
                    pc.field(self.id_field) <= pc.scalar(iter_end)
                )
                # pc.bit_wise_and(pc.field(self.id_field), pc.scalar(num_ranks - 1)) == pc.scalar(rank)

            self.scanner = self.delta_table.to_pyarrow_dataset().scanner(
                columns=self.fields, filter=_filter
            )
        return self.scanner

    def __iter__(self):
        if self.loaded:
            self.stop()
        self.init_loading(self.length, self.path)
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
