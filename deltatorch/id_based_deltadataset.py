import io
import logging
import queue
import threading
from queue import Queue
from threading import Thread
import random
from time import sleep
from typing import Optional, Callable

import numpy as np
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import get_worker_info
import pyarrow.compute as pc

from deltatorch import DeltaIterableDataset

logger = logging.getLogger(__name__)


class IDBasedDeltaDataset(DeltaIterableDataset):
    def __init__(
        self,
        path: str,
        length: int,
        id_field: str,
        src_field: str,
        target_field: str,
        # batch_size: int = None,
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
    ):
        super().__init__(
            path,
            length,
            src_field,
            target_field,
            apply_src_numpy_shape,
            load_pil,
            transform,
            target_transform,
            use_fixed_rank,
            rank,
            num_ranks,
            num_workers,
            shuffle,
            timeout,
            queue_size,
        )
        self.id_field = id_field

    def init_loading(self, length, path):
        self.delta_table = None
        self.scanner = None
        self.queue = Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.start = 0
        if length is not None and length > 0:
            self.end = length
        else:
            self.end = self.count()
        logger.debug(f"Dataset for path {path}. Count:{self.end}")
        _info = get_worker_info()
        if self.use_fixed_rank:
            new_start = self.rank * self.end / self.num_ranks
            new_end = (self.rank + 1) * self.end / self.num_ranks
            self.start = new_start
            self.end = new_end
            logger.debug(
                f"Using fixed rank.  Current rank: {self.rank} World size: {self.num_ranks}"
            )
            logger.debug(f"Start: {self.start} End: {self.end}")
        elif _info is not None:
            self.rank = _info.id
            self.num_ranks = _info.num_workers
            logger.debug(
                f"Detected DDP process. Current rank: {self.rank} World size: {self.num_ranks}"
            )
            new_start = self.rank * self.end / self.num_ranks
            new_end = (self.rank + 1) * self.end / self.num_ranks
            logger.debug(
                "This rank will use the follosing set of rows: {self.start}-{self.end}"
            )
            self.start = new_start
            self.end = new_end

        self.workers = [
            Thread(
                target=self.worker_fn,
                args=(
                    self.path,
                    self.start + i * (self.end - self.start) / self.num_workers,
                    self.start + (i + 1) * (self.end - self.start) / self.num_workers,
                    self.queue,
                    self.stop_event,
                    self.shuffle,
                    self.id_field,
                    self.fields,
                    self.apply_src_numpy_shape,
                    self.load_pil,
                    self.src_field,
                    self.target_field,
                    self.transform,
                    self.target_transform,
                    self.timeout,
                ),
                daemon=True,
            )
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        self.loaded = True

    @staticmethod
    def worker_fn(
        path: str,
        start: int,
        end: int,
        q: Queue,
        event: threading.Event,
        shuffle: bool,
        id_field: str,
        fields,
        apply_src_numpy_shape,
        load_pil,
        src_field,
        target_field,
        transform,
        target_transform,
        timeout,
    ):
        try:
            logger.debug(f"Started worker for path {path} and range: {start}-{end}")
            # i = 0
            _filter = None
            if end > 0 and start >= 0:
                _filter = (pc.field(id_field) >= pc.scalar(start)) & (
                    pc.field(id_field) <= pc.scalar(end)
                )
            delta_table = DeltaTable(path)
            scanner = delta_table.to_pyarrow_dataset().scanner(
                columns=fields, filter=_filter
            )
            while not event.is_set():
                for rb in scanner.to_reader():
                    num_rows = rb.num_rows
                    # pylist = rb.to_pylist()
                    indexes = list(range(num_rows))
                    if shuffle:
                        random.shuffle(indexes)

                    for i in indexes:
                        item = rb.slice(offset=i, length=1).to_pylist()[0]
                        # pylist[i]
                        if apply_src_numpy_shape is not None:
                            item[src_field] = np.frombuffer(
                                item[src_field], dtype=np.uint8
                            ).reshape(apply_src_numpy_shape)
                        if load_pil:
                            item[src_field] = Image.open(io.BytesIO(item[src_field]))
                        if transform is not None:
                            item[src_field] = transform(item[src_field])
                        if target_transform is not None:
                            item[target_field] = target_transform(item[target_field])
                        try:
                            q.put(
                                (item[src_field], item[target_field]),
                                block=True,
                                timeout=timeout,
                            )
                            i += 1
                            if event.is_set():
                                return
                        except queue.Full:
                            logger.debug("full")
                            sleep(1)
                        except Exception as e:
                            print(e)

        except Exception as e:
            print(e)

    def __len__(self):
        if not self.loaded:
            self.init_loading(self.length, self.path)
        return int(self.end - self.start)

    def stop(self):
        self.stop_event.set()
        self.loaded = False
        self.delta_table = None
        self.scanner = None
        self.queue = None
        if self.workers:
            for w in self.workers:
                w.join(self.timeout)
        self.workers = []

    def __del__(self):
        self.stop()
