import io
import logging
import queue
import threading
from queue import Queue
from threading import Thread
import random
from typing import Optional, Callable

import numpy as np
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import get_worker_info

from deltatorch import DeltaIterableDataset

logger = logging.getLogger(__name__)


class SkipReadDeltaDataset(DeltaIterableDataset):
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
        queue_size: int = 250000,
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

    def init_loading(self, length, path):
        self.delta_table = None
        self.scanner = None
        self.queue = Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        _info = get_worker_info()

        if self.use_fixed_rank:
            self.num_chunks = self.num_workers * self.num_ranks
            assert self.rank > 0
        elif _info is not None:
            self.rank = _info.id
            self.num_ranks = _info.num_workers
            self.num_chunks = self.num_workers * self.num_ranks
        else:
            self.num_chunks = self.num_workers
            self.rank = 1

        self.workers = [
            Thread(
                target=self.worker_fn,
                args=(
                    self.path,
                    self.queue,
                    self.stop_event,
                    self.rank * i,
                    self.num_chunks,
                    self.shuffle,
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
        q: Queue,
        event: threading.Event,
        chunk: int,
        num_chunks: int,
        shuffle: bool,
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
            delta_table = DeltaTable(path)
            scanner = delta_table.to_pyarrow_dataset().scanner(
                columns=fields,
            )
            while not event.is_set():
                batch_id = 0
                batch_iter = iter(scanner.to_reader())
                while True:
                    rb = next(batch_iter, None)
                    if rb is None:
                        return
                    batch_id += 1
                    if batch_id % num_chunks != chunk:
                        continue
                    logger.debug(
                        f"Worker {chunk} is processing batch {batch_id} with {rb.num_rows} rows."
                    )
                    num_rows = rb.num_rows
                    indexes = list(range(num_rows))
                    if shuffle:
                        random.shuffle(indexes)

                    for i in indexes:
                        item = rb.slice(offset=i, length=1).to_pylist()[0]
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
                            if event.is_set():
                                return

                        except queue.Full:
                            logger.debug("full")
                        except Exception as e:
                            logger.error(str(e), e)

        except Exception as e:
            print(e)

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
