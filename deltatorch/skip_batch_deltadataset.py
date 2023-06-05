import io
import logging
import queue
import threading
from queue import Queue
from threading import Thread
import random
from typing import Optional, Callable, List

import numpy as np
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import get_worker_info

from deltatorch import DeltaIterableDataset
from deltatorch.deltadataset import FieldSpec

logger = logging.getLogger(__name__)


class SkipReadDeltaDataset(DeltaIterableDataset):
    def __init__(
        self,
        path: str,
        fields: List[FieldSpec],
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        num_workers: int = 2,
        shuffle: bool = False,
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

    def init_loading(self, path):
        self.delta_table = None
        self.scanner = None
        self.queue = Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.init_boundaries(self.path)

        self.num_chunks = self.num_workers * self.num_ranks

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
                    self.field_specs,
                    self.arrow_fields,
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
        fields_specs,
        arrow_fields,
        timeout,
    ):
        try:
            delta_table = DeltaTable(path)
            scanner = delta_table.to_pyarrow_dataset().scanner(
                columns=arrow_fields,
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
                        item = DeltaIterableDataset.decode_and_transform_record(
                            item, fields_specs
                        )

                        try:
                            q.put(
                                item,
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
