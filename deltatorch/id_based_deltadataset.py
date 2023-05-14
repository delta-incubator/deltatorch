import io
import logging
import queue
import threading
import traceback
from queue import Queue
from threading import Thread
import random
from time import sleep
from typing import Optional, Callable, List, Dict

from deltalake import DeltaTable
import pyarrow.compute as pc

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
        timeout: int = 15,
        queue_size: int = 25000,
    ):
        super().__init__(
            path,
            fields,
            use_fixed_rank,
            rank,
            num_ranks,
            num_workers,
            shuffle,
            timeout,
            queue_size,
        )
        self.id_field = id_field

    def init_loading(self, path):
        self.delta_table = None
        self.scanner = None
        self.queue = Queue(maxsize=self.queue_size)
        self.stop_event = threading.Event()
        self.init_boundaries(path)

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
        start: int,
        end: int,
        q: Queue,
        event: threading.Event,
        shuffle: bool,
        id_field: str,
        field_specs: List[FieldSpec],
        arrow_fields: Dict,
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
                columns=arrow_fields, filter=_filter
            )
            # while not event.is_set():
            for rb in scanner.to_reader():
                if event.is_set():
                    break
                num_rows = rb.num_rows
                # pylist = rb.to_pylist()
                indexes = list(range(num_rows))
                if shuffle:
                    random.shuffle(indexes)

                for i in indexes:
                    item = rb.slice(offset=i, length=1).to_pylist()[0]
                    item = DeltaIterableDataset.decode_and_transform_record(
                        item, field_specs
                    )
                    try:
                        q.put(
                            item,
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
                        traceback.print_exception(e)

        except Exception as e:
            traceback.print_exception(e)

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
