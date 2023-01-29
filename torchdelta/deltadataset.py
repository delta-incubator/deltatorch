import io
import math
import queue
import threading
from queue import Queue
from threading import Thread
from queue import Empty
import random
from time import sleep

import numpy as np
from PIL import Image
from deltalake import DeltaTable
from torch.utils.data import get_worker_info, Dataset, IterableDataset, DataLoader
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import List, Optional, Callable

from torchvision.datasets import VisionDataset


class DeltaIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        # fields: List[str],
        id_field: str,
        src_field: str,
        target_field: str,
        batch_size: int = None,
        apply_src_numpy_shape=None,
        load_pil: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        num_workers: int = 2,
        shuffle: bool = False,
    ) -> None:

        super().__init__()
        self.path = path
        self.fields = {
            src_field: ds.field(src_field),
            target_field: ds.field(target_field),
        }  # {field_name: ds.field(field_name) for field_name in fields}
        self.id_field = id_field
        self.src_field = src_field
        self.target_field = target_field
        self.use_fixed_rank = use_fixed_rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.apply_src_numpy_shape = apply_src_numpy_shape
        self.load_pil = load_pil
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle

        self.delta_table = None
        self.scanner = None
        self.start = 0
        self.end = self.count()

        _info = get_worker_info()

        if self.use_fixed_rank:
            new_start = self.rank * self.end / self.num_ranks
            new_end = (self.rank + 1) * self.end / self.num_ranks
            self.start = new_start
            self.end = new_end
            print(f'Using fixed rank.  Current rank: {self.rank} World size: {self.num_ranks}')
            print(f'Start: {self.start} End: {self.end}')
        elif _info is not None:
            self.rank = _info.id
            self.num_ranks = _info.num_workers
            print(f'Detected DDP process. Current rank: {self.rank} World size: {self.num_ranks}')
            new_start = self.rank * self.end / self.num_ranks
            new_end = (self.rank + 1) * self.end / self.num_ranks
            print('This rank will use the follosing set of rows: {self.start}-{self.end}')
            self.start = new_start
            self.end = new_end

        self.queue = Queue(maxsize=20000)
        self.stop_event = threading.Event()
        
        self.delta_table = None
        self.scanner = None
        self.workers = [
            Thread(
                target=self.worker_fn,
                args=(
                    self.path,
                    self.start + i * (self.end-self.start) / self.num_workers,
                    self.start + (i + 1) * (self.end-self.start) / self.num_workers,
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
                ),
                daemon=True,
            )
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()


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
    ):
        try:
            print("Started worker: ", start, "-", end)
            i = 0
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
                    pylist = rb.to_pylist()
                    indexes = list(range(len(pylist)))
                    if shuffle:
                        random.shuffle(indexes)

                    for i in indexes:
                        item = pylist[i]
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
                                timeout=60,
                            )
                            i += 1
                            if event.is_set():
                              return
                        except queue.Full:
                            print("full")
                        except Exception as e:
                            print(e)
            # print("Finished reading: ", i, " ", start, " ", end)
            # sleep(1000)
        except Exception as e:
            print(e)

    def count(self):
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
        i = self.start
        while True:
            try:
                item = self.queue.get(block=True, timeout=15)
                yield item
                i += 1
                if i >= self.end:
                    return
            except Empty:
                alive = False
                print(self.workers)
                for w in self.workers:
                    if w.is_alive():
                        alive = True
                        print("\nEmpty ", i)
                if not alive:
                    print("\Exiting ", i)
                    return

    def __len__(self):
        return self.end-self.start

    def __del__(self):
        self.stop_event.set()
