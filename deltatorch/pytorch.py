from typing import Optional, Callable

from torch.utils.data import DataLoader
from .id_based_deltadataset import IDBasedDeltaDataset


def create_pytorch_dataloader(
    path: str,
    length: int,
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
    timeout: int = 15,
    queue_size: int = 25000,
):
    dataset = IDBasedDeltaDataset(
        path,
        length,
        id_field,
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
