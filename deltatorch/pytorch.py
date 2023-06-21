from typing import List, Optional, Tuple, Any

from pyarrow.dataset import Expression
from torch.utils.data import DataLoader

from .deltadataset import FieldSpec
from .id_based_deltadataset import IDBasedDeltaDataset


def create_pytorch_dataloader(
    path: str,
    id_field: str,
    fields: List[FieldSpec],
    partition_filter: Optional[List[Tuple[str, str, Any]]] = None,
    version: Optional[int] = None,
    batch_size: int = 32,
    use_fixed_rank: bool = False,
    rank: int = None,
    num_ranks: int = None,
    num_workers: int = 2,
    shuffle: bool = False,
    drop_last: bool = False,
    **pytorch_dataloader_kwargs
):
    """Create a PyTorch DataLoader.

    This method will do the following two steps:
      1) Create an iterable dataset out of delta tables.
      2) Create a PyTorch DataLoader based on the itearable dataset created in (1)

    :param path: Path to the DeltaLake table
    :param id_field: Autoincrement ID field.delta table must have an autoincrement ID
    field.This field is used by deltatorch for sharding and parallel data loading
    :param fields: List of  `detatorch.FieldSpec` used for training
    :param version: Delta Table Version to load
    :param partition_filter: A list of partition filters
    :param batch_size: The number of items to return per batch. Default ``32``.
    :param rank: Rank of the current process within the current distributed
            group. This value is set by ``torch.distributed.get_rank()``
            when ``use_fixed_rank`` is set to `False`.
    :param num_ranks: number of processes in the current distributed process group.
           This value is set by ``torch.distributed.get_world_size()``
           when ``use_fixed_rank`` is set to `False`.
    :param use_fixed_rank: If this values is set to `True`, `rank` and `num_ranks`
    should be set explicitly by user
    :param num_workers: An int for the number of workers to use in dataloader
    :param shuffle: set to ``True`` to have the data reshuffled
            inside the record batches (default: ``False``).
    :param drop_last: set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    :param pytorch_dataloader_kwargs: arguments for `torch.utils.data.DataLoader`,
        exclude these arguments: ``batch_size``, ``num_workers``, ``shuffle``,
        ``drop_last``.

    :return: `torch.utils.data.DataLoader` object.
    """
    dataset = IDBasedDeltaDataset(
        path,
        id_field,
        fields,
        version,
        partition_filter,
        use_fixed_rank,
        rank,
        num_ranks,
        num_workers,
        shuffle,
        batch_size,
        drop_last,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        **pytorch_dataloader_kwargs
    )
