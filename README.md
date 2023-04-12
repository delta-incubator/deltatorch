# TorchDelta

## Concept

`torchdelta` allows users to directly use  `DeltaLake` tables as a data source for training using PyTorch. 
Using  `torchdelta` users can create a PyTorch  `DataLoader` and use it to load the training data. 
We support distributed training using PyTorch DDP as well. 

## Requirements

- Python Version \> 3.8
- `pip` or `conda`

## Installation

- with `pip`:

```
pip install git+https://github.com/mshtelma/torchdelta
```

## Getting started
To utilise `torchdelta` at first we will need a DeltaLake table containing  training data we would like to use for training your PyTorch deep learning model. 
There is a requirement: this table must have an autoincrement ID field. This field is used by `torchdelta` for sharding and parallelization of loading. 
After that we can use `create_pytorch_dataloader` function to create PyTorch DataLoader which can be directly used during the training process. 
Below you can find an example of creating a DataLoader for the following table schema :
```sql
CREATE TABLE TRAINING_DATA 
(   
    image BINARY,   
    label BIGINT,   
    id INT
) 
USING delta LOCATION 'path' 
```

After the table is ready we can use the `create_pytorch_dataloader` function to create a PyTorch DataLoader :
```python
from torchdelta import create_pytorch_dataloader

def create_data_loader(path:str, length:int, batch_size:int):

    return create_pytorch_dataloader(
        path,
        length,
        src_field="image",
        target_field="label",
        id_field="id",
        load_pil=True,
        num_workers=2,
        shuffle=True,
        use_fixed_rank=True,
        batch_size=batch_size,
    )
```