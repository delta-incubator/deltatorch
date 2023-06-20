# deltatorch

![![image](https://github.com/mshtelma/deltatorch/workflows/build/badge.svg)](https://github.com/mshtelma/deltatorch/actions/workflows/ci.yml/badge.svg)
![![image](https://github.com/mshtelma/deltatorch/workflows/build/badge.svg)](https://github.com/mshtelma/deltatorch/actions/workflows/black.yml/badge.svg)
![![image](https://github.com/mshtelma/deltatorch/workflows/build/badge.svg)](https://github.com/mshtelma/deltatorch/actions/workflows/lint.yml/badge.svg)

## Concept

`deltatorch` allows users to directly use  `DeltaLake` tables as a data source for training using PyTorch. 
Using  `deltatorch`, users can create a PyTorch  `DataLoader` to load the training data. 
We support distributed training using PyTorch DDP as well. 

## Why yet another data-loading framework?

- Many Deep Learning projects are struggling with efficient data loading, especially with tabular datasets or datasets containing many small images
- Classical Big Data formats like Parquet can help with this issue, but are hard to operate:
  * writers might block readers
  * Failed write can make the whole dataset unreadable
  * More complicated projects might ingest data all the time, even during training

Delta Lake storage format solves all these issues, but PyTorch has no direct support for `DeltaLake` datasets.
`deltatorch` introduces such support and allows users to use `DeltaLake` for training Deep Learning models using PyTorch.

## Usage

### Requirements

- Python Version \> 3.8
- `pip` or `conda`

### Installation

- with `pip`:

```
pip install  git+https://github.com/delta-incubator/deltatorch
```
### Create PyTorch DataLoader to read our DeltaLake table

To utilize `deltatorch` at first, we will need a DeltaLake table containing training data we would like to use for training your PyTorch deep learning model. 
There is a requirement: this table must have an autoincrement ID field. This field is used by `deltatorch` for sharding and parallelization of loading. 
After that, we can use the `create_pytorch_dataloader` function to create PyTorch DataLoader, which can be used directly during training. 
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
from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

def create_data_loader(path:str, batch_size:int):

    return create_pytorch_dataloader(
        # Path to the DeltaLake table
        path,
        # Autoincrement ID field
        id_field="id",
        # Fields which will be used during training
        fields=[
            FieldSpec("image",
                      # Load image using Pillow
                      load_image_using_pil=True, 
                      # PyTorch Transform
                      transform=transform),
            FieldSpec("label"),
        ],
        # Number of readers 
        num_workers=2,
        # Shuffle data inside the record batches
        shuffle=True,
        # Batch size        
        batch_size=batch_size,
    )
```
