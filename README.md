# TFRecord reader

## Installation
pip3 install tfrecord

## Usage

Use TFRecordDataset to read TFRecord files in PyTorch.
```python
import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/path/to/data.tfrecord"
dataset = TFRecordDataset(tfrecord_path, None, {"image": "byte", "label": "float"})
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
```

Use MultiTFRecordDataset to read multiple TFRecord files. This class samples from given tfrecord files with given probability.
```python
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset

tfrecord_pattern = "/path/to/{}.tfrecord"
index_pattern = "/path/to/{}.index"
splits = {
    "dataset1": 0.8,
    "dataset2": 0.2,
}
dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, {"image": "byte", "label": "int"})
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
```

You can create an index file which allows random access. Index file must be provided when using multiple workers, otherwise the loader may return duplicate records.
```
python3 -m tfrecord.tools.tfrecord2idx <tfrecord path> <index path>
```
