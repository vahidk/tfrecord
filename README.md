# TFRecord reader

Usage:

Use TFRecordDataset to read TFRecord files in PyTorch.
```python
import torch
from tfrecord.torch dataset import TFRecordDataset, MultiTFRecordDataset

dataset = TFRecordDataset(tfrecord_path, None, {"image": "byte", "label": "float"})
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
```

Optionally create a index file which allows random access useful for parallel reading.
```
python3 -m tools.tfrecord2idx <tfrecord path> <index path>
```
