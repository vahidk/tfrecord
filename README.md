# TFRecord reader

## Installation
pip3 install tfrecord

## Usage

It's recommended to create an index file for each TFRecord file. Index file must be provided when using multiple workers, otherwise the loader may return duplicate records.
```
python3 -m tfrecord.tools.tfrecord2idx <tfrecord path> <index path>
```


Use TFRecordDataset to read TFRecord files in PyTorch.
```python
import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/path/to/data.tfrecord"
index_path = None
description = {"image": "byte", "label": "float"}
dataset = TFRecordDataset(tfrecord_path, index_path, description)
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
description = {"image": "byte", "label": "int"}
dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data)
```

Creating tfrecord files:
```python
import tfrecord

writer = tfrecord.TFRecordWriter("/path/to/data.tfrecord")
writer.write({
    "image": (image_bytes, "byte"),
    "label": (label, "float"),
    "index": (index, "int")
})
writer.close()
```
Note: To write tfrecord files you also need an additional dependency:
```
pip3 install crc32c
```

Reading tfrecord files in python:
```python
import tfrecord

loader = tfrecord.tfrecord_loader("/path/to/data.tfrecord", None, {
    "image": "byte",
    "label": "float",
    "index": "int"
})
for record in loader:
    print(record["label"])
```

### Transforming input

You can optionally pass a function as `transform` argument to perform post processing of features before returning. 
This can for example be used to decode images or normalize colors to a certain range or pad variable length sequence.
 
```python
import tfrecord
import cv2

def decode_image(features):
    # get BGR image from bytes
    features["image"] = cv2.imdecode(features["image"], -1)
    return features


description = {
    "image": "bytes",
}

dataset = tfrecord.torch.TFRecordDataset("/path/to/data.tfrecord",
                                         index_path=None,
                                         description=description,
                                         transform=decode_image)

data = next(iter(dataset))
print(data)
```