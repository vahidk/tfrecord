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

There are use cases when you might not want to return your input directly after reading from the tfrecord files.
A trivial example could be to normalize your input, if you haven't saved the normalized feature in the files.

You can easily achieve this by leveraging the `transform_func` argument.
There might also be use cases to remove a field after performing a transformation, this can be leveraged using the `removed_fields`.

Both the arguments mentioned above are available in all readers including `tfrecord.tfrecord_loader` and `tfrecord.torch.TFRecordDataset`.
 
```python
import tfrecord

def transformation(features):
    # modify an existing feature
    features["feature_1"] = features["feature_1"] * 2
    # create a new feature
    features["feature_2"] = features["index"] * 4
    return features


description = {
    "index": "int",
    "feature_1": "float",
}

dataset = tfrecord.torch.TFRecordDataset("/path/to/data.tfrecord",
                                         index_path=None,
                                         transform_func=transformation,
                                         removed_fields=["index"])

data = next(iter(dataset))

assert set(data.keys()) == {"feature_1", "feature_2"}, "Expected keys don't match."
```