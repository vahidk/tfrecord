# TFRecord reader and writer

This library allows reading and writing tfrecord files efficiently in python. The library also provides an IterableDataset reader of tfrecord files for PyTorch. Currently uncompressed and compressed gzip TFRecords are supported.

## Installation

```
pip3 install 'tfrecord[torch]'
```

## Usage

It's recommended to create an index file for each TFRecord file. Index file must be provided when using multiple workers, otherwise the loader may return duplicate records. You can create an index file for an individual tfrecord file with this utility program:
```
python3 -m tfrecord.tools.tfrecord2idx <tfrecord path> <index path>
```

To create "*.tfidnex" files for all "*.tfrecord" files in a directory run:
```
tfrecord2idx <data dir>
```

## Reading & Writing tf.train.Example

### Reading tf.Example records in PyTorch
Use TFRecordDataset to read TFRecord files in PyTorch.
```python
import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/tmp/data.tfrecord"
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

tfrecord_pattern = "/tmp/{}.tfrecord"
index_pattern = "/tmp/{}.index"
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

### Infinite and finite PyTorch dataset

By default, `MultiTFRecordDataset` is infinite, meaning that it samples the data forever. You can make it finite by providing the appropriate flag
```
dataset = MultiTFRecordDataset(..., infinite=False)
```

### Shuffling the data

Both TFRecordDataset and MultiTFRecordDataset automatically shuffle the data when you provide a queue size.
```
dataset = TFRecordDataset(..., shuffle_queue_size=1024)
```

### Transforming input data

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

dataset = tfrecord.torch.TFRecordDataset("/tmp/data.tfrecord",
                                         index_path=None,
                                         description=description,
                                         transform=decode_image)

data = next(iter(dataset))
print(data)
```

### Writing tf.Example records in Python
```python
import tfrecord

writer = tfrecord.TFRecordWriter("/tmp/data.tfrecord")
writer.write({
    "image": (image_bytes, "byte"),
    "label": (label, "float"),
    "index": (index, "int")
})
writer.close()
```

### Reading tf.Example records in Python
```python
import tfrecord

loader = tfrecord.tfrecord_loader("/tmp/data.tfrecord", None, {
    "image": "byte",
    "label": "float",
    "index": "int"
})
for record in loader:
    print(record["label"])
```

## Reading & Writing tf.train.SequenceExample

SequenceExamples can be read and written using the same methods shown above with an extra argument
(`sequence_description` for reading and `sequence_datum` for writing) which cause the respective
read/write functions to treat the data as a SequenceExample.

### Writing SequenceExamples to file

```python
import tfrecord

writer = tfrecord.TFRecordWriter("/tmp/data.tfrecord")
writer.write({'length': (3, 'int'), 'label': (1, 'int')},
             {'tokens': ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], 'int'), 'seq_labels': ([0, 1, 1], 'int')})
writer.write({'length': (3, 'int'), 'label': (1, 'int')},
             {'tokens': ([[0, 0, 1], [1, 0, 0]], 'int'), 'seq_labels': ([0, 1], 'int')})
writer.close()
```

### Reading SequenceExamples in python

Reading from a SequenceExample yeilds a tuple containing two elements.

```python
import tfrecord

context_description = {"length": "int", "label": "int"}
sequence_description = {"tokens": "int", "seq_labels": "int"}
loader = tfrecord.tfrecord_loader("/tmp/data.tfrecord", None,
                                  context_description,
                                  sequence_description=sequence_description)

for context, sequence_feats in loader:
    print(context["label"])
    print(sequence_feats["seq_labels"])
```

### Read SequenceExamples in PyTorch

As described in the section on `Transforming Input`, one can pass a function as the `transform` argument to
perform post processing of features. This should be used especially for the sequence features as these are
variable length sequence and need to be padded out before being batched.

```python
import torch
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset

PAD_WIDTH = 5
def pad_sequence_feats(data):
    context, features = data
    for k, v in features.items():
        features[k] = np.pad(v, ((0, PAD_WIDTH - len(v)), (0, 0)), 'constant')
    return (context, features)

context_description = {"length": "int", "label": "int"}
sequence_description = {"tokens": "int ", "seq_labels": "int"}
dataset = TFRecordDataset("/tmp/data.tfrecord",
                          index_path=None,
			  description=context_description,
			  transform=pad_sequence_feats,
			  sequence_description=sequence_description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
data = next(iter(loader))
print(data)
```

Alternatively, you could choose to implement a custom `collate_fn` in order to assemble the batch,
for example, to perform dynamic padding.

```python
import torch
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset

def collate_fn(batch):
    from torch.utils.data._utils import collate
    from torch.nn.utils import rnn
    context, feats = zip(*batch)
    feats_ = {k: [torch.Tensor(d[k]) for d in feats] for k in feats[0]}
    return (collate.default_collate(context),
            {k: rnn.pad_sequence(f, True) for (k, f) in feats_.items()})

context_description = {"length": "int", "label": "int"}
sequence_description = {"tokens": "int ", "seq_labels": "int"}
dataset = TFRecordDataset("/tmp/data.tfrecord",
                          index_path=None,
			  description=context_description,
			  transform=pad_sequence_feats,
			  sequence_description=sequence_description)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
data = next(iter(loader))
print(data)
```
