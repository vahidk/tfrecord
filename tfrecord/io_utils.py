"""I/O utils."""

import functools
import io
import struct

import numpy as np

from tfrecord import example_pb2
from tfrecord import iterator_utils


def serialize_tf_example(datum):
    """Serialize example into tfrecord.Example proto. 

    Args:
        Datum is a dictionary of tuples of form (value, dtype). dtype can be "byte", "float" or "int".
    Returns:
        Serialized tfrecord.example to bytes.
    """
    features = {}
    for key, (value, dtype) in datum.items():
        feature = {
            "byte": lambda f: example_pb2.Feature(bytes_list=example_pb2.BytesList(value=f)),
            "float": lambda f: example_pb2.Feature(float_list=example_pb2.FloatList(value=f)),
            "int": lambda f: example_pb2.Feature(int64_list=example_pb2.Int64List(value=f))
        }[dtype](value)
        features[key] = feature

    example_proto = example_pb2.Example(
        features=example_pb2.Features(feature=features))
    return example_proto.SerializeToString()


def tfrecord_iterator(data_path, index_path=None):
    """Iterate over a tfrecord file given tfrecord and index file."""
    file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024*1024)

    def read_record():
        if file.readinto(length_bytes) != 8:
            return None
        if file.readinto(crc_bytes) != 4:
            return None
        length, = struct.unpack("q", length_bytes)
        if length > len(datum_bytes):
            datum_bytes.zfill(int(length * 1.5))
        datum_bytes_view = memoryview(datum_bytes)[:length]
        if file.readinto(datum_bytes_view) != length:
            return None
        if file.readinto(crc_bytes) != 4:
            return None
        return datum_bytes_view

    if index_path is not None:
        index = np.loadtxt(index_path, dtype=np.int64)
        if index.ndim > 1:
            index = index[:, 0]
        offset = np.random.choice(index)
        file.seek(offset)

        while True:
            datum = read_record()
            if datum is None:
                break
            yield datum

        file.seek(0)
        while True:
            datum = read_record()
            if datum is None:
                break
            yield datum
            if file.tell() >= offset:
                break

    else:
        while True:
            datum = read_record()
            if datum is None:
                break
            yield datum

    file.close()


def tfrecord_loader(data_path, index_path=None , description=None):
    """Create an iterator from a tfrecord dataset. 

    Args:
        data_path: Input data path.
        index_path: Index path..
        description: A dictionary of key and values where keys are the name of the features and values correspond to
                     data type. The data type can be "byte", "float" or "int".
    Returns:
        An iterator that generates individual data records.
    """
    record_iterator = tfrecord_iterator(data_path, index_path)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        features = {}
        for key, typename in description.items():
            tf_typename = {
                "byte": "bytes_list",
                "float": "float_list",
                "int": "int64_list"
            }[typename]
            if key not in example.features.feature:
                raise ValueError("Key {} doesn't exist.".format(key))
            value = getattr(example.features.feature[key], tf_typename).value
            if typename == "byte":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif typename == "float":
                value = np.array(value, dtype=np.float32)
            elif typename == "int":
                value = np.array(value, dtype=np.int32)
            features[key] = value

        yield features


def multi_tfrecord_loader(data_pattern, index_pattern, splits, description):
    """Create an iterator by reading and merging multiple tfrecord datasets.

    Args:
        data_pattern: Input data path pattern.
        index_pattern: Input index path pattern.
        splits: Dictionary of keys and values, the key is used in conjunction with pattern to construct the path, the
                values determine the contribution of each split to the batch.
        description: Description of data. See tfrecord_loader.
    Returns:
        A repeating iterator that generates batches of data.
    """
    loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                 index_path=index_pattern.format(split), description=description)
               for split in splits.keys()]
    return iterator_utils.sample_iterators(loaders, list(splits.values()))
