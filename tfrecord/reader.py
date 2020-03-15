"""Reader utils."""

import functools
import io
import os
import struct

import numpy as np

from tfrecord import example_pb2
from tfrecord import iterator_utils


def tfrecord_iterator(data_path, index_path=None, shard=None):
    """Create an iterator over tfrecord dataset.

    Args:
      data_path: TFRecord file path.
      index_path: Index file path.
      shard: A tuple (index, count) representing the shard information.
    Returns:
      An iterator over the dataset.
    """
    file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024*1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    if index_path is None:
        yield from read_records()
    else:
        index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        if shard is None:
            offset = np.random.choice(index)
            yield from read_records(offset)
            yield from read_records(0, offset)
        else:
            num_records = len(index)
            shard_idx, shard_count = shard
            start_index = (num_records * shard_idx) // shard_count
            end_index = (num_records * (shard_idx + 1)) // shard_count
            start_byte = index[start_index]
            end_byte = index[end_index] if end_index < num_records else None
            yield from read_records(start_byte, end_byte)

    file.close()


def tfrecord_loader(data_path, index_path, description=None, shard=None):
    """Create an iterator from a tfrecord dataset.

    Args:
        data_path: Path of the input data.
        index_path: Path of index file. This can be set to None if not available.
        description: List of keys to extract from each record. If None, then all
            features contained in the file are extracted. (default: None)
        shard: A tuple (index, count) representing the shard information. (default : None)
    Returns:
        An iterator that generates individual data records.
    """
    record_iterator = tfrecord_iterator(data_path, index_path, shard)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        features = {}
        all_keys = list(example.features.feature.keys())
        if description is None:
            description = all_keys
        for key in description.keys():
            if key not in all_keys:
                raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")
            # NOTE: We assume that each key in the example has only one field
            # (either "bytes_list", "float_list", or "int64_list")!
            field = example.features.feature[key].ListFields()[0]
            tf_typename, value = field[0].name, field[1].value

            # Decode raw bytes into respective data types
            if tf_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif tf_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif tf_typename == "int64_list":
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
