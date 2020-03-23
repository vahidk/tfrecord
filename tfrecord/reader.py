"""Reader utils."""

import functools
import io
import os
import struct
import typing

import numpy as np

from tfrecord import example_pb2
from tfrecord import iterator_utils


def tfrecord_iterator(data_path: str,
                      index_path: typing.Optional[str] = None,
                      shard: typing.Optional[typing.Tuple[int, int]] = None
                      ) -> typing.Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.

    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
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


def tfrecord_loader(data_path: str,
                    index_path: str,
                    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                    shard: typing.Optional[typing.Tuple[int, int]] = None
                    ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """
    record_iterator = tfrecord_iterator(data_path, index_path, shard)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        all_keys = list(example.features.feature.keys())
        if description is None:
            description = dict.fromkeys(all_keys, None)
        elif isinstance(description, list):
            description = dict.fromkeys(description, None)

        features = {}
        for key, typename in description.items():
            if key not in all_keys:
                raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")
            # NOTE: We assume that each key in the example has only one field
            # (either "bytes_list", "float_list", or "int64_list")!
            field = example.features.feature[key].ListFields()[0]
            inferred_typename, value = field[0].name, field[1].value
            if typename is not None:
                typename_mapping = {
                    "byte": "bytes_list",
                    "float": "float_list",
                    "int": "int64_list"
                }
                tf_typename = typename_mapping[typename]
                if tf_typename != inferred_typename:
                    reversed_mapping = {v:k for k, v in typename_mapping.items()}
                    raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                                    f"(should be '{reversed_mapping[inferred_typename]}').")

            # Decode raw bytes into respective data types
            if inferred_typename == "bytes_list":
                value = np.frombuffer(value[0], dtype=np.uint8)
            elif inferred_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif inferred_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value
        yield features


def multi_tfrecord_loader(data_pattern: str,
                          index_pattern: str,
                          splits: typing.Dict[str, float],
                          description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None
                          ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    NOTE: Sharding is currently unavailable for the multi tfrecord loader.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str, optional, default=None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
    loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                 index_path=index_pattern.format(split) \
                                     if index_pattern is not None else None,
                                 description=description)
               for split in splits.keys()]
    return iterator_utils.sample_iterators(loaders, list(splits.values()))
