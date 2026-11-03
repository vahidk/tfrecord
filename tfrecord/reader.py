"""Reader utils."""

import functools
import gzip
import io
import os
import typing

import numpy as np

from tfrecord import _record, example_pb2, iterator_utils


def tfrecord_iterator(
    data_path: str,
    index_path: typing.Optional[str] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
) -> typing.Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.

    Each yielded memoryview owns a reference to its record's immutable bytes
    and remains valid after the iterator advances. Both TFRecord checksums
    are validated before yielding a record.

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
    if compression_type not in (None, "gzip"):
        raise ValueError("compression_type should be either 'gzip' or None")
    if shard is not None:
        shard_idx, shard_count = shard
        if shard_count <= 0 or not 0 <= shard_idx < shard_count:
            raise ValueError("shard must satisfy 0 <= index < count")

    opener = gzip.open if compression_type == "gzip" else io.open
    with opener(data_path, "rb") as file:
        def read_records(start_offset=None, end_offset=None):
            if start_offset is not None:
                file.seek(start_offset)
            if end_offset is None and compression_type is None:
                end_offset = os.fstat(file.fileno()).st_size
            while end_offset is None or file.tell() < end_offset:
                payload = _record.read_record(file, end_offset)
                if payload is None:
                    break
                yield memoryview(payload)

        if index_path is None:
            yield from read_records()
        else:
            if os.path.getsize(index_path) == 0:
                return
            index = np.loadtxt(index_path, dtype=np.int64, ndmin=2)
            if index.size == 0:
                return
            if index.shape[1] != 2:
                raise ValueError("Index must contain offset and record-length columns")
            index = index[:, 0]
            if shard is None:
                offset = int(np.random.choice(index))
                yield from read_records(offset)
                yield from read_records(0, offset)
            else:
                num_records = len(index)
                start_index = (num_records * shard_idx) // shard_count
                end_index = (num_records * (shard_idx + 1)) // shard_count
                if start_index == end_index:
                    return
                start_byte = int(index[start_index])
                end_byte = int(index[end_index]) if end_index < num_records else None
                yield from read_records(start_byte, end_byte)


def process_feature(feature: example_pb2.Feature, typename: str, typename_mapping: dict, key: str):
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(
                f"Incompatible type '{typename}' for `{key}` "
                f"(should be '{reversed_mapping[inferred_typename]}')."
            )

    if inferred_typename == "bytes_list":
        if len(value) == 1:
            value = value[0]
        else:
            # Fixed-width NumPy byte strings strip trailing NUL bytes when
            # indexed or converted back to Python. Protobuf bytes are binary.
            value = np.array(value, dtype=object)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int64)
    return value


def extract_feature_dict(features, description, typename_mapping):
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = functools.partial(
                process_feature, typename=typename, typename_mapping=typename_mapping, key=key
            )
            return list(map(fn, feature))

    elif isinstance(features, example_pb2.Features):
        features = features.feature

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename, typename_mapping, key)

    else:
        raise TypeError(
            f"Incompatible type: features should be either of type "
            f"example_pb2.Features or example_pb2.FeatureLists and "
            f"not {type(features)}"
        )

    all_keys = list(features.keys())

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features


def example_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
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

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """

    typename_mapping = {"byte": "bytes_list", "float": "float_list", "int": "int64_list"}

    record_iterator = tfrecord_iterator(
        data_path=data_path,
        index_path=index_path,
        shard=shard,
        compression_type=compression_type,
    )

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        yield extract_feature_dict(example.features, description, typename_mapping)


def sequence_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    context_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    features_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
) -> typing.Iterable[
    typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, typing.List[np.ndarray]]]
]:
    """Create an iterator over the (decoded) sequence examples contained within
    the dataset.

    Decodes raw bytes of both the context and features (contained within the
    dataset) into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    context_description: list or dict of str, optional, default=None
        List of keys or dict (key, value) pairs to extract from the
        the context of each record. The keys represent the name of the
        features and the values ("byte", "float" or "int") correspond
        to the data type. If dtypes are provided, then they are verified
        against the inferred type for compatibility purposes. If None
        (default), then all features contained in the file are extracted.

    features_description: list or dict of str, optional, default=None
        Same as `context_description`, but applies to the features of
        each record.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    Yields:
    -------
    A tuple of (context, features) for an individual record.

    context: dict of {str, np.ndarray}
        Decoded bytes of the context features into its respective data
        type.

    features: dict of {str, np.ndarray}
        Decoded bytes of the sequence features into its respective data
        type.
    """
    typename_mapping = {"byte": "bytes_list", "float": "float_list", "int": "int64_list"}

    record_iterator = tfrecord_iterator(
        data_path=data_path,
        index_path=index_path,
        shard=shard,
        compression_type=compression_type,
    )

    for record in record_iterator:
        example = example_pb2.SequenceExample()
        example.ParseFromString(record)

        context = extract_feature_dict(example.context, context_description, typename_mapping)
        features = extract_feature_dict(
            example.feature_lists, features_description, typename_mapping
        )

        yield context, features


def tfrecord_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    compression_type: typing.Optional[str] = None,
) -> typing.Iterable[
    typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, typing.List[np.ndarray]]],
    ]
]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        or an empty list or dictionary, then all features contained in
        the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    Yields:
    -------
    features: dict of {str, value}
        Decoded bytes of the features into its respective data type (for
        an individual record). `value` is either going to be an np.ndarray
        in the instance of an `Example` and a list of np.ndarray in the
        instance of a `SequenceExample`.
    """
    if sequence_description is not None:
        return sequence_loader(
            data_path=data_path,
            index_path=index_path,
            context_description=description,
            features_description=sequence_description,
            shard=shard,
            compression_type=compression_type,
        )
    return example_loader(
        data_path=data_path,
        index_path=index_path,
        description=description,
        shard=shard,
        compression_type=compression_type,
    )


def multi_tfrecord_loader(
    data_pattern: str,
    index_pattern: typing.Union[str, None],
    splits: typing.Dict[str, float],
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    compression_type: typing.Optional[str] = None,
    infinite: bool = True,
    shard: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Iterable[
    typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, typing.List[np.ndarray]]],
    ]
]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
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

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
    loaders = [
        functools.partial(
            tfrecord_loader,
            data_path=data_pattern.format(split),
            index_path=index_pattern.format(split) if index_pattern is not None else None,
            description=description,
            sequence_description=sequence_description,
            compression_type=compression_type,
            shard=shard
        )
        for split in splits.keys()
    ]
    return iterator_utils.sample_iterators(loaders, list(splits.values()), infinite=infinite)
