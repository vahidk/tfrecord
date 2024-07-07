"""Writer utils."""

import io
import struct
import typing

import crc32c
import numpy as np

from tfrecord import example_pb2


class TFRecordWriter:
    """Opens a TFRecord file for writing.

    Params:
    -------
    data_path: str
        Path to the tfrecord file.
    """

    def __init__(self, data_path: str) -> None:
        self.file = io.open(data_path, "wb")

    def close(self) -> None:
        """Close the tfrecord file."""
        self.file.close()

    def write(self, datum: typing.Dict[str, typing.Tuple[typing.Any, str]],
              sequence_datum: typing.Union[typing.Dict[str, typing.Tuple[typing.List[typing.Any], str]], None] = None,
              ) -> None:
        """Write an example into tfrecord file. Either as a Example
        SequenceExample depending on the presence of `sequence_datum`.
        If `sequence_datum` is None (by default), this writes a Example
        to file. Otherwise, it writes a SequenceExample to file, assuming
        `datum` to be the context and `sequence_datum` to be the sequential
        features.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "byte", "float" or "int".
        sequence_datum: dict
            By default, it is set to None. If this value is present, then the
            Dictionary of tuples of the form (value, dtype). dtype can be
            "byte", "float" or "int". value should be the sequential features.
        """
        if sequence_datum is None:
            record = TFRecordWriter.serialize_tf_example(datum)
        else:
            record = TFRecordWriter.serialize_tf_sequence_example(datum, sequence_datum)

        length = len(record)
        length_bytes = struct.pack("<Q", length)
        self.file.write(length_bytes)
        self.file.write(TFRecordWriter.masked_crc(length_bytes))
        self.file.write(record)
        self.file.write(TFRecordWriter.masked_crc(record))

    @staticmethod
    def masked_crc(data: bytes) -> bytes:
        """CRC checksum."""
        mask = 0xa282ead8
        crc = crc32c.crc32c(data)
        masked = ((crc >> 15) | (crc << 17)) + mask
        masked = np.uint32(masked & np.iinfo(np.uint32).max)
        masked_bytes = struct.pack("<I", masked)
        return masked_bytes

    @staticmethod
    def serialize_tf_example(datum: typing.Dict[str, typing.Tuple[typing.Any, str]]) -> bytes:
        """Serialize example into tfrecord.Example proto.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "byte", "float" or "int".

        Returns:
        --------
        proto: bytes
            Serialized tfrecord.example to bytes.
        """
        feature_map = {
            "byte": lambda f: example_pb2.Feature(
                bytes_list=example_pb2.BytesList(value=f)),
            "float": lambda f: example_pb2.Feature(
                float_list=example_pb2.FloatList(value=f)),
            "int": lambda f: example_pb2.Feature(
                int64_list=example_pb2.Int64List(value=f))
        }

        def serialize(value, dtype):
            if not isinstance(value, (list, tuple, np.ndarray)):
                value = [value]
            return feature_map[dtype](value)

        features = {key: serialize(value, dtype) for key, (value, dtype) in datum.items()}
        example_proto = example_pb2.Example(features=example_pb2.Features(feature=features))
        return example_proto.SerializeToString()

    @staticmethod
    def serialize_tf_sequence_example(context_datum: typing.Dict[str, typing.Tuple[typing.Any, str]],
                                      features_datum: typing.Dict[str, typing.Tuple[typing.List[typing.Any], str]],
                                      ) -> bytes:
        """Serialize sequence example into tfrecord.SequenceExample proto.

        Params:
        -------
        context_datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "byte", "float" or int.

        features_datum: dict
            Same as `context_datum`, but for the features.

        Returns:
        --------
        proto: bytes
            Serialized tfrecord.SequenceExample to bytes.
        """
        feature_map = {
            "byte": lambda f: example_pb2.Feature(
                bytes_list=example_pb2.BytesList(value=f)),
            "float": lambda f: example_pb2.Feature(
                float_list=example_pb2.FloatList(value=f)),
            "int": lambda f: example_pb2.Feature(
                int64_list=example_pb2.Int64List(value=f))
        }

        def serialize(value, dtype):
            if not isinstance(value, (list, tuple, np.ndarray)):
                value = [value]
            return feature_map[dtype](value)

        def serialize_repeated(value, dtype):
            feature_list = example_pb2.FeatureList()
            for v in value:
                feature_list.feature.append(serialize(v, dtype))
            return feature_list

        context = {key: serialize(value, dtype) for key, (value, dtype) in context_datum.items()}
        features = {key: serialize_repeated(value, dtype) for key, (value, dtype) in features_datum.items()}

        context = example_pb2.Features(feature=context)
        features = example_pb2.FeatureLists(feature_list=features)
        proto = example_pb2.SequenceExample(context=context, feature_lists=features)
        return proto.SerializeToString()
