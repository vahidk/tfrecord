"""Writer utils."""

import io
import struct
import typing

import numpy as np
try:
    import crc32c
except ImportError:
    crc32c = None

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

    def write(self, datum: typing.Dict[str, typing.Tuple[typing.Any, str]]) -> None:
        """Write an example into tfrecord file.

        Params:
        -------
        datum: dict
            Dictionary of tuples of form (value, dtype). dtype can be
            "byte", "float" or "int".
        """
        record = TFRecordWriter.serialize_tf_example(datum)
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
        crc = crc32c.crc32(data)
        masked = ((crc >> 15) | (crc << 17)) + mask
        masked = np.uint32(masked)
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
        serialize = {
            "byte": lambda f: example_pb2.Feature(
                bytes_list=example_pb2.BytesList(value=[f])),
            "float": lambda f: example_pb2.Feature(
                float_list=example_pb2.FloatList(value=f)),
            "int": lambda f: example_pb2.Feature(
                int64_list=example_pb2.Int64List(value=f))
        }

        features = {key: serialize[dtype](value) for key, (value, dtype) in datum.items()}
        example_proto = example_pb2.Example(features=example_pb2.Features(feature=features))
        return example_proto.SerializeToString()
