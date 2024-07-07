import unittest
from unittest.mock import mock_open, patch

import numpy as np
from tfrecord.reader import (
    example_loader,
    sequence_loader,
    tfrecord_iterator,
    process_feature,
)

from tfrecord import example_pb2


class TestFeatureProcessing(unittest.TestCase):

    def setUp(self):
        self.feature_bytes = example_pb2.Feature(
            bytes_list=example_pb2.BytesList(value=[b"test"])
        )
        self.feature_float = example_pb2.Feature(
            float_list=example_pb2.FloatList(value=[1.0])
        )
        self.feature_int = example_pb2.Feature(
            int64_list=example_pb2.Int64List(value=[1])
        )

    def test_process_feature_bytes(self):
        result = process_feature(
            self.feature_bytes, "byte", {"byte": "bytes_list"}, "key"
        )
        self.assertEqual(result, b"test")

    def test_process_feature_float(self):
        result = process_feature(
            self.feature_float, "float", {"float": "float_list"}, "key"
        )
        np.testing.assert_array_equal(result, np.array([1.0], dtype=np.float32))

    def test_process_feature_int(self):
        result = process_feature(self.feature_int, "int", {"int": "int64_list"}, "key")
        np.testing.assert_array_equal(result, np.array([1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
