import os
import tempfile
import unittest

import numpy as np

from tfrecord.reader import example_loader, tfrecord_iterator
from tfrecord.writer import TFRecordWriter


class TestReadWrite(unittest.TestCase):

    def write_tfrecord(self, filename, records):
        writer = TFRecordWriter(filename)
        for datum in records:
            writer.write(datum)
        writer.close()

    def read_tfrecord(self, filename):
        iterator = tfrecord_iterator(filename)
        records = list(iterator)
        return records

    def test_write_and_read_integers(self):
        datum = {"int_key": (123, "int")}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            self.write_tfrecord(filename, [datum])

        records = self.read_tfrecord(filename)

        self.assertEqual(len(records), 1)
        example = list(example_loader(filename, None))
        np.testing.assert_array_equal(
            example[0]["int_key"], np.array([123], dtype=np.int64)
        )

        os.remove(filename)

    def test_write_and_read_floats(self):
        datum = {"float_key": (1.23, "float")}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            self.write_tfrecord(filename, [datum])

        records = self.read_tfrecord(filename)

        self.assertEqual(len(records), 1)
        example = list(example_loader(filename, None))
        np.testing.assert_array_equal(
            example[0]["float_key"], np.array([1.23], dtype=np.float32)
        )

        os.remove(filename)

    def test_write_and_read_string_arrays(self):
        datum = {"string_key": ([b"test1", b"test2"], "byte")}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            self.write_tfrecord(filename, [datum])

        records = self.read_tfrecord(filename)

        self.assertEqual(len(records), 1)
        example = list(example_loader(filename, None))
        self.assertEqual(example[0]["string_key"], b"test1")

        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
