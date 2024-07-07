import unittest
import tempfile
import os
import numpy as np

from tfrecord.reader import tfrecord_iterator
from tfrecord.writer import TFRecordWriter


class TestTFRecordWriter(unittest.TestCase):

    def test_tfrecord_writer_write_example(self):
        datum = {"key": (b"value", "byte")}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            writer = TFRecordWriter(filename)
            writer.write(datum)
            writer.close()

        iterator = tfrecord_iterator(filename)
        records = list(iterator)
        self.assertEqual(records[0], b"\n\x12\n\x10\n\x03key\x12\t\n\x07\n\x05value")
        os.remove(filename)

    def test_tfrecord_writer_write_sequence_example(self):
        datum = {"key": (b"value", "byte")}
        sequence_datum = {"seq_key": ([b"seq_value"], "byte")}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = temp_file.name
            writer = TFRecordWriter(filename)
            writer.write(datum, sequence_datum)
            writer.close()

        iterator = tfrecord_iterator(filename)
        records = list(iterator)
        self.assertTrue(records[0].tobytes().startswith(b"\n\x12\n\x10\n\x03key\x12\t\n\x07\n\x05value"))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
