import tempfile
import unittest
from contextlib import contextmanager

import torch

from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.torch.dataset import MultiTFRecordDataset
from tfrecord.writer import TFRecordWriter


@contextmanager
def write_tfrecord_with_index(records):
    with tempfile.NamedTemporaryFile(delete=False) as record_temp_file:
        record_file = record_temp_file.name
        writer = TFRecordWriter(record_file)
        for datum in records:
            writer.write(datum)
        writer.close()
        create_index(record_file, f"{record_file}.idx")
        yield record_file


class TestMultiTFRecordDataset(unittest.TestCase):

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_has_correct_number_of_records_with_any_number_of_process(self):
        """
        Test that the MultiTFRecordDataset generates the correct number of records
        when running in a single process or in multiple processes (num_workers >= 2).
        """
        with (
            write_tfrecord_with_index([{"key": (b"1", "byte")}] * 1000) as record_file_1,
            write_tfrecord_with_index([{"key": (b"2", "byte")}] * 1000) as record_file_2,
        ):
            record_pattern = "{}"
            index_pattern = "{}.idx"
            split = {record_file_1: 0.8, record_file_2: 0.2}
            dataset = MultiTFRecordDataset(record_pattern, index_pattern, split, infinite=False)
            for num_workers in [0, 2, 8]:
                with self.subTest("num_workers", num_workers=num_workers):
                    total_records = 0
                    loader = torch.utils.data.DataLoader(dataset, batch_size=9, num_workers=num_workers)
                    for batch in loader:
                        total_records += len(batch["key"])

                    self.assertEqual(total_records, 2000)


if __name__ == "__main__":
    unittest.main()
