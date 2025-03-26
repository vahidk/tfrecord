from contextlib import contextmanager
import unittest
import tempfile

import numpy as np
from torch.utils.data import DataLoader

from tfrecord import example_pb2
from tfrecord.reader import process_feature
from tfrecord.torch.dataset import MultiTFRecordDataset
from tfrecord.writer import TFRecordWriter
from tfrecord.tools.tfrecord2idx import create_index


@contextmanager
def write_tfrecord_with_index(records):
    with tempfile.NamedTemporaryFile(delete=False) as record_temp_file, tempfile.NamedTemporaryFile(delete=False) as index_temp_file:
        record_file = record_temp_file.name
        index_file = index_temp_file.name
        writer = TFRecordWriter(record_file)
        for datum in records:
            writer.write(datum)
        writer.close()
        create_index(record_file, f"{record_file}.idx")
        yield record_file


class TestMultiTFRecordDataset(unittest.TestCase):

    def test_has_correct_number_of_records_with_any_number_of_process(self):
        """
        Test that the MultiTFRecordDataset generates the correct number of records
        when running in a single process or in multiple processes (num_workers >= 2).
        """
        with (
            write_tfrecord_with_index([{"key": (b"1", "byte")}]*1000) as record_file_1,
            write_tfrecord_with_index([{"key": (b"2", "byte")}]*1000) as record_file_2
        ):
            record_pattern = "{}"
            index_pattern = "{}.idx"
            # Note that the split does not matter so much in the infinite=False case, 
            # the iterator will stop when all records are read, the split will just
            # influence the order in which the records are read.
            split = {
                record_file_1: 0.8,
                record_file_2: 0.2
            }
            dataset = MultiTFRecordDataset(
                record_pattern,
                index_pattern,
                split,
                infinite=False
            )
            for num_workers in [0, 2, 8]:
                with self.subTest("num_workers", num_workers=num_workers):
                    total_records = 0
                    loader = DataLoader(dataset, batch_size=9, num_workers=num_workers)
                    for batch in loader:
                        total_records += len(batch["key"])

                    self.assertEqual(total_records, 2000)

    def test_samples_correctly_with_any_number_of_processes(self):
        """
        Test that the MultiTFRecordDataset generates records with the correct sampling 
        ratio when running in single or multiple processes (num_workers >= 2).
        """
        with (
            write_tfrecord_with_index([{"key": (b"1", "byte")}]*1000) as record_file_1,
            write_tfrecord_with_index([{"key": (b"2", "byte")}]*1000) as record_file_2
        ):
            record_pattern = "{}"
            index_pattern = "{}.idx"
            split = {
                record_file_1: 0.8,
                record_file_2: 0.2
            }
            dataset = MultiTFRecordDataset(
                record_pattern,
                index_pattern,
                split,
                infinite=True
            )
            for num_workers in [0, 2, 8]:
                with self.subTest("num_workers", num_workers=num_workers):
                    total_records = 0
                    key_1_count = 0
                    key_2_count = 0
                    loader = DataLoader(dataset, batch_size=9, num_workers=num_workers)

                    for batch in loader:
                        total_records += len(batch["key"])
                        key_1_count += np.sum(np.array(batch["key"]) == b"1")
                        key_2_count += np.sum(np.array(batch["key"]) == b"2")
                        if total_records > 2000:
                            break
                        
                    self.assertGreater(total_records, 2000)
                    # Those assertions have each a ~1e-8 probability of failing because 
                    # the sampling is random.
                    self.assertTrue(0.75 < key_1_count / total_records < 0.85)
                    self.assertTrue(0.15 < key_2_count / total_records < 0.25)


if __name__ == "__main__":
    unittest.main()
