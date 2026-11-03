from pathlib import Path
import pickle
import tempfile
import unittest

import torch

from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.torch.dataset import TFRecordDataset
from tfrecord.writer import TFRecordWriter


def keep_numpy(batch):
    # Keep this test focused on dataset spawning, without requiring PyTorch's
    # platform-specific shared-memory tensor allocator.
    return batch


class TestDatasetMultiprocessing(unittest.TestCase):
    def test_default_transform_supports_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.tfrecord"
            index = Path(directory) / "data.index"
            writer = TFRecordWriter(str(path))
            try:
                for value in range(4):
                    writer.write({"value": (value, "int")})
            finally:
                writer.close()
            create_index(str(path), str(index))
            dataset = pickle.loads(pickle.dumps(TFRecordDataset(str(path), str(index))))
            loader = torch.utils.data.DataLoader(
                dataset, num_workers=2, multiprocessing_context="spawn", collate_fn=keep_numpy
            )
            values = [batch[0]["value"].item() for batch in loader]
            self.assertEqual(sorted(values), list(range(4)))
