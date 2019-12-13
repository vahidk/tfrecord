import numpy as np
import torch.utils.data
from tfrecord import reader
from tfrecord import iterator_utils


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, index_path, description, shuffle_queue_size=False):
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = reader.tfrecord_loader(
            self.data_path, self.index_path, self.description, shard)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        return it


class MultiTFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_pattern, index_pattern, splits, description, shuffle_queue_size=None):
        super().__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_loader(
            self.data_pattern, self.index_pattern, self.splits, self.description)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        return it
