import torch.utils.data
from tfrecord import io_utils
from tfrecord import iterator_utils


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, index_path, description, shuffle_queue_size=False):
        super().__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size

    def __iter__(self):
        it = io_utils.tfrecord_loader(
            self.data_path, self.index_path, self.description)
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
        it = io_utils.multi_tfrecord_loader(
            self.data_pattern, self.index_pattern, self.splits, self.description)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        return it
