from typing import Dict, Optional

import numpy as np

from torch.utils.data import IterableDataset
from torch.utils.data._utils import worker

from tfrecord import reader
from tfrecord import iterator_utils


class TFRecordDataset(IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str
        The path to the index file.

    description: dict of (str, str), optional, default=None
        Dictionary of (key, value) pairs where keys are the name of the
        features and values correspond to data type. The data type can
        be "byte", "float" or "int".

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    """
    def __init__(self,
                 data_path: str,
                 index_path: str,
                 description: Optional[Dict[str, str]] = None,
                 shuffle_queue_size: Optional[int] = None) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size

    def __iter__(self):
        worker_info = worker.get_worker_info()
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


class MultiTFRecordDataset(IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    index_pattern: str, optional, default=None
        Input index path pattern.

    description: dict of str
        Dictionary of (key, value) pairs where keys are the name of the
        features and values correspond to data type. The data type can
        be "byte", "float" or "int".

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    """

    def __init__(self,
                 data_pattern: str,
                 splits: Dict[str, int],
                 index_pattern: Optional[str] = None,
                 description: Optional[Dict[str, str]] = None,
                 shuffle_queue_size: Optional[int] = None) -> None:
        super(MultiTFRecordDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size

    def __iter__(self):
        worker_info = worker.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_loader(
            self.data_pattern, self.index_pattern, self.splits, self.description)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        return it
