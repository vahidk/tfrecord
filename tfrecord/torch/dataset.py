"""Load tfrecord files into torch datasets."""

import typing
import numpy as np

import torch.utils.data

from tfrecord import reader
from tfrecord import iterator_utils


class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    """
    def __init__(self,
                 data_path: str,
                 index_path: str,
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = None) -> None:
        super(TFRecordDataset, self).__init__()
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
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    """

    def __init__(self,
                 data_pattern: str,
                 index_pattern: str,
                 splits: typing.Dict[str, float],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = None) -> None:
        super(MultiTFRecordDataset, self).__init__()
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
