"""Load tfrecord files into torch datasets."""

import typing

import numpy as np
import torch.utils.data

from tfrecord import iterator_utils, reader


class TFRecordMapDataset(torch.utils.data.Dataset):
    """Parse (generic) TFRecords dataset into `Dataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str
        The path to the index file. Necessary for TFRecordMapDataset

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

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(
        self,
        data_path: str,
        index_path: str,
        description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        shuffle_queue_size: typing.Optional[int] = None,
        transform: typing.Callable[[dict], typing.Any] = None,
        sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        compression_type: typing.Optional[str] = None,
    ) -> None:
        super(TFRecordMapDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type
        self.index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        self.it = None 

    def _close(self):
        self.it.send(-1)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None

        if self.it is None:
            self.it = reader.tfrecord_loader(
                data_path=self.data_path,
                index_path=self.index_path,
                description=self.description,
                shard=shard,
                sequence_description=self.sequence_description,
                compression_type=self.compression_type,
            )
            next(self.it)
        if self.shuffle_queue_size:
            self.it = iterator_utils.shuffle_iterator(self.it, self.shuffle_queue_size)
        record = self.it.send(index)
        if self.transform:
            record = self.transform(record)
        return record

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
        else:
            shard = 0, 1 
        return len(self.index) / shard[1]

class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
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

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(
        self,
        data_path: str,
        index_path: typing.Union[str, None],
        description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        shuffle_queue_size: typing.Optional[int] = None,
        transform: typing.Callable[[dict], typing.Any] = None,
        sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        compression_type: typing.Optional[str] = None,
    ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = reader.tfrecord_loader(
            data_path=self.data_path,
            index_path=self.index_path,
            description=self.description,
            shard=shard,
            sequence_description=self.sequence_description,
            compression_type=self.compression_type,
        )
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it


class MultiTFRecordDataset(torch.utils.data.IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
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

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the Dataset should be infinite or not
    """

    def __init__(
        self,
        data_pattern: str,
        index_pattern: typing.Union[str, None],
        splits: typing.Dict[str, float],
        description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        shuffle_queue_size: typing.Optional[int] = None,
        transform: typing.Callable[[dict], typing.Any] = None,
        sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        compression_type: typing.Optional[str] = None,
        infinite: bool = True,
    ) -> None:
        super(MultiTFRecordDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type
        self.infinite = infinite

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_loader(
            data_pattern=self.data_pattern,
            index_pattern=self.index_pattern,
            splits=self.splits,
            description=self.description,
            sequence_description=self.sequence_description,
            compression_type=self.compression_type,
            infinite=self.infinite,
        )
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it
