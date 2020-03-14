"""Iterator utils."""

import warnings
from typing import Any, Callable, Iterable, Iterator, List

import numpy as np


def sample_iterators(itrs: List[Iterator], ratios: List[int]) -> Iterable[Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    itrs: list of itrators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """

    def cycle(iterator_fn: Callable):
        """Create a repeating iterator from an iterator generator."""
        while True:
            for element in iterator_fn():
                yield element

    itrs = [cycle(iterator) for iterator in itrs]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    while True:
        choice = np.random.choice(len(ratios), p=ratios)
        yield next(itrs[choice])


def shuffle_iterator(itr: Iterator, queue_size: int) -> Iterable[Any]:
    """Shuffle elements contained in an iterator.

    Params:
    -------
    itr: iterator
        The iterator.

    queue_size: int
        Length of buffer. Determines how many records are queued to
        sample from.

    Yields:
    -------
    item: Any
        Decoded bytes of the features into its respective data type (for
        an individual record) from an iterator.
    """
    buffer = []
    try:
        for _ in range(queue_size):
            buffer.append(next(itr))
    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (s={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(itr)
            yield item
        except StopIteration:
            yield buffer.pop(index)
