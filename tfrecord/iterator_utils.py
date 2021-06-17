"""Iterator utils."""

from __future__ import division

import typing
import warnings

import numpy as np


def cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element


def sample_iterators(iterators: typing.List[typing.Iterator],
                     ratios: typing.List[int],
                     infinite: bool = True) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.
    
    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    if infinite:
        iterators = [cycle(iterator) for iterator in iterators]
    else:
        iterators = [iterator() for iterator in iterators]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    while iterators:
        choice = np.random.choice(len(ratios), p=ratios)
        try:
            yield next(iterators[choice])
        except StopIteration:
            if iterators:
                del iterators[choice]
                ratios = np.delete(ratios, choice)
                ratios = ratios / ratios.sum()



def shuffle_iterator(iterator: typing.Iterator,
                     queue_size: int) -> typing.Iterable[typing.Any]:
    """Shuffle elements contained in an iterator.

    Params:
    -------
    iterator: iterator
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
            buffer.append(next(iterator))
    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (N={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)
