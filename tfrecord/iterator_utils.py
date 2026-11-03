"""Iterator utils."""

from __future__ import division

import typing
import warnings

import numpy as np


def cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        yielded = False
        for element in iterator_fn():
            yielded = True
            yield element
        if not yielded:
            return


def sample_iterators(
    iterators: typing.List[typing.Callable[[], typing.Iterator]],
    ratios: typing.List[float],
    infinite: bool = True,
) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of callables
        Factories returning a fresh iterator for each file. Empty sources
        are removed, including when infinite=True.

    ratios: list of float
        Finite, nonnegative sampling weights. Zero-weight sources are not
        opened. At least one weight must be positive for nonempty inputs.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.ndim != 1 or len(ratios) != len(iterators):
        raise ValueError("Provide one sampling weight per iterator")
    if not np.all(np.isfinite(ratios)) or np.any(ratios < 0):
        raise ValueError("Sampling weights must be finite and nonnegative")
    if not iterators:
        return
    enabled = ratios > 0
    if not np.any(enabled):
        raise ValueError("At least one sampling weight must be positive")
    iterators = [
        cycle(factory) if infinite else iter(factory())
        for factory, active in zip(iterators, enabled) if active
    ]
    ratios = ratios[enabled]

    def normalize(weights):
        # Scale before summing so large, finite weights cannot overflow.
        probabilities = weights / weights.max()
        return probabilities / probabilities.sum()

    probabilities = normalize(ratios)
    while iterators:
        choice = np.random.choice(len(ratios), p=probabilities)
        try:
            yield next(iterators[choice])
        except StopIteration:
            del iterators[choice]
            ratios = np.delete(ratios, choice)
            if iterators:
                probabilities = normalize(ratios)


def shuffle_iterator(iterator: typing.Iterator, queue_size: int) -> typing.Iterable[typing.Any]:
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
        warnings.warn(
            "Number of elements in the iterator is less than the " f"queue size (N={queue_size})."
        )
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)
