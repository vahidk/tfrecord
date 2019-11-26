import typing
import numpy as np


def cycle(iterator_fn: typing.Callable):
    """Create a repeating iterator from an iterator generator.."""
    while True:
        for element in iterator_fn():
            yield element


def sample_iterators(iterators: typing.List[typing.Iterable], ratios):
    """Create an iterator by sampling from a list of iterators with given ratios."""
    iterators = [cycle(iterator) for iterator in iterators]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    while True:
        choice = np.random.choice(len(ratios), p=ratios)
        yield next(iterators[choice])


def shuffle_iterator(iterator: typing.Iterable, queue_size):
    """Shuffle elements of an iterator."""
    buffer = []
    try:
        for _ in range(queue_size):
            buffer.append(next(iterator))
    except StopIteration:
        print("Warning: Elements in the iterator are less than queue size.")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)
