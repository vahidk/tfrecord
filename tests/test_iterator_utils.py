import itertools
import unittest

from tfrecord.iterator_utils import cycle, sample_iterators


class TestSampling(unittest.TestCase):
    def test_empty_cycle_stops(self):
        calls = 0

        def empty():
            nonlocal calls
            calls += 1
            if calls > 3:
                self.fail("Empty iterator was restarted indefinitely")
            return iter(())

        self.assertEqual(list(cycle(empty)), [])

    def test_nonempty_cycle_repeats(self):
        self.assertEqual(list(itertools.islice(cycle(lambda: iter([1, 2])), 5)), [1, 2, 1, 2, 1])

    def test_zero_weight_sources_are_never_opened(self):
        def disabled():
            self.fail("Zero-weight source was opened")

        factories = [lambda: iter([1, 2]), disabled]
        self.assertEqual(list(sample_iterators(factories, [1.0, 0.0], infinite=False)), [1, 2])
        self.assertEqual(list(itertools.islice(sample_iterators(factories, [1.0, 0.0]), 3)), [1, 2, 1])

    def test_infinite_sampling_skips_empty_sources(self):
        sources = [lambda: iter(()), lambda: iter([1, 2])]
        self.assertEqual(list(itertools.islice(sample_iterators(sources, [1, 1]), 4)), [1, 2, 1, 2])

    def test_finite_sampling_exhausts_every_enabled_source(self):
        sources = [lambda: iter([1, 2]), lambda: iter([3, 4])]
        self.assertEqual(sorted(sample_iterators(sources, [0.8, 0.2], infinite=False)), [1, 2, 3, 4])

    def test_empty_sampling_stops(self):
        self.assertEqual(list(sample_iterators([], [])), [])
        self.assertEqual(list(sample_iterators([lambda: iter(())], [1])), [])

    def test_invalid_weights_are_rejected(self):
        for ratios in ([0], [-1], [float("nan")], [float("inf")], [1, 2], [[1]]):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                next(sample_iterators([lambda: iter([1])], ratios))

    def test_large_finite_weights_do_not_overflow(self):
        sources = [lambda: iter([1]), lambda: iter([2])]
        self.assertEqual(sorted(sample_iterators(sources, [1e308, 1e308], infinite=False)), [1, 2])
