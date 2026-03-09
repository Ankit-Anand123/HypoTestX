"""
Unit tests for hypotestx.tests.correlation
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypotestx.tests.correlation import (
    pearson_correlation, spearman_correlation, point_biserial_correlation,
)
from hypotestx.core.exceptions import InsufficientDataError, DataFormatError
from hypotestx.core.result import HypoResult


class TestPearsonCorrelation(unittest.TestCase):

    def setUp(self):
        self.x_pos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y_pos = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]
        self.x_neg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y_neg = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    def test_returns_hyporesult(self):
        r = pearson_correlation(self.x_pos, self.y_pos)
        self.assertIsInstance(r, HypoResult)

    def test_positive_significant(self):
        r = pearson_correlation(self.x_pos, self.y_pos)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_negative_correlation(self):
        r = pearson_correlation(self.x_neg, self.y_neg)
        self.assertLess(r.effect_size, 0)

    def test_effect_size_range(self):
        r = pearson_correlation(self.x_pos, self.y_pos)
        self.assertGreaterEqual(abs(r.effect_size), 0)
        self.assertLessEqual(abs(r.effect_size), 1)

    def test_perfect_correlation_r_is_one(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_correlation(x, y)
        self.assertAlmostEqual(abs(r.effect_size), 1.0, places=5)

    def test_no_correlation(self):
        # Near-zero correlation: shuffled y unrelated to x
        x = list(range(1, 21))
        y = [5, 8, 3, 6, 9, 2, 7, 4, 8, 5, 3, 7, 6, 9, 4, 5, 8, 3, 6, 9]
        r = pearson_correlation(x, y)
        self.assertFalse(r.is_significant)

    def test_ci_returned(self):
        r = pearson_correlation(self.x_pos, self.y_pos)
        self.assertIsNotNone(r.confidence_interval)

    def test_altered_alpha(self):
        r = pearson_correlation(self.x_pos, self.y_pos, alpha=0.01)
        self.assertEqual(r.alpha, 0.01)

    def test_alternative_two_sided(self):
        r = pearson_correlation(self.x_pos, self.y_pos, alternative='two-sided')
        self.assertIsInstance(r, HypoResult)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            pearson_correlation([1], [2])

    def test_p_value_range(self):
        r = pearson_correlation(self.x_pos, self.y_pos)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)


class TestSpearmanCorrelation(unittest.TestCase):

    def setUp(self):
        self.x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]

    def test_returns_hyporesult(self):
        r = spearman_correlation(self.x, self.y)
        self.assertIsInstance(r, HypoResult)

    def test_significant(self):
        r = spearman_correlation(self.x, self.y)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_effect_size_range(self):
        r = spearman_correlation(self.x, self.y)
        self.assertGreaterEqual(abs(r.effect_size), 0)
        self.assertLessEqual(abs(r.effect_size), 1)

    def test_monotone_perfect(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 50, 100, 200]  # monotone, not linear
        r = spearman_correlation(x, y)
        self.assertAlmostEqual(abs(r.effect_size), 1.0, places=5)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            spearman_correlation([1], [2])


class TestPointBiserialCorrelation(unittest.TestCase):

    def setUp(self):
        # Binary group and continuous measure
        # point_biserial_correlation(continuous, binary)
        self.cont   = [2, 3, 4, 5, 6, 14, 15, 16, 17, 18]
        self.binary = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_returns_hyporesult(self):
        r = point_biserial_correlation(self.cont, self.binary)
        self.assertIsInstance(r, HypoResult)

    def test_significant(self):
        r = point_biserial_correlation(self.cont, self.binary)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_effect_size_range(self):
        r = point_biserial_correlation(self.cont, self.binary)
        self.assertGreaterEqual(abs(r.effect_size), 0)
        self.assertLessEqual(abs(r.effect_size), 1)

    def test_non_binary_raises(self):
        # Second arg must be binary; three-valued second arg should raise
        with self.assertRaises((DataFormatError, ValueError)):
            point_biserial_correlation([4, 5, 6], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
