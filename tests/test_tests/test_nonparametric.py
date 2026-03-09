"""
Unit tests for hypotestx.tests.nonparametric
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypotestx.tests.nonparametric import (
    mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis,
)
from hypotestx.core.exceptions import InsufficientDataError
from hypotestx.core.result import HypoResult


class TestMannWhitneyU(unittest.TestCase):

    def setUp(self):
        self.low  = [1, 2, 3, 4, 5]
        self.high = [6, 7, 8, 9, 10]

    def test_returns_hyporesult(self):
        r = mann_whitney_u(self.low, self.high)
        self.assertIsInstance(r, HypoResult)

    def test_significant_separation(self):
        r = mann_whitney_u(self.low, self.high)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_no_difference(self):
        same = [5, 6, 5, 6, 5]
        r = mann_whitney_u(same, same)
        self.assertFalse(r.is_significant)

    def test_effect_size_range(self):
        r = mann_whitney_u(self.low, self.high)
        # rank-biserial r in [-1, 1]
        self.assertGreaterEqual(abs(r.effect_size), 0)
        self.assertLessEqual(abs(r.effect_size), 1)

    def test_alternative_greater(self):
        r = mann_whitney_u(self.low, self.high, alternative="less")
        self.assertLess(r.p_value, 0.05)

    def test_p_value_range(self):
        r = mann_whitney_u(self.low, self.high)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            mann_whitney_u([1], [2, 3])


class TestWilcoxonSignedRank(unittest.TestCase):

    def setUp(self):
        # After values consistently higher
        self.before = [4, 5, 6, 7, 8, 5, 6, 7, 5, 6]
        self.after  = [7, 8, 9, 10, 11, 8, 9, 10, 8, 9]

    def test_returns_hyporesult(self):
        r = wilcoxon_signed_rank(self.before, self.after)
        self.assertIsInstance(r, HypoResult)

    def test_paired_significant(self):
        r = wilcoxon_signed_rank(self.before, self.after)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_one_sample(self):
        # Test one-sample version (mu=0)
        diffs = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]   # all positive
        r = wilcoxon_signed_rank(diffs, mu=0)
        self.assertTrue(r.is_significant)

    def test_no_change(self):
        # All differences are zero: Wilcoxon cannot be performed — should raise
        same = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]
        with self.assertRaises((ValueError, InsufficientDataError)):
            wilcoxon_signed_rank(same, same)

    def test_p_value_range(self):
        r = wilcoxon_signed_rank(self.before, self.after)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            wilcoxon_signed_rank([5], [6])


class TestKruskalWallis(unittest.TestCase):

    def setUp(self):
        self.g1 = [1, 2, 3, 4, 5]
        self.g2 = [10, 11, 12, 13, 14]
        self.g3 = [20, 21, 22, 23, 24]

    def test_returns_hyporesult(self):
        r = kruskal_wallis(self.g1, self.g2, self.g3)
        self.assertIsInstance(r, HypoResult)

    def test_significant_groups(self):
        r = kruskal_wallis(self.g1, self.g2, self.g3)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_no_difference(self):
        same = [5, 6, 5, 6, 5]
        r = kruskal_wallis(same, same, same)
        # Should not be significant for identical groups
        self.assertFalse(r.is_significant)

    def test_requires_two_groups(self):
        with self.assertRaises((InsufficientDataError, ValueError)):
            kruskal_wallis(self.g1)

    def test_degrees_of_freedom(self):
        r = kruskal_wallis(self.g1, self.g2, self.g3)
        self.assertEqual(r.degrees_of_freedom, 2)  # k-1 = 3-1 = 2

    def test_effect_size_range(self):
        r = kruskal_wallis(self.g1, self.g2, self.g3)
        self.assertGreaterEqual(r.effect_size, 0)
        self.assertLessEqual(r.effect_size, 1)


if __name__ == '__main__':
    unittest.main()
