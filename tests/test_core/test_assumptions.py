"""
Unit tests for hypotestx.core.assumptions
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.core.assumptions import (
    bartlett_test,
    check_equal_variances,
    check_normality,
    jarque_bera,
    levene_test,
    shapiro_wilk,
)
from hypotestx.core.result import HypoResult

# --- Sample data ---------------------------------------------------------

# Approximately normal (mean=5, std=1), n=30
NORMAL_DATA = [
    4.63,
    5.12,
    5.45,
    4.78,
    5.23,
    4.89,
    5.67,
    4.34,
    5.01,
    5.56,
    4.22,
    5.88,
    4.99,
    5.32,
    4.71,
    5.15,
    4.55,
    5.43,
    5.02,
    4.87,
    5.29,
    4.61,
    5.18,
    4.93,
    5.07,
    5.41,
    4.75,
    5.25,
    4.98,
    5.10,
]

# Strongly right-skewed (exponential-like)
SKEWED_DATA = [
    0.02,
    0.05,
    0.07,
    0.12,
    0.18,
    0.25,
    0.50,
    0.90,
    1.50,
    2.80,
    4.20,
    6.50,
    9.00,
    15.0,
    25.0,
    40.0,
    70.0,
]


class TestShapiroWilk(unittest.TestCase):

    def test_returns_hyporesult(self):
        r = shapiro_wilk(NORMAL_DATA)
        self.assertIsInstance(r, HypoResult)

    def test_normal_data_not_rejected(self):
        r = shapiro_wilk(NORMAL_DATA)
        self.assertFalse(
            r.is_significant,
            f"Normal data flagged: W={r.statistic:.4f}, p={r.p_value:.4f}",
        )

    def test_skewed_data_rejected(self):
        r = shapiro_wilk(SKEWED_DATA)
        self.assertTrue(
            r.is_significant,
            f"Skewed data not flagged: W={r.statistic:.4f}, p={r.p_value:.4f}",
        )

    def test_w_statistic_range(self):
        r = shapiro_wilk(NORMAL_DATA)
        self.assertGreater(r.statistic, 0)
        self.assertLessEqual(r.statistic, 1)

    def test_small_sample_minimum(self):
        with self.assertRaises(Exception):
            shapiro_wilk([5.0, 6.0])  # n < 3 should raise

    def test_alpha_propagated(self):
        r = shapiro_wilk(NORMAL_DATA, alpha=0.01)
        self.assertEqual(r.alpha, 0.01)


class TestLeveneTest(unittest.TestCase):

    def setUp(self):
        # Equal-variance groups: both have similar spread
        self.equal_var = [[5, 6, 5, 6, 5, 6, 5], [5, 6, 6, 5, 6, 5, 6]]
        # Unequal-variance groups: first is tight, second is very spread
        self.unequal_var = [
            [4.9, 5.0, 5.1, 5.0, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9],
            [1, 10, 30, 70, 120, 5, 50, 15, 80, 25],
        ]

    def test_returns_hyporesult(self):
        r = levene_test(*self.equal_var)
        self.assertIsInstance(r, HypoResult)

    def test_equal_variances_not_rejected(self):
        r = levene_test(*self.equal_var)
        self.assertFalse(r.is_significant, f"Equal-var flagged: p={r.p_value:.4f}")

    def test_unequal_variances_rejected(self):
        r = levene_test(*self.unequal_var)
        self.assertTrue(r.is_significant, f"Unequal-var not flagged: p={r.p_value:.4f}")

    def test_requires_two_or_more_groups(self):
        with self.assertRaises(Exception):
            levene_test([1, 2, 3])

    def test_f_statistic_positive(self):
        r = levene_test(*self.equal_var)
        self.assertGreaterEqual(r.statistic, 0)


class TestBartlettTest(unittest.TestCase):

    def setUp(self):
        self.equal_var = [[5, 6, 5, 6, 5, 6, 5], [5, 6, 6, 5, 6, 5, 6]]
        self.unequal_var = [
            [4.9, 5.0, 5.1, 5.0, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9],
            [1, 10, 30, 70, 120, 5, 50, 15, 80, 25],
        ]

    def test_returns_hyporesult(self):
        r = bartlett_test(*self.equal_var)
        self.assertIsInstance(r, HypoResult)

    def test_equal_variances_not_rejected(self):
        r = bartlett_test(*self.equal_var)
        self.assertFalse(r.is_significant, f"Bartlett flagged equal vars: p={r.p_value:.4f}")

    def test_unequal_variances_detected(self):
        r = bartlett_test(*self.unequal_var)
        self.assertTrue(r.is_significant, f"Bartlett missed unequal vars: p={r.p_value:.4f}")

    def test_chi2_statistic_positive(self):
        r = bartlett_test(*self.equal_var)
        self.assertGreaterEqual(r.statistic, 0)


class TestJarqueBera(unittest.TestCase):

    def test_returns_hyporesult(self):
        r = jarque_bera(NORMAL_DATA)
        self.assertIsInstance(r, HypoResult)

    def test_normal_data_not_rejected(self):
        r = jarque_bera(NORMAL_DATA)
        self.assertFalse(
            r.is_significant,
            f"JB normal flagged: JB={r.statistic:.4f}, p={r.p_value:.4f}",
        )

    def test_skewed_data_rejected(self):
        r = jarque_bera(SKEWED_DATA)
        self.assertTrue(
            r.is_significant,
            f"JB skewed not flagged: JB={r.statistic:.4f}, p={r.p_value:.4f}",
        )

    def test_jb_statistic_nonnegative(self):
        r = jarque_bera(NORMAL_DATA)
        self.assertGreaterEqual(r.statistic, 0)


class TestConvenienceWrappers(unittest.TestCase):

    def test_check_normality_normal(self):
        is_normal, r = check_normality(NORMAL_DATA)
        self.assertIsInstance(is_normal, bool)
        self.assertIsInstance(r, HypoResult)
        self.assertTrue(is_normal)

    def test_check_normality_skewed(self):
        is_normal, _ = check_normality(SKEWED_DATA)
        self.assertFalse(is_normal)

    def test_check_equal_variances_equal(self):
        g1 = [5, 6, 5, 6, 5, 6, 5, 6]
        g2 = [5, 6, 6, 5, 6, 5, 6, 5]
        equal, r = check_equal_variances(g1, g2)
        self.assertIsInstance(equal, bool)
        self.assertIsInstance(r, HypoResult)
        self.assertTrue(equal)

    def test_check_equal_variances_unequal(self):
        g1 = [4.9, 5.0, 5.1, 5.0, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9]
        g2 = [1, 10, 30, 70, 120, 5, 50, 15, 80, 25]
        equal, _ = check_equal_variances(g1, g2)
        self.assertFalse(equal)


if __name__ == "__main__":
    unittest.main()
