"""
Unit tests for hypotestx.tests.categorical
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.core.exceptions import DataFormatError
from hypotestx.core.result import HypoResult
from hypotestx.tests.categorical import chi_square_test, fisher_exact_test


class TestChiSquareTest(unittest.TestCase):

    def setUp(self):
        # Clearly different proportions
        self.table_sig = [[50, 10], [10, 50]]
        # Nearly even — expectation of no association
        self.table_ns = [[25, 25], [25, 25]]

    def test_returns_hyporesult(self):
        r = chi_square_test(self.table_sig)
        self.assertIsInstance(r, HypoResult)

    def test_significant_table(self):
        r = chi_square_test(self.table_sig)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_no_association(self):
        r = chi_square_test(self.table_ns)
        self.assertFalse(r.is_significant)

    def test_p_value_in_range(self):
        r = chi_square_test(self.table_sig)
        self.assertGreaterEqual(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)

    def test_effect_size_cramers_v(self):
        r = chi_square_test(self.table_sig)
        self.assertGreaterEqual(r.effect_size, 0)
        self.assertLessEqual(r.effect_size, 1)

    def test_degrees_of_freedom_2x2(self):
        r = chi_square_test(self.table_sig)
        self.assertEqual(r.degrees_of_freedom, 1)

    def test_degrees_of_freedom_3x2(self):
        table_3x2 = [[10, 20], [30, 5], [15, 15]]
        r = chi_square_test(table_3x2)
        self.assertEqual(r.degrees_of_freedom, 2)

    def test_larger_table(self):
        table = [[10, 20, 5], [5, 15, 20]]
        r = chi_square_test(table)
        self.assertIsInstance(r, HypoResult)

    def test_yates_correction(self):
        # 2x2 — Yates default should not raise
        r = chi_square_test(self.table_sig, correction=True)
        self.assertIsInstance(r, HypoResult)

    def test_invalid_table_raises(self):
        with self.assertRaises(Exception):
            chi_square_test([[5]])  # 1x1 table

    def test_zero_cell(self):
        # Zero cell should not raise but may warn
        table = [[50, 0], [0, 50]]
        r = chi_square_test(table)
        self.assertIsInstance(r, HypoResult)


class TestFisherExactTest(unittest.TestCase):

    def setUp(self):
        self.table_sig = [[14, 0], [1, 11]]
        self.table_ns = [[10, 10], [10, 10]]

    def test_returns_hyporesult(self):
        r = fisher_exact_test(self.table_sig)
        self.assertIsInstance(r, HypoResult)

    def test_significant_table(self):
        r = fisher_exact_test(self.table_sig)
        self.assertTrue(r.is_significant, f"p={r.p_value:.4f}")

    def test_no_association(self):
        r = fisher_exact_test(self.table_ns)
        self.assertFalse(r.is_significant)

    def test_p_value_range(self):
        r = fisher_exact_test(self.table_sig)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)

    def test_odds_ratio_available(self):
        r = fisher_exact_test(self.table_sig)
        # odds ratio stored in statistic or effect_size
        val = r.statistic if r.effect_size is None else r.effect_size
        self.assertGreater(val, 0)

    def test_non_2x2_raises(self):
        with self.assertRaises((DataFormatError, ValueError)):
            fisher_exact_test([[5, 5, 5], [5, 5, 5]])


if __name__ == "__main__":
    unittest.main()
