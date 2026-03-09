"""
Unit tests for hypotestx.tests.parametric
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypotestx.tests.parametric import (
    one_sample_ttest, two_sample_ttest, paired_ttest, anova_one_way,
)
from hypotestx.core.exceptions import InsufficientDataError, InvalidAlternativeError
from hypotestx.core.result import HypoResult


class TestOneSampleTTest(unittest.TestCase):

    def setUp(self):
        # Mean ~ 12, mu0 = 10  -> should be significant
        self.data = [10.5, 12.1, 11.8, 13.2, 12.5, 10.9, 11.6, 13.0,
                     12.3, 11.7, 12.8, 10.8, 13.1, 12.0, 11.9]

    def test_returns_hyporesult(self):
        r = one_sample_ttest(self.data, mu=10)
        self.assertIsInstance(r, HypoResult)

    def test_significant_result(self):
        r = one_sample_ttest(self.data, mu=10, alpha=0.05)
        self.assertTrue(r.is_significant, f"Expected significant, p={r.p_value:.4f}")

    def test_not_significant_for_matching_mu(self):
        r = one_sample_ttest(self.data, mu=12.0, alpha=0.05)
        self.assertFalse(r.is_significant, f"Expected not significant, p={r.p_value:.4f}")

    def test_effect_size_cohens_d(self):
        r = one_sample_ttest(self.data, mu=10)
        self.assertIsNotNone(r.effect_size)
        self.assertIn("cohen", r.effect_size_name.lower())

    def test_confidence_interval_contains_mean(self):
        r = one_sample_ttest(self.data, mu=10)
        sample_mean = sum(self.data) / len(self.data)
        self.assertLessEqual(r.confidence_interval[0], sample_mean)
        self.assertGreaterEqual(r.confidence_interval[1], sample_mean)

    def test_alternative_greater(self):
        r = one_sample_ttest(self.data, mu=10, alternative="greater")
        self.assertLess(r.p_value, 0.05)

    def test_alternative_less(self):
        r = one_sample_ttest(self.data, mu=15, alternative="less")
        self.assertLess(r.p_value, 0.05)

    def test_degrees_of_freedom(self):
        r = one_sample_ttest(self.data, mu=10)
        self.assertEqual(r.degrees_of_freedom, len(self.data) - 1)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            one_sample_ttest([5.0], mu=5)

    def test_invalid_alternative_raises(self):
        with self.assertRaises(InvalidAlternativeError):
            one_sample_ttest(self.data, mu=10, alternative="invalid")

    def test_statistic_sign(self):
        # Data mean > mu0, so t > 0
        r = one_sample_ttest(self.data, mu=10)
        self.assertGreater(r.statistic, 0)

    def test_p_value_range(self):
        r = one_sample_ttest(self.data, mu=10)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1)


class TestTwoSampleTTest(unittest.TestCase):

    def setUp(self):
        self.control   = [5, 6, 7, 6, 5, 6, 7, 5, 6, 7]
        self.treatment = [9, 10, 8, 11, 9, 10, 8, 9, 10, 11]

    def test_significant_difference(self):
        r = two_sample_ttest(self.control, self.treatment)
        self.assertTrue(r.is_significant)

    def test_equal_var_flag(self):
        r_student = two_sample_ttest(self.control, self.treatment, equal_var=True)
        r_welch   = two_sample_ttest(self.control, self.treatment, equal_var=False)
        # Both should be significant; p-values close
        self.assertTrue(r_student.is_significant)
        self.assertTrue(r_welch.is_significant)

    def test_no_difference(self):
        same = [5, 6, 7, 5, 6, 7, 5, 6, 7, 6]
        r = two_sample_ttest(same, same)
        self.assertFalse(r.is_significant)

    def test_effect_size_present(self):
        r = two_sample_ttest(self.control, self.treatment)
        self.assertIsNotNone(r.effect_size)

    def test_sample_sizes(self):
        r = two_sample_ttest(self.control, self.treatment)
        self.assertIsNotNone(r.sample_sizes)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            two_sample_ttest([1], [2, 3])


class TestPairedTTest(unittest.TestCase):

    def setUp(self):
        # Differences must NOT all be equal (std=0 raises for t-test)
        self.before = [4, 5, 6, 7, 8, 6, 7, 8, 5, 6]
        self.after  = [7, 9, 8, 10, 11, 9, 10, 11, 8, 9]

    def test_significant_improvement(self):
        r = paired_ttest(self.before, self.after)
        self.assertTrue(r.is_significant)

    def test_unequal_lengths_raises(self):
        from hypotestx.core.exceptions import DataFormatError
        with self.assertRaises((ValueError, DataFormatError)):
            paired_ttest([1, 2, 3], [1, 2])

    def test_no_change(self):
        # All-identical differences cause std=0: function raises, confirming no measurable change
        same = [5, 6, 7, 8, 9]
        with self.assertRaises((ValueError, InsufficientDataError)):
            paired_ttest(same, same)


class TestAnovaOneWay(unittest.TestCase):

    def setUp(self):
        self.g1 = [5, 6, 7, 6, 5]
        self.g2 = [10, 11, 12, 10, 11]
        self.g3 = [15, 16, 17, 15, 16]

    def test_significant_groups(self):
        r = anova_one_way(self.g1, self.g2, self.g3)
        self.assertTrue(r.is_significant)

    def test_no_difference(self):
        g = [5, 6, 5, 6, 5]
        r = anova_one_way(g, g, g)
        self.assertFalse(r.is_significant)

    def test_requires_two_groups(self):
        with self.assertRaises((InsufficientDataError, ValueError)):
            anova_one_way(self.g1)

    def test_effect_size_eta_squared(self):
        r = anova_one_way(self.g1, self.g2, self.g3)
        self.assertIsNotNone(r.effect_size)
        self.assertGreater(r.effect_size, 0)
        self.assertLessEqual(r.effect_size, 1)

    def test_df_tuple(self):
        r = anova_one_way(self.g1, self.g2, self.g3)
        self.assertIsInstance(r.degrees_of_freedom, tuple)
        self.assertEqual(r.degrees_of_freedom[0], 2)  # k-1 = 3-1 = 2


if __name__ == '__main__':
    unittest.main()
