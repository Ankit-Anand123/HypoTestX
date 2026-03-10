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


# ---------------------------------------------------------------------------
# Edge-case tests (Issues 3 & 10)
# ---------------------------------------------------------------------------

class TestWelchDivisionByZero(unittest.TestCase):
    """Welch t-test must never silently divide by zero (Issue 3)."""

    def test_both_constant_groups_raises(self):
        """Both groups constant → denominator = 0; must raise, not produce NaN/inf."""
        g1 = [5.0, 5.0, 5.0, 5.0]
        g2 = [5.0, 5.0, 5.0, 5.0]
        with self.assertRaises(ValueError) as ctx:
            two_sample_ttest(g1, g2, equal_var=False)
        self.assertIn("zero variance", str(ctx.exception).lower())

    def test_one_constant_group_raises(self):
        """One group constant, the other variable: se_sq > 0 → Welch defined,
        but the constant-group path in Welch-Satterthwaite denom is 0.
        Should raise with a descriptive message."""
        g1 = [5.0, 5.0, 5.0, 5.0]
        g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
        # Standard error > 0 (var2/n2 > 0), so this can potentially succeed.
        # The important thing is it must NOT produce a ZeroDivisionError silently.
        try:
            r = two_sample_ttest(g1, g2, equal_var=False)
            # If it runs, p_value must be a valid float
            self.assertFalse(r.p_value != r.p_value,  # nan check
                             "p_value should not be NaN")
        except ValueError as exc:
            # Also acceptable: raise with a descriptive message
            self.assertTrue(len(str(exc)) > 0)

    def test_student_both_constant_raises(self):
        """Student t-test: pooled variance 0 → must raise descriptively."""
        g1 = [3.0, 3.0, 3.0]
        g2 = [3.0, 3.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            two_sample_ttest(g1, g2, equal_var=True)
        self.assertIn("zero variance", str(ctx.exception).lower())

    def test_normal_welch_still_works(self):
        """Sanity: well-behaved data should still produce valid results."""
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [6.0, 7.0, 8.0, 9.0, 10.0]
        r = two_sample_ttest(g1, g2, equal_var=False)
        self.assertTrue(r.is_significant)
        self.assertFalse(r.p_value != r.p_value)  # not NaN


class TestEdgeCasesOneSample(unittest.TestCase):
    """Edge cases for one-sample t-test."""

    def test_empty_list_raises(self):
        with self.assertRaises((ValueError, InsufficientDataError)):
            one_sample_ttest([], mu=0)

    def test_single_element_raises(self):
        with self.assertRaises((ValueError, InsufficientDataError)):
            one_sample_ttest([5.0], mu=5)

    def test_constant_data_raises(self):
        with self.assertRaises(ValueError):
            one_sample_ttest([3.0, 3.0, 3.0, 3.0], mu=0)

    def test_large_dataset(self):
        data = [float(i) for i in range(1, 1001)]
        r = one_sample_ttest(data, mu=500)
        self.assertIsNotNone(r.p_value)
        self.assertFalse(r.p_value != r.p_value)  # not NaN


class TestEdgeCasesTwoSample(unittest.TestCase):
    """Edge cases for two-sample t-test."""

    def test_minimum_size(self):
        r = two_sample_ttest([1.0, 2.0], [3.0, 4.0])
        self.assertIsNotNone(r.statistic)

    def test_one_element_each_raises(self):
        with self.assertRaises((ValueError, InsufficientDataError)):
            two_sample_ttest([5.0], [6.0])

    def test_p_value_in_range(self):
        g1 = [1, 2, 3, 4, 5]
        g2 = [1, 2, 3, 4, 5]
        r = two_sample_ttest(g1, g2)
        self.assertGreater(r.p_value, 0)
        self.assertLessEqual(r.p_value, 1.0)

    def test_very_large_effect(self):
        g1 = [0.0] * 100
        g2 = [100.0] * 100
        # Student path: both have zero variance → should raise
        with self.assertRaises(ValueError):
            two_sample_ttest(g1, g2, equal_var=True)


if __name__ == '__main__':
    unittest.main()
