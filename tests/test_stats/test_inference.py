"""
Tests for hypotestx.stats.inference — CIs and z-test.
"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.stats.inference import (
    confidence_interval_mean,
    confidence_interval_proportion,
    confidence_interval_difference_of_means,
    z_test_one_sample,
)


def approx(a, b, tol=1e-3):
    return abs(a - b) < tol


DATA = [2.1, 2.5, 3.0, 2.8, 3.3, 2.9, 3.1, 2.7, 3.0, 2.6]
GROUP_A = [5.0, 6.0, 7.0, 5.5, 6.5, 6.0]
GROUP_B = [3.0, 4.0, 3.5, 4.5, 3.0, 4.0]


# ---------------------------------------------------------------------------
# confidence_interval_mean
# ---------------------------------------------------------------------------

class TestConfidenceIntervalMean:
    def test_returns_tuple(self):
        lo, hi = confidence_interval_mean(DATA)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_lt_upper(self):
        lo, hi = confidence_interval_mean(DATA)
        assert lo < hi

    def test_mean_inside_ci(self):
        lo, hi = confidence_interval_mean(DATA, confidence=0.95)
        m = sum(DATA) / len(DATA)
        assert lo <= m <= hi

    def test_wider_at_higher_confidence(self):
        lo90, hi90 = confidence_interval_mean(DATA, confidence=0.90)
        lo99, hi99 = confidence_interval_mean(DATA, confidence=0.99)
        assert (hi99 - lo99) > (hi90 - lo90)

    def test_alternative_greater_upper_is_inf(self):
        lo, hi = confidence_interval_mean(DATA, alternative="greater")
        assert hi == float("inf")
        assert lo < float("inf")

    def test_alternative_less_lower_is_neg_inf(self):
        lo, hi = confidence_interval_mean(DATA, alternative="less")
        assert lo == float("-inf")
        assert hi > float("-inf")

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_mean(DATA, confidence=1.5)

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_mean(DATA, alternative="bigger")

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_mean([5.0])


# ---------------------------------------------------------------------------
# confidence_interval_proportion
# ---------------------------------------------------------------------------

class TestConfidenceIntervalProportion:
    def test_wilson_returns_valid_range(self):
        lo, hi = confidence_interval_proportion(40, 100)
        assert 0.0 <= lo <= hi <= 1.0

    def test_normal_method(self):
        lo, hi = confidence_interval_proportion(40, 100, method="normal")
        assert 0.0 <= lo <= hi <= 1.0

    def test_prop_inside_wilson_ci(self):
        lo, hi = confidence_interval_proportion(30, 100, confidence=0.95)
        assert lo <= 0.30 <= hi

    def test_zero_successes(self):
        lo, hi = confidence_interval_proportion(0, 100)
        assert lo < 0.001   # Wilson CI lower bound near but not exactly 0
        assert hi >= 0.0

    def test_all_successes(self):
        lo, hi = confidence_interval_proportion(100, 100)
        assert lo >= 0.0
        assert hi == 1.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_proportion(5, 0)

    def test_invalid_successes_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_proportion(-1, 100)

    def test_successes_exceeds_n_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_proportion(110, 100)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_proportion(10, 100, method="bayes")


# ---------------------------------------------------------------------------
# confidence_interval_difference_of_means
# ---------------------------------------------------------------------------

class TestConfidenceIntervalDifferenceOfMeans:
    def test_returns_tuple(self):
        lo, hi = confidence_interval_difference_of_means(GROUP_A, GROUP_B)
        assert isinstance(lo, float), isinstance(hi, float)

    def test_lower_lt_upper(self):
        lo, hi = confidence_interval_difference_of_means(GROUP_A, GROUP_B)
        assert lo < hi

    def test_true_diff_inside_ci(self):
        lo, hi = confidence_interval_difference_of_means(GROUP_A, GROUP_B, 0.95)
        diff = sum(GROUP_A) / len(GROUP_A) - sum(GROUP_B) / len(GROUP_B)
        assert lo <= diff <= hi

    def test_small_group_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_difference_of_means([1.0], GROUP_B)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            confidence_interval_difference_of_means(GROUP_A, GROUP_B, 0.0)


# ---------------------------------------------------------------------------
# z_test_one_sample
# ---------------------------------------------------------------------------

class TestZTestOneSample:
    def test_returns_tuple(self):
        z, p = z_test_one_sample(DATA, mu0=2.8)
        assert isinstance(z, float)
        assert isinstance(p, float)

    def test_p_value_in_range(self):
        _, p = z_test_one_sample(DATA, mu0=2.8)
        assert 0.0 <= p <= 1.0

    def test_known_null_is_true_has_large_pvalue(self):
        # If null = true mean, p should be large (close to 1 for two-sided)
        m = sum(DATA) / len(DATA)
        _, p = z_test_one_sample(DATA, mu0=m)
        assert p > 0.5

    def test_clearly_wrong_null_has_small_pvalue(self):
        # Use known sigma to avoid se=0 from constant data
        large_data = [10.0] * 30
        _, p = z_test_one_sample(large_data, mu0=0.0, sigma=1.0)
        assert p < 0.001

    def test_known_sigma(self):
        z, p = z_test_one_sample(DATA, mu0=2.8, sigma=0.5)
        assert isinstance(z, float)
        assert 0.0 <= p <= 1.0

    def test_alternative_greater(self):
        data = [10.0] * 20
        _, p = z_test_one_sample(data, mu0=5.0, sigma=1.0, alternative="greater")
        assert p < 0.001

    def test_alternative_less(self):
        data = [0.0] * 20
        _, p = z_test_one_sample(data, mu0=5.0, sigma=1.0, alternative="less")
        assert p < 0.001

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            z_test_one_sample([], mu0=0.0)

    def test_constant_data_with_known_sigma(self):
        # constant data with sigma=0 -> se=0, should raise
        with pytest.raises(ValueError):
            z_test_one_sample([5.0] * 10, mu0=5.0, sigma=0.0)

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError):
            z_test_one_sample(DATA, mu0=0.0, alternative="invalid")
