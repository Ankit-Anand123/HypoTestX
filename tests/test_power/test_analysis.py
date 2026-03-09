"""
Tests for hypotestx.power — power analysis and sample size calculation.
"""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.power.analysis import (
    power_ttest_one_sample,
    power_ttest_two_sample,
    power_anova,
)
from hypotestx.power.sample_size import (
    n_ttest_one_sample,
    n_ttest_two_sample,
)


def approx(a, b, tol=0.01):
    return abs(a - b) < tol


class TestPowerTTestOneSample:
    def test_returns_float_in_01(self):
        result = power_ttest_one_sample(effect_size=0.5, n=30, alpha=0.05)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_larger_n_higher_power(self):
        p1 = power_ttest_one_sample(effect_size=0.5, n=30, alpha=0.05)
        p2 = power_ttest_one_sample(effect_size=0.5, n=100, alpha=0.05)
        assert p2 > p1

    def test_larger_effect_higher_power(self):
        p1 = power_ttest_one_sample(effect_size=0.2, n=50, alpha=0.05)
        p2 = power_ttest_one_sample(effect_size=0.8, n=50, alpha=0.05)
        assert p2 > p1

    def test_large_effect_large_n_near_1(self):
        result = power_ttest_one_sample(effect_size=1.0, n=200, alpha=0.05)
        assert result > 0.99

    def test_alternative_one_sided(self):
        p2 = power_ttest_one_sample(effect_size=0.5, n=30, alpha=0.05,
                                    alternative="greater")
        p1 = power_ttest_one_sample(effect_size=0.5, n=30, alpha=0.05,
                                    alternative="two-sided")
        assert p2 >= p1


class TestPowerTTestTwoSample:
    def test_basic(self):
        result = power_ttest_two_sample(effect_size=0.5, n1=30, alpha=0.05)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_larger_n_higher_power(self):
        p1 = power_ttest_two_sample(effect_size=0.5, n1=30, alpha=0.05)
        p2 = power_ttest_two_sample(effect_size=0.5, n1=100, alpha=0.05)
        assert p2 > p1


class TestPowerANOVA:
    def test_basic(self):
        result = power_anova(effect_size=0.25, n_per_group=20, k=3, alpha=0.05)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_larger_n_higher_power(self):
        p1 = power_anova(effect_size=0.25, n_per_group=20, k=3, alpha=0.05)
        p2 = power_anova(effect_size=0.25, n_per_group=80, k=3, alpha=0.05)
        assert p2 > p1


class TestSampleSizeOneSample:
    def test_returns_positive_int(self):
        n = n_ttest_one_sample(effect_size=0.5, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_effect_smaller_n(self):
        n1 = n_ttest_one_sample(effect_size=0.2, alpha=0.05, power=0.80)
        n2 = n_ttest_one_sample(effect_size=0.8, alpha=0.05, power=0.80)
        assert n1 > n2

    def test_higher_power_larger_n(self):
        n1 = n_ttest_one_sample(effect_size=0.5, alpha=0.05, power=0.80)
        n2 = n_ttest_one_sample(effect_size=0.5, alpha=0.05, power=0.95)
        assert n2 > n1

    def test_consistency_with_power(self):
        """Sample size computed for power=0.80 should achieve ~0.80 power."""
        n = n_ttest_one_sample(effect_size=0.5, alpha=0.05, power=0.80)
        achieved = power_ttest_one_sample(effect_size=0.5, n=n, alpha=0.05)
        assert achieved >= 0.75


class TestSampleSizeTwoSample:
    def test_returns_positive_int(self):
        n = n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0

    def test_two_sample_larger_than_one(self):
        n1 = n_ttest_one_sample(effect_size=0.5, alpha=0.05, power=0.80)
        n2 = n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.80)
        assert n2 >= n1
