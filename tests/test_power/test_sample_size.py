"""
Tests for sample-size calculation functions in hypotestx.power.sample_size.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hypotestx.power.sample_size import (
    _solve_n,
    n_anova,
    n_chi_square,
    n_correlation,
    n_ttest_paired,
    sample_size_summary,
)


class TestSolveN:
    def test_bad_target_power_zero(self):
        with pytest.raises(ValueError, match="strictly between"):
            _solve_n(lambda n, **kw: 0.5, target_power=0.0)

    def test_bad_target_power_one(self):
        with pytest.raises(ValueError, match="strictly between"):
            _solve_n(lambda n, **kw: 0.5, target_power=1.0)


class TestNTTestPaired:
    def test_returns_int(self):
        n = n_ttest_paired(effect_size=0.5)
        assert isinstance(n, int)
        assert n > 2

    def test_large_effect_needs_fewer(self):
        n_med = n_ttest_paired(effect_size=0.5)
        n_large = n_ttest_paired(effect_size=0.8)
        assert n_med > n_large

    def test_one_sided(self):
        n_two = n_ttest_paired(effect_size=0.5, alternative="two-sided")
        n_one = n_ttest_paired(effect_size=0.5, alternative="greater")
        assert n_two >= n_one


class TestNAnova:
    def test_basic(self):
        n = n_anova(effect_size=0.25, k=3)
        assert isinstance(n, int)
        assert n > 2

    def test_more_groups_needs_fewer_per_group(self):
        n3 = n_anova(effect_size=0.25, k=3)
        n6 = n_anova(effect_size=0.25, k=6)
        # more groups, same total N target, fewer per group
        assert n6 <= n3

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="at least 2 groups"):
            n_anova(effect_size=0.25, k=1)


class TestNChiSquare:
    def test_basic(self):
        n = n_chi_square(effect_size=0.3, df=1)
        assert isinstance(n, int)
        assert n > 2

    def test_larger_effect_fewer_n(self):
        n_small = n_chi_square(effect_size=0.1, df=1)
        n_large = n_chi_square(effect_size=0.5, df=1)
        assert n_small > n_large


class TestNCorrelation:
    def test_basic(self):
        n = n_correlation(r=0.3)
        assert isinstance(n, int)
        assert n >= 4

    def test_larger_r_fewer_n(self):
        n_small = n_correlation(r=0.1)
        n_large = n_correlation(r=0.5)
        assert n_small > n_large


class TestSampleSizeSummary:
    def test_one_sample_t(self):
        result = sample_size_summary("one_sample_t", 0.5)
        assert isinstance(result, str)
        assert "one sample t" in result.lower()
        assert "Effect size" in result

    def test_paired_t(self):
        result = sample_size_summary("paired_t", 0.5)
        assert "pairs" in result

    def test_two_sample_t(self):
        result = sample_size_summary("two_sample_t", 0.5)
        assert "per group" in result

    def test_anova(self):
        result = sample_size_summary("anova", 0.25, k=3)
        assert "per group" in result

    def test_chi_square(self):
        result = sample_size_summary("chi_square", 0.3, df=1)
        assert "total" in result

    def test_correlation(self):
        result = sample_size_summary("correlation", 0.3)
        assert "total" in result

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            sample_size_summary("bad_type", 0.5)
