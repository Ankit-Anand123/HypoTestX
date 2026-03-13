"""
Extra power analysis tests — coverage for paired_t, chi_square, correlation,
power_summary, and error branches not exercised by the base test file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hypotestx.power.analysis import (
    power_anova,
    power_chi_square,
    power_correlation,
    power_summary,
    power_ttest_one_sample,
    power_ttest_paired,
    power_ttest_two_sample,
)


# ── error branches ────────────────────────────────────────────────────────────

class TestPowerErrorBranches:
    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="alternative"):
            power_ttest_one_sample(0.5, 30, alternative="bad")

    def test_one_sample_n_too_small(self):
        with pytest.raises(ValueError, match="at least 2"):
            power_ttest_one_sample(0.5, 1)

    def test_two_sample_n1_too_small(self):
        with pytest.raises(ValueError):
            power_ttest_two_sample(0.5, 1, 1)

    def test_anova_k_too_small(self):
        with pytest.raises(ValueError, match="at least 2 groups"):
            power_anova(0.25, 20, 1)

    def test_anova_n_per_group_too_small(self):
        with pytest.raises(ValueError, match="n_per_group"):
            power_anova(0.25, 1, 3)


# ── paired t-test ─────────────────────────────────────────────────────────────

class TestPowerTTestPaired:
    def test_returns_float_in_01(self):
        result = power_ttest_paired(0.5, 30)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_large_effect_large_n_near_1(self):
        result = power_ttest_paired(1.0, 100)
        assert result > 0.99

    def test_one_sided(self):
        p1 = power_ttest_paired(0.5, 30, alternative="two-sided")
        p2 = power_ttest_paired(0.5, 30, alternative="greater")
        assert p2 >= p1


# ── chi-square ────────────────────────────────────────────────────────────────

class TestPowerChiSquare:
    def test_basic(self):
        result = power_chi_square(0.3, 100, df=1)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_larger_n_higher_power(self):
        p1 = power_chi_square(0.3, 50, df=1)
        p2 = power_chi_square(0.3, 500, df=1)
        assert p2 > p1

    def test_larger_effect_higher_power(self):
        p1 = power_chi_square(0.1, 100, df=1)
        p2 = power_chi_square(0.5, 100, df=1)
        assert p2 > p1

    def test_multiple_df(self):
        result = power_chi_square(0.3, 100, df=3)
        assert 0.0 < result < 1.0


# ── correlation ───────────────────────────────────────────────────────────────

class TestPowerCorrelation:
    def test_basic(self):
        result = power_correlation(0.3, 50)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_large_r_near_1(self):
        result = power_correlation(0.8, 50)
        assert result > 0.95

    def test_perfect_r_returns_1(self):
        result = power_correlation(1.0, 30)
        assert result == 1.0

    def test_n_too_small(self):
        with pytest.raises(ValueError):
            power_correlation(0.3, 3)

    def test_one_sided(self):
        p1 = power_correlation(0.3, 50, alternative="two-sided")
        p2 = power_correlation(0.3, 50, alternative="greater")
        assert p2 >= p1


# ── power_summary ─────────────────────────────────────────────────────────────

class TestPowerSummary:
    def test_one_sample_t(self):
        result = power_summary("one_sample_t", 0.5, n=30)
        assert isinstance(result, str)
        assert "Power" in result

    def test_paired_t(self):
        result = power_summary("paired_t", 0.5, n=30)
        assert "Power" in result

    def test_two_sample_t(self):
        result = power_summary("two_sample_t", 0.5, n=30)
        assert "Power" in result

    def test_anova(self):
        result = power_summary("anova", 0.25, n=20, k=3)
        assert "Power" in result

    def test_chi_square(self):
        result = power_summary("chi_square", 0.3, n=100, df=1)
        assert "Power" in result

    def test_correlation(self):
        result = power_summary("correlation", 0.3, n=50)
        assert "Power" in result

    def test_low_power_warning(self):
        result = power_summary("one_sample_t", 0.1, n=5)
        assert "Power < 0.80" in result

    def test_adequate_power(self):
        result = power_summary("one_sample_t", 1.0, n=200)
        assert "ok" in result.lower()

    def test_invalid_test_type(self):
        with pytest.raises(ValueError):
            power_summary("bad_type", 0.5, n=30)

    def test_missing_n_raises(self):
        with pytest.raises(ValueError, match="n is required"):
            power_summary("one_sample_t", 0.5)
