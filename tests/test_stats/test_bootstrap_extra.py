"""
Additional tests for hypotestx.stats.bootstrap to improve coverage.
Tests bootstrap_two_sample_ci, bootstrap_test, permutation_test, bca method.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.math.statistics import mean as _mean
from hypotestx.math.statistics import median as _median
from hypotestx.stats.bootstrap import (
    bootstrap_ci,
    bootstrap_test,
    bootstrap_two_sample_ci,
    permutation_test,
)

DATA = [2.0, 3.5, 4.1, 2.8, 3.9, 4.5, 3.3, 2.6, 4.0, 3.7]
GROUP_HIGH = [8.0, 9.5, 8.5, 10.0, 9.0, 8.8, 9.3, 10.2, 8.7, 9.1]
GROUP_LOW = [3.0, 4.0, 3.5, 4.5, 3.0, 4.0, 3.5, 4.0, 3.8, 4.2]


# ---------------------------------------------------------------------------
# bootstrap_two_sample_ci
# ---------------------------------------------------------------------------


class TestBootstrapTwoSampleCI:
    def test_returns_three_values(self):
        lo, hi, boot = bootstrap_two_sample_ci(GROUP_HIGH, GROUP_LOW, n_resamples=500, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert isinstance(boot, list)

    def test_lower_lt_upper(self):
        lo, hi, _ = bootstrap_two_sample_ci(GROUP_HIGH, GROUP_LOW, n_resamples=500, seed=42)
        assert lo < hi

    def test_true_diff_inside_ci(self):
        lo, hi, _ = bootstrap_two_sample_ci(
            GROUP_HIGH, GROUP_LOW, ci=0.95, n_resamples=2000, seed=7
        )
        true_diff = _mean(GROUP_HIGH) - _mean(GROUP_LOW)
        assert lo <= true_diff <= hi

    def test_custom_diff_fn(self):
        def diff_median(a, b):
            return _median(a) - _median(b)

        lo, hi, boot = bootstrap_two_sample_ci(
            GROUP_HIGH, GROUP_LOW, diff_fn=diff_median, n_resamples=500, seed=1
        )
        assert lo < hi
        assert len(boot) == 500

    def test_boot_dist_length(self):
        _, _, boot = bootstrap_two_sample_ci(GROUP_HIGH, GROUP_LOW, n_resamples=300, seed=0)
        assert len(boot) == 300


# ---------------------------------------------------------------------------
# bootstrap_test (one-sample)
# ---------------------------------------------------------------------------


class TestBootstrapTest:
    def test_returns_three_values(self):
        p, obs, boot = bootstrap_test(DATA, _mean, null_value=3.5, n_resamples=500, seed=42)
        assert isinstance(p, float)
        assert isinstance(obs, float)
        assert isinstance(boot, list)

    def test_p_value_in_range(self):
        p, _, _ = bootstrap_test(DATA, _mean, null_value=3.5, n_resamples=1000, seed=42)
        assert 0.0 <= p <= 1.0

    def test_clearly_wrong_null_low_pvalue(self):
        big_data = [10.0] * 30
        p, _, _ = bootstrap_test(big_data, _mean, null_value=0.0, n_resamples=1000, seed=42)
        assert p < 0.05

    def test_true_null_high_pvalue(self):
        # null = observed mean -> should usually be non-significant
        obs_mean = _mean(DATA)
        p, _, _ = bootstrap_test(DATA, _mean, null_value=obs_mean, n_resamples=1000, seed=42)
        assert p > 0.05

    def test_alternative_greater(self):
        high_data = [10.0] * 20
        p, _, _ = bootstrap_test(
            high_data,
            _mean,
            null_value=5.0,
            alternative="greater",
            n_resamples=500,
            seed=0,
        )
        assert p < 0.05

    def test_alternative_less(self):
        low_data = [1.0] * 20
        p, _, _ = bootstrap_test(
            low_data, _mean, null_value=5.0, alternative="less", n_resamples=500, seed=0
        )
        assert p < 0.05

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError):
            bootstrap_test(DATA, _mean, null_value=3.0, alternative="invalid", n_resamples=100)

    def test_boot_dist_length(self):
        _, _, boot = bootstrap_test(DATA, _mean, n_resamples=400, seed=5)
        assert len(boot) == 400


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_returns_three_values(self):
        p, obs, perm = permutation_test(GROUP_HIGH, GROUP_LOW, n_resamples=500, seed=42)
        assert isinstance(p, float)
        assert isinstance(obs, float)
        assert isinstance(perm, list)

    def test_p_value_in_range(self):
        p, _, _ = permutation_test(GROUP_HIGH, GROUP_LOW, n_resamples=1000, seed=42)
        assert 0.0 <= p <= 1.0

    def test_identical_groups_high_pvalue(self):
        p, _, _ = permutation_test(DATA, DATA, n_resamples=1000, seed=42)
        assert p > 0.05

    def test_well_separated_groups_low_pvalue(self):
        p, _, _ = permutation_test(GROUP_HIGH, GROUP_LOW, n_resamples=1000, seed=42)
        assert p < 0.05

    def test_alternative_greater(self):
        p, _, _ = permutation_test(
            GROUP_HIGH, GROUP_LOW, alternative="greater", n_resamples=500, seed=0
        )
        assert p < 0.05

    def test_alternative_less(self):
        p, _, _ = permutation_test(
            GROUP_LOW, GROUP_HIGH, alternative="less", n_resamples=500, seed=0
        )
        assert p < 0.05

    def test_custom_statistic_fn(self):
        def diff_median(a, b):
            return _median(a) - _median(b)

        p, obs, perm = permutation_test(
            GROUP_HIGH, GROUP_LOW, statistic_fn=diff_median, n_resamples=500, seed=1
        )
        assert isinstance(p, float)

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError):
            permutation_test(DATA, DATA, alternative="bad", n_resamples=100)

    def test_perm_dist_length(self):
        _, _, perm = permutation_test(GROUP_HIGH, GROUP_LOW, n_resamples=250, seed=0)
        assert len(perm) == 250


# ---------------------------------------------------------------------------
# bootstrap_ci — bca method
# ---------------------------------------------------------------------------


class TestBootstrapCIBca:
    def test_bca_returns_valid_ci(self):
        lo, hi, boot = bootstrap_ci(DATA, _mean, n_resamples=500, method="bca", seed=42)
        assert lo < hi
        assert isinstance(boot, list)

    def test_bca_covers_mean(self):
        lo, hi, _ = bootstrap_ci(DATA, _mean, ci=0.95, n_resamples=2000, method="bca", seed=99)
        assert lo <= _mean(DATA) <= hi

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(DATA, _mean, method="jacknife", n_resamples=100)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], _mean, n_resamples=100)
