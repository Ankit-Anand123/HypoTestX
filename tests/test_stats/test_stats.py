"""
Tests for hypotestx.stats — bootstrap, descriptive, distributions, inference.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.math.statistics import mean as _mean
from hypotestx.stats.bootstrap import (
    bootstrap_ci,
    bootstrap_mean_ci,
    bootstrap_two_sample_ci,
)
from hypotestx.stats.descriptive import DescriptiveStats, describe, frequency_table


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


DATA10 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


class TestBootstrapMeanCI:
    def test_returns_two_floats(self):
        lo, hi = bootstrap_mean_ci(DATA10, n_resamples=200, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_ci_covers_true_mean(self):
        lo, hi = bootstrap_mean_ci(DATA10, n_resamples=2000, ci=0.95, seed=42)
        true_mean = sum(DATA10) / len(DATA10)
        assert lo <= true_mean <= hi


class TestBootstrapCI:
    def test_returns_three_values(self):
        lo, hi, boot = bootstrap_ci(DATA10, _mean, n_resamples=500, seed=42)
        assert lo < hi
        assert isinstance(boot, list)
        assert len(boot) == 500

    def test_ci_covers_true_mean(self):
        lo, hi, _ = bootstrap_ci(DATA10, _mean, n_resamples=2000, ci=0.95, seed=42)
        true_mean = sum(DATA10) / len(DATA10)
        assert lo <= true_mean <= hi

    def test_wider_at_higher_confidence(self):
        lo90, hi90, _ = bootstrap_ci(DATA10, _mean, n_resamples=500, ci=0.90, seed=42)
        lo99, hi99, _ = bootstrap_ci(DATA10, _mean, n_resamples=500, ci=0.99, seed=42)
        assert (hi99 - lo99) >= (hi90 - lo90)


class TestDescribe:
    def test_returns_descriptive_stats(self):
        result = describe(DATA10)
        assert isinstance(result, DescriptiveStats)

    def test_has_required_attributes(self):
        result = describe(DATA10)
        for attr in ("mean", "std", "min", "max", "n"):
            assert hasattr(result, attr)

    def test_correct_values(self):
        result = describe(DATA10)
        assert approx(result.mean, 5.5)
        assert result.min == 1.0
        assert result.max == 10.0
        assert result.n == 10


class TestFrequencyTable:
    def test_basic(self):
        data = ["A", "B", "A", "A", "B", "C"]
        ft = frequency_table(data)
        assert isinstance(ft, list)
        # Each element should be (value, count, pct)
        assert len(ft) == 3

    def test_sorted_by_count(self):
        data = ["B", "A", "A", "B", "C", "B"]
        ft = frequency_table(data)
        counts = [row[1] for row in ft]
        # First entry should be the most frequent
        assert counts[0] >= counts[-1]

    def test_percentages_sum_to_100(self):
        data = ["A", "B", "A", "C"]
        ft = frequency_table(data)
        total_pct = sum(row[2] for row in ft)
        assert approx(total_pct, 100.0, tol=0.01)
