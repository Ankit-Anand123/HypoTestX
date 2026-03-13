"""
Additional tests for hypotestx.stats.descriptive to improve coverage.
Tests DescriptiveStats fully, five_number_summary, detect_outliers, compare_groups.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.stats.descriptive import (
    DescriptiveStats,
    compare_groups,
    describe,
    detect_outliers,
    five_number_summary,
    frequency_table,
)

DATA = [2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 5.0, 7.0, 9.0, 1.0]
SKEWED = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 10.0, 50.0, 100.0]
WITH_OUTLIERS = [5.0, 5.5, 6.0, 5.8, 6.2, 5.9, 6.1, 5.7, 100.0, -80.0]


# ---------------------------------------------------------------------------
# DescriptiveStats — attributes
# ---------------------------------------------------------------------------


class TestDescriptiveStatsAttributes:
    def setup_method(self):
        self.ds = DescriptiveStats(DATA)

    def test_n(self):
        assert self.ds.n == 10

    def test_mean(self):
        assert abs(self.ds.mean - 5.5) < 1e-10

    def test_min_max(self):
        assert self.ds.min == 1.0
        assert self.ds.max == 10.0

    def test_range(self):
        assert abs(self.ds.range - 9.0) < 1e-10

    def test_std_positive(self):
        assert self.ds.std > 0

    def test_variance_positive(self):
        assert self.ds.variance > 0

    def test_iqr_positive(self):
        assert self.ds.iqr >= 0

    def test_p05_lt_p95(self):
        assert self.ds.p05 < self.ds.p95

    def test_sem_positive(self):
        assert self.ds.sem > 0

    def test_cv_positive(self):
        assert self.ds.cv > 0

    def test_name_stored(self):
        ds = DescriptiveStats(DATA, name="scores")
        assert ds.name == "scores"


# ---------------------------------------------------------------------------
# DescriptiveStats — methods
# ---------------------------------------------------------------------------


class TestDescriptiveStatsMethods:
    def test_summary_returns_str(self):
        s = DescriptiveStats(DATA).summary()
        assert isinstance(s, str)

    def test_summary_verbose_returns_str(self):
        s = DescriptiveStats(DATA).summary(verbose=True)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_to_dict_keys(self):
        d = DescriptiveStats(DATA).to_dict()
        for key in ["n", "mean", "std", "median", "min", "max", "iqr"]:
            assert key in d

    def test_to_dict_values_match(self):
        ds = DescriptiveStats(DATA)
        d = ds.to_dict()
        assert d["n"] == ds.n
        assert abs(d["mean"] - ds.mean) < 1e-12

    def test_str_dunder(self):
        s = str(DescriptiveStats(DATA))
        assert isinstance(s, str)

    def test_repr_dunder(self):
        r = repr(DescriptiveStats(DATA))
        assert "DescriptiveStats" in r

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            DescriptiveStats([])


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_returns_descriptive_stats(self):
        assert isinstance(describe(DATA), DescriptiveStats)

    def test_verbose_prints_and_returns(self, capsys):
        ds = describe(DATA, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert isinstance(ds, DescriptiveStats)

    def test_custom_name(self):
        ds = describe(DATA, name="my_col")
        assert ds.name == "my_col"


# ---------------------------------------------------------------------------
# five_number_summary
# ---------------------------------------------------------------------------


class TestFiveNumberSummary:
    def test_returns_dict(self):
        assert isinstance(five_number_summary(DATA), dict)

    def test_keys(self):
        d = five_number_summary(DATA)
        assert set(d.keys()) == {"min", "q1", "median", "q3", "max"}

    def test_ordering(self):
        d = five_number_summary(DATA)
        assert d["min"] <= d["q1"] <= d["median"] <= d["q3"] <= d["max"]

    def test_min_max_correct(self):
        d = five_number_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert d["min"] == 1.0
        assert d["max"] == 5.0

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            five_number_summary([5.0])


# ---------------------------------------------------------------------------
# detect_outliers — iqr method
# ---------------------------------------------------------------------------


class TestDetectOutliersIQR:
    def test_returns_indices_values_meta(self):
        idx, vals, meta = detect_outliers(WITH_OUTLIERS, method="iqr")
        assert isinstance(idx, list)
        assert isinstance(vals, list)
        assert isinstance(meta, dict)

    def test_detects_outliers(self):
        idx, vals, _ = detect_outliers(WITH_OUTLIERS, method="iqr")
        assert len(idx) >= 1
        assert 100.0 in vals or -80.0 in vals

    def test_no_outliers_in_normal_data(self):
        normal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        idx, _, _ = detect_outliers(normal, method="iqr", threshold=3.0)
        assert len(idx) == 0

    def test_custom_threshold(self):
        idx_tight, _, _ = detect_outliers(WITH_OUTLIERS, method="iqr", threshold=1.0)
        idx_loose, _, _ = detect_outliers(WITH_OUTLIERS, method="iqr", threshold=3.0)
        assert len(idx_tight) >= len(idx_loose)


# ---------------------------------------------------------------------------
# detect_outliers — zscore method
# ---------------------------------------------------------------------------


class TestDetectOutliersZScore:
    def test_returns_indices_values_meta(self):
        idx, vals, meta = detect_outliers(SKEWED, method="zscore")
        assert isinstance(idx, list)
        assert isinstance(meta, dict)
        assert "mean" in meta

    def test_detects_extreme_values(self):
        idx, vals, _ = detect_outliers(SKEWED, method="zscore", threshold=2.0)
        # With threshold=2.0 the high values (50.0, 100.0) should be flagged
        assert len(idx) > 0

    def test_constant_data_no_outliers(self):
        idx, _, _ = detect_outliers([5.0] * 20, method="zscore")
        assert len(idx) == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            detect_outliers(DATA, method="mahalanobis")


# ---------------------------------------------------------------------------
# frequency_table
# ---------------------------------------------------------------------------


class TestFrequencyTable:
    def test_returns_list_of_tuples(self):
        t = frequency_table([1, 2, 2, 3, 3, 3])
        assert isinstance(t, list)
        assert all(len(row) == 3 for row in t)

    def test_counts_correct(self):
        t = frequency_table([1, 2, 2, 3, 3, 3])
        d = {v: c for v, c, _ in t}
        assert d[1] == 1
        assert d[2] == 2
        assert d[3] == 3

    def test_percentages_sum_to_100(self):
        t = frequency_table([1, 2, 2, 3])
        total_pct = sum(pct for _, _, pct in t)
        assert abs(total_pct - 100.0) < 1e-9

    def test_empty_returns_empty(self):
        assert frequency_table([]) == []


# ---------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------


class TestCompareGroups:
    def test_returns_string(self):
        result = compare_groups(DATA, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result, str)

    def test_three_groups(self):
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 5.0, 6.0]
        g3 = [7.0, 8.0, 9.0]
        result = compare_groups(g1, g2, g3)
        assert isinstance(result, str)

    def test_custom_names(self):
        result = compare_groups(DATA, [1.0, 2.0, 3.0, 4.0, 5.0], names=["Control", "Treatment"])
        assert "Control" in result
        assert "Treatment" in result

    def test_wrong_name_count_raises(self):
        with pytest.raises(ValueError):
            compare_groups(DATA, [1.0, 2.0], names=["only_one"])

    def test_contains_mean_row(self):
        result = compare_groups(DATA, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert "mean" in result.lower()
