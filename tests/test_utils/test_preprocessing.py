"""
Tests for hypotestx.utils.preprocessing — all transformation functions.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hypotestx.utils.preprocessing import (
    apply,
    center,
    log_transform,
    normalize,
    rank_transform,
    robust_scale,
    standardize,
    winsorize,
)


class TestStandardize:
    def test_mean_zero(self):
        result = standardize([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(result) / len(result)
        assert abs(mean) < 1e-10

    def test_unit_variance(self):
        result = standardize([10.0, 20.0, 30.0])
        mean = sum(result) / len(result)
        var = sum((x - mean) ** 2 for x in result) / (len(result) - 1)
        assert abs(var - 1.0) < 1e-9

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            standardize([5.0])

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            standardize([3.0, 3.0, 3.0])

    def test_ddof_zero(self):
        result = standardize([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        assert len(result) == 5


class TestNormalize:
    def test_min_becomes_zero(self):
        result = normalize([0.0, 5.0, 10.0])
        assert abs(result[0] - 0.0) < 1e-10

    def test_max_becomes_one(self):
        result = normalize([0.0, 5.0, 10.0])
        assert abs(result[-1] - 1.0) < 1e-10

    def test_custom_range(self):
        result = normalize([0.0, 10.0], low=2.0, high=4.0)
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[-1] - 4.0) < 1e-10

    def test_constant_data_midpoint(self):
        result = normalize([5.0, 5.0, 5.0])
        assert all(abs(x - 0.5) < 1e-10 for x in result)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize([])


class TestWinsorize:
    def test_clips_extremes(self):
        data = list(range(1, 21))
        result = winsorize(data, limits=(0.1, 0.1))
        assert len(result) == 20
        assert min(result) >= min(data)
        assert max(result) <= max(data)

    def test_no_clipping(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = winsorize(data, limits=(0.0, 0.0))
        assert result == data

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            winsorize([1.0])

    def test_bad_lower_limit_raises(self):
        with pytest.raises(ValueError):
            winsorize([1.0, 2.0, 3.0], limits=(0.6, 0.1))


class TestLogTransform:
    def test_natural_ln1(self):
        result = log_transform([1.0, math.e])
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9

    def test_base_10(self):
        result = log_transform([1.0, 10.0, 100.0], base="10")
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9
        assert abs(result[2] - 2.0) < 1e-9

    def test_base_2(self):
        result = log_transform([1.0, 2.0, 4.0], base="2")
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9
        assert abs(result[2] - 2.0) < 1e-9

    def test_shift(self):
        result = log_transform([0.0, 1.0], shift=1.0)
        assert abs(result[0] - 0.0) < 1e-9

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match="base"):
            log_transform([1.0, 2.0], base="e")

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            log_transform([-1.0, 1.0])


class TestRankTransform:
    def test_basic_order(self):
        result = rank_transform([30.0, 10.0, 20.0])
        assert result[1] == 1.0
        assert result[2] == 2.0
        assert result[0] == 3.0

    def test_ties_average(self):
        result = rank_transform([1.0, 1.0, 3.0], method="average")
        assert result[0] == result[1] == 1.5
        assert result[2] == 3.0

    def test_ties_min(self):
        result = rank_transform([1.0, 1.0, 3.0], method="min")
        assert result[0] == 1.0
        assert result[1] == 1.0

    def test_ties_max(self):
        result = rank_transform([1.0, 1.0, 3.0], method="max")
        assert result[0] == 2.0
        assert result[1] == 2.0

    def test_ordinal(self):
        result = rank_transform([1.0, 1.0, 3.0], method="ordinal")
        # tied values get consecutive ranks in order of appearance
        assert result[2] == 3.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            rank_transform([1.0, 2.0, 3.0], method="invalid")


class TestCenter:
    def test_mean_is_zero(self):
        result = center([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(result) / len(result)
        assert abs(mean) < 1e-10

    def test_preserves_length(self):
        result = center([10.0, 20.0, 30.0])
        assert len(result) == 3


class TestRobustScale:
    def test_basic(self):
        result = robust_scale([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5

    def test_median_is_zero(self):
        result = robust_scale([1.0, 2.0, 3.0, 4.0, 5.0])
        # median of [1,2,3,4,5] is 3, so result[2] should be 0
        assert abs(result[2] - 0.0) < 1e-9

    def test_zero_iqr_raises(self):
        with pytest.raises(ValueError, match="IQR"):
            robust_scale([5.0, 5.0, 5.0, 5.0, 5.0])


class TestApply:
    def test_sqrt(self):
        result = apply([1.0, 4.0, 9.0], math.sqrt)
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 2.0) < 1e-9
        assert abs(result[2] - 3.0) < 1e-9

    def test_square(self):
        result = apply([2.0, 3.0], lambda x: x * x)
        assert abs(result[0] - 4.0) < 1e-9
        assert abs(result[1] - 9.0) < 1e-9
