"""
Tests for hypotestx.math.statistics — descriptive statistics functions.
"""

import math

from hypotestx.math.statistics import (
    correlation,
    covariance,
    iqr,
    kurtosis,
    mad,
    mean,
    median,
    mode,
    percentile,
    quartiles,
    range_stat,
    skewness,
    std,
    trimmed_mean,
    variance,
)


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


SYMMETRIC = [1.0, 2.0, 3.0, 4.0, 5.0]
UNIFORM = [2.0, 2.0, 2.0, 2.0, 2.0]
SKEWED = [1.0, 1.0, 1.0, 2.0, 10.0]


class TestMean:
    def test_basic(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_float(self):
        assert approx(mean([1.5, 2.5]), 2.0)

    def test_single(self):
        assert mean([7.0]) == 7.0

    def test_negative(self):
        assert mean([-2.0, 0.0, 2.0]) == 0.0


class TestMedian:
    def test_odd_length(self):
        assert median([3, 1, 2]) == 2.0

    def test_even_length(self):
        assert approx(median([1, 2, 3, 4]), 2.5)

    def test_single(self):
        assert median([5]) == 5.0


class TestMode:
    def test_single_mode(self):
        result = mode([1, 2, 2, 3])
        assert 2 in result or result == [2.0]

    def test_uniform_all_modes(self):
        result = mode([1, 2, 3])
        assert len(result) == 3


class TestVariance:
    def test_known(self):
        # var([2,4,4,4,5,5,7,9]) = 4.0 (sample)
        assert approx(variance([2, 4, 4, 4, 5, 5, 7, 9]), 4.571428, tol=1e-4)

    def test_constant_zero(self):
        assert variance(UNIFORM) == 0.0

    def test_symmetric(self):
        v = variance(SYMMETRIC)
        assert v > 0

    def test_population(self):
        v = variance([2, 4, 4, 4, 5, 5, 7, 9], ddof=0)
        assert approx(v, 4.0, tol=1e-6)


class TestStd:
    def test_equals_sqrt_variance(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert approx(std(data), math.sqrt(variance(data)))

    def test_constant_zero(self):
        assert std(UNIFORM) == 0.0


class TestCovariance:
    def test_positive_cov(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert covariance(x, y) > 0

    def test_negative_cov(self):
        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        assert covariance(x, y) < 0

    def test_zero_cov(self):
        x = [1.0, 2.0, 3.0]
        y = [2.0, 2.0, 2.0]
        assert covariance(x, y) == 0.0


class TestCorrelation:
    def test_perfect_positive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert approx(correlation(x, y), 1.0)

    def test_perfect_negative(self):
        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        assert approx(correlation(x, y), -1.0)

    def test_range(self):
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        y = [2.0, 4.0, 3.0, 1.0, 5.0]
        r = correlation(x, y)
        assert -1.0 <= r <= 1.0


class TestSkewnessKurtosis:
    def test_symmetric_skewness_near_zero(self):
        data = list(range(1, 101))
        assert abs(skewness(data)) < 0.1

    def test_right_skewed(self):
        data = [1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 30.0]
        assert skewness(data) > 0

    def test_kurtosis_type(self):
        assert isinstance(kurtosis(SYMMETRIC), float)


class TestPercentile:
    def test_0th(self):
        assert percentile(SYMMETRIC, 0) == 1.0

    def test_100th(self):
        assert percentile(SYMMETRIC, 100) == 5.0

    def test_50th(self):
        assert approx(percentile(SYMMETRIC, 50), 3.0, tol=0.5)


class TestQuartiles:
    def test_returns_three(self):
        q = quartiles([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(q) == 3

    def test_ordering(self):
        q1, q2, q3 = quartiles([1, 2, 3, 4, 5, 6, 7, 8])
        assert q1 <= q2 <= q3


class TestIQR:
    def test_symmetric(self):
        result = iqr([1, 2, 3, 4, 5, 6, 7, 8])
        assert result > 0


class TestRangeStat:
    def test_basic(self):
        assert range_stat([1, 2, 3, 4, 5]) == 4.0

    def test_negative(self):
        assert range_stat([-5, -3, -1, 0, 2]) == 7.0


class TestMAD:
    def test_constant_zero(self):
        assert mad([3.0, 3.0, 3.0]) == 0.0

    def test_positive(self):
        assert mad([1.0, 2.0, 3.0, 4.0, 5.0]) > 0


class TestTrimmedMean:
    def test_same_as_mean_for_zero_trim(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert approx(trimmed_mean(data, 0.0), mean(data))

    def test_reduces_outlier_effect(self):
        with_outlier = [1.0, 2.0, 3.0, 4.0, 100.0]
        without_trimmed = trimmed_mean(with_outlier, 0.2)
        assert without_trimmed < mean(with_outlier)
