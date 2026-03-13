"""
Tests for hypotestx.utils.data_utils.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hypotestx.utils.data_utils import (
    are_paired,
    coerce_numeric,
    detect_missing,
    drop_missing,
    group_by,
    split_groups,
    summary_table,
    validate_sample_data,
)


class TestCoerceNumeric:
    def test_basic_types(self):
        result = coerce_numeric([1, "2.5", 3.0])
        assert result == [1.0, 2.5, 3.0]

    def test_drop_none(self):
        result = coerce_numeric([1.0, None, 3.0], drop_invalid=True)
        assert result == [1.0, 3.0]

    def test_drop_nan(self):
        result = coerce_numeric([1.0, float("nan"), 3.0], drop_invalid=True)
        assert result == [1.0, 3.0]

    def test_error_on_bad_value(self):
        with pytest.raises(ValueError):
            coerce_numeric([1, "abc", 3])


class TestDetectMissing:
    def test_none_missing(self):
        n, idx = detect_missing([1.0, 2.0, 3.0])
        assert n == 0
        assert idx == []

    def test_none_value(self):
        n, idx = detect_missing([1.0, None, 3.0])
        assert n == 1
        assert 1 in idx

    def test_nan_value(self):
        n, idx = detect_missing([1.0, float("nan"), 3.0])
        assert n == 1
        assert 1 in idx

    def test_multiple_missing(self):
        n, idx = detect_missing([None, 2.0, float("nan")])
        assert n == 2


class TestDropMissing:
    def test_basic(self):
        (a, b), dropped = drop_missing([1.0, None, 3.0], [4.0, 5.0, 6.0])
        assert dropped == 1
        assert a == [1.0, 3.0]
        assert b == [4.0, 6.0]

    def test_no_missing(self):
        (a,), dropped = drop_missing([1.0, 2.0, 3.0])
        assert dropped == 0
        assert a == [1.0, 2.0, 3.0]

    def test_no_columns(self):
        result, n = drop_missing()
        assert result == []
        assert n == 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            drop_missing([1.0, 2.0], [3.0])


class TestGroupBy:
    def test_basic(self):
        result = group_by([1.0, 2.0, 3.0, 4.0], ["a", "b", "a", "b"])
        assert sorted(result["a"]) == [1.0, 3.0]
        assert sorted(result["b"]) == [2.0, 4.0]

    def test_three_groups(self):
        result = group_by([1.0, 2.0, 3.0], ["x", "y", "x"])
        assert result["x"] == [1.0, 3.0]
        assert result["y"] == [2.0]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            group_by([1.0, 2.0], ["a"])


class TestSplitGroups:
    def test_multiple_list_args(self):
        result = split_groups([1.0, 2.0], [3.0, 4.0])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_single_list_of_lists(self):
        result = split_groups([[1.0, 2.0], [3.0, 4.0]])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_type_coercion(self):
        result = split_groups([1, 2, 3])
        assert result == [[1.0, 2.0, 3.0]]


class TestValidateSampleData:
    def test_basic(self):
        result = validate_sample_data([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            validate_sample_data([1.0], min_size=2)

    def test_allow_missing_drops_none(self):
        result = validate_sample_data([1.0, None, 3.0], allow_missing=True)
        assert result == [1.0, 3.0]

    def test_custom_min_size(self):
        with pytest.raises(ValueError):
            validate_sample_data([1.0, 2.0], min_size=5)


class TestSummaryTable:
    def test_two_groups(self):
        result = summary_table([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert isinstance(result, str)
        assert "Group 1" in result
        assert "Group 2" in result

    def test_custom_names(self):
        result = summary_table([1.0, 2.0, 3.0], names=["Control"])
        assert "Control" in result

    def test_single_element_group(self):
        # std of single element should not crash
        result = summary_table([5.0])
        assert isinstance(result, str)


class TestArePaired:
    def test_same_length_true(self):
        assert are_paired([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) is True

    def test_different_length_false(self):
        assert are_paired([1.0, 2.0], [3.0]) is False

    def test_too_short_false(self):
        assert are_paired([1.0], [2.0]) is False
