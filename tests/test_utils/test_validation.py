"""
Tests for hypotestx.utils.validation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.utils.validation import (
    validate_alpha,
    validate_alternative,
    validate_categorical_column,
    validate_columns,
    validate_dataframe,
    validate_numeric_column,
    validate_probability,
    validate_sample_size,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GOOD_DICT = {
    "age": [25, 30, 35, 40, 45],
    "salary": [50.0, 60.0, 70.0, 80.0, 90.0],
    "gender": ["M", "F", "M", "F", "M"],
}


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------


class TestValidateDataframe:
    def test_valid_dict(self):
        validate_dataframe(GOOD_DICT)  # should not raise

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            validate_dataframe(None)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            validate_dataframe({})

    def test_zero_rows_raises(self):
        with pytest.raises(ValueError):
            validate_dataframe({"a": []})

    def test_inconsistent_lengths_raises(self):
        with pytest.raises(ValueError):
            validate_dataframe({"a": [1, 2], "b": [1, 2, 3]})

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            validate_dataframe([1, 2, 3])

    def test_unsupported_type_raises_string(self):
        with pytest.raises(TypeError):
            validate_dataframe("not a dataframe")


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------


class TestValidateColumns:
    def test_existing_columns_pass(self):
        validate_columns(GOOD_DICT, "age", "salary")  # should not raise

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            validate_columns(GOOD_DICT, "nonexistent")

    def test_partial_missing_raises(self):
        with pytest.raises(KeyError):
            validate_columns(GOOD_DICT, "age", "missing_col")

    def test_multiple_missing_raises(self):
        with pytest.raises(KeyError):
            validate_columns(GOOD_DICT, "x", "y")


# ---------------------------------------------------------------------------
# validate_numeric_column
# ---------------------------------------------------------------------------


class TestValidateNumericColumn:
    def test_valid_int_column(self):
        validate_numeric_column(GOOD_DICT, "age")  # should not raise

    def test_valid_float_column(self):
        validate_numeric_column(GOOD_DICT, "salary")  # should not raise

    def test_string_column_raises(self):
        with pytest.raises(ValueError):
            validate_numeric_column(GOOD_DICT, "gender")

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            validate_numeric_column(GOOD_DICT, "nope")

    def test_empty_column_raises(self):
        d = {"x": []}
        with pytest.raises((ValueError, KeyError)):
            # dict has a column but validate_dataframe already blocks empty,
            # here the column itself is empty after passing in directly
            validate_numeric_column({"x": []}, "x")


# ---------------------------------------------------------------------------
# validate_categorical_column
# ---------------------------------------------------------------------------


class TestValidateCategoricalColumn:
    def test_string_column_passes(self):
        validate_categorical_column(GOOD_DICT, "gender")  # should not raise

    def test_numeric_column_passes(self):
        # permissive — any column is acceptable
        validate_categorical_column(GOOD_DICT, "age")

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            validate_categorical_column(GOOD_DICT, "nope")


# ---------------------------------------------------------------------------
# validate_sample_size
# ---------------------------------------------------------------------------


class TestValidateSampleSize:
    def test_sufficient_size_passes(self):
        validate_sample_size([1, 2, 3], min_size=2)  # should not raise

    def test_exact_min_passes(self):
        validate_sample_size([1, 2], min_size=2)

    def test_insufficient_size_raises(self):
        with pytest.raises(ValueError):
            validate_sample_size([1], min_size=2)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_sample_size([], min_size=1)

    def test_custom_min_raises(self):
        with pytest.raises(ValueError):
            validate_sample_size([1, 2, 3], min_size=5)


# ---------------------------------------------------------------------------
# validate_alpha
# ---------------------------------------------------------------------------


class TestValidateAlpha:
    def test_typical_values_pass(self):
        for a in [0.01, 0.05, 0.1, 0.001]:
            validate_alpha(a)  # should not raise

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            validate_alpha(0.0)

    def test_one_raises(self):
        with pytest.raises(ValueError):
            validate_alpha(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_alpha(-0.05)

    def test_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            validate_alpha(1.5)


# ---------------------------------------------------------------------------
# validate_probability
# ---------------------------------------------------------------------------


class TestValidateProbability:
    def test_valid_values(self):
        for p in [0.0, 0.5, 1.0, 0.99]:
            validate_probability(p)  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_probability(-0.1)

    def test_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            validate_probability(1.1)


# ---------------------------------------------------------------------------
# validate_alternative
# ---------------------------------------------------------------------------


class TestValidateAlternative:
    def test_valid_two_sided(self):
        validate_alternative("two-sided")

    def test_valid_greater(self):
        validate_alternative("greater")

    def test_valid_less(self):
        validate_alternative("less")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            validate_alternative("bigger")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_alternative("")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            validate_alternative("Two-Sided")
