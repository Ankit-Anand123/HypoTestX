"""
Tests for custom exception classes in hypotestx.core.exceptions.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from hypotestx.core.exceptions import (
    AssumptionViolationError,
    DataFormatError,
    HypoTestXError,
    InsufficientDataError,
    InvalidAlternativeError,
    ParseError,
    UnsupportedTestError,
)


class TestHypoTestXError:
    def test_is_exception(self):
        err = HypoTestXError("base error")
        assert isinstance(err, Exception)

    def test_message_accessible(self):
        err = HypoTestXError("something went wrong")
        assert "something went wrong" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(HypoTestXError):
            raise HypoTestXError("test")


class TestInsufficientDataError:
    def test_default_message(self):
        err = InsufficientDataError()
        assert "Not enough" in str(err)
        assert err.message

    def test_custom_message(self):
        err = InsufficientDataError("Only 1 observation")
        assert "Only 1 observation" in str(err)
        assert err.message == "Only 1 observation"

    def test_is_hypotestx_error(self):
        assert isinstance(InsufficientDataError(), HypoTestXError)


class TestAssumptionViolationError:
    def test_assumption_in_message(self):
        err = AssumptionViolationError("normality")
        assert "normality" in str(err)
        assert err.assumption == "normality"

    def test_with_extra_message(self):
        err = AssumptionViolationError("normality", message="Shapiro p = 0.01")
        assert "Shapiro" in str(err)

    def test_without_extra_message(self):
        err = AssumptionViolationError("homoscedasticity")
        assert "homoscedasticity" in str(err)
        # no extra message should not raise
        assert err.message


class TestInvalidAlternativeError:
    def test_alternative_in_message(self):
        err = InvalidAlternativeError("bad-alt")
        assert "bad-alt" in str(err)
        assert err.alternative == "bad-alt"

    def test_valid_list_in_message(self):
        err = InvalidAlternativeError("x")
        assert "two-sided" in str(err)

    def test_custom_valid_list(self):
        err = InvalidAlternativeError("bad", valid=["greater", "less"])
        assert "greater" in str(err)


class TestParseError:
    def test_hypothesis_in_message(self):
        err = ParseError("Is A > B?")
        assert "Is A > B?" in str(err)
        assert err.hypothesis == "Is A > B?"

    def test_with_reason(self):
        err = ParseError("??", reason="no keywords found")
        assert "no keywords" in str(err)

    def test_without_reason(self):
        err = ParseError("test")
        # should not raise
        assert "test" in str(err)


class TestUnsupportedTestError:
    def test_test_type_in_message(self):
        err = UnsupportedTestError("bayesian_ttest")
        assert "bayesian_ttest" in str(err)
        assert err.test_type == "bayesian_ttest"

    def test_is_hypotestx_error(self):
        assert isinstance(UnsupportedTestError("x"), HypoTestXError)


class TestDataFormatError:
    def test_basic(self):
        err = DataFormatError("unexpected format")
        assert isinstance(err, HypoTestXError)
        assert "unexpected format" in str(err)
