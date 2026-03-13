"""
Tests for hypotestx.core.testsuite — TestSuite class.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.core.testsuite import TestSuite
from hypotestx.tests.correlation import pearson_correlation
from hypotestx.tests.parametric import one_sample_ttest, two_sample_ttest

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

DATA_A = [8.0, 9.0, 10.0, 11.0, 9.5, 10.5, 9.0, 10.0, 11.5, 8.5]
DATA_B = [6.0, 7.0, 6.5, 7.5, 6.0, 7.0, 6.5, 7.0, 7.5, 6.0]


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestTestSuiteInit:
    def test_default_name(self):
        ts = TestSuite()
        assert ts.name == "TestSuite"

    def test_custom_name(self):
        ts = TestSuite("My Analysis")
        assert ts.name == "My Analysis"

    def test_initially_empty(self):
        ts = TestSuite()
        assert ts._tests == []
        assert ts._results == []


# ---------------------------------------------------------------------------
# add() and fluent chaining
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_returns_self(self):
        ts = TestSuite()
        result = ts.add(lambda: one_sample_ttest(DATA_A, mu=8.0))
        assert result is ts  # fluent return

    def test_add_multiple(self):
        ts = TestSuite()
        ts.add(lambda: one_sample_ttest(DATA_A, mu=8.0))
        ts.add(lambda: one_sample_ttest(DATA_B, mu=5.0))
        assert len(ts._tests) == 2

    def test_chained_add(self):
        ts = (
            TestSuite()
            .add(lambda: one_sample_ttest(DATA_A, mu=8.0))
            .add(lambda: one_sample_ttest(DATA_B, mu=5.0))
        )
        assert len(ts._tests) == 2


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_list(self):
        ts = TestSuite()
        ts.add(lambda: one_sample_ttest(DATA_A, mu=8.0))
        results = ts.run()
        assert isinstance(results, list)
        assert len(results) == 1

    def test_run_each_is_hypo_result(self):
        from hypotestx.core.result import HypoResult

        ts = TestSuite()
        ts.add(lambda: one_sample_ttest(DATA_A, mu=0.0))
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        results = ts.run()
        for r in results:
            assert isinstance(r, HypoResult)

    def test_run_empty_suite(self):
        ts = TestSuite()
        results = ts.run()
        assert results == []

    def test_results_stored(self):
        ts = TestSuite()
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        ts.run()
        assert len(ts._results) == 1


# ---------------------------------------------------------------------------
# n_significant
# ---------------------------------------------------------------------------


class TestNSignificant:
    def test_counts_significant(self):
        ts = TestSuite()
        # Clear group difference -> significant
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        # Same group vs itself -> not significant
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_A))
        ts.run()
        assert ts.n_significant >= 1

    def test_zero_before_run(self):
        ts = TestSuite()
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        # Not run yet
        assert ts.n_significant == 0


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_returns_string(self):
        ts = TestSuite("Demo")
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        ts.run()
        s = ts.summary()
        assert isinstance(s, str)

    def test_contains_suite_name(self):
        ts = TestSuite("MySuite")
        ts.run()
        s = ts.summary()
        assert "MySuite" in s

    def test_contains_test_count(self):
        ts = TestSuite()
        ts.add(lambda: two_sample_ttest(DATA_A, DATA_B))
        ts.add(lambda: pearson_correlation(DATA_A, DATA_B))
        ts.run()
        s = ts.summary()
        assert "2" in s

    def test_empty_summary(self):
        ts = TestSuite("Empty")
        ts.run()
        s = ts.summary()
        assert "0" in s
