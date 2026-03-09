"""
Tests for hypotestx.core.engine — analyze() dispatcher and helpers.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from hypotestx.core.engine import (
    analyze,
    _col_to_list,
    _column_names,
    _unique_values,
    _extract_groups,
    _build_contingency_table,
)
from hypotestx.core.result import HypoResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GENDER_SALARY = {
    "gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"] * 6,
    "salary": [70, 75, 80, 72, 78, 60, 65, 70, 63, 68] * 6,
    "age":    [30, 35, 40, 32, 37, 28, 32, 38, 29, 34] * 6,
    "dept":   ["Eng", "Sales", "Eng", "Sales", "Eng",
                "HR",  "HR",   "Sales", "HR",  "Eng"] * 6,
}

SIMPLE_NUMERIC = {
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "y": [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0],
}

# ---------------------------------------------------------------------------
# _col_to_list
# ---------------------------------------------------------------------------

class TestColToList:
    def test_dict(self):
        d = {"a": [1, 2, 3]}
        assert _col_to_list(d, "a") == [1, 2, 3]

    def test_missing_col_raises(self):
        d = {"a": [1]}
        with pytest.raises(KeyError):
            _col_to_list(d, "nope")

    def test_pandas(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({"x": [10, 20, 30]})
        assert _col_to_list(df, "x") == [10, 20, 30]


# ---------------------------------------------------------------------------
# _column_names
# ---------------------------------------------------------------------------

class TestColumnNames:
    def test_dict(self):
        d = {"a": [], "b": [], "c": []}
        assert set(_column_names(d)) == {"a", "b", "c"}

    def test_pandas(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({"p": [1], "q": [2]})
        assert set(_column_names(df)) == {"p", "q"}


# ---------------------------------------------------------------------------
# _unique_values
# ---------------------------------------------------------------------------

class TestUniqueValues:
    def test_basic(self):
        d = {"g": ["A", "B", "A", "C", "B"]}
        vals = _unique_values(d, "g")
        assert sorted(vals) == ["A", "B", "C"]

    def test_numeric(self):
        d = {"n": [3, 1, 2, 1, 3]}
        assert _unique_values(d, "n") == [1, 2, 3]


# ---------------------------------------------------------------------------
# _extract_groups
# ---------------------------------------------------------------------------

class TestExtractGroups:
    def test_two_groups(self):
        groups = _extract_groups(GENDER_SALARY, "gender", "salary")
        assert set(groups.keys()) == {"M", "F"}
        assert all(isinstance(v, list) for v in groups.values())
        assert all(isinstance(x, float) for v in groups.values() for x in v)

    def test_selected_groups(self):
        groups = _extract_groups(GENDER_SALARY, "gender", "salary",
                                 group_values=["M"])
        assert list(groups.keys()) == ["M"]


# ---------------------------------------------------------------------------
# _build_contingency_table
# ---------------------------------------------------------------------------

class TestBuildContingencyTable:
    def test_2x2(self):
        d = {
            "treatment": ["A", "A", "B", "B", "A", "B"],
            "outcome":   ["Y", "N", "Y", "Y", "Y", "N"],
        }
        table = _build_contingency_table(d, "treatment", "outcome")
        # rows = A, B; cols = N, Y
        total = sum(table[i][j] for i in range(len(table))
                    for j in range(len(table[0])))
        assert total == 6

    def test_symmetric_totals(self):
        g = GENDER_SALARY
        table = _build_contingency_table(g, "gender", "dept")
        row_total = sum(sum(r) for r in table)
        assert row_total == len(g["gender"])


# ---------------------------------------------------------------------------
# analyze() — routing and dispatch
# ---------------------------------------------------------------------------

class TestAnalyzeRouting:
    """Test that analyze() routes to the correct test and returns HypoResult."""

    def test_returns_hypo_result(self):
        result = analyze(GENDER_SALARY, "Do males earn more than females?")
        assert isinstance(result, HypoResult)

    def test_two_sample_ttest(self):
        result = analyze(GENDER_SALARY, "Do males earn more than females?")
        assert "t-test" in result.test_name.lower() or "welch" in result.test_name.lower()
        assert 0.0 <= result.p_value <= 1.0
        assert result.is_significant  # the data has a clear difference

    def test_pearson_correlation(self):
        result = analyze(SIMPLE_NUMERIC, "Is x correlated with y?")
        assert "pearson" in result.test_name.lower() or "correlation" in result.test_name.lower()
        assert 0.0 <= result.p_value <= 1.0

    def test_spearman_correlation(self):
        def mock_fn(msgs):
            return ('{"test": "spearman", "x_column": "x", "y_column": "y",'
                    ' "alternative": "two-sided", "confidence": 0.9}')
        result = analyze(SIMPLE_NUMERIC, "rank correlation", backend=mock_fn)
        assert "spearman" in result.test_name.lower()

    def test_chi_square(self):
        result = analyze(GENDER_SALARY,
                         "Is there an association between gender and dept?")
        assert "chi" in result.test_name.lower()
        assert 0.0 <= result.p_value <= 1.0

    def test_one_sample_ttest(self):
        def mock_fn(msgs):
            return ('{"test": "one_sample_ttest", "value_column": "salary",'
                    ' "mu": 70.0, "alternative": "two-sided", "confidence": 0.9}')
        result = analyze(GENDER_SALARY, "Is mean salary 70?", backend=mock_fn)
        assert "one" in result.test_name.lower() or "sample" in result.test_name.lower()

    def test_paired_ttest(self):
        before_after = {
            "before": [10.0, 12.0, 11.0, 9.0, 13.0, 14.0, 10.0, 11.0],
            "after":  [12.0, 15.0, 11.5, 12.0, 14.0, 16.5, 11.0, 12.0],
        }
        def mock_fn(msgs):
            return ('{"test": "paired_ttest", "x_column": "before",'
                    ' "y_column": "after", "alternative": "less", "confidence": 0.9}')
        result = analyze(before_after, "Did before improve to after?",
                         backend=mock_fn)
        assert "paired" in result.test_name.lower()

    def test_anova(self):
        three_groups = {
            "region": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "score":  [5.0, 6.0, 5.5, 6.5, 5.0] * 4 +
                      [7.0, 8.0, 7.5, 8.5, 7.0] * 4 +
                      [3.0, 4.0, 3.5, 4.5, 3.0] * 4,
        }
        def mock_fn(msgs):
            return ('{"test": "anova", "group_column": "region",'
                    ' "value_column": "score", "confidence": 0.85}')
        result = analyze(three_groups, "Compare regions", backend=mock_fn)
        assert "anova" in result.test_name.lower() or "analysis" in result.test_name.lower()

    def test_mann_whitney(self):
        def mock_fn(msgs):
            return ('{"test": "mann_whitney", "group_column": "gender",'
                    ' "value_column": "salary", "alternative": "two-sided",'
                    ' "confidence": 0.85}')
        result = analyze(GENDER_SALARY, "non-parametric comparison",
                         backend=mock_fn)
        assert "mann" in result.test_name.lower() or "whitney" in result.test_name.lower()

    def test_kruskal_wallis(self):
        three_groups = {
            "region": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "score":  [5.0, 6.0, 5.5, 6.5, 5.0] * 4 +
                      [7.0, 8.0, 7.5, 8.5, 7.0] * 4 +
                      [3.0, 4.0, 3.5, 4.5, 3.0] * 4,
        }
        def mock_fn(msgs):
            return ('{"test": "kruskal_wallis", "group_column": "region",'
                    ' "value_column": "score", "confidence": 0.85}')
        result = analyze(three_groups, "kruskal test", backend=mock_fn)
        assert "kruskal" in result.test_name.lower()

    def test_fisher_exact(self):
        d = {
            "treatment": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "outcome":   ["Y", "Y", "Y", "N", "Y", "N", "N", "N"],
        }
        def mock_fn(msgs):
            return ('{"test": "fisher", "x_column": "treatment",'
                    ' "y_column": "outcome", "alternative": "two-sided",'
                    ' "confidence": 0.85}')
        result = analyze(d, "fisher test", backend=mock_fn)
        assert "fisher" in result.test_name.lower()

    def test_point_biserial(self):
        def mock_fn(msgs):
            return ('{"test": "point_biserial", "x_column": "salary",'
                    ' "y_column": "gender", "confidence": 0.85}')
        result = analyze(GENDER_SALARY, "point biserial", backend=mock_fn)
        assert "biserial" in result.test_name.lower() or "correlation" in result.test_name.lower()


class TestAnalyzeBackends:
    """Test backend resolution in analyze()."""

    def test_none_backend_uses_fallback(self):
        result = analyze(GENDER_SALARY, "Do males earn more?", backend=None)
        assert isinstance(result, HypoResult)

    def test_fallback_string(self):
        result = analyze(GENDER_SALARY, "Do males earn more?", backend="fallback")
        assert isinstance(result, HypoResult)

    def test_callable_backend(self):
        def my_fn(msgs):
            return ('{"test": "two_sample_ttest", "group_column": "gender",'
                    ' "value_column": "salary", "alternative": "two-sided",'
                    ' "confidence": 0.9}')
        result = analyze(GENDER_SALARY, "test", backend=my_fn)
        assert isinstance(result, HypoResult)

    def test_backend_instance(self):
        from hypotestx.core.llm import FallbackBackend
        b = FallbackBackend()
        result = analyze(GENDER_SALARY, "Do males earn more?", backend=b)
        assert isinstance(result, HypoResult)

    def test_invalid_backend_raises(self):
        with pytest.raises((ValueError, TypeError)):
            analyze(GENDER_SALARY, "test", backend="nonexistent_xyz_abc")

    def test_alpha_forwarded(self):
        result = analyze(GENDER_SALARY, "Do males earn more?",
                         backend=None, alpha=0.01)
        assert result.alpha == 0.01

    def test_verbose_does_not_crash(self, capsys):
        analyze(GENDER_SALARY, "Do males earn more?", verbose=True)
        captured = capsys.readouterr()
        assert "HypoTestX" in captured.out

    def test_pandas_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame(GENDER_SALARY)
        result = analyze(df, "Do males earn more than females?")
        assert isinstance(result, HypoResult)


class TestAnalyzeEdgeCases:
    """Edge cases and error paths."""

    def test_missing_column_raises(self):
        def mock_fn(msgs):
            return ('{"test": "two_sample_ttest", "group_column": "no_such_col",'
                    ' "value_column": "salary", "confidence": 0.9}')
        with pytest.raises((KeyError, ValueError)):
            analyze(GENDER_SALARY, "test", backend=mock_fn)

    def test_result_has_required_attributes(self):
        result = analyze(GENDER_SALARY, "Do males earn more?")
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "effect_size")
        assert hasattr(result, "is_significant")
        assert hasattr(result, "alpha")
        assert callable(result.summary)
