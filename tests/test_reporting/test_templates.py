"""
Tests for hypotestx.reporting.templates — render_apa, render_plain, render_one_line.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.reporting.templates import (
    APA_TEMPLATES,
    PLAIN_TEMPLATE,
    ONE_LINE_TEMPLATE,
    render_apa,
    render_plain,
    render_one_line,
    _significance_word,
    _effect_size_str,
)


# Minimal context shared across tests
BASIC_CTX = {
    "test_name": "Two-sample t-test",
    "statistic": 3.45,
    "p_value": 0.003,
    "alpha": 0.05,
    "is_significant": True,
    "effect_size": 0.72,
}


# ---------------------------------------------------------------------------
# _significance_word
# ---------------------------------------------------------------------------

class TestSignificanceWord:
    def test_significant_true(self):
        assert "significant" in _significance_word(True)

    def test_significant_false(self):
        assert "not" in _significance_word(False)


# ---------------------------------------------------------------------------
# _effect_size_str
# ---------------------------------------------------------------------------

class TestEffectSizeStr:
    def test_none_returns_na(self):
        assert _effect_size_str(None) == "N/A"

    def test_float_returns_formatted(self):
        s = _effect_size_str(0.5)
        assert "0.5" in s  # rounds to 4 decimal places

    def test_nan_returns_na(self):
        assert _effect_size_str(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# APA_TEMPLATES existence
# ---------------------------------------------------------------------------

class TestTemplateRegistry:
    def test_all_expected_keys_present(self):
        expected = {
            "two_sample_ttest", "one_sample_ttest", "paired_ttest",
            "anova", "chi_square", "pearson", "spearman",
            "mann_whitney", "kruskal_wallis", "fisher", "generic",
        }
        for key in expected:
            assert key in APA_TEMPLATES, f"Missing key: {key}"

    def test_each_template_is_str(self):
        for k, v in APA_TEMPLATES.items():
            assert isinstance(v, str), f"Template {k!r} is not a string"


# ---------------------------------------------------------------------------
# render_apa
# ---------------------------------------------------------------------------

class TestRenderApa:
    def test_generic_fallback(self):
        ctx = {
            "test_name": "Some Test",
            "statistic": 1.23,
            "p_value": 0.045,
            "is_significant": True,
        }
        result = render_apa("unknown_test_xyz", ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pearson_key(self):
        ctx = {
            "x_column": "height", "y_column": "weight",
            "df": 98, "statistic": 0.65, "p_value": 0.001,
            "is_significant": True,
        }
        result = render_apa("pearson", ctx)
        assert "height" in result
        assert "weight" in result

    def test_two_sample_ttest_key(self):
        ctx = {
            "value_column": "salary",
            "group1": "Male", "group2": "Female",
            "mean1": 70.0, "sd1": 5.0, "mean2": 65.0, "sd2": 4.5,
            "df": 98, "statistic": 2.5, "p_value": 0.014,
            "is_significant": True,
        }
        result = render_apa("two_sample_ttest", ctx)
        assert "Male" in result or "salary" in result

    def test_significance_word_injected(self):
        ctx = {
            "test_name": "T", "statistic": 1.0, "p_value": 0.5,
            "is_significant": False,
        }
        result = render_apa("generic", ctx)
        assert "not" in result.lower()

    def test_effect_size_sentence_defaults_empty_when_none(self):
        ctx = {
            "test_name": "T", "statistic": 1.0, "p_value": 0.5,
            "is_significant": True, "effect_size": None,
        }
        result = render_apa("generic", ctx)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# render_plain
# ---------------------------------------------------------------------------

class TestRenderPlain:
    def test_returns_str(self):
        result = render_plain(BASIC_CTX)
        assert isinstance(result, str)

    def test_contains_test_name(self):
        result = render_plain(BASIC_CTX)
        assert BASIC_CTX["test_name"] in result

    def test_contains_p_value(self):
        result = render_plain(BASIC_CTX)
        assert "0.0030" in result

    def test_conclusion_reject(self):
        result = render_plain(BASIC_CTX)
        assert "Reject" in result

    def test_conclusion_fail_to_reject(self):
        ctx = dict(BASIC_CTX)
        ctx["is_significant"] = False
        result = render_plain(ctx)
        assert "Fail" in result

    def test_no_effect_size(self):
        ctx = dict(BASIC_CTX)
        ctx["effect_size"] = None
        result = render_plain(ctx)
        assert "N/A" in result


# ---------------------------------------------------------------------------
# render_one_line
# ---------------------------------------------------------------------------

class TestRenderOneLine:
    def test_returns_str(self):
        result = render_one_line(BASIC_CTX)
        assert isinstance(result, str)

    def test_single_line(self):
        result = render_one_line(BASIC_CTX)
        assert "\n" not in result

    def test_contains_statistic(self):
        result = render_one_line(BASIC_CTX)
        assert "3.45" in result

    def test_contains_p_value(self):
        result = render_one_line(BASIC_CTX)
        assert "0.0030" in result

    def test_conclusion_in_result(self):
        result = render_one_line(BASIC_CTX)
        assert "Reject" in result
