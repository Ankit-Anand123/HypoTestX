"""
Tests for hypotestx.reporting.formatters.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.reporting.formatters import (
    apa_stat,
    effect_interpretation_table,
    format_ci,
    format_effect,
    format_p,
)


class TestFormatP:
    def test_below_threshold(self):
        assert format_p(0.0001) == "< 0.001"

    def test_above_threshold(self):
        result = format_p(0.05)
        assert "0.050" in result

    def test_custom_threshold(self):
        result = format_p(0.005, threshold=0.01)
        assert "< 0.01" in result

    def test_at_threshold(self):
        # p == threshold: NOT below, so should format normally
        result = format_p(0.001, threshold=0.001)
        assert "0.001" in result


class TestFormatCI:
    def test_default_95_pct(self):
        result = format_ci(1.234, 5.678)
        assert "95% CI" in result

    def test_values_present(self):
        result = format_ci(1.234, 5.678)
        assert "1.234" in result
        assert "5.678" in result

    def test_custom_level(self):
        result = format_ci(0.1, 0.9, level=0.99)
        assert "99%" in result

    def test_custom_decimal_places(self):
        result = format_ci(1.0, 2.0, decimal_places=1)
        assert "1.0" in result


class TestFormatEffect:
    def test_without_magnitude(self):
        result = format_effect("d", 0.5)
        assert "d = 0.500" in result

    def test_with_magnitude(self):
        result = format_effect("Cohen d", 0.5, magnitude="medium")
        assert "medium" in result
        assert "0.500" in result


class TestApaStat:
    def test_t_with_df_and_p(self):
        result = apa_stat("t", 2.34, df=29, p=0.026)
        assert "t(29)" in result
        assert "2.34" in result

    def test_f_with_tuple_df(self):
        result = apa_stat("F", 5.12, df=(2, 45), p=0.008)
        assert "F(2, 45)" in result

    def test_with_effect_name_and_value(self):
        result = apa_stat("t", 2.34, df=29, p=0.026, effect_name="d", effect_value=0.43)
        assert "d = 0.430" in result

    def test_with_n_as_int(self):
        result = apa_stat("t", 2.0, n=40)
        assert "n = 40" in result

    def test_with_n_as_tuple(self):
        result = apa_stat("t", 2.0, n=(20, 20))
        assert "N = 40" in result

    def test_p_very_small_apa_format(self):
        result = apa_stat("t", 5.0, p=0.0001)
        assert "< .001" in result

    def test_chi_square_symbol(self):
        result = apa_stat("chi2", 7.5, df=2)
        assert "chi2(2)" in result

    def test_unknown_test_name_passthrough(self):
        result = apa_stat("custom_stat", 3.14)
        assert "custom_stat" in result

    def test_no_df(self):
        result = apa_stat("z", 1.96)
        assert "z = 1.96" in result


class TestEffectInterpretationTable:
    def test_returns_string(self):
        result = effect_interpretation_table()
        assert isinstance(result, str)

    def test_contains_cohen_d(self):
        result = effect_interpretation_table()
        assert "Cohen" in result

    def test_has_multiple_rows(self):
        result = effect_interpretation_table()
        lines = result.split("\n")
        assert len(lines) > 5
