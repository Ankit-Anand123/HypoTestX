"""
Tests for hypotestx.utils.formatting.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.utils.formatting import (
    format_confidence_interval,
    format_effect_size,
    format_p_value,
)


class TestFormatPValue:
    def test_very_small_below_threshold(self):
        assert "< 0.001" in format_p_value(0.0001)

    def test_between_001_and_01(self):
        result = format_p_value(0.005)
        assert "0.0050" in result

    def test_between_001_and_005(self):
        result = format_p_value(0.03)
        assert "0.030" in result

    def test_large_p(self):
        result = format_p_value(0.2)
        assert "0.200" in result


class TestFormatEffectSize:
    def test_negligible(self):
        result = format_effect_size(0.1)
        assert "negligible" in result

    def test_small(self):
        result = format_effect_size(0.3)
        assert "small" in result

    def test_medium(self):
        result = format_effect_size(0.6)
        assert "medium" in result

    def test_large(self):
        result = format_effect_size(1.0)
        assert "large" in result

    def test_negative_classified_by_abs(self):
        result = format_effect_size(-1.0)
        assert "large" in result

    def test_other_effect_type(self):
        result = format_effect_size(0.3, effect_type="eta-squared")
        assert "0.300" in result


class TestFormatConfidenceInterval:
    def test_default_95pct(self):
        result = format_confidence_interval((0.1, 0.9))
        assert "95%" in result

    def test_values_in_output(self):
        result = format_confidence_interval((0.1234, 0.8765))
        assert "0.1234" in result
        assert "0.8765" in result

    def test_custom_confidence_level(self):
        result = format_confidence_interval((1.0, 2.0), confidence_level=0.99)
        assert "99%" in result
