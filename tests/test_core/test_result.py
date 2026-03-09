"""
Unit tests for hypotestx.core.result.HypoResult
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hypotestx.core.result import HypoResult


def _make_result(**kwargs):
    defaults = dict(
        test_name="t-test",
        statistic=3.5,
        p_value=0.002,
        alpha=0.05,
        effect_size=0.8,
        effect_size_name="Cohen d",
        degrees_of_freedom=19,
        sample_sizes=20,
        confidence_interval=(1.2, 5.8),
        alternative="two-sided",
    )
    defaults.update(kwargs)
    return HypoResult(**defaults)


class TestHypoResultIsSignificant(unittest.TestCase):

    def test_significant_when_p_less_than_alpha(self):
        r = _make_result(p_value=0.001, alpha=0.05)
        self.assertTrue(r.is_significant)

    def test_not_significant_when_p_greater(self):
        r = _make_result(p_value=0.3, alpha=0.05)
        self.assertFalse(r.is_significant)

    def test_boundary_equal_alpha_not_significant(self):
        # p == alpha is conventionally not significant
        r = _make_result(p_value=0.05, alpha=0.05)
        self.assertFalse(r.is_significant)

    def test_custom_alpha(self):
        r = _make_result(p_value=0.04, alpha=0.01)
        self.assertFalse(r.is_significant)


class TestHypoResultEffectMagnitude(unittest.TestCase):

    def test_small_effect(self):
        # Cohen's d: 0.2 <= d < 0.5 is 'small'
        r = _make_result(effect_size=0.3, effect_size_name="Cohen d")
        self.assertEqual(r.effect_magnitude, "small")

    def test_medium_effect(self):
        r = _make_result(effect_size=0.5, effect_size_name="Cohen d")
        self.assertEqual(r.effect_magnitude, "medium")

    def test_large_effect(self):
        r = _make_result(effect_size=0.9, effect_size_name="Cohen d")
        self.assertEqual(r.effect_magnitude, "large")

    def test_none_effect_size(self):
        r = _make_result(effect_size=None)
        # When no effect size provided, magnitude description says 'not calculated'
        self.assertIn(r.effect_magnitude, (None, "not calculated", "N/A"))


class TestHypoResultSummary(unittest.TestCase):

    def test_summary_returns_string(self):
        r = _make_result()
        s = r.summary()
        self.assertIsInstance(s, str)

    def test_summary_contains_test_name(self):
        r = _make_result(test_name="Pearson r")
        s = r.summary()
        self.assertIn("Pearson r", s)

    def test_summary_contains_p_value(self):
        r = _make_result(p_value=0.002)
        s = r.summary()
        self.assertIn("0.002", s)

    def test_summary_significant_label(self):
        r = _make_result(p_value=0.001)
        s = r.summary()
        self.assertIn("significant", s.lower())


class TestHypoResultToDict(unittest.TestCase):

    def test_to_dict_returns_dict(self):
        r = _make_result()
        d = r.to_dict()
        self.assertIsInstance(d, dict)

    def test_to_dict_has_p_value(self):
        r = _make_result(p_value=0.042)
        d = r.to_dict()
        self.assertIn("p_value", d)
        self.assertAlmostEqual(d["p_value"], 0.042, places=6)

    def test_to_dict_has_is_significant(self):
        r = _make_result(p_value=0.001)
        d = r.to_dict()
        # is_significant may be stored as a key or computable from p_value
        if "is_significant" in d:
            self.assertTrue(d["is_significant"])
        else:
            self.assertLess(d["p_value"], d.get("alpha", 0.05))

    def test_to_dict_has_effect_size(self):
        r = _make_result(effect_size=0.5)
        d = r.to_dict()
        self.assertIn("effect_size", d)

    def test_to_dict_roundtrip_statistic(self):
        r = _make_result(statistic=2.718)
        d = r.to_dict()
        self.assertAlmostEqual(d["statistic"], 2.718, places=5)


class TestHypoResultRepr(unittest.TestCase):

    def test_repr_returns_string(self):
        r = _make_result()
        self.assertIsInstance(repr(r), str)

    def test_repr_contains_test_name(self):
        r = _make_result(test_name="Mann-Whitney U")
        self.assertIn("Mann-Whitney U", repr(r))


if __name__ == '__main__':
    unittest.main()
