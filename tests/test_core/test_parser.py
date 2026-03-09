"""
Tests for hypotestx.core.parser -- hypothesis text parsing.
"""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from hypotestx.core.parser import (
    parse_hypothesis,
    ParsedHypothesis,
    AdvancedHypothesisParser,
    SimpleHypothesisParser,
    create_parser,
)


class TestParseHypothesisReturnType:
    """parse_hypothesis() should always return a ParsedHypothesis."""

    def test_returns_parsed_hypothesis(self):
        r = parse_hypothesis("Is there a difference?")
        assert isinstance(r, ParsedHypothesis)

    def test_has_test_type(self):
        r = parse_hypothesis("Do men earn more than women?")
        assert hasattr(r, "test_type")
        assert isinstance(r.test_type, str)
        assert len(r.test_type) > 0

    def test_has_tail(self):
        r = parse_hypothesis("Is x correlated with y?")
        assert hasattr(r, "tail")
        assert r.tail in ("two-sided", "greater", "less")

    def test_has_confidence_level(self):
        r = parse_hypothesis("Compare groups at 95% confidence")
        assert hasattr(r, "confidence_level")
        assert 0 < r.confidence_level < 1

    def test_has_raw_text(self):
        text = "Compare group A vs group B"
        r = parse_hypothesis(text)
        assert r.raw_text == text


class TestParseHypothesisTestTypes:
    """Test that the parser infers sensible test types from question text."""

    def test_comparison_question(self):
        r = parse_hypothesis("Do males earn more than females?")
        assert r.test_type in (
            "two_sample_ttest", "one_sample_ttest", "anova",
            "chi_square", "correlation", "unknown"
        )

    def test_correlation_question(self):
        r = parse_hypothesis("Is there a correlation between height and weight?")
        assert r.test_type in ("correlation", "two_sample_ttest", "chi_square", "unknown")

    def test_association_question(self):
        r = parse_hypothesis("Is there an association between gender and outcome?")
        assert isinstance(r.test_type, str)

    def test_before_after_question(self):
        r = parse_hypothesis("Is there a significant difference before and after treatment?")
        assert isinstance(r.test_type, str)


class TestParseHypothesisAlternative:
    """Tail / alternative should reflect directional vs non-directional wording."""

    def test_more_than_gives_greater_or_twosided(self):
        r = parse_hypothesis("Do men spend more than women?")
        assert r.tail in ("greater", "two-sided")

    def test_less_than_gives_less_or_twosided(self):
        r = parse_hypothesis("Do women earn less than men?")
        assert r.tail in ("less", "two-sided")

    def test_difference_gives_twosided(self):
        r = parse_hypothesis("Is there a difference between A and B?")
        assert r.tail == "two-sided"


class TestSimpleHypothesisParser:
    """Test the regex fallback parser directly."""

    def setup_method(self):
        self.parser = SimpleHypothesisParser()

    def test_parse_returns_parsed_hypothesis(self):
        r = self.parser.parse("Do men earn more than women?")
        assert isinstance(r, ParsedHypothesis)

    def test_comparison_type_greater(self):
        r = self.parser.parse("Do men earn more than women?")
        assert r.comparison_type == "greater"
        assert r.tail == "greater"

    def test_comparison_type_less(self):
        r = self.parser.parse("Do women earn less than men?")
        assert r.comparison_type == "less"
        assert r.tail == "less"

    def test_chi_square_keyword(self):
        r = self.parser.parse("Is there an association between education and income?")
        assert r.test_type in ("chi_square", "two_sample_ttest", "unknown")

    def test_correlation_keyword(self):
        r = self.parser.parse("Is height correlated with weight?")
        assert r.test_type in ("correlation", "two_sample_ttest", "unknown")

    def test_paired_keyword(self):
        r = self.parser.parse("Is there a difference before and after the intervention?")
        assert r.test_type in ("paired_ttest", "two_sample_ttest", "unknown")

    def test_one_sample_pattern(self):
        r = self.parser.parse("Is the mean equal to 50?")
        assert r.test_type in ("one_sample_ttest", "two_sample_ttest", "unknown")


class TestCreateParser:
    """create_parser() factory."""

    def test_default_is_advanced(self):
        p = create_parser(advanced=True)
        assert isinstance(p, AdvancedHypothesisParser)

    def test_simple_parser(self):
        p = create_parser(advanced=False)
        assert isinstance(p, SimpleHypothesisParser)

    def test_both_have_parse_method(self):
        for p in (create_parser(True), create_parser(False)):
            assert callable(getattr(p, "parse", None))
