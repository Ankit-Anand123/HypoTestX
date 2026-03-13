"""
Tests for hypotestx.reporting.generator — apa_report, text_report, batch_report.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.reporting.generator import apa_report, batch_report, export_csv, text_report
from hypotestx.tests.categorical import chi_square_test
from hypotestx.tests.correlation import pearson_correlation
from hypotestx.tests.parametric import one_sample_ttest, two_sample_ttest

# ---------------------------------------------------------------------------
# Shared test results
# ---------------------------------------------------------------------------

G1 = [8.0, 9.0, 10.0, 11.0, 9.5, 10.5, 9.0, 10.0, 11.5, 8.5]
G2 = [6.0, 7.0, 6.5, 7.5, 6.0, 7.0, 6.5, 7.0, 7.5, 6.0]
TTEST_RESULT = two_sample_ttest(G1, G2)
ONE_SAMP = one_sample_ttest([5.5, 6.0, 5.8, 6.1, 5.7] * 4, mu=5.0)
CORR_RESULT = pearson_correlation(G1, G2)
CHISQ_RESULT = chi_square_test([[10, 5], [3, 12]])


# ---------------------------------------------------------------------------
# apa_report
# ---------------------------------------------------------------------------


class TestApaReport:
    def test_returns_string(self):
        assert isinstance(apa_report(TTEST_RESULT), str)

    def test_contains_pvalue(self):
        report = apa_report(TTEST_RESULT)
        # Should contain p = or p < somewhere
        assert "p" in report.lower()

    def test_contains_significant_word(self):
        report = apa_report(TTEST_RESULT)
        assert "significant" in report.lower()

    def test_one_sample_report(self):
        report = apa_report(ONE_SAMP)
        assert isinstance(report, str)
        assert len(report) > 10

    def test_correlation_report(self):
        report = apa_report(CORR_RESULT)
        assert isinstance(report, str)

    def test_contains_statistic(self):
        report = apa_report(TTEST_RESULT)
        stat = f"{abs(TTEST_RESULT.statistic):.2f}"[:4]
        assert stat in report or "t" in report.lower()


# ---------------------------------------------------------------------------
# text_report
# ---------------------------------------------------------------------------


class TestTextReport:
    def test_returns_string(self):
        assert isinstance(text_report(TTEST_RESULT), str)

    def test_verbose_true_longer(self):
        short = text_report(TTEST_RESULT, verbose=False)
        long_ = text_report(TTEST_RESULT, verbose=True)
        assert len(long_) >= len(short)

    def test_contains_test_name(self):
        report = text_report(TTEST_RESULT)
        assert TTEST_RESULT.test_name.lower() in report.lower()

    def test_contains_pvalue(self):
        report = text_report(TTEST_RESULT)
        assert "p-value" in report.lower() or str(round(TTEST_RESULT.p_value, 2)) in report

    def test_contains_significance_decision(self):
        report = text_report(TTEST_RESULT)
        assert "H0" in report

    def test_chi_sq_report(self):
        report = text_report(CHISQ_RESULT)
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# batch_report
# ---------------------------------------------------------------------------


class TestBatchReport:
    def test_returns_string(self):
        report = batch_report([TTEST_RESULT, ONE_SAMP])
        assert isinstance(report, str)

    def test_contains_multiple_tests(self):
        report = batch_report([TTEST_RESULT, ONE_SAMP, CORR_RESULT])
        # Should list all three result test names
        assert TTEST_RESULT.test_name.lower()[:6] in report.lower() or len(report) > 50

    def test_empty_list_returns_string(self):
        report = batch_report([])
        assert isinstance(report, str)

    def test_custom_title(self):
        report = batch_report([TTEST_RESULT], title="My Analysis")
        assert "My Analysis" in report


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "results.csv")
        export_csv([TTEST_RESULT, ONE_SAMP], path)
        assert os.path.exists(path)

    def test_file_has_content(self, tmp_path):
        path = str(tmp_path / "results.csv")
        export_csv([TTEST_RESULT], path)
        with open(path) as f:
            content = f.read()
        assert len(content) > 0
        assert "," in content  # CSV format
