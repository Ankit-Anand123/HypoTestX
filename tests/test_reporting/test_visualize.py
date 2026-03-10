"""
Tests for hypotestx.explore.visualize and HypoResult.plot().

Matplotlib is an optional dependency.  Tests that need it are skipped
when it is not installed.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.tests.parametric import two_sample_ttest, one_sample_ttest
from hypotestx.explore.visualize import (
    plot_result,
    plot_distributions,
    plot_p_value,
    generate_report,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for tests
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Shared results
G1 = [8.0, 9.0, 10.0, 11.0, 9.5, 10.5, 9.0, 10.0]
G2 = [6.0, 7.0,  6.5,  7.5,  6.0,  7.0,  6.5,  7.0]
RESULT = two_sample_ttest(G1, G2)
ONE_SAMP = one_sample_ttest([5.5, 6.0, 5.8, 6.1, 5.7, 5.4, 5.9, 6.2], mu=5.0)


@unittest.skipUnless(HAS_MPL, "matplotlib not installed")
class TestPlotResult(unittest.TestCase):

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_returns_figure(self):
        import matplotlib.figure
        fig = plot_result(RESULT)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_auto_kind_two_group(self):
        import matplotlib.figure
        fig = plot_result(RESULT, kind="auto")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_p_value_kind(self):
        import matplotlib.figure
        fig = plot_result(ONE_SAMP, kind="p_value")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_hypo_result_plot_method(self):
        import matplotlib.figure
        fig = RESULT.plot()
        self.assertIsInstance(fig, matplotlib.figure.Figure)


@unittest.skipUnless(HAS_MPL, "matplotlib not installed")
class TestPlotDistributions(unittest.TestCase):

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_box_plot(self):
        import matplotlib.figure
        fig = plot_distributions([G1, G2], labels=["A", "B"], kind="box")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_bar_plot(self):
        import matplotlib.figure
        fig = plot_distributions([G1, G2], labels=["A", "B"], kind="bar")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_default_labels(self):
        import matplotlib.figure
        fig = plot_distributions([G1, G2])
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_single_group(self):
        import matplotlib.figure
        fig = plot_distributions([G1])
        self.assertIsInstance(fig, matplotlib.figure.Figure)


@unittest.skipUnless(HAS_MPL, "matplotlib not installed")
class TestPlotPValue(unittest.TestCase):

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_two_sided(self):
        import matplotlib.figure
        fig = plot_p_value(0.03, alpha=0.05, alternative="two-sided")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_greater(self):
        import matplotlib.figure
        fig = plot_p_value(0.10, alpha=0.05, alternative="greater")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_less(self):
        import matplotlib.figure
        fig = plot_p_value(0.10, alpha=0.05, alternative="less")
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_with_test_statistic(self):
        import matplotlib.figure
        fig = plot_p_value(0.04, alpha=0.05, test_statistic=2.1)
        self.assertIsInstance(fig, matplotlib.figure.Figure)


class TestGenerateReport(unittest.TestCase):

    def test_html_contains_test_name(self):
        html = generate_report(RESULT, fmt="html")
        self.assertIn(RESULT.test_name, html)

    def test_html_contains_pvalue(self):
        html = generate_report(RESULT, fmt="html")
        self.assertIn("p_value", html.lower().replace("-", "_"))

    def test_html_is_valid_html(self):
        html = generate_report(RESULT, fmt="html")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)

    def test_text_format(self):
        text = generate_report(RESULT, fmt="text")
        self.assertIn(RESULT.test_name, text)

    def test_save_html(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        ) as f:
            tmp_path = f.name
        try:
            generate_report(RESULT, path=tmp_path, fmt="html")
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
        finally:
            os.unlink(tmp_path)


class TestMissingMatplotlib(unittest.TestCase):
    """When matplotlib is absent, plotting raises ImportError with guidance."""

    def test_import_error_message(self):
        if HAS_MPL:
            self.skipTest("matplotlib is present; skipping absence test")
        with self.assertRaises(ImportError) as ctx:
            plot_result(RESULT)
        self.assertIn("matplotlib", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
