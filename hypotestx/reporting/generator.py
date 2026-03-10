"""
Report generator for hypothesis test results.

Provides APA-style text reports from HypoResult objects and from lists of
HypoResult objects (batch reports).

Provides
--------
apa_report(result)              -> str   APA-style paragraph
text_report(result, verbose)    -> str   detailed plain-text report
batch_report(results, title)    -> str   multi-test summary table
export_csv(results, path)       -> None  write batch results to CSV
"""
from typing import List, Optional
from ..core.result import HypoResult
from .formatters import apa_stat, format_p, format_ci, format_effect


# ---------------------------------------------------------------------------
# APA-style paragraph
# ---------------------------------------------------------------------------

# Map test names to APA symbol names
_APA_SYMBOLS = {
    "one-sample t-test":                  ("t", "d"),
    "two-sample t-test":                  ("t", "d"),
    "welch's t-test":                     ("t", "d"),
    "paired t-test":                      ("t", "d"),
    "one-way anova":                      ("F", "eta-squared"),
    "mann-whitney u test":                ("U", "rank-biserial r"),
    "wilcoxon signed-rank test":          ("W", "r"),
    "kruskal-wallis test":                ("H", "eta-squared"),
    "chi-square test of independence":    ("chi2", "phi"),
    "chi-square goodness-of-fit test":    ("chi2", "w"),
    "fisher's exact test":                ("OR", None),
    "pearson correlation":                ("r", "r"),
    "spearman correlation":               ("rho", "rho"),
    "point-biserial correlation":         ("r_pb", "r_pb"),
    "shapiro-wilk normality test":        ("W", None),
    "levene test for equal variances":    ("F", None),
    "bartlett test for equal variances":  ("B", None),
    "jarque-bera normality test":         ("JB", None),
}


def _lookup_symbols(test_name: str):
    """Return (stat_symbol, effect_symbol) for a given test name."""
    key = test_name.lower().strip()
    for k, v in _APA_SYMBOLS.items():
        if k in key or key in k:
            return v
    return (test_name[:6], None)


def apa_report(result: HypoResult) -> str:
    """
    Generate an APA-style results paragraph for a single HypoResult.

    Parameters
    ----------
    result : HypoResult from any test function

    Returns
    -------
    str : APA-style citation suitable for use in a Results section

    Example
    -------
    An independent-samples t-test revealed a significant difference between
    groups, t(28) = 3.45, p = .001, d = 0.62 (medium).
    """
    stat_sym, eff_sym = _lookup_symbols(result.test_name)
    df = result.degrees_of_freedom

    # Build inline citation
    citation = apa_stat(
        stat_sym,
        result.statistic,
        df=df,
        p=result.p_value,
        effect_name=eff_sym if result.effect_size is not None and eff_sym else None,
        effect_value=result.effect_size if result.effect_size is not None else None,
    )

    sig_word   = "significant" if result.is_significant else "non-significant"
    direction  = ""
    if result.interpretation:
        # Grab the first sentence of the interpretation for context
        first_sent = result.interpretation.split(".")[0].strip()
        direction  = f" {first_sent}."

    # CI phrase
    ci_phrase = ""
    if result.confidence_interval is not None:
        ci_level = int((1 - result.alpha) * 100)
        ci_phrase = (f" A {ci_level}% confidence interval for the effect was "
                     f"[{result.confidence_interval[0]:.3f}, "
                     f"{result.confidence_interval[1]:.3f}].")

    report = (
        f"A {result.test_name.lower()} was conducted. "
        f"The result was {sig_word} ({citation}).{direction}{ci_phrase}"
    )
    return report


# ---------------------------------------------------------------------------
# Detailed plain-text report
# ---------------------------------------------------------------------------

def text_report(result: HypoResult, verbose: bool = True) -> str:
    """
    Generate a detailed plain-text report for a single HypoResult.

    Parameters
    ----------
    result  : HypoResult
    verbose : include sample sizes, assumptions, data summary (default True)

    Returns
    -------
    str : multi-line report
    """
    lines = []
    width = 60

    lines.append("=" * width)
    lines.append(f"  {result.test_name}")
    lines.append("=" * width)
    lines.append("")

    # --- Hypothesis ---
    lines.append("HYPOTHESIS")
    lines.append("-" * width)
    lines.append(f"  H0: No effect / no difference")
    lines.append(f"  H1: Alternative ({result.alternative})")
    lines.append(f"  Significance level (alpha): {result.alpha}")
    lines.append("")

    # --- Test statistics ---
    lines.append("TEST RESULTS")
    lines.append("-" * width)
    lines.append(f"  Test statistic : {result.statistic:.4f}")
    lines.append(f"  p-value        : {format_p(result.p_value)}")
    if result.degrees_of_freedom is not None:
        df = result.degrees_of_freedom
        if isinstance(df, tuple):
            df_str = f"({', '.join(str(d) for d in df)})"
        else:
            df_str = str(df)
        lines.append(f"  df             : {df_str}")
    decision = "REJECT H0 (significant)" if result.is_significant else "FAIL TO REJECT H0 (not significant)"
    lines.append(f"  Decision       : {decision}")
    lines.append("")

    # --- Effect size ---
    if result.effect_size is not None:
        lines.append("EFFECT SIZE")
        lines.append("-" * width)
        lines.append(f"  {result.effect_size_name}: {result.effect_size:.4f} ({result.effect_magnitude})")
        lines.append("")

    # --- Confidence interval ---
    if result.confidence_interval is not None:
        ci_level = int((1 - result.alpha) * 100)
        lines.append(f"CONFIDENCE INTERVAL ({ci_level}%)")
        lines.append("-" * width)
        lines.append(f"  [{result.confidence_interval[0]:.4f},  {result.confidence_interval[1]:.4f}]")
        lines.append("")

    if verbose:
        # --- Sample sizes ---
        if result.sample_sizes is not None:
            lines.append("SAMPLE")
            lines.append("-" * width)
            if isinstance(result.sample_sizes, tuple):
                lines.append(f"  n per group : {result.sample_sizes}")
                lines.append(f"  total N     : {sum(result.sample_sizes)}")
            else:
                lines.append(f"  n : {result.sample_sizes}")
            lines.append("")

        # --- Assumptions ---
        if result.assumptions_met:
            lines.append("ASSUMPTION CHECKS")
            lines.append("-" * width)
            for assumption, met in result.assumptions_met.items():
                status = "Met" if met else "Violated (!)"
                lines.append(f"  {assumption:<28}: {status}")
            lines.append("")

        # --- Data summary ---
        if result.data_summary:
            lines.append("DATA SUMMARY")
            lines.append("-" * width)
            for key, value in result.data_summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key:<28}: {value:.4f}")
                else:
                    lines.append(f"  {key:<28}: {value}")
            lines.append("")

    # --- APA citation ---
    lines.append("APA CITATION")
    lines.append("-" * width)
    lines.append(f"  {apa_report(result)}")
    lines.append("")

    # --- Interpretation ---
    if result.interpretation:
        lines.append("INTERPRETATION")
        lines.append("-" * width)
        lines.append(f"  {result.interpretation}")
        lines.append("")

    lines.append("=" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch report (multiple tests)
# ---------------------------------------------------------------------------

def batch_report(
    results: List[HypoResult],
    title: str = "Hypothesis Testing Results",
    show_effect: bool = True,
) -> str:
    """
    Generate a summary table for multiple test results.

    Parameters
    ----------
    results     : list of HypoResult objects
    title       : report title
    show_effect : include effect size columns (default True)

    Returns
    -------
    str : formatted summary table
    """
    if not results:
        return "No results to display."

    lines = []
    lines.append("=" * 80)
    lines.append(f"  {title}")
    lines.append("=" * 80)
    lines.append("")

    # Header
    col_test  = 32
    col_stat  = 10
    col_df    = 8
    col_p     = 10
    col_sig   = 5

    header = (
        f"{'Test':<{col_test}}"
        f"{'Statistic':>{col_stat}}"
        f"{'df':>{col_df}}"
        f"{'p-value':>{col_p}}"
        f"{'Sig':>{col_sig}}"
    )
    if show_effect:
        header += f"  {'Effect':<20}"

    lines.append(header)
    lines.append("-" * len(header))

    sig_count = 0
    for r in results:
        df_str = ""
        if r.degrees_of_freedom is not None:
            df = r.degrees_of_freedom
            if isinstance(df, tuple):
                df_str = "(" + ",".join(str(d) for d in df) + ")"
            else:
                df_str = str(df)

        sig_marker = "* " if r.is_significant else "  "
        if r.is_significant:
            sig_count += 1

        row = (
            f"{r.test_name[:col_test]:<{col_test}}"
            f"{r.statistic:>{col_stat}.4f}"
            f"{df_str:>{col_df}}"
            f"{format_p(r.p_value):>{col_p}}"
            f"{sig_marker:>{col_sig}}"
        )
        if show_effect and r.effect_size is not None:
            eff_str = f"{r.effect_size_name} = {r.effect_size:.3f} ({r.effect_magnitude})"
            row += f"  {eff_str:<20}"

        lines.append(row)

    lines.append("-" * len(header))
    lines.append(f"  * significant at alpha = {results[0].alpha}")
    lines.append(f"  {sig_count}/{len(results)} tests significant")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    results: List[HypoResult],
    path: str,
    sep: str = ",",
) -> None:
    """
    Write a batch of HypoResult objects to a CSV file.

    Parameters
    ----------
    results : list of HypoResult
    path    : output file path
    sep     : delimiter (default ',')
    """
    if not results:
        raise ValueError("results list is empty")

    fieldnames = [
        "test_name", "statistic", "p_value", "is_significant", "alpha",
        "alternative", "degrees_of_freedom", "sample_sizes",
        "effect_size", "effect_size_name", "effect_magnitude",
        "ci_lower", "ci_upper",
    ]

    rows = []
    for r in results:
        ci_lower = r.confidence_interval[0] if r.confidence_interval else ""
        ci_upper = r.confidence_interval[1] if r.confidence_interval else ""
        rows.append({
            "test_name":           r.test_name,
            "statistic":           round(r.statistic, 6),
            "p_value":             round(r.p_value, 6),
            "is_significant":      r.is_significant,
            "alpha":               r.alpha,
            "alternative":         r.alternative,
            "degrees_of_freedom":  r.degrees_of_freedom or "",
            "sample_sizes":        r.sample_sizes or "",
            "effect_size":         round(r.effect_size, 6) if r.effect_size is not None else "",
            "effect_size_name":    r.effect_size_name or "",
            "effect_magnitude":    r.effect_magnitude if r.effect_size is not None else "",
            "ci_lower":            round(ci_lower, 6) if ci_lower != "" else "",
            "ci_upper":            round(ci_upper, 6) if ci_upper != "" else "",
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        # Write header
        f.write(sep.join(fieldnames) + "\n")
        for row in rows:
            f.write(sep.join(str(row[k]) for k in fieldnames) + "\n")


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def export_html(
    result: "HypoResult",
    path: Optional[str] = None,
) -> str:
    """
    Generate a self-contained HTML report for a single :class:`HypoResult`.

    Delegates to :func:`hypotestx.explore.visualize.generate_report` so that
    an embedded plot is included when matplotlib is installed.

    Parameters
    ----------
    result : HypoResult
    path   : optional output file path (e.g. ``"report.html"``).
             If *None*, the HTML string is returned without saving.

    Returns
    -------
    str : HTML content
    """
    from ..explore.visualize import generate_report
    return generate_report(result, path=path, fmt="html")


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_pdf(
    result: "HypoResult",
    path: str,
) -> None:
    """
    Save a PDF report for a single :class:`HypoResult`.

    Requires ``weasyprint``::

        pip install weasyprint

    Parameters
    ----------
    result : HypoResult
    path   : output file path (e.g. ``"report.pdf"``).
    """
    from ..explore.visualize import generate_report
    generate_report(result, path=path, fmt="pdf")


__all__ = [
    "apa_report", "text_report", "batch_report",
    "export_csv", "export_html", "export_pdf",
]
