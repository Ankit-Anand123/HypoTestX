"""
Report templates for HypoTestX output.

Provides string templates and rendering for:
  - APA-style statistical reports
  - Plaintext summary reports
  - Minimal one-line summaries

All templates are pure Python — no Jinja2 or other dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# APA template
# ---------------------------------------------------------------------------

APA_TEMPLATES: Dict[str, str] = {
    "two_sample_ttest": (
        "An independent-samples t-test was conducted to compare {value_column} "
        "between {group1} and {group2}. The results indicate a "
        "{significance} difference between {group1} (M = {mean1:.2f}, "
        "SD = {sd1:.2f}) and {group2} (M = {mean2:.2f}, SD = {sd2:.2f}), "
        "t({df}) = {statistic:.3f}, p = {p_value:.4f}."
        "{effect_size_sentence}"
    ),
    "one_sample_ttest": (
        "A one-sample t-test was conducted to determine whether the mean of "
        "{value_column} differed significantly from {mu0}. "
        "The test was {significance}, "
        "t({df}) = {statistic:.3f}, p = {p_value:.4f}, "
        "M = {sample_mean:.2f}, SD = {sample_sd:.2f}."
    ),
    "paired_ttest": (
        "A paired-samples t-test was conducted to evaluate whether the mean "
        "difference between {col1} and {col2} was zero. "
        "The test was {significance}, "
        "t({df}) = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "anova": (
        "A one-way ANOVA was conducted to compare {value_column} across "
        "{k} groups ({groups}). "
        "The results were {significance}, "
        "F({df_between}, {df_within}) = {statistic:.3f}, p = {p_value:.4f}."
        "{effect_size_sentence}"
    ),
    "chi_square": (
        "A chi-square test of independence was conducted between "
        "{x_column} and {y_column}. "
        "The association was {significance}, "
        "chi^2({df}) = {statistic:.3f}, p = {p_value:.4f}."
        "{effect_size_sentence}"
    ),
    "pearson": (
        "A Pearson correlation coefficient was computed between "
        "{x_column} and {y_column}. "
        "The correlation was {significance}, "
        "r({df}) = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "spearman": (
        "Spearman's rank-order correlation was computed between "
        "{x_column} and {y_column}. "
        "The correlation was {significance}, "
        "rs({df}) = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "mann_whitney": (
        "A Mann-Whitney U test was conducted to compare "
        "{group1} and {group2} on {value_column}. "
        "The test was {significance}, "
        "U = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "kruskal_wallis": (
        "A Kruskal-Wallis H test was conducted to compare "
        "{value_column} across {k} groups ({groups}). "
        "The test was {significance}, "
        "H = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "fisher": (
        "Fisher's exact test was conducted to assess the association between "
        "{x_column} and {y_column}. "
        "The test was {significance}, "
        "OR = {statistic:.3f}, p = {p_value:.4f}."
    ),
    "generic": (
        "{test_name} was conducted. "
        "The result was {significance}, "
        "stat = {statistic:.3f}, p = {p_value:.4f}."
    ),
}


# ---------------------------------------------------------------------------
# Plain-text template
# ---------------------------------------------------------------------------

PLAIN_TEMPLATE = """\
Test        : {test_name}
Statistic   : {statistic:.4f}
p-value     : {p_value:.4f}
Alpha       : {alpha}
Conclusion  : {conclusion}
Effect size : {effect_size_str}
"""

ONE_LINE_TEMPLATE = "{test_name}: stat={statistic:.4f}, p={p_value:.4f} -> {conclusion}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _significance_word(is_significant: bool) -> str:
    return "statistically significant" if is_significant else "not statistically significant"


def _effect_size_sentence(effect_size: Optional[float]) -> str:
    if effect_size is None:
        return ""
    return f" The effect size was {effect_size:.3f}."


def _effect_size_str(effect_size: Optional[float]) -> str:
    if effect_size is None or (isinstance(effect_size, float) and effect_size != effect_size):
        return "N/A"
    return f"{effect_size:.4f}"


def render_apa(
    test_key: str,
    context: Dict[str, Any],
) -> str:
    """
    Render an APA-style report sentence using the template for *test_key*.

    Parameters
    ----------
    test_key : key into APA_TEMPLATES (e.g. 'two_sample_ttest')
    context  : dict of values to interpolate

    Returns
    -------
    str
    """
    template = APA_TEMPLATES.get(test_key, APA_TEMPLATES["generic"])
    ctx = dict(context)
    ctx.setdefault("significance", _significance_word(ctx.get("is_significant", False)))
    ctx.setdefault("effect_size_sentence", _effect_size_sentence(ctx.get("effect_size")))
    try:
        return template.format_map(ctx)
    except (KeyError, ValueError):
        # Fall back to generic if template keys don't match
        return APA_TEMPLATES["generic"].format_map(
            {
                "test_name": ctx.get("test_name", test_key),
                "significance": ctx.get("significance", ""),
                "statistic": ctx.get("statistic", float("nan")),
                "p_value": ctx.get("p_value", float("nan")),
            }
        )


def render_plain(context: Dict[str, Any]) -> str:
    """
    Render a plain-text summary block.

    Parameters
    ----------
    context : dict with keys test_name, statistic, p_value, alpha,
              is_significant, effect_size
    """
    ctx = dict(context)
    ctx.setdefault("conclusion", "Reject H0" if ctx.get("is_significant") else "Fail to reject H0")
    ctx.setdefault("effect_size_str", _effect_size_str(ctx.get("effect_size")))
    return PLAIN_TEMPLATE.format_map(ctx)


def render_one_line(context: Dict[str, Any]) -> str:
    """Render a compact one-line summary."""
    ctx = dict(context)
    ctx.setdefault("conclusion", "Reject H0" if ctx.get("is_significant") else "Fail to reject H0")
    return ONE_LINE_TEMPLATE.format_map(ctx)


__all__ = [
    "APA_TEMPLATES",
    "PLAIN_TEMPLATE",
    "ONE_LINE_TEMPLATE",
    "render_apa",
    "render_plain",
    "render_one_line",
]
