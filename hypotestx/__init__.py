"""
HypoTestX: Natural Language Hypothesis Testing Library
"""

# ── Core ──────────────────────────────────────────────────────────────────────
from .core.parser import parse_hypothesis, create_parser
from .core.result import HypoResult
from .core.exceptions import (
    HypoTestXError,
    InsufficientDataError,
    AssumptionViolationError,
    InvalidAlternativeError,
    ParseError,
    UnsupportedTestError,
    DataFormatError,
)

# ── Parametric tests ──────────────────────────────────────────────────────────
from .tests.parametric import (
    one_sample_ttest,
    two_sample_ttest,
    paired_ttest,
    anova_one_way,
)

# ── Non-parametric tests ──────────────────────────────────────────────────────
from .tests.nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
)

# ── Categorical tests ─────────────────────────────────────────────────────────
from .tests.categorical import (
    chi_square_test,
    fisher_exact_test,
)

# ── Correlation tests ─────────────────────────────────────────────────────────
from .tests.correlation import (
    pearson_correlation,
    spearman_correlation,
    point_biserial_correlation,
)

# ── Math utilities ────────────────────────────────────────────────────────────
from .math.statistics import mean, std, variance, correlation
from .math.distributions import Normal, StudentT, ChiSquare, F

# ── Assumption checks ────────────────────────────────────────────────────────
from .core.assumptions import (
    shapiro_wilk, levene_test, bartlett_test, jarque_bera,
    check_normality, check_equal_variances,
)

# ── Power analysis ────────────────────────────────────────────────────────────
from .power.analysis import (
    power_ttest_one_sample, power_ttest_two_sample, power_ttest_paired,
    power_anova, power_chi_square, power_correlation, power_summary,
)
from .power.sample_size import (
    n_ttest_one_sample, n_ttest_two_sample, n_ttest_paired,
    n_anova, n_chi_square, n_correlation, sample_size_summary,
)

# ── Descriptive stats & bootstrap ────────────────────────────────────────────
from .stats.descriptive import (
    DescriptiveStats, describe,
    five_number_summary, detect_outliers, frequency_table, compare_groups,
)
from .stats.bootstrap import (
    bootstrap_ci, bootstrap_two_sample_ci, bootstrap_mean_ci,
    bootstrap_test, permutation_test,
)

# ── Reporting ────────────────────────────────────────────────────────────────
from .reporting.generator import (
    apa_report, text_report, batch_report,
    export_csv, export_html, export_pdf,
)
from .reporting.formatters import (
    format_p, format_ci, format_effect, apa_stat, effect_interpretation_table,
)

# ── Visualization (optional — requires matplotlib) ───────────────────────────
from .explore.visualize import (
    plot_result,
    plot_distributions,
    plot_p_value,
    generate_report,
)

# ── Utils ─────────────────────────────────────────────────────────────────────
from .utils.data_utils import (
    coerce_numeric, detect_missing, drop_missing,
    group_by, split_groups, validate_sample_data,
    summary_table, are_paired,
)
from .utils.preprocessing import (
    standardize, normalize, winsorize, log_transform,
    rank_transform, center, robust_scale,
)

__version__ = "1.0.5"
__author__ = "Ankit"

# ── LLM-powered natural language interface ───────────────────────────────────
from .core.engine import analyze
from .core.llm import (
    get_backend,
    LLMBackend,
    CallableBackend,
    RoutingResult,
    SchemaInfo,
    OllamaBackend,
    OpenAICompatBackend,
    GeminiBackend,
    HuggingFaceBackend,
    FallbackBackend,
)

# ── Natural language interface (legacy regex parser) ─────────────────────────

def test(hypothesis: str, data=None, **kwargs):
    """
    Main natural language interface for hypothesis testing.

    Parse *hypothesis* in plain English and automatically route to the
    appropriate statistical test.

    Args:
        hypothesis: Natural language hypothesis statement
        data: Optional pandas DataFrame (or Series / list) with the data
        **kwargs: Override parameters forwarded to the detected test:
            alpha (float): significance level (default 0.05)
            alternative (str): 'two-sided', 'greater', or 'less'
            mu (float): hypothesised mean for one-sample t-test
            equal_var (bool): assume equal variances for two-sample t-test
            before / after (list): paired data arrays
            method (str): 'non-parametric' to force a non-parametric test

    Returns:
        HypoResult with test results

    Examples:
        >>> result = htx.test("Do males spend more than females?", data=df)
        >>> result = htx.test("Is the mean different from 100?", data=df['values'])
        >>> result = htx.test("Is there an association between gender and preference?", data=df)
    """
    parsed = parse_hypothesis(hypothesis, data)

    alpha       = kwargs.get('alpha', parsed.confidence_level)
    alternative = kwargs.get('alternative', parsed.tail)
    method      = kwargs.get('method', 'parametric')

    # ── one-sample t-test ────────────────────────────────────────
    if parsed.test_type == "one_sample_ttest":
        if data is None:
            raise ValueError("Data is required for statistical testing")
        values = _coerce_to_list(data, "data")
        mu = kwargs.get('mu', 0.0)
        return one_sample_ttest(values, mu=mu, alpha=alpha, alternative=alternative)

    # ── two-sample t-test / Welch's / Mann-Whitney ────────────────
    elif parsed.test_type == "two_sample_ttest":
        if data is None or parsed.group_column is None or parsed.value_column is None:
            raise ValueError("Data with group and value columns required for two-sample test")

        unique_groups = list(data[parsed.group_column].unique())
        if len(unique_groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(unique_groups)}")

        g1 = list(data[data[parsed.group_column] == unique_groups[0]][parsed.value_column].values)
        g2 = list(data[data[parsed.group_column] == unique_groups[1]][parsed.value_column].values)

        if method == 'non-parametric':
            return mann_whitney_u(g1, g2, alpha=alpha, alternative=alternative)

        equal_var = kwargs.get('equal_var', True)
        return two_sample_ttest(g1, g2, alpha=alpha, alternative=alternative, equal_var=equal_var)

    # ── paired t-test ─────────────────────────────────────────────
    elif parsed.test_type == "paired_ttest":
        before = kwargs.get('before')
        after  = kwargs.get('after')
        if before is None or after is None:
            raise ValueError("Paired t-test requires 'before' and 'after' keyword arguments")
        if method == 'non-parametric':
            return wilcoxon_signed_rank(before, after, alpha=alpha, alternative=alternative)
        return paired_ttest(before, after, alpha=alpha, alternative=alternative)

    # ── one-way ANOVA / Kruskal-Wallis ────────────────────────────
    elif parsed.test_type == "anova":
        if data is None or parsed.group_column is None or parsed.value_column is None:
            raise ValueError("Data with group and value columns required for ANOVA")

        groups = [
            list(data[data[parsed.group_column] == g][parsed.value_column].values)
            for g in data[parsed.group_column].unique()
        ]
        if method == 'non-parametric':
            return kruskal_wallis(*groups, alpha=alpha)
        return anova_one_way(*groups, alpha=alpha)

    # ── chi-square test of independence ──────────────────────────
    elif parsed.test_type == "chi_square":
        if data is None or parsed.group_column is None or parsed.value_column is None:
            raise ValueError("Data with two categorical columns required for chi-square test")

        row_vals = list(data[parsed.group_column].unique())
        col_vals = list(data[parsed.value_column].unique())
        table = [
            [int((data[data[parsed.group_column] == r][parsed.value_column] == c).sum())
             for c in col_vals]
            for r in row_vals
        ]
        return chi_square_test(table, alpha=alpha)

    # ── correlation ───────────────────────────────────────────────
    elif parsed.test_type == "correlation":
        if data is None:
            raise ValueError("Data is required for correlation test")

        if parsed.group_column and parsed.value_column:
            x = list(data[parsed.group_column].values)
            y = list(data[parsed.value_column].values)
        else:
            num_cols = [col for col in data.columns
                        if str(data[col].dtype) in ('int64', 'float64', 'int32', 'float32')]
            if len(num_cols) < 2:
                raise ValueError("Need at least two numeric columns for a correlation test")
            x = list(data[num_cols[0]].values)
            y = list(data[num_cols[1]].values)

        if method == 'non-parametric':
            return spearman_correlation(x, y, alpha=alpha, alternative=alternative)
        return pearson_correlation(x, y, alpha=alpha, alternative=alternative)

    else:
        raise UnsupportedTestError(parsed.test_type)


# ── Convenience wrappers ──────────────────────────────────────────────────────

def ttest_1samp(data, mu=0.0, **kwargs):
    """One-sample t-test.  See ``one_sample_ttest`` for full docs."""
    return one_sample_ttest(_coerce_to_list(data, "data"), mu=mu, **kwargs)


def ttest_2samp(group1, group2, **kwargs):
    """Two-sample t-test (Student's or Welch's).  See ``two_sample_ttest``."""
    return two_sample_ttest(
        _coerce_to_list(group1, "group1"),
        _coerce_to_list(group2, "group2"),
        **kwargs,
    )


def ttest_paired(before, after, **kwargs):
    """Paired t-test.  See ``paired_ttest`` for full docs."""
    return paired_ttest(
        _coerce_to_list(before, "before"),
        _coerce_to_list(after, "after"),
        **kwargs,
    )


def welch_ttest(group1, group2, **kwargs):
    """Welch's t-test (unequal variances).  Shortcut for ttest_2samp with equal_var=False."""
    kwargs['equal_var'] = False
    return ttest_2samp(group1, group2, **kwargs)


def anova_1way(*groups, **kwargs):
    """One-way ANOVA.  See ``anova_one_way`` for full docs."""
    return anova_one_way(*groups, **kwargs)


def mannwhitney(group1, group2, **kwargs):
    """Mann-Whitney U test.  See ``mann_whitney_u`` for full docs."""
    return mann_whitney_u(
        _coerce_to_list(group1, "group1"),
        _coerce_to_list(group2, "group2"),
        **kwargs,
    )


def wilcoxon(x, y=None, **kwargs):
    """Wilcoxon signed-rank test.  See ``wilcoxon_signed_rank`` for full docs."""
    x = _coerce_to_list(x, "x")
    if y is not None:
        y = _coerce_to_list(y, "y")
    return wilcoxon_signed_rank(x, y, **kwargs)


def kruskal(*groups, **kwargs):
    """Kruskal-Wallis H test.  See ``kruskal_wallis`` for full docs."""
    return kruskal_wallis(*groups, **kwargs)


def chi2_test(observed, **kwargs):
    """Chi-square test of independence or goodness-of-fit.  See ``chi_square_test``."""
    return chi_square_test(observed, **kwargs)


def fisher_exact(table, **kwargs):
    """Fisher's exact test for 2×2 tables.  See ``fisher_exact_test``."""
    return fisher_exact_test(table, **kwargs)


def pearson(x, y, **kwargs):
    """Pearson correlation test.  See ``pearson_correlation`` for full docs."""
    return pearson_correlation(
        _coerce_to_list(x, "x"),
        _coerce_to_list(y, "y"),
        **kwargs,
    )


def spearman(x, y, **kwargs):
    """Spearman rank correlation test.  See ``spearman_correlation`` for full docs."""
    return spearman_correlation(
        _coerce_to_list(x, "x"),
        _coerce_to_list(y, "y"),
        **kwargs,
    )


def pointbiserial(continuous, binary, **kwargs):
    """Point-biserial correlation.  See ``point_biserial_correlation`` for full docs."""
    return point_biserial_correlation(
        _coerce_to_list(continuous, "continuous"),
        list(binary),
        **kwargs,
    )


# ── Internal utility ──────────────────────────────────────────────────────────

def _coerce_to_list(data, name: str = "data"):
    """Convert pandas Series / array-like to a plain Python list."""
    if isinstance(data, list):
        return data
    if hasattr(data, 'tolist'):      # numpy array or pandas Series
        return data.tolist()
    if hasattr(data, 'values'):      # pandas Series fallback
        return list(data.values)
    try:
        return list(data)
    except TypeError:
        raise DataFormatError(
            f"'{name}' must be a list or iterable, got {type(data).__name__}"
        )


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # LLM-powered natural language interface
    "analyze",
    # LLM backend factory & base class
    "get_backend",
    "LLMBackend",
    "CallableBackend",
    "RoutingResult",
    "SchemaInfo",
    # Built-in backends
    "OllamaBackend",
    "OpenAICompatBackend",
    "GeminiBackend",
    "HuggingFaceBackend",
    "FallbackBackend",
    # Legacy natural language interface
    "test",
    # Core classes
    "HypoResult",
    "parse_hypothesis",
    "create_parser",
    # Exceptions
    "HypoTestXError",
    "InsufficientDataError",
    "AssumptionViolationError",
    "InvalidAlternativeError",
    "ParseError",
    "UnsupportedTestError",
    "DataFormatError",
    # Parametric — short names (README API)
    "ttest_1samp",
    "ttest_2samp",
    "ttest_paired",
    "welch_ttest",
    "anova_1way",
    # Parametric — full names
    "one_sample_ttest",
    "two_sample_ttest",
    "paired_ttest",
    "anova_one_way",
    # Non-parametric — short names
    "mannwhitney",
    "wilcoxon",
    "kruskal",
    # Non-parametric — full names
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    # Categorical — short names
    "chi2_test",
    "fisher_exact",
    # Categorical — full names
    "chi_square_test",
    "fisher_exact_test",
    # Correlation — short names
    "pearson",
    "spearman",
    "pointbiserial",
    # Correlation — full names
    "pearson_correlation",
    "spearman_correlation",
    "point_biserial_correlation",
    # Math utilities
    "mean",
    "std",
    "variance",
    "correlation",
    "Normal",
    "StudentT",
    "ChiSquare",
    "F",
    # Assumption checks
    "shapiro_wilk", "levene_test", "bartlett_test", "jarque_bera",
    "check_normality", "check_equal_variances",
    # Power analysis
    "power_ttest_one_sample", "power_ttest_two_sample", "power_ttest_paired",
    "power_anova", "power_chi_square", "power_correlation", "power_summary",
    # Sample size
    "n_ttest_one_sample", "n_ttest_two_sample", "n_ttest_paired",
    "n_anova", "n_chi_square", "n_correlation", "sample_size_summary",
    # Descriptive stats
    "DescriptiveStats", "describe",
    "five_number_summary", "detect_outliers", "frequency_table", "compare_groups",
    # Bootstrap
    "bootstrap_ci", "bootstrap_two_sample_ci", "bootstrap_mean_ci",
    "bootstrap_test", "permutation_test",
    # Reporting
    "apa_report", "text_report", "batch_report", "export_csv",
    "format_p", "format_ci", "format_effect", "apa_stat",
    "effect_interpretation_table",
    # Utils
    "coerce_numeric", "detect_missing", "drop_missing",
    "group_by", "split_groups", "validate_sample_data",
    "summary_table", "are_paired",
    "standardize", "normalize", "winsorize", "log_transform",
    "rank_transform", "center", "robust_scale",
]