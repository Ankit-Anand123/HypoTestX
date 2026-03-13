"""
Regex/rule-based fallback backend — zero dependencies, zero API calls.

Used automatically when no LLM backend is specified.
Handles the most common question patterns with reasonable accuracy.
For production use or complex questions, switch to a real LLM backend.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..base import LLMBackend, RoutingResult, SchemaInfo

# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

_TESTS_BY_KEYWORD = [
    # (regex, test_key, default_alternative)   — searched in order
    # Correlations
    (
        "correlat|related to|linear relationship|predict.*from|association.*numeric"
        "|scatter",
        "pearson",
        "two-sided",
    ),
    ("rank.*corr|spearman|monotone|ordinal corr", "spearman", "two-sided"),
    # Categorical association
    (
        "association|independent|chi.square|chi2|contingency|"
        "relationship.*categor|categor.*relationship",
        "chi_square",
        "two-sided",
    ),
    ("fisher", "fisher", "two-sided"),
    # One-sample
    (
        r"mean.*equal|equal.*mean|mean.*differ.*\d|differ.*\d.*mean|"
        r"average.*\d|\d.*average|test.*mean|mean.*test|"
        "is.*mean|population mean",
        "one_sample_ttest",
        "two-sided",
    ),
    # Paired
    (
        "before.*after|pre.*post|paired|repeated.*measure|"
        "within.*subject|change over time|same.*subject",
        "paired_ttest",
        "two-sided",
    ),
    # ANOVA / multi-group
    (
        "anova|more than two|three.*group|multiple.*group|"
        "several.*group|across.*group|among.*group",
        "anova",
        "two-sided",
    ),
    # Kruskal-Wallis
    ("kruskal|non.param.*group|group.*non.param", "kruskal_wallis", "two-sided"),
    # Two-sample
    (
        "compar.*mean|mean.*compar|differ.*group|group.*differ|"
        "between.*group|higher.*than|lower.*than|more.*than.*less.*than|"
        "male.*female|female.*male|group.*a.*group.*b|"
        "two.*group|independen.*sample",
        "two_sample_ttest",
        "two-sided",
    ),
]

_DIRECTION_KEYWORDS = {
    "greater": r"\bhigher\b|\bmore\b|\bgreater\b|\blarger\b|\bexceed\b|\babove\b",
    "less": r"\blower\b|\bless\b|\bsmaller\b|\bbelow\b|\bunder\b|\bfewer\b",
}

_TWO_SAMPLE_KEYS = {"two_sample_ttest", "mann_whitney", "anova", "kruskal_wallis"}

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class FallbackBackend(LLMBackend):
    """
    Pure regex routing — no LLM, no internet, no dependencies.

    Accuracy is lower than an LLM but it always works offline and is
    extremely fast.  Use it for quick experiments or when no LLM is
    available.
    """

    name = "fallback"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        The fallback backend does not call an LLM.
        ``route()`` is overridden directly instead.
        """
        return ""

    def route(
        self,
        question: str,
        schema: SchemaInfo,
        extra_context: str = "",
        warn_fallback: bool = True,
    ) -> RoutingResult:
        """Bypass LLM and route via regex rules."""
        import warnings

        if warn_fallback:
            warnings.warn(
                f'\n[HypoTestX] Using built-in regex fallback to route: "{question}"\n'
                "  Confidence is limited (~0.6). For better accuracy use a real LLM backend:\n"
                '    hx.analyze(df, question, backend="gemini", api_key="...")\n'
                '    hx.analyze(df, question, backend="ollama")  # free, offline\n'
                "  Suppress this with: warn_fallback=False",
                UserWarning,
                stacklevel=4,
            )
        r = RoutingResult()
        lower = question.lower()

        # ── Detect test type ────────────────────────────────────────────
        for pattern, test, default_alt in _TESTS_BY_KEYWORD:
            if re.search(pattern, lower):
                r.test = test
                r.alternative = default_alt
                break

        if not r.test:
            r.test = "two_sample_ttest"  # safest default

        # ── Detect direction ────────────────────────────────────────────
        # Skip for two-sample tests: group ordering is alphabetical (from
        # sorted unique values), so "greater" / "less" would be misleading
        # unless the first-mentioned group happens to be first alphabetically.
        if r.test not in _TWO_SAMPLE_KEYS:
            for direction, regex in _DIRECTION_KEYWORDS.items():
                if re.search(regex, lower):
                    r.alternative = direction
                    break

        # ── Detect mu for one-sample ────────────────────────────────────
        if r.test == "one_sample_ttest":
            m = re.search(r"(\d+\.?\d*)", question)
            if m:
                r.mu = float(m.group(1))

        # ── Map question words to schema columns ────────────────────────
        r.value_column, r.group_column = _match_columns(lower, schema)

        _CORRELATION_TESTS = {"pearson", "spearman", "point_biserial"}
        _CATEGORICAL_TESTS = {"chi_square", "fisher"}

        if r.test in _CORRELATION_TESTS:
            # For correlation, x and y should both be numeric columns.
            # _match_columns puts matched numerics in value_column (first match).
            # Pull a second numeric column from mentioned or from schema.
            mentioned_num = _mentioned_numerics(lower, schema)
            if len(mentioned_num) >= 2:
                r.x_column = mentioned_num[0]
                r.y_column = mentioned_num[1]
            elif len(mentioned_num) == 1:
                nums = list(schema.numerics.keys())
                r.x_column = mentioned_num[0]
                # pick first numeric that is different
                r.y_column = next((c for c in nums if c != r.x_column), None)
            else:
                nums = list(schema.numerics.keys())
                r.x_column = nums[0] if len(nums) > 0 else None
                r.y_column = nums[1] if len(nums) > 1 else None
            r.value_column = r.x_column
            r.group_column = r.y_column
        elif r.test in _CATEGORICAL_TESTS:
            # Both columns should be categorical
            mentioned_cat = _mentioned_categoricals(lower, schema)
            if len(mentioned_cat) >= 2:
                r.x_column = mentioned_cat[0]
                r.y_column = mentioned_cat[1]
            elif len(mentioned_cat) == 1:
                cats = list(schema.categoricals.keys())
                r.x_column = mentioned_cat[0]
                r.y_column = next((c for c in cats if c != r.x_column), None)
            else:
                cats = list(schema.categoricals.keys())
                r.x_column = cats[0] if len(cats) > 0 else None
                r.y_column = cats[1] if len(cats) > 1 else None
            r.group_column = r.x_column
            r.value_column = r.y_column
        else:
            r.x_column = r.group_column or r.value_column
            r.y_column = r.value_column if r.x_column != r.value_column else None

        r.reasoning = "(routed by regex fallback — no LLM used)"
        r.confidence = 0.6
        r.routing_source = "fallback"
        return r


# ---------------------------------------------------------------------------
# Column matching helpers
# ---------------------------------------------------------------------------


def _match_columns(
    question_lower: str,
    schema: SchemaInfo,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristically assign value_column (numeric) and group_column (categorical)
    by finding schema column names that appear in the question text.
    """
    mentioned_numeric = []
    mentioned_categ = []

    for col in schema.columns:
        col_l = col.lower().replace("_", " ")
        if col_l in question_lower or col.lower() in question_lower:
            if col in schema.numerics:
                mentioned_numeric.append(col)
            elif col in schema.categoricals:
                mentioned_categ.append(col)

    value_col = mentioned_numeric[0] if mentioned_numeric else None
    group_col = mentioned_categ[0] if mentioned_categ else None

    # If nothing matched explicitly, use first numeric / first categorical
    if value_col is None and schema.numerics:
        value_col = list(schema.numerics.keys())[0]
    if group_col is None and schema.categoricals:
        group_col = list(schema.categoricals.keys())[0]

    return value_col, group_col


def _mentioned_numerics(question_lower: str, schema: SchemaInfo) -> List[str]:
    """Return all numeric schema columns whose name appears in the question."""
    found = []
    for col in schema.columns:
        if col not in schema.numerics:
            continue
        col_l = col.lower().replace("_", " ")
        if col_l in question_lower or col.lower() in question_lower:
            found.append(col)
    return found


def _mentioned_categoricals(question_lower: str, schema: SchemaInfo) -> List[str]:
    """Return all categorical schema columns whose name appears in the question."""
    found = []
    for col in schema.columns:
        if col not in schema.categoricals:
            continue
        col_l = col.lower().replace("_", " ")
        if col_l in question_lower or col.lower() in question_lower:
            found.append(col)
    return found
