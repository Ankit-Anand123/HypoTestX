"""
Prompt templates for HypoTestX's LLM routing layer.

All prompts are plain strings so they are easy to read, audit, and override.
Nothing here calls any LLM — that is the backend's job.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SchemaInfo


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a statistical analysis routing expert embedded in the HypoTestX library.

Your ONLY job is to read a plain-English question about a dataset and decide
which statistical test to run and which columns to use.

## Available tests

| test key              | When to use |
|-----------------------|-------------|
| one_sample_ttest      | One group, test if mean equals a number (e.g. "Is the mean height 170?") |
| two_sample_ttest      | Two independent groups, compare means (e.g. "Do males earn more than females?") |
| paired_ttest          | Same subjects measured twice, compare before/after means |
| anova                 | Three or more independent groups, compare means |
| mann_whitney          | Two groups, non-parametric (use when data is non-normal or ordinal) |
| wilcoxon              | Two paired measurements, non-parametric |
| kruskal_wallis        | Three or more groups, non-parametric |
| chi_square            | Two categorical variables, test for association / independence |
| fisher                | Two categorical variables (2x2 table only), small sample sizes |
| pearson               | Two continuous variables, test linear correlation |
| spearman              | Two variables (one/both ordinal or non-normal), rank correlation |
| point_biserial        | One binary (0/1) variable and one continuous variable |

## Decision rules

1. If the question asks to "compare means" between exactly TWO named groups
   inside one column -> two_sample_ttest (or mann_whitney if ordinal/non-normal).
2. If THREE or more groups -> anova (or kruskal_wallis).
3. "Before and after", "pre and post", "change over time" on the SAME subjects -> paired_ttest.
4. "Association", "relationship", "independent" between two categorical columns -> chi_square.
5. "Correlation", "related to", "predict" between two numeric columns -> pearson or spearman.
6. "Is the mean equal to / greater than / less than {number}" -> one_sample_ttest.

## alternative field

- "two-sided" : question asks "different / any change"
- "greater"   : first group / variable is hypothesised HIGHER
- "less"      : first group / variable is hypothesised LOWER

## Output format

Reply with ONLY a JSON object — no markdown prose before/after it.

```json
{
  "test": "<test key from table above>",
  "value_column": "<name of the numeric response column, or null>",
  "group_column": "<name of the grouping/categorical column, or null>",
  "x_column": "<first variable for correlation, or null>",
  "y_column": "<second variable for correlation, or null>",
  "group_values": ["<group A label>", "<group B label>"] or null,
  "alternative": "two-sided" | "greater" | "less",
  "alpha": 0.05,
  "mu": <null or numeric value for one-sample test>,
  "equal_var": false,
  "method": "parametric" | "non-parametric",
  "reasoning": "<one sentence explaining your choice>"
}
```

Rules:
- Only use column names that exist in the provided schema.
- If a column does not exist, set it to null.
- Do NOT invent column names.
- Never explain or apologise — output JSON only.
"""


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def build_schema(df) -> "SchemaInfo":
    """
    Build a ``SchemaInfo`` snapshot from a DataFrame (pandas or polars).
    Works without importing pandas/polars at module level.
    """
    from .base import SchemaInfo

    info = SchemaInfo()

    # ── dict fallback (used in tests and simple scripts) ────────────────
    if isinstance(df, dict):
        info.columns = list(df.keys())
        first_col = info.columns[0] if info.columns else None
        info.n_rows = len(df[first_col]) if first_col is not None else 0
        for col, vals in df.items():
            # Detect numeric vs categorical
            non_null = [v for v in vals if v is not None]
            if non_null and isinstance(non_null[0], (int, float)):
                info.dtypes[col] = "float64"
                info.numerics[col] = {
                    "min":  float(min(non_null)),
                    "max":  float(max(non_null)),
                    "mean": float(sum(non_null) / len(non_null)),
                }
            else:
                info.dtypes[col] = "object"
                unique_vals = list(dict.fromkeys(str(v) for v in non_null))[:20]
                info.categoricals[col] = unique_vals
        return info

    info.n_rows = len(df)

    # Support both pandas and polars
    try:
        # pandas
        info.columns = list(df.columns)
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            info.dtypes[col] = dtype_str
            if dtype_str in ("object", "category", "string", "bool"):
                vals = [str(v) for v in df[col].dropna().unique()[:20]]
                info.categoricals[col] = vals
            elif "int" in dtype_str or "float" in dtype_str:
                series = df[col].dropna()
                if len(series) > 0:
                    info.numerics[col] = {
                        "min":  float(series.min()),
                        "max":  float(series.max()),
                        "mean": float(series.mean()),
                    }
    except AttributeError:
        # polars (has .schema dict)
        try:
            info.columns = df.columns
            for col, dtype in df.schema.items():
                dtype_str = str(dtype)
                info.dtypes[col] = dtype_str
                col_series = df[col].drop_nulls()
                if "Utf8" in dtype_str or "Categorical" in dtype_str or "Boolean" in dtype_str:
                    vals = [str(v) for v in col_series.unique().to_list()[:20]]
                    info.categoricals[col] = vals
                elif "Int" in dtype_str or "Float" in dtype_str:
                    if len(col_series) > 0:
                        info.numerics[col] = {
                            "min":  float(col_series.min()),
                            "max":  float(col_series.max()),
                            "mean": float(col_series.mean()),
                        }
        except Exception:
            pass

    return info


def build_system_prompt() -> str:
    """Return the system prompt (constant)."""
    return SYSTEM_PROMPT


def build_user_prompt(
    question: str,
    schema: "SchemaInfo",
    extra_context: str = "",
) -> str:
    """
    Build the user-turn prompt that includes the dataset schema and the
    question.
    """
    lines = [f"Dataset: {schema.n_rows} rows"]
    lines.append("")
    lines.append("Columns:")

    for col in schema.columns:
        dtype = schema.dtypes.get(col, "unknown")
        if col in schema.categoricals:
            uniques = schema.categoricals[col]
            preview = ", ".join(f'"{v}"' for v in uniques[:8])
            if len(uniques) > 8:
                preview += f", ... ({len(uniques)} unique)"
            lines.append(f"  - {col!r} [{dtype}]: {preview}")
        elif col in schema.numerics:
            n = schema.numerics[col]
            lines.append(
                f"  - {col!r} [{dtype}]: "
                f"min={n['min']:.2f}, max={n['max']:.2f}, mean={n['mean']:.2f}"
            )
        else:
            lines.append(f"  - {col!r} [{dtype}]")

    if extra_context:
        lines.append("")
        lines.append("Additional context:")
        lines.append(extra_context)

    lines.append("")
    lines.append(f'Question: "{question}"')
    lines.append("")
    lines.append("Reply with JSON only.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt fragments used by backends that need a simpler single-string prompt
# (e.g. completion-only models)
# ---------------------------------------------------------------------------

def build_completion_prompt(question: str, schema: "SchemaInfo") -> str:
    """
    Combines system + user into a single string for completion-style APIs.
    """
    system = build_system_prompt()
    user   = build_user_prompt(question, schema)
    return f"{system}\n\n---\n\n{user}\n\nJSON answer:\n"
