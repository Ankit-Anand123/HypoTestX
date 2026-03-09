"""
HypoTestX Dispatch Engine
=========================

``analyze(df, question, ...)`` is the single entry-point for the natural-language
interface.  It:

1. Resolves the backend (string shorthand → LLMBackend instance).
2. Builds a ``SchemaInfo`` snapshot of the DataFrame.
3. Asks the backend to parse the question into a ``RoutingResult``.
4. Extracts the required columns / groups from the DataFrame.
5. Calls the matching statistical test function.
6. Returns a ``HypoResult``.

The dispatcher supports both pandas and polars DataFrames, and gracefully
falls back to the regex-based ``FallbackBackend`` when no backend is given.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .result import HypoResult

# --------------------------------------------------------------------------- #
#  Internal helpers – DataFrame abstraction (pandas *or* polars, no import)   #
# --------------------------------------------------------------------------- #

def _col_to_list(df: Any, col: str) -> List:
    """
    Extract *col* from *df* and return a plain Python list.

    Works with pandas DataFrames, polars DataFrames, and any dict-like
    mapping (useful for unit tests).
    """
    if col not in _column_names(df):
        raise KeyError(
            f"Column '{col}' not found in DataFrame. "
            f"Available columns: {_column_names(df)}"
        )
    # pandas
    if hasattr(df, "iloc"):
        return df[col].tolist()
    # polars
    if hasattr(df, "to_pandas"):
        return df[col].to_list()
    # dict-like fallback (used in tests)
    return list(df[col])


def _column_names(df: Any) -> List[str]:
    """Return column names of *df* as a plain list of strings."""
    if hasattr(df, "columns"):
        cols = df.columns
        # pandas returns Index; polars returns list
        return list(cols)
    if hasattr(df, "keys"):
        return list(df.keys())
    raise TypeError(f"Cannot determine column names from {type(df).__name__!r}")


def _filter_df_by_value(df: Any, col: str, val: Any) -> Any:
    """Return a sub-DataFrame (or dict) where *col* == *val*."""
    if hasattr(df, "iloc"):  # pandas
        return df[df[col] == val]
    if hasattr(df, "filter"):  # polars
        return df.filter(df[col] == val)
    # dict fallback
    idx = [i for i, v in enumerate(df[col]) if v == val]
    return {c: [df[c][i] for i in idx] for c in df}


def _unique_values(df: Any, col: str) -> List:
    """Return sorted unique values in *col*."""
    vals = _col_to_list(df, col)
    seen: dict = {}
    for v in vals:
        seen[v] = True
    try:
        return sorted(seen.keys())
    except TypeError:
        return list(seen.keys())


def _extract_groups(
    df: Any,
    group_col: str,
    value_col: str,
    group_values: Optional[List] = None,
) -> Dict[Any, List[float]]:
    """
    Split *value_col* by *group_col*.

    Returns an ordered dict  { group_label: [values] }.
    If *group_values* is provided, only those groups are included (in order).
    """
    if group_values:
        keys = list(group_values)
    else:
        keys = _unique_values(df, group_col)

    groups: Dict[Any, List[float]] = {}
    for k in keys:
        sub = _filter_df_by_value(df, group_col, k)
        nums = _col_to_list(sub, value_col)
        groups[k] = [float(v) for v in nums]
    return groups


def _build_contingency_table(
    df: Any, row_col: str, col_col: str
) -> List[List[int]]:
    """
    Build a cross-tabulation (contingency table) from two categorical columns.

    Returns a 2-D list of counts  [ [n11, n12, ...], [n21, n22, ...], ... ]
    with rows corresponding to unique values of *row_col* and columns to
    unique values of *col_col*.
    """
    row_vals = _unique_values(df, row_col)
    col_vals = _unique_values(df, col_col)
    rv_list = _col_to_list(df, row_col)
    cv_list = _col_to_list(df, col_col)

    # Build freq map
    counts: Dict[Any, Dict[Any, int]] = {r: {c: 0 for c in col_vals} for r in row_vals}
    for r, c in zip(rv_list, cv_list):
        if r in counts and c in counts[r]:
            counts[r][c] += 1

    table = [[counts[r][c] for c in col_vals] for r in row_vals]
    return table


# --------------------------------------------------------------------------- #
#  Dispatch table                                                               #
# --------------------------------------------------------------------------- #

def _dispatch(routing, df: Any, alpha: float, verbose: bool) -> HypoResult:
    """
    Execute the test specified by *routing* against *df*.

    Imports are done lazily inside the function so that the top-level
    ``hypotestx`` package does not incur circular-import risks during loading.
    """
    from ..tests.parametric import (
        one_sample_ttest,
        two_sample_ttest,
        paired_ttest,
        anova_one_way,
    )
    from ..tests.nonparametric import (
        mann_whitney_u,
        wilcoxon_signed_rank,
        kruskal_wallis,
    )
    from ..tests.correlation import (
        pearson_correlation,
        spearman_correlation,
        point_biserial_correlation,
    )
    from ..tests.categorical import chi_square_test, fisher_exact_test

    test      = routing.test or "two_sample_ttest"
    alt       = routing.alternative or "two-sided"
    eff_alpha = routing.alpha if routing.alpha not in (None, 0.0) else alpha
    mu        = routing.mu if routing.mu is not None else 0.0
    equal_var = routing.equal_var if routing.equal_var is not None else True

    if verbose:
        print(f"[HypoTestX] Routing -> test={test!r}, confidence={routing.confidence:.2f}")
        if routing.reasoning:
            print(f"[HypoTestX] Reasoning: {routing.reasoning}")

    # ------------------------------------------------------------------ #
    # One-sample t-test                                                    #
    # ------------------------------------------------------------------ #
    if test == "one_sample_ttest":
        col = routing.value_column or routing.x_column
        if not col:
            raise ValueError("one_sample_ttest requires value_column in routing result")
        data = [float(v) for v in _col_to_list(df, col)]
        return one_sample_ttest(data, mu=mu, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Two-sample t-test                                                    #
    # ------------------------------------------------------------------ #
    if test in ("two_sample_ttest", "student_ttest", "welch_ttest"):
        groups = _resolve_two_groups(routing, df, test_name=test)
        g1, g2 = groups
        return two_sample_ttest(
            g1, g2,
            alpha=eff_alpha,
            alternative=alt,
            equal_var=(not test.startswith("welch")) and equal_var,
        )

    # ------------------------------------------------------------------ #
    # Paired t-test                                                        #
    # ------------------------------------------------------------------ #
    if test == "paired_ttest":
        x_col, y_col = _resolve_paired_columns(routing, df, test_name=test)
        x = [float(v) for v in _col_to_list(df, x_col)]
        y = [float(v) for v in _col_to_list(df, y_col)]
        return paired_ttest(x, y, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # One-way ANOVA                                                        #
    # ------------------------------------------------------------------ #
    if test in ("anova", "anova_one_way", "one_way_anova"):
        groups = _resolve_all_groups(routing, df, test_name=test)
        return anova_one_way(*groups, alpha=eff_alpha)

    # ------------------------------------------------------------------ #
    # Mann-Whitney U                                                       #
    # ------------------------------------------------------------------ #
    if test in ("mann_whitney", "mann_whitney_u"):
        groups = _resolve_two_groups(routing, df, test_name=test)
        g1, g2 = groups
        return mann_whitney_u(g1, g2, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Wilcoxon signed-rank                                                 #
    # ------------------------------------------------------------------ #
    if test in ("wilcoxon", "wilcoxon_signed_rank"):
        if routing.x_column and routing.y_column:
            x = [float(v) for v in _col_to_list(df, routing.x_column)]
            y = [float(v) for v in _col_to_list(df, routing.y_column)]
            return wilcoxon_signed_rank(x, y=y, mu=mu, alpha=eff_alpha, alternative=alt)
        col = routing.value_column or routing.x_column
        if not col:
            raise ValueError("wilcoxon_signed_rank requires value_column or x_column/y_column")
        data = [float(v) for v in _col_to_list(df, col)]
        return wilcoxon_signed_rank(data, mu=mu, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Kruskal-Wallis                                                       #
    # ------------------------------------------------------------------ #
    if test in ("kruskal_wallis", "kruskal"):
        groups = _resolve_all_groups(routing, df, test_name=test)
        return kruskal_wallis(*groups, alpha=eff_alpha)

    # ------------------------------------------------------------------ #
    # Chi-square                                                           #
    # ------------------------------------------------------------------ #
    if test in ("chi_square", "chi_square_test", "chi2"):
        x_col = routing.x_column or routing.group_column
        y_col = routing.y_column or routing.value_column
        if not x_col or not y_col:
            raise ValueError(
                "chi_square_test requires two categorical columns "
                "(x_column and y_column in routing result)"
            )
        table = _build_contingency_table(df, x_col, y_col)
        return chi_square_test(table, alpha=eff_alpha)

    # ------------------------------------------------------------------ #
    # Fisher's exact test                                                  #
    # ------------------------------------------------------------------ #
    if test in ("fisher", "fisher_exact", "fisher_exact_test"):
        x_col = routing.x_column or routing.group_column
        y_col = routing.y_column or routing.value_column
        if not x_col or not y_col:
            raise ValueError(
                "fisher_exact_test requires two categorical columns "
                "(x_column and y_column in routing result)"
            )
        table = _build_contingency_table(df, x_col, y_col)
        return fisher_exact_test(table, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Pearson correlation                                                  #
    # ------------------------------------------------------------------ #
    if test in ("pearson", "pearson_correlation"):
        x_col, y_col = _resolve_xy_columns(routing, df, test_name=test)
        x = [float(v) for v in _col_to_list(df, x_col)]
        y = [float(v) for v in _col_to_list(df, y_col)]
        return pearson_correlation(x, y, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Spearman correlation                                                 #
    # ------------------------------------------------------------------ #
    if test in ("spearman", "spearman_correlation"):
        x_col, y_col = _resolve_xy_columns(routing, df, test_name=test)
        x = [float(v) for v in _col_to_list(df, x_col)]
        y = [float(v) for v in _col_to_list(df, y_col)]
        return spearman_correlation(x, y, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Point-biserial correlation                                           #
    # ------------------------------------------------------------------ #
    if test in ("point_biserial", "point_biserial_correlation"):
        x_col, y_col = _resolve_xy_columns(routing, df, test_name=test)
        x = [float(v) for v in _col_to_list(df, x_col)]
        y = _col_to_list(df, y_col)  # binary col — no float() conversion
        return point_biserial_correlation(x, y, alpha=eff_alpha, alternative=alt)

    # ------------------------------------------------------------------ #
    # Unknown – fall back to two-sample t-test with a warning             #
    # ------------------------------------------------------------------ #
    import warnings
    warnings.warn(
        f"Unknown test key '{test}'; falling back to two_sample_ttest. "
        "If the routing was correct, please open an issue.",
        RuntimeWarning,
        stacklevel=4,
    )
    groups = _resolve_two_groups(routing, df, test_name="two_sample_ttest")
    g1, g2 = groups
    return two_sample_ttest(g1, g2, alpha=eff_alpha, alternative=alt)


# --------------------------------------------------------------------------- #
#  Column-resolution helpers                                                   #
# --------------------------------------------------------------------------- #

def _resolve_two_groups(routing, df: Any, test_name: str) -> List[List[float]]:
    """
    Return [group1_values, group2_values] from the routing result.

    Strategy:
    - If group_column + value_column are given: split value_column by
      group_column (taking the first 2 groups, or those in group_values).
    - If x_column + y_column are given: treat them as two paired numeric cols.
    """
    if routing.group_column and routing.value_column:
        gv = list(routing.group_values or [])
        groups_dict = _extract_groups(df, routing.group_column, routing.value_column, gv or None)
        vals = list(groups_dict.values())
        if len(vals) < 2:
            raise ValueError(
                f"{test_name}: need at least 2 groups in '{routing.group_column}', "
                f"found {len(vals)}: {list(groups_dict.keys())}"
            )
        return [vals[0], vals[1]]

    if routing.x_column and routing.y_column:
        x = [float(v) for v in _col_to_list(df, routing.x_column)]
        y = [float(v) for v in _col_to_list(df, routing.y_column)]
        return [x, y]

    raise ValueError(
        f"{test_name}: routing result must include either "
        "(group_column + value_column) or (x_column + y_column). "
        f"Got: {routing!r}"
    )


def _resolve_all_groups(routing, df: Any, test_name: str) -> List[List[float]]:
    """
    Return a list of group value lists (all groups found in group_column).
    """
    if routing.group_column and routing.value_column:
        gv = list(routing.group_values or [])
        groups_dict = _extract_groups(df, routing.group_column, routing.value_column, gv or None)
        return list(groups_dict.values())

    raise ValueError(
        f"{test_name}: routing result must include group_column + value_column. "
        f"Got: {routing!r}"
    )


def _resolve_paired_columns(routing, df: Any, test_name: str):
    """Return (x_col, y_col) names for paired tests."""
    if routing.x_column and routing.y_column:
        return routing.x_column, routing.y_column
    if routing.group_column and routing.value_column:
        # Some LLMs emit group/value instead of x/y for paired tests
        return routing.group_column, routing.value_column
    raise ValueError(
        f"{test_name}: routing result must include x_column + y_column. "
        f"Got: {routing!r}"
    )


def _resolve_xy_columns(routing, df: Any, test_name: str):
    """Return (x_col, y_col) names for correlation tests."""
    if routing.x_column and routing.y_column:
        return routing.x_column, routing.y_column
    if routing.value_column and routing.group_column:
        return routing.value_column, routing.group_column
    raise ValueError(
        f"{test_name}: routing result must include x_column + y_column. "
        f"Got: {routing!r}"
    )


# --------------------------------------------------------------------------- #
#  Public entry-point                                                          #
# --------------------------------------------------------------------------- #

_BACKEND_KWARGS = frozenset({
    # universal
    "api_key", "model", "timeout", "temperature", "max_tokens",
    # Ollama
    "host", "options",
    # HuggingFace
    "token", "use_local", "device", "load_kwargs",
    # OpenAI-compatible (groq / openai / together / mistral / perplexity)
    "base_url", "provider", "extra_headers",
})


def analyze(
    df: Any,
    question: str,
    backend: Any = None,
    alpha: float = 0.05,
    verbose: bool = False,
    **kwargs,
) -> HypoResult:
    """
    Natural-language hypothesis testing.

    Parses *question* in the context of *df*'s schema and automatically
    selects and executes the most appropriate statistical test.

    Parameters
    ----------
    df : pandas.DataFrame | polars.DataFrame
        The dataset to analyse.
    question : str
        A plain-English hypothesis question, e.g.
        ``"Do males earn more than females?"`` or
        ``"Is age correlated with salary?"``.
    backend : str | LLMBackend | callable | None
        LLM to use for intent parsing.
        - ``None`` (default) — fast regex-based FallbackBackend (no API key)
        - ``"ollama"``       — local Ollama (llama3.2 by default)
        - ``"gemini"``       — Google Gemini free tier
        - ``"groq"``         — Groq free tier (OpenAI-compatible)
        - ``"openai"``       — OpenAI API
        - Any ``LLMBackend`` subclass instance.
        - Any ``callable(messages) -> str``.

        Pass any backend constructor kwargs directly to ``analyze()``:

        .. list-table::
           :header-rows: 1

           * - kwarg
             - backends
             - notes
           * - ``api_key``
             - gemini, openai, groq, together, mistral, perplexity
             - required for cloud providers
           * - ``model``
             - all
             - override the default model name/ID
           * - ``timeout``
             - all (default: 60 s)
             - HTTP / inference timeout in seconds
           * - ``temperature``
             - gemini, openai-compat, huggingface
             - sampling temperature (0 = deterministic)
           * - ``max_tokens``
             - gemini, openai-compat, huggingface
             - max tokens in the LLM response
           * - ``host``
             - ollama
             - server URL (default ``http://localhost:11434``)
           * - ``options``
             - ollama
             - dict forwarded to Ollama model options
           * - ``token``
             - huggingface
             - HF access token for Inference API
           * - ``use_local``
             - huggingface
             - load model locally via ``transformers``
           * - ``device``
             - huggingface local
             - ``"cpu"`` or ``"cuda"``
           * - ``base_url``
             - openai-compat
             - override API base URL (e.g. Azure endpoint)
           * - ``provider``
             - openai-compat
             - shorthand: ``"groq"``, ``"together"``, ``"mistral"``, etc.
           * - ``extra_headers``
             - openai-compat
             - additional HTTP headers dict
    alpha : float
        Significance level (default 0.05).
    verbose : bool
        Print routing info and LLM reasoning to stdout.

    Returns
    -------
    HypoResult
        Full result object with statistic, p-value, effect size, decision,
        and human-readable summary.

    Examples
    --------
    >>> # Regex fallback — no API key, works offline
    >>> result = hx.analyze(df, "Do males earn more than females?")
    >>> print(result.summary())

    >>> # Gemini — free tier; pick any gemini-2.x model
    >>> result = hx.analyze(
    ...     df, "Is there a salary difference between genders?",
    ...     backend="gemini", api_key="AIza...",
    ...     model="gemini-2.0-flash",  # or "gemini-2.0-flash-lite"
    ...     temperature=0.0, max_tokens=512, timeout=30,
    ... )

    >>> # Groq — free tier, very fast
    >>> result = hx.analyze(
    ...     df, "Do departments differ in performance?",
    ...     backend="groq", api_key="gsk_...",
    ...     model="llama-3.3-70b-versatile",  # default; override freely
    ...     temperature=0.0, max_tokens=512,
    ... )

    >>> # OpenAI
    >>> result = hx.analyze(
    ...     df, "Is salary correlated with tenure?",
    ...     backend="openai", api_key="sk-...",
    ...     model="gpt-4o-mini",  # or "gpt-4o"
    ...     temperature=0.0, max_tokens=256,
    ... )

    >>> # Together AI / Mistral / Perplexity (OpenAI-compatible)
    >>> result = hx.analyze(
    ...     df, "Compare groups A and B",
    ...     backend="together", api_key="...",
    ...     model="meta-llama/Llama-3-70b-chat-hf",
    ... )

    >>> # Custom OpenAI-compatible endpoint (Azure, vLLM, LiteLLM, …)
    >>> result = hx.analyze(
    ...     df, "Compare groups",
    ...     backend="openai", api_key="...",
    ...     base_url="https://my-az-endpoint.openai.azure.com/openai/v1",
    ...     model="gpt-4o",
    ... )

    >>> # Ollama — local, no API key
    >>> result = hx.analyze(
    ...     df, "Compare groups A and B",
    ...     backend="ollama",
    ...     model="mistral",       # default: llama3.2
    ...     host="http://localhost:11434",
    ...     timeout=120,
    ... )

    >>> # HuggingFace Inference API (cloud, free tier)
    >>> result = hx.analyze(
    ...     df, "Are gender and department related?",
    ...     backend="huggingface", token="hf_...",
    ...     model="HuggingFaceH4/zephyr-7b-beta",
    ... )

    >>> # HuggingFace local (requires: pip install transformers torch)
    >>> result = hx.analyze(
    ...     df, "Is income different across regions?",
    ...     backend="huggingface",
    ...     model="microsoft/Phi-3.5-mini-instruct",
    ...     use_local=True, device="cuda",  # or "cpu"
    ... )

    >>> # Bring your own callable
    >>> result = hx.analyze(
    ...     df, "Is age correlated with salary?",
    ...     backend=lambda msgs: my_llm_fn(msgs[-1]["content"]),
    ... )
    """
    from .llm import get_backend, build_schema

    # Separate backend-constructor kwargs from test kwargs
    backend_kwargs = {k: v for k, v in kwargs.items() if k in _BACKEND_KWARGS}

    backend_instance = get_backend(backend, **backend_kwargs)

    schema = build_schema(df)

    if verbose:
        print(f"[HypoTestX] Schema: {schema.n_rows} rows, "
              f"columns: {schema.columns}")
        print(f"[HypoTestX] Backend: {type(backend_instance).__name__}")
        print(f"[HypoTestX] Question: {question!r}")

    routing = backend_instance.route(question, schema)

    return _dispatch(routing, df, alpha=alpha, verbose=verbose)
