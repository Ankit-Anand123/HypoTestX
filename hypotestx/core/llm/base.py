"""
LLM Backend abstractions for HypoTestX.

All LLM backends implement ``LLMBackend``.  The only method that must be
overridden is ``chat()``, which receives a list of OpenAI-style message dicts
and returns the assistant text string.

Custom backends
---------------
Subclass ``LLMBackend`` and implement ``chat()``, then pass an instance to
``analyze()``:

    class MyBackend(LLMBackend):
        def chat(self, messages):
            # call your model / API
            return response_text

    result = hx.analyze(df, "Is height correlated with weight?",
                         backend=MyBackend())

You can also pass any callable that accepts ``(messages: list) -> str``:

    result = hx.analyze(df, "...", backend=my_chat_fn)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Structured routing result
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """
    Structured intent extracted from a user question by an LLM or fallback
    parser.  The engine uses this to fetch the correct columns from the
    DataFrame and call the right test function.
    """

    # Which statistical test to run
    test: str = ""  # e.g. "two_sample_ttest"

    # Column names in the dataframe
    value_column: Optional[str] = None  # the numeric / response variable
    group_column: Optional[str] = None  # the grouping / categorical variable
    x_column: Optional[str] = None  # alias for first var in correlation
    y_column: Optional[str] = None  # alias for second var in correlation

    # Specific group labels to compare (subset of group_column's values)
    group_values: Optional[List[str]] = None  # e.g. ["Male", "Female"]

    # Test parameters
    alternative: str = "two-sided"  # "two-sided" | "greater" | "less"
    alpha: Optional[float] = None  # None means "use analyze()'s alpha"
    mu: Optional[float] = None  # hypothesised mean (one-sample t-test)
    equal_var: bool = False  # Student vs Welch
    correction: bool = True  # Yates correction for chi-square
    method: str = "parametric"  # "parametric" | "non-parametric"

    # Meta
    reasoning: str = ""  # LLM's explanation of its choice
    confidence: float = 1.0  # 0.0–1.0; LLM = 1.0, regex fallback = 0.6
    routing_source: str = "llm"  # "llm" or "fallback"
    raw_response: str = ""  # full LLM output for debugging


@dataclass
class SchemaInfo:
    """
    Summary of a DataFrame passed to the LLM as context.
    Built by ``build_schema()`` in prompts.py.
    """

    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    n_rows: int = 0
    # For categorical columns: unique values (up to 20)
    categoricals: Dict[str, List[Any]] = field(default_factory=dict)
    # For numeric columns: (min, max, mean)
    numerics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """
    Abstract LLM backend.

    Subclass this and implement ``chat()`` to integrate any LLM.
    The default ``route()`` method handles prompt building, JSON extraction,
    validation, and returning a ``RoutingResult`` — you only need to supply
    the raw API call.
    """

    # Name shown in error messages / repr
    name: str = "custom"

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a list of OpenAI-style messages and return the assistant reply.

        Args:
            messages: List of dicts with 'role' ('system' | 'user' | 'assistant')
                      and 'content' (str) keys.

        Returns:
            The model's text response.
        """

    # ------------------------------------------------------------------
    # Route — called by engine.py
    # ------------------------------------------------------------------

    def route(
        self,
        question: str,
        schema: "SchemaInfo",  # noqa: F821
        extra_context: str = "",
        warn_fallback: bool = True,
    ) -> RoutingResult:
        """
        Build prompts, call the LLM, parse the JSON response.

        This method is final — override ``chat()`` instead.
        """
        from .prompts import build_system_prompt, build_user_prompt  # local import

        system = build_system_prompt()
        user = build_user_prompt(question, schema, extra_context)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            raw = self.chat(messages)
        except Exception as exc:
            raise RuntimeError(
                f"[HypoTestX] LLM backend '{self.name}' raised an error: {exc}"
            ) from exc

        result = _parse_routing_json(raw)
        result.raw_response = raw
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend='{self.name}'>"


# ---------------------------------------------------------------------------
# Callable wrapper — lets users pass a plain function as backend
# ---------------------------------------------------------------------------


class CallableBackend(LLMBackend):
    """Wraps any ``callable(messages) -> str`` as an ``LLMBackend``."""

    name = "callable"

    def __init__(self, fn):
        if not callable(fn):
            raise TypeError("CallableBackend requires a callable(messages) -> str")
        self._fn = fn

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self._fn(messages)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _parse_routing_json(raw: str) -> RoutingResult:
    """
    Extract a ``RoutingResult`` from the raw LLM text.

    Tries (in order):
    1. Parse the first JSON code block  (```json ... ```)
    2. Parse bare JSON object  { ... }
    3. Regex fallback on key fields
    """
    text = raw.strip()

    # 1. code-fence block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return _json_to_result(m.group(1))

    # 2. bare JSON object (first { ... } that spans the whole answer)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return _json_to_result(m.group(0))

    # 3. partial regex fallback
    return _regex_fallback(text)


def _json_to_result(json_str: str) -> RoutingResult:
    """Parse a JSON string into a RoutingResult, tolerating minor issues."""
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        data: Dict = json.loads(cleaned)
    except json.JSONDecodeError:
        return _regex_fallback(json_str)

    r = RoutingResult()
    r.test = str(data.get("test", data.get("test_type", ""))).lower()
    r.value_column = data.get("value_column") or data.get("outcome_column")
    r.group_column = data.get("group_column") or data.get("grouping_column")
    r.x_column = data.get("x_column") or r.group_column
    r.y_column = data.get("y_column") or r.value_column
    r.group_values = data.get("group_values")
    r.alternative = str(data.get("alternative", "two-sided")).lower()
    r.alpha = float(data.get("alpha", 0.05))
    r.mu = float(data["mu"]) if "mu" in data and data["mu"] is not None else None
    r.equal_var = bool(data.get("equal_var", False))
    r.correction = bool(data.get("correction", True))
    r.method = str(data.get("method", "parametric")).lower()
    r.reasoning = str(data.get("reasoning", ""))
    r.confidence = float(data.get("confidence", 1.0))
    return r


_TEST_KEYWORDS = {
    "one_sample_ttest": ["one.sample", "one_sample", "single.sample"],
    "two_sample_ttest": ["two.sample", "two_sample", "independent", "unpaired"],
    "paired_ttest": ["paired", "repeated", "before.after", "within"],
    "anova": ["anova", "one.way", "multiple.group", "three.or.more"],
    "kruskal_wallis": ["kruskal", "kruskal.wallis"],
    "mann_whitney": ["mann", "whitney", "wilcoxon.rank"],
    "wilcoxon": ["wilcoxon.signed", "signed.rank"],
    "chi_square": [
        "chi.square",
        "chi2",
        "categorical",
        "contingency",
        "association",
        "independence",
    ],
    "fisher": ["fisher"],
    "pearson": ["pearson", "linear.corr"],
    "spearman": ["spearman", "rank.corr", "monotone"],
    "correlation": ["correlat"],
}


def _regex_fallback(text: str) -> RoutingResult:
    """Last-resort regex scan of the raw LLM output."""
    lower = text.lower()
    r = RoutingResult()
    for test, kws in _TEST_KEYWORDS.items():
        if any(re.search(kw, lower) for kw in kws):
            r.test = test
            break
    if "greater" in lower:
        r.alternative = "greater"
    elif "less" in lower:
        r.alternative = "less"
    r.reasoning = "(extracted via regex fallback)"
    return r
