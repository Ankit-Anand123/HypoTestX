"""
LLM sub-package for HypoTestX.

Public API
----------
get_backend(spec, **kwargs) -> LLMBackend
    Resolve a backend from a string, class, instance, or callable.

build_schema(df) -> SchemaInfo
    Snapshot a DataFrame into a SchemaInfo for the prompt.
"""

from __future__ import annotations

from typing import Any

from .backends import (
    FallbackBackend,
    GeminiBackend,
    HuggingFaceBackend,
    OllamaBackend,
    OpenAICompatBackend,
)
from .base import CallableBackend, LLMBackend, RoutingResult, SchemaInfo
from .prompts import build_schema, build_system_prompt, build_user_prompt

# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_STRING_MAP = {
    "ollama": OllamaBackend,
    "openai": lambda **kw: OpenAICompatBackend(provider="openai", **kw),
    "groq": lambda **kw: OpenAICompatBackend(provider="groq", **kw),
    "together": lambda **kw: OpenAICompatBackend(provider="together", **kw),
    "perplexity": lambda **kw: OpenAICompatBackend(provider="perplexity", **kw),
    "mistral": lambda **kw: OpenAICompatBackend(provider="mistral", **kw),
    "azure": lambda **kw: OpenAICompatBackend(provider="azure", **kw),
    "gemini": GeminiBackend,
    "huggingface": HuggingFaceBackend,
    "hf": HuggingFaceBackend,
    "fallback": FallbackBackend,
    "regex": FallbackBackend,
    "none": FallbackBackend,
}


def get_backend(spec: Any = None, **kwargs) -> LLMBackend:
    """
    Resolve *spec* to a concrete ``LLMBackend`` instance.

    Parameters
    ----------
    spec : str | LLMBackend | callable | None
        - ``None`` / ``"fallback"``  → FallbackBackend (regex, offline)
        - ``"gemini"``               → GeminiBackend
        - ``"ollama"``               → OllamaBackend
        - ``"openai"``               → OpenAICompatBackend(provider="openai")
        - ``"groq"``                 → OpenAICompatBackend(provider="groq")
        - ``"together"``             → OpenAICompatBackend(provider="together")
        - ``"mistral"``              → OpenAICompatBackend(provider="mistral")
        - ``"perplexity"``           → OpenAICompatBackend(provider="perplexity")
        - ``"huggingface"``          → HuggingFaceBackend
        - An ``LLMBackend`` instance → returned as-is
        - A ``callable``             → wrapped in CallableBackend

    **kwargs
        Forwarded verbatim to the backend constructor.  Supported kwargs:

        .. list-table::
           :header-rows: 1
           :widths: 18 38 22

           * - kwarg
             - backends
             - default
           * - ``api_key``
             - gemini, openai, groq, together, ...
             - (required)
           * - ``model``
             - all
             - provider default
           * - ``timeout``
             - all
             - 60 s
           * - ``temperature``
             - gemini, openai-compat, huggingface
             - 0.0
           * - ``max_tokens``
             - gemini, openai-compat, huggingface
             - 512
           * - ``host``
             - ollama
             - localhost:11434
           * - ``options``
             - ollama
             - {"temperature": 0}
           * - ``token``
             - huggingface
             - (required)
           * - ``use_local``
             - huggingface
             - False
           * - ``device``
             - huggingface (local)
             - "cpu"
           * - ``base_url``
             - openai-compat
             - provider default
           * - ``provider``
             - openai-compat
             - "openai"
           * - ``extra_headers``
             - openai-compat
             - None

    Examples
    --------
    >>> from hypotestx.core.llm import get_backend
    >>> b = get_backend("gemini", api_key="AIza...", model="gemini-2.0-flash-lite")
    >>> b = get_backend("groq",   api_key="gsk_...", model="llama-3.3-70b-versatile")
    >>> b = get_backend("openai", api_key="sk-...",  model="gpt-4o", temperature=0.2)
    >>> b = get_backend("ollama", model="mistral", host="http://localhost:11434")
    >>> b = get_backend("huggingface", token="hf_...", model="HuggingFaceH4/zephyr-7b-beta")
    >>> b = get_backend("huggingface", model="microsoft/Phi-3.5-mini-instruct",
    ...                  use_local=True, device="cuda")
    >>> b = get_backend("together", api_key="...", model="meta-llama/Llama-3-70b-chat-hf")
    >>> b = get_backend("mistral",  api_key="...", model="mistral-large-latest")
    """
    if spec is None:
        return FallbackBackend()

    if isinstance(spec, LLMBackend):
        return spec

    # Duck-type: accept any object that exposes a .route() method even if it
    # doesn't formally inherit from LLMBackend (useful for testing stubs and
    # third-party wrappers).
    if hasattr(spec, "route") and callable(getattr(spec, "route")) and not isinstance(spec, type):
        return spec  # type: ignore[return-value]

    if callable(spec) and not isinstance(spec, type):
        return CallableBackend(spec)

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in _STRING_MAP:
            raise ValueError(f"Unknown backend '{spec}'. " f"Choose from: {', '.join(_STRING_MAP)}")
        cls_or_fn = _STRING_MAP[key]
        return cls_or_fn(**kwargs)

    if isinstance(spec, type) and issubclass(spec, LLMBackend):
        return spec(**kwargs)

    raise TypeError(
        f"backend must be None, a string, an LLMBackend instance, "
        f"or a callable — got {type(spec).__name__!r}"
    )


__all__ = [
    # Factory
    "get_backend",
    # Base classes / data classes
    "LLMBackend",
    "CallableBackend",
    "RoutingResult",
    "SchemaInfo",
    # Prompt helpers
    "build_schema",
    "build_system_prompt",
    "build_user_prompt",
    # Concrete backends
    "OllamaBackend",
    "OpenAICompatBackend",
    "GeminiBackend",
    "HuggingFaceBackend",
    "FallbackBackend",
]
