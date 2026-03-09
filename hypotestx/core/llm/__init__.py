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

from .base import LLMBackend, RoutingResult, SchemaInfo, CallableBackend
from .prompts import build_schema, build_system_prompt, build_user_prompt
from .backends import (
    OllamaBackend,
    OpenAICompatBackend,
    GeminiBackend,
    HuggingFaceBackend,
    FallbackBackend,
)

# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_STRING_MAP = {
    "ollama":       OllamaBackend,
    "openai":       lambda **kw: OpenAICompatBackend(provider="openai", **kw),
    "groq":         lambda **kw: OpenAICompatBackend(provider="groq",   **kw),
    "together":     lambda **kw: OpenAICompatBackend(provider="together", **kw),
    "perplexity":   lambda **kw: OpenAICompatBackend(provider="perplexity", **kw),
    "mistral":      lambda **kw: OpenAICompatBackend(provider="mistral",  **kw),
    "gemini":       GeminiBackend,
    "huggingface":  HuggingFaceBackend,
    "hf":           HuggingFaceBackend,
    "fallback":     FallbackBackend,
    "regex":        FallbackBackend,
    "none":         FallbackBackend,
}


def get_backend(spec: Any = None, **kwargs) -> LLMBackend:
    """
    Resolve *spec* to a concrete ``LLMBackend`` instance.

    *spec* can be:

    - ``None``              → FallbackBackend (no LLM, regex routing)
    - ``"ollama"``          → OllamaBackend(**kwargs)
    - ``"groq"``            → OpenAICompatBackend(provider="groq", **kwargs)
    - ``"openai"``          → OpenAICompatBackend(provider="openai", **kwargs)
    - ``"together"``        → OpenAICompatBackend(provider="together", **kwargs)
    - ``"perplexity"``      → OpenAICompatBackend(provider="perplexity", **kwargs)
    - ``"mistral"``         → OpenAICompatBackend(provider="mistral", **kwargs)
    - ``"gemini"``          → GeminiBackend(**kwargs)
    - ``"huggingface"``     → HuggingFaceBackend(**kwargs)
    - ``"fallback"``        → FallbackBackend()
    - An ``LLMBackend`` instance  → returned as-is
    - A ``callable``        → wrapped in CallableBackend

    Extra keyword arguments are forwarded to the backend constructor.
    """
    if spec is None:
        return FallbackBackend()

    if isinstance(spec, LLMBackend):
        return spec

    if callable(spec) and not isinstance(spec, type):
        return CallableBackend(spec)

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in _STRING_MAP:
            raise ValueError(
                f"Unknown backend '{spec}'. "
                f"Choose from: {', '.join(_STRING_MAP)}"
            )
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
