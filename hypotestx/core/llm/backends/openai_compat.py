"""
OpenAI-compatible backend.

Works with any API that follows the OpenAI chat-completion format:
    - OpenAI            (https://platform.openai.com)
    - Groq              (https://console.groq.com)  — free tier, very fast
    - Together AI       (https://www.together.ai)   — many open-source models
    - Perplexity        (https://www.perplexity.ai)
    - Mistral AI        (https://mistral.ai)
    - Azure OpenAI      (set base_url to your Azure endpoint)
    - Local llama.cpp   (--server mode exposes OpenAI-compatible API)
    - vLLM, LiteLLM, etc.

Usage:
    # OpenAI
    result = hx.analyze(df, "...", backend="openai", api_key="sk-...")

    # Groq (free tier)
    result = hx.analyze(df, "...", backend="groq",
                         api_key="gsk_...",
                         model="llama-3.3-70b-versatile")

    # Generic OpenAI-compatible
    backend = OpenAICompatBackend(
        api_key   = "...",
        base_url  = "https://api.together.xyz/v1",
        model     = "meta-llama/Llama-3-70b-chat-hf",
    )
    result = hx.analyze(df, "...", backend=backend)
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional

from ..base import LLMBackend


# Known provider shorthand configs
_PROVIDER_CONFIGS: Dict[str, Dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "default_model": "llama-3.1-sonar-small-128k-online",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
    },
}


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible chat-completion API.

    Args:
        api_key:      API key / bearer token.
        base_url:     Base URL ending in ``/v1`` (e.g. ``https://api.groq.com/openai/v1``).
        model:        Model name.
        provider:     Shorthand: ``"openai"``, ``"groq"``, ``"together"``,
                      ``"perplexity"``, ``"mistral"``.  Sets base_url + model
                      automatically if not specified.
        timeout:      HTTP timeout in seconds (default: 60).
        temperature:  Sampling temperature (default: 0 for deterministic routing).
        max_tokens:   Maximum tokens in the response (default: 512).
        extra_headers: Additional HTTP headers dict.
    """

    name = "openai_compat"

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        model: str = "",
        provider: str = "openai",
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        cfg = _PROVIDER_CONFIGS.get(provider.lower(), _PROVIDER_CONFIGS["openai"])
        self.api_key     = api_key
        self.base_url    = (base_url or cfg["base_url"]).rstrip("/")
        self.model       = model or cfg["default_model"]
        self.provider    = provider
        self.timeout     = timeout
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.extra_headers = extra_headers or {}
        self.name        = provider.lower()

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI-compatible /chat/completions endpoint."""
        url     = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }).encode("utf-8")

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)

        req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"[{self.name}] HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"[{self.name}] Connection error: {exc.reason}"
            ) from exc

        return data["choices"][0]["message"]["content"]

    def __repr__(self) -> str:
        return (
            f"<OpenAICompatBackend provider='{self.provider}' "
            f"model='{self.model}'>"
        )
