"""
OpenAI-compatible backend.

Works with any API that follows the OpenAI chat-completion format:
    - OpenAI            (https://platform.openai.com)
    - Groq              (https://console.groq.com)  — free tier, very fast
    - Together AI       (https://www.together.ai)   — many open-source models
    - Perplexity        (https://www.perplexity.ai)
    - Mistral AI        (https://mistral.ai)
    - Azure OpenAI      (set provider="azure" or base_url to your Azure endpoint)
    - Local llama.cpp   (--server mode exposes OpenAI-compatible API)
    - vLLM, LiteLLM, etc.

Usage:
    # OpenAI
    result = hx.analyze(df, "...", backend="openai", api_key="sk-...")

    # Groq (free tier)
    result = hx.analyze(df, "...", backend="groq",
                         api_key="gsk_...",
                         model="llama-3.3-70b-versatile")

    # Azure OpenAI
    result = hx.analyze(df, "...", backend="azure",
                         api_key="<azure-api-key>",
                         base_url="https://<resource>.openai.azure.com",
                         model="<deployment-name>",
                         api_version="2024-02-01")

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
    # Azure OpenAI — base_url and model (deployment name) must be supplied by
    # the user; this entry only carries the default API version.
    "azure": {
        "base_url": "",          # must be provided: https://<resource>.openai.azure.com
        "default_model": "",     # must be provided: deployment name
        "api_version": "2024-02-01",
    },
}


def _is_azure_url(url: str) -> bool:
    """Return True when *url* looks like an Azure OpenAI endpoint."""
    return ".openai.azure.com" in url.lower() or "azure" in url.lower()


class OpenAICompatBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible chat-completion API.

    Args:
        api_key:      API key / bearer token.
                      For Azure this is the ``api-key`` header value.
        base_url:     Base URL ending in ``/v1`` (e.g. ``https://api.groq.com/openai/v1``).
                      For Azure: ``https://<resource>.openai.azure.com`` (no trailing path).
        model:        Model name.  For Azure this is the *deployment name*.
        provider:     Shorthand: ``"openai"``, ``"groq"``, ``"together"``,
                      ``"perplexity"``, ``"mistral"``, ``"azure"``.
                      Sets base_url + model automatically if not specified.
        timeout:      HTTP timeout in seconds (default: 60).
        temperature:  Sampling temperature (default: 0 for deterministic routing).
        max_tokens:   Maximum tokens in the response (default: 512).
        extra_headers: Additional HTTP headers dict.
        api_version:  Azure API version string (default: ``"2024-02-01"``).
                      Only used when provider is ``"azure"`` or base_url is an
                      Azure endpoint.
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
        api_version: str = "",
    ):
        cfg = _PROVIDER_CONFIGS.get(provider.lower(), _PROVIDER_CONFIGS["openai"])
        self.api_key     = api_key
        self.base_url    = (base_url or cfg.get("base_url", "")).rstrip("/")
        self.model       = model or cfg.get("default_model", "")
        self.provider    = provider
        self.timeout     = timeout
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.extra_headers = extra_headers or {}
        self.name        = provider.lower()

        # Azure-specific: determine whether this is an Azure endpoint
        self._is_azure = (
            provider.lower() == "azure"
            or _is_azure_url(self.base_url)
        )
        # API version (only meaningful for Azure)
        self.api_version = (
            api_version
            or cfg.get("api_version", "")
            or "2024-02-01"
        )

        if self._is_azure:
            if not self.base_url:
                raise ValueError(
                    "Azure OpenAI requires base_url "
                    "(e.g. 'https://<resource>.openai.azure.com'). "
                    "Pass base_url='https://...' to get_backend() or OpenAICompatBackend()."
                )
            if not self.model:
                raise ValueError(
                    "Azure OpenAI requires model to be set to the deployment name. "
                    "Pass model='<deployment-name>' to get_backend() or OpenAICompatBackend()."
                )

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def _build_url(self) -> str:
        """Return the fully-qualified chat/completions URL for this provider."""
        if self._is_azure:
            # Azure URL format:
            # https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<ver>
            return (
                f"{self.base_url}/openai/deployments/{self.model}"
                f"/chat/completions?api-version={self.api_version}"
            )
        return f"{self.base_url}/chat/completions"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI-compatible /chat/completions endpoint."""
        url = self._build_url()

        # Azure uses a fixed deployment name in the URL, so it must not
        # appear again in the JSON body.
        body: Dict = {
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        if not self._is_azure:
            body["model"] = self.model

        payload = json.dumps(body).encode("utf-8")

        if self._is_azure:
            # Azure authenticates with the api-key header (not Bearer token)
            headers = {
                "Content-Type": "application/json",
                "api-key":      self.api_key,
            }
        else:
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
            body_txt = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"[{self.name}] HTTP {exc.code}: {body_txt}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"[{self.name}] Connection error: {exc.reason}"
            ) from exc

        return data["choices"][0]["message"]["content"]

    def __repr__(self) -> str:
        if self._is_azure:
            return (
                f"<OpenAICompatBackend provider='azure' "
                f"deployment='{self.model}' api_version='{self.api_version}'>"
            )
        return (
            f"<OpenAICompatBackend provider='{self.provider}' "
            f"model='{self.model}'>"
        )
