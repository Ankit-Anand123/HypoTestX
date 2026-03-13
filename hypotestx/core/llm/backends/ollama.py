"""
Ollama backend — free, local, open-source LLM inference.

Install:  https://ollama.com  (Windows/Mac/Linux)
Run:      ollama serve          (starts on http://localhost:11434)
Pull:     ollama pull llama3.2  (or mistral, gemma2, phi4, etc.)

Usage:
    result = hx.analyze(df, "...", backend="ollama")
    result = hx.analyze(df, "...", backend="ollama", model="mistral")
    result = hx.analyze(df, "...", backend=OllamaBackend(model="gemma2"))

Recommended free models (small but capable):
    llama3.2     ~2 GB  fastest
    mistral      ~4 GB  good quality
    gemma2       ~5 GB  very accurate
    phi4         ~9 GB  best reasoning
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from ..base import LLMBackend

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_HOST = "http://localhost:11434"
_CHAT_ENDPOINT = "/api/chat"
_TAGS_ENDPOINT = "/api/tags"


class OllamaBackend(LLMBackend):
    """
    Ollama backend — fully local, zero API cost.

    Args:
        model:   Ollama model name (default: ``llama3.2``).
        host:    Base URL of the Ollama server (default: ``http://localhost:11434``).
        timeout: Request timeout in seconds (default: 120).
        options: Extra Ollama model options dict, e.g. ``{"temperature": 0}``.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        timeout: int = 120,
        options: Optional[Dict] = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.options = options or {"temperature": 0}

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat request to the local Ollama server."""
        self._check_server()

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self.options,
            }
        ).encode("utf-8")

        url = self.host + _CHAT_ENDPOINT
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"[Ollama] Could not reach {url}. " "Make sure Ollama is running: `ollama serve`"
            ) from exc

        return data["message"]["content"]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _check_server(self) -> None:
        """Raise a helpful error if Ollama is not reachable."""
        try:
            urllib.request.urlopen(self.host + _TAGS_ENDPOINT, timeout=5)
        except Exception:
            raise RuntimeError(
                "[Ollama] Server not reachable at "
                f"{self.host}.\n"
                "  1. Install Ollama: https://ollama.com\n"
                "  2. Start it:       ollama serve\n"
                f"  3. Pull a model:   ollama pull {self.model}"
            )

    def available_models(self) -> List[str]:
        """Return list of locally available model names."""
        try:
            url = self.host + _TAGS_ENDPOINT
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def auto_select_model(self) -> str:
        """
        Pick the best locally available model.
        Preference order: phi4, gemma2, mistral, llama3.2, (anything else).
        """
        available = self.available_models()
        if not available:
            return self.model  # let the request fail with a clear error
        preference = ["phi4", "gemma2", "mistral", "llama3.2"]
        for pref in preference:
            for m in available:
                if pref in m:
                    return m
        return available[0]

    def __repr__(self) -> str:
        return f"<OllamaBackend model='{self.model}' host='{self.host}'>"
