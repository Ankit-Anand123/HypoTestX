"""
Google Gemini backend.

Free tier:  Gemini 2.0 Flash — 1 million token context, 1500 requests/day free.
Get a key:  https://aistudio.google.com/app/apikey (no credit card needed)

Usage:
    result = hx.analyze(df, "...", backend="gemini", api_key="AIza...")

    # Choose a specific model
    result = hx.analyze(df, "...",
                         backend=GeminiBackend(api_key="AIza...",
                                               model="gemini-2.0-flash-lite"))

Available free models (as of 2026):
    gemini-2.0-flash          fast, high quality — recommended default
    gemini-2.0-flash-lite     smallest / fastest / lowest quota cost
    gemini-1.5-pro            most capable (lower free quota)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from ..base import LLMBackend

_DEFAULT_MODEL = "gemini-2.0-flash"
_BASE_URL = "https://generativelanguage.googleapis.com/v1/models"


class GeminiBackend(LLMBackend):
    """
    Google Gemini backend via the Generative Language REST API.

    No SDK required — uses only the Python standard library.

    Args:
        api_key:     Google AI Studio API key.
        model:       Model name (default: ``gemini-2.0-flash``).
        timeout:     HTTP timeout seconds (default: 60).
        temperature: Sampling temperature (default: 0).
        max_tokens:  Maximum output tokens (default: 512).
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the Gemini ``generateContent`` endpoint.

        The OpenAI message list is converted to Gemini's ``contents`` format:
        ``system`` roles are prepended to the first user message text.
        """
        system_parts = []
        gemini_contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                text = (
                    "\n\n".join(system_parts) + "\n\n" + content
                    if system_parts
                    else content
                )
                system_parts = []  # consumed
                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": text}],
                    }
                )
            elif role == "assistant":
                gemini_contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": content}],
                    }
                )

        payload = json.dumps(
            {
                "contents": gemini_contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            }
        ).encode("utf-8")

        url = f"{_BASE_URL}/{self.model}:generateContent" f"?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"[Gemini] HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"[Gemini] Connection error: {exc.reason}") from exc

        # Extract text from response
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"[Gemini] Unexpected response format: {data}") from exc

    def __repr__(self) -> str:
        return f"<GeminiBackend model='{self.model}'>"
