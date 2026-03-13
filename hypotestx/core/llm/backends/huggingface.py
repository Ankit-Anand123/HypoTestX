"""
HuggingFace backend — two modes:

1. Inference API  (``use_local=False``, default)
   - Free tier with rate limits
   - No GPU needed — model runs on HF servers
   - Get token: https://huggingface.co/settings/tokens

2. Local transformers  (``use_local=True``)
   - Requires:  pip install transformers torch
   - Model downloaded once to ~/.cache/huggingface
   - Runs on your CPU/GPU, completely offline after download

Recommended free models for routing:
    microsoft/Phi-3.5-mini-instruct   ~7 GB  best quality locally
    Qwen/Qwen2.5-3B-Instruct          ~6 GB  good balance
    TinyLlama/TinyLlama-1.1B-Chat-v1.0  ~2 GB  very fast, lower quality

Usage:
    # Inference API (cloud, free)
    result = hx.analyze(df, "...",
                         backend=HuggingFaceBackend(token="hf_..."))

    # Local inference
    result = hx.analyze(df, "...",
                         backend=HuggingFaceBackend(
                             model="microsoft/Phi-3.5-mini-instruct",
                             use_local=True))
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from ..base import LLMBackend

_HF_API_URL = "https://api-inference.huggingface.co/models"
_DEFAULT_API_MODEL = "HuggingFaceH4/zephyr-7b-beta"
_DEFAULT_LOCAL_MODEL = "microsoft/Phi-3.5-mini-instruct"


class HuggingFaceBackend(LLMBackend):
    """
    HuggingFace backend (Inference API or local transformers).

    Args:
        token:       HF access token (required for Inference API; optional locally).
        model:       Model repo ID.
        use_local:   If True, load the model locally via ``transformers``.
        timeout:     HTTP timeout for Inference API (default: 60).
        max_tokens:  Maximum new tokens (default: 512).
        device:      PyTorch device for local inference (``"cpu"`` or ``"cuda"``).
        load_kwargs: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained()``.
    """

    name = "huggingface"

    def __init__(
        self,
        token: str = "",
        model: str = "",
        use_local: bool = False,
        timeout: int = 60,
        max_tokens: int = 512,
        device: str = "cpu",
        load_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.token = token
        self.use_local = use_local
        self.model = model or (
            _DEFAULT_LOCAL_MODEL if use_local else _DEFAULT_API_MODEL
        )
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.device = device
        self.load_kwargs = load_kwargs or {}
        self._pipeline = None  # lazy-loaded local pipeline

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.use_local:
            return self._local_chat(messages)
        return self._api_chat(messages)

    # ------------------------------------------------------------------ #
    # Inference API path                                                   #
    # ------------------------------------------------------------------ #

    def _api_chat(self, messages: List[Dict[str, str]]) -> str:
        """Use the HuggingFace Inference API (text-generation task)."""
        # Flatten messages into a single prompt (chat template best-effort)
        prompt = _format_messages_as_prompt(messages)

        payload = json.dumps(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False,
                },
            }
        ).encode("utf-8")

        url = f"{_HF_API_URL}/{self.model}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 503:
                raise RuntimeError(
                    f"[HuggingFace] Model '{self.model}' is loading. "
                    "Wait ~20s and retry."
                ) from exc
            raise RuntimeError(f"[HuggingFace] HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"[HuggingFace] Connection error: {exc.reason}") from exc

        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text", str(data))
        return str(data)

    # ------------------------------------------------------------------ #
    # Local transformers path                                              #
    # ------------------------------------------------------------------ #

    def _local_chat(self, messages: List[Dict[str, str]]) -> str:
        """Run inference locally via the transformers library."""
        pipe = self._get_pipeline()

        try:
            # Modern chat-template API
            output = pipe(
                messages,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return output[0]["generated_text"]
        except TypeError:
            # Older models: flatten to text
            prompt = _format_messages_as_prompt(messages)
            output = pipe(
                prompt,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return output[0]["generated_text"]

    def _get_pipeline(self):
        """Lazy-load (and cache) the local transformers pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Local HuggingFace inference requires transformers + torch:\n"
                "  pip install transformers torch"
            ) from exc

        print(
            f"[HypoTestX] Loading '{self.model}' locally on {self.device}. "
            "First run downloads the model weights (~GB)."
        )
        kwargs = {"device": self.device, **self.load_kwargs}
        if self.token:
            kwargs["token"] = self.token
        self._pipeline = pipeline("text-generation", model=self.model, **kwargs)
        return self._pipeline

    def __repr__(self) -> str:
        mode = "local" if self.use_local else "api"
        return f"<HuggingFaceBackend model='{self.model}' mode='{mode}'>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_messages_as_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Simple chat-template fallback for models that don't support a message list.
    Uses ChatML format widely supported by open-source models.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
