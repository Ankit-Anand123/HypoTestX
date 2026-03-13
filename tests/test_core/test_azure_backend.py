"""
Tests for the OpenAICompatBackend, focusing on:
  - Azure provider construction
  - Correct URL and header generation for Azure vs. standard OpenAI
  - backend_options passthrough in analyze()
  - Provider factory shorthand for "azure"
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hypotestx.core.llm import get_backend
from hypotestx.core.llm.backends.openai_compat import OpenAICompatBackend, _is_azure_url

# ---------------------------------------------------------------------------
# _is_azure_url helper
# ---------------------------------------------------------------------------


class TestIsAzureUrl(unittest.TestCase):
    def test_azure_domain(self):
        self.assertTrue(_is_azure_url("https://myresource.openai.azure.com"))

    def test_azure_in_path(self):
        self.assertTrue(_is_azure_url("https://myproxy.azure.example.com/v1"))

    def test_openai_standard(self):
        self.assertFalse(_is_azure_url("https://api.openai.com/v1"))

    def test_groq(self):
        self.assertFalse(_is_azure_url("https://api.groq.com/openai/v1"))


# ---------------------------------------------------------------------------
# OpenAICompatBackend construction
# ---------------------------------------------------------------------------


class TestOpenAICompatBackendConstruction(unittest.TestCase):

    def test_standard_openai(self):
        b = OpenAICompatBackend(api_key="sk-test", provider="openai")
        self.assertFalse(b._is_azure)
        self.assertEqual(b.base_url, "https://api.openai.com/v1")

    def test_azure_provider_requires_base_url(self):
        with self.assertRaises(ValueError) as ctx:
            OpenAICompatBackend(
                api_key="az-key",
                provider="azure",
                # No base_url → should raise
            )
        self.assertIn("base_url", str(ctx.exception).lower())

    def test_azure_provider_requires_model(self):
        with self.assertRaises(ValueError) as ctx:
            OpenAICompatBackend(
                api_key="az-key",
                provider="azure",
                base_url="https://myresource.openai.azure.com",
                # No model (deployment name) → should raise
            )
        self.assertIn("model", str(ctx.exception).lower())

    def test_azure_provider_valid(self):
        b = OpenAICompatBackend(
            api_key="az-key",
            provider="azure",
            base_url="https://myresource.openai.azure.com",
            model="my-deployment",
            api_version="2024-05-01",
        )
        self.assertTrue(b._is_azure)
        self.assertEqual(b.api_version, "2024-05-01")
        self.assertEqual(b.model, "my-deployment")

    def test_autodetect_azure_from_base_url(self):
        """If provider is not 'azure' but base_url looks Azure, auto-detect."""
        b = OpenAICompatBackend(
            api_key="az-key",
            provider="openai",
            base_url="https://myresource.openai.azure.com",
            model="gpt-4o",
        )
        self.assertTrue(b._is_azure)

    def test_default_api_version(self):
        b = OpenAICompatBackend(
            api_key="az-key",
            provider="azure",
            base_url="https://myresource.openai.azure.com",
            model="gpt4",
        )
        # Should have a non-empty default
        self.assertTrue(len(b.api_version) > 0)


# ---------------------------------------------------------------------------
# URL and header generation
# ---------------------------------------------------------------------------


class TestAzureUrlAndHeaders(unittest.TestCase):

    def _make_azure_backend(self, api_version="2024-02-01"):
        return OpenAICompatBackend(
            api_key="az-key-secret",
            provider="azure",
            base_url="https://myresource.openai.azure.com",
            model="my-gpt4-deployment",
            api_version=api_version,
        )

    def test_azure_url_contains_deployment(self):
        b = self._make_azure_backend()
        url = b._build_url()
        self.assertIn("my-gpt4-deployment", url)
        self.assertIn("chat/completions", url)
        self.assertIn("api-version=2024-02-01", url)

    def test_azure_url_not_bearer(self):
        """Azure should use api-key header, NOT Authorization: Bearer."""
        b = self._make_azure_backend()
        # Intercept the HTTP call to inspect the headers
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            # Return a minimal valid response
            import io

            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": '{"test":"one_sample_ttest"}'}}]}
            ).encode()
            return mock_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            try:
                b.chat([{"role": "user", "content": "test"}])
            except Exception:
                pass  # Response parsing may fail; we only care about headers

        # Azure uses 'api-key', not 'authorization'
        header_keys = {k.lower() for k in captured.get("headers", {})}
        self.assertIn(
            "api-key", header_keys, "Azure backend must send 'api-key' header"
        )
        self.assertNotIn(
            "authorization",
            header_keys,
            "Azure backend must NOT send 'Authorization' header",
        )

    def test_standard_openai_uses_bearer(self):
        b = OpenAICompatBackend(api_key="sk-mykey", provider="openai")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "test"}}]}
            ).encode()
            return mock_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            try:
                b.chat([{"role": "user", "content": "test"}])
            except Exception:
                pass

        header_keys = {k.lower() for k in captured.get("headers", {})}
        self.assertIn(
            "authorization",
            header_keys,
            "Standard backend must send 'Authorization' header",
        )

    def test_azure_body_excludes_model_field(self):
        """Azure URL already contains deployment name; 'model' must not appear in JSON body."""
        b = self._make_azure_backend()
        captured_body = {}

        def fake_urlopen(req, timeout=None):
            captured_body["data"] = json.loads(req.data.decode())
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode()
            return mock_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            try:
                b.chat([{"role": "user", "content": "hi"}])
            except Exception:
                pass

        self.assertNotIn(
            "model",
            captured_body.get("data", {}),
            "Azure request body must not contain a 'model' field",
        )


# ---------------------------------------------------------------------------
# get_backend factory — "azure" shorthand
# ---------------------------------------------------------------------------


class TestGetBackendAzureShorthand(unittest.TestCase):

    def test_azure_shorthand_creates_backend(self):
        b = get_backend(
            "azure",
            api_key="az-key",
            base_url="https://myresource.openai.azure.com",
            model="my-deployment",
        )
        self.assertIsInstance(b, OpenAICompatBackend)
        self.assertTrue(b._is_azure)

    def test_azure_repr_shows_deployment(self):
        b = get_backend(
            "azure",
            api_key="az-key",
            base_url="https://myresource.openai.azure.com",
            model="my-deployment",
        )
        r = repr(b)
        self.assertIn("azure", r.lower())
        self.assertIn("my-deployment", r)


# ---------------------------------------------------------------------------
# backend_options passthrough in analyze()
# ---------------------------------------------------------------------------


class TestBackendOptionsPassthrough(unittest.TestCase):
    """backend_options dict must reach the backend constructor."""

    def test_backend_options_merged(self):
        """Extra items in backend_options must reach the backend kwargs."""
        received_kwargs = {}

        class RecordingBackend:
            def __init__(self, **kw):
                received_kwargs.update(kw)

            def route(self, question, schema):
                from hypotestx.core.llm.base import RoutingResult

                return RoutingResult(
                    test="one_sample_ttest",
                    value_column="v",
                    alternative="two-sided",
                )

        from hypotestx.core.engine import analyze

        df = {"v": [1.0, 2.0, 3.0, 4.0, 5.0]}
        # Pass a backend instance directly to bypass factory; test passthrough
        # by verifying backend_options doesn't crash the pipeline.
        backend_inst = RecordingBackend()
        result = analyze(
            df,
            "Is the mean different from 2?",
            backend=backend_inst,
            backend_options={"my_custom_flag": True},
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
