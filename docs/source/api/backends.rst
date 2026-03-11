LLM Backends
============

Abstract Base Class
-------------------

.. autoclass:: hypotestx.core.llm.base.LLMBackend
   :members:
   :show-inheritance:

.. autoclass:: hypotestx.core.llm.base.RoutingResult
   :members:
   :show-inheritance:

.. autoclass:: hypotestx.core.llm.base.SchemaInfo
   :members:
   :show-inheritance:

Callable Wrapper
----------------

.. autoclass:: hypotestx.core.llm.base.CallableBackend
   :members:
   :show-inheritance:

Built-in Regex Fallback
-----------------------

.. autoclass:: hypotestx.core.llm.backends.fallback.FallbackBackend
   :members:
   :show-inheritance:

Google Gemini
-------------

.. autoclass:: hypotestx.core.llm.backends.gemini.GeminiBackend
   :members:
   :show-inheritance:

OpenAI-Compatible (OpenAI / Groq / Together / Mistral / Perplexity / Azure)
----------------------------------------------------------------------------

.. autoclass:: hypotestx.core.llm.backends.openai_compat.OpenAICompatBackend
   :members:
   :show-inheritance:

Local Ollama
------------

.. autoclass:: hypotestx.core.llm.backends.ollama.OllamaBackend
   :members:
   :show-inheritance:

HuggingFace
-----------

.. autoclass:: hypotestx.core.llm.backends.huggingface.HuggingFaceBackend
   :members:
   :show-inheritance:

Backend Factory
---------------

.. autofunction:: hypotestx.core.llm.get_backend

.. autofunction:: hypotestx.core.llm.build_schema
