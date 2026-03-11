``analyze()`` — Natural Language Entry Point
============================================

.. autofunction:: hypotestx.core.engine.analyze

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``df``
     - ``DataFrame``
     - pandas or polars DataFrame containing the data to analyse.
   * - ``question``
     - ``str``
     - Plain-English hypothesis question, e.g.
       ``"Do males earn more than females?"``.
   * - ``backend``
     - ``str | LLMBackend | callable | None``
     - LLM to use for intent parsing. ``None`` (default) = built-in regex
       fallback. Accepts ``"gemini"``, ``"groq"``, ``"openai"``,
       ``"ollama"``, ``"azure"``, ``"together"``, ``"mistral"``,
       ``"perplexity"``, ``"huggingface"``, an ``LLMBackend`` instance,
       or any ``callable(messages) -> str``.
   * - ``alpha``
     - ``float``
     - Significance level. Default ``0.05``.
   * - ``verbose``
     - ``bool``
     - If ``True``, prints routing info and LLM reasoning to stdout.
   * - ``warn_fallback``
     - ``bool``
     - Emit a ``UserWarning`` when the regex fallback is used. Default ``True``.
   * - ``api_key``
     - ``str``
     - API key forwarded to the backend constructor.
   * - ``model``
     - ``str``
     - Model name/ID forwarded to the backend constructor.
   * - ``temperature``
     - ``float``
     - Sampling temperature (gemini / openai-compat / huggingface).
   * - ``max_tokens``
     - ``int``
     - Max tokens for the LLM response.
   * - ``timeout``
     - ``int``
     - HTTP timeout in seconds (default: 60).
   * - ``host``
     - ``str``
     - Ollama server URL (default: ``http://localhost:11434``).
   * - ``base_url``
     - ``str``
     - Override API base URL (openai-compat / azure).
   * - ``api_version``
     - ``str``
     - Azure API version (default: ``"2024-02-01"``).

Returns
-------
:class:`hypotestx.core.result.HypoResult`
    Full result object with statistic, p-value, effect size, confidence
    interval, routing metadata, and human-readable summary.

Examples
--------

.. code-block:: python

   import hypotestx as hx
   import pandas as pd

   df = pd.read_csv("survey.csv")

   # Regex fallback — no API key
   result = hx.analyze(df, "Do males earn more than females?")
   print(result.summary())

   # Gemini free tier
   result = hx.analyze(
       df,
       "Is there a salary difference between engineering and sales?",
       backend="gemini",
       api_key="AIza...",
       model="gemini-2.0-flash",
       temperature=0.0,
   )

   # Suppress fallback warning
   result = hx.analyze(df, "Is age correlated with salary?", warn_fallback=False)
