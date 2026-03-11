# LLM Backends

HypoTestX uses LLM backends to parse plain-English questions into structured
routing decisions.  All backends implement the `LLMBackend` abstract base class —
you can swap them with a single keyword argument, or build your own.

---

## Backend Summary

| `backend=` string | Provider | Cost | Default model | Extra deps |
|---|---|---|---|---|
| `None` / `"fallback"` | Built-in regex | Free, offline | — | None |
| `"ollama"` | Local Ollama | Free, offline | `llama3.2` | Ollama app |
| `"gemini"` | Google Gemini | Free (1 500 req/day) | `gemini-2.0-flash` | None |
| `"groq"` | Groq Cloud | Free tier | `llama-3.3-70b-versatile` | None |
| `"openai"` | OpenAI | Paid | `gpt-4o-mini` | None |
| `"azure"` | Azure OpenAI | Paid | *(deployment name)* | None |
| `"together"` | Together AI | Free tier | `meta-llama/Llama-3-70b-chat-hf` | None |
| `"mistral"` | Mistral AI | Free tier | `mistral-small-latest` | None |
| `"perplexity"` | Perplexity AI | Free tier | `llama-3.1-sonar-small-128k-online` | None |
| `"huggingface"` | HF Inference API / local | Free tier / Local | `zephyr-7b-beta` | `transformers` (local only) |

---

## Common kwargs

All backends accept these keyword arguments via `hx.analyze()`:

| kwarg | applicable backends | description |
|---|---|---|
| `api_key` | gemini, groq, openai, together, mistral, perplexity, azure | Required for cloud providers |
| `model` | all | Override the default model name / ID |
| `temperature` | gemini, openai-compat, huggingface | Sampling temperature; `0` = deterministic |
| `max_tokens` | gemini, openai-compat, huggingface | Max tokens in the LLM response |
| `timeout` | all | HTTP timeout in seconds (default: `60`) |
| `host` | ollama | Server URL (default: `http://localhost:11434`) |
| `options` | ollama | Dict forwarded to Ollama model options |
| `token` | huggingface | HF access token for Inference API |
| `use_local` | huggingface | Load model locally via `transformers` |
| `device` | huggingface local | `"cpu"` or `"cuda"` |
| `base_url` | openai-compat, azure | Override the API base URL |
| `api_version` | azure | Azure API version (default: `"2024-02-01"`) |
| `extra_headers` | openai-compat | Additional HTTP headers dict |
| `backend_options` | all | Dict of extra backend-specific kwargs (passthrough) |

---

## Code Examples

### Regex Fallback (default, offline, no API key)

```python
import hypotestx as hx

result = hx.analyze(df, "Do males earn more than females?")
# Uses FallbackBackend automatically — no API key needed
# routing_confidence = 0.6
```

To suppress the routing warning:

```python
result = hx.analyze(df, "Do males earn more?", warn_fallback=False)
```

---

### Google Gemini

```python
import os, hypotestx as hx

result = hx.analyze(
    df,
    "Is there a salary difference between engineering and sales?",
    backend="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-2.0-flash",        # or "gemini-2.0-flash-lite"
    temperature=0.0,
    max_tokens=512,
)
```

---

### Groq (free tier, very fast)

```python
result = hx.analyze(
    df,
    "Is employee satisfaction correlated with tenure?",
    backend="groq",
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
    temperature=0.0,
)
```

---

### OpenAI

```python
result = hx.analyze(
    df,
    "Is salary correlated with years of experience?",
    backend="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",              # or "gpt-4o"
    temperature=0.0,
    max_tokens=256,
)
```

---

### Ollama (local, offline, free)

```python
result = hx.analyze(
    df,
    "Are there differences in performance scores across teams?",
    backend="ollama",
    model="phi4",                     # default: llama3.2
    host="http://localhost:11434",
    timeout=120,
)
```

---

### Azure OpenAI

```python
result = hx.analyze(
    df,
    "Do departments differ in performance?",
    backend="azure",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url="https://<resource>.openai.azure.com",
    model="<deployment-name>",
    api_version="2024-02-01",
)
```

---

### Together AI

```python
result = hx.analyze(
    df,
    "Do groups differ?",
    backend="together",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="meta-llama/Llama-3-70b-chat-hf",
)
```

---

### Mistral AI

```python
result = hx.analyze(
    df,
    "Is there an association between region and sales tier?",
    backend="mistral",
    api_key=os.environ["MISTRAL_API_KEY"],
    model="mistral-small-latest",
)
```

---

### Perplexity AI

```python
result = hx.analyze(
    df,
    "Compare satisfaction across customer segments",
    backend="perplexity",
    api_key=os.environ["PERPLEXITY_API_KEY"],
    model="llama-3.1-sonar-small-128k-online",
)
```

---

### HuggingFace Inference API (cloud, free tier)

```python
result = hx.analyze(
    df,
    "Are gender and department related?",
    backend="huggingface",
    token=os.environ["HF_TOKEN"],
    model="HuggingFaceH4/zephyr-7b-beta",
)
```

### HuggingFace Local

```bash
pip install transformers torch
```

```python
result = hx.analyze(
    df,
    "Is income different across regions?",
    backend="huggingface",
    model="microsoft/Phi-3.5-mini-instruct",
    use_local=True,
    device="cuda",   # or "cpu"
)
```

---

### Custom callable

Wrap any `callable(messages: list) -> str` as a backend:

```python
result = hx.analyze(
    df,
    "Is height correlated with weight?",
    backend=lambda msgs: my_llm_function(msgs[-1]["content"]),
)
```

---

### Custom LLMBackend subclass

Subclass `LLMBackend` to integrate any LLM that's not yet built-in:

```python
import hypotestx as hx

class MyCompanyLLM(hx.LLMBackend):
    name = "my_llm"

    def chat(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system", "content": ...},
                   {"role": "user",   "content": ...}]
        Must return a JSON string matching the RoutingResult schema.
        """
        prompt = messages[-1]["content"]
        return my_internal_api.complete(prompt)

result = hx.analyze(df, "Is satisfaction higher in Q4?", backend=MyCompanyLLM())
```

The `chat()` method only needs to return a valid JSON routing response — all
prompt construction, JSON extraction, and validation is handled by the base class
`route()` method.

---

## Custom OpenAI-compatible Endpoint

For self-hosted models (vLLM, LiteLLM, Ollama OpenAI mode, …):

```python
result = hx.analyze(
    df,
    "Compare groups",
    backend="openai",
    api_key="any-string",              # required field even if unused
    base_url="https://my-vllm/v1",
    model="my-fine-tuned-model",
)
```

---

## Security: API Key Best Practices

**Never hard-code API keys in source code or commit them to version control.**

```python
import os

# Load from environment
result = hx.analyze(
    df, "Do groups differ?",
    backend="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
)
```

With `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()   # reads .env file into os.environ
import os, hypotestx as hx

result = hx.analyze(df, "...", backend="groq",
                    api_key=os.environ["GROQ_API_KEY"])
```

Add `.env` to your `.gitignore` to prevent key leaks.
