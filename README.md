# HypoTestX

**Natural Language Hypothesis Testing — Powered by LLMs or Pure Regex**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/Ankit-Anand123/HypoTestX)
[![Tests](https://img.shields.io/badge/tests-532%20passing-brightgreen.svg)](https://github.com/Ankit-Anand123/HypoTestX)
[![Version](https://img.shields.io/badge/version-1.0.5-blue.svg)](https://pypi.org/project/hypotestx/)

> **Ask a statistical question in plain English. HypoTestX routes it to the right test — with or without an LLM.**

HypoTestX gives you two ways to run hypothesis tests:

- **Direct API** — call any of 12 statistical tests explicitly with full parameter control.
- **Natural language interface** — pass a plain-English question and a DataFrame to `analyze()`. HypoTestX parses the intent (via a regex fallback or a real LLM), picks the right test, extracts the right columns, and returns a full `HypoResult`.

The mathematical core is **pure Python** — no NumPy, no SciPy, no compiled extensions required.

---

## Key Features

### Natural Language Interface — `analyze()`
```python
import hypotestx as hx
import pandas as pd

df = pd.read_csv('survey.csv')

# Zero config — built-in regex router, no API key needed
result = hx.analyze(df, "Do males earn more than females?")
print(result.summary())
```

### Plug-in LLM Backends
Swap in any LLM with a single keyword argument — no code changes:

```python
# Google Gemini (free tier, 1500 req/day) — pick any gemini-2.x model
result = hx.analyze(df, "Is age correlated with salary?",
                    backend="gemini", api_key="AIza...",
                    model="gemini-2.0-flash")   # or "gemini-2.0-flash-lite"

# Groq (free tier, OpenAI-compatible) — pick any supported model
result = hx.analyze(df, "Is there an association between gender and dept?",
                    backend="groq", api_key="gsk_...",
                    model="llama-3.3-70b-versatile")

# OpenAI — specify model and token budget
result = hx.analyze(df, "Do groups differ?",
                    backend="openai", api_key="sk-...",
                    model="gpt-4o-mini", temperature=0.0)

# Local Ollama (completely offline) — choose any pulled model
result = hx.analyze(df, "Compare satisfaction across regions?",
                    backend="ollama", model="mistral")

# Bring your own callable
result = hx.analyze(df, "Any question",
                    backend=lambda msgs: my_llm(msgs[-1]["content"]))
```

### Pure Python Mathematics
- **Zero dependencies** for all statistical computations
- All test functions and distributions implemented from scratch
- Complete transparency — read the source to see exactly how statistics work
- All LLM HTTP calls use only `urllib.request` from the standard library

### Dual Mode Design
```python
# Natural language — let HypoTestX choose the test
hx.analyze(df, "Is there a difference between group A and B?")

# Direct API — explicit control over every parameter
hx.ttest_2samp(group1, group2, equal_var=False, alpha=0.01)
```

### Comprehensive Statistical Toolkit
- **Parametric tests**: one-sample, two-sample, paired t-tests, one-way ANOVA
- **Non-parametric tests**: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis
- **Categorical tests**: chi-square (independence + GoF), Fisher's exact
- **Correlation**: Pearson, Spearman, point-biserial
- **Effect sizes**: Cohen's d, eta-squared, Cramer's V, rank-biserial r
- **Power analysis**: sample size calculations, post-hoc power

---

## Quick Start

### Installation

```bash
pip install hypotestx
```

No mandatory external dependencies — all statistical maths and HTTP calls are pure Python stdlib.  
Optional extras:

```bash
# For local Ollama backend (free, offline)
# 1. Install Ollama from https://ollama.com
# 2. Pull a model:  ollama pull llama3.2

# For HuggingFace local inference (optional)
pip install transformers torch

# For visualization helpers (optional)
pip install matplotlib
```

### Basic Usage

```python
import hypotestx as hx
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Ask questions naturally — no API key required (regex fallback)
result = hx.analyze(df, "Do customers in region A spend more than region B?")

# Get comprehensive results
print(result.summary())
# [ Welch's t-test (unequal variances) ]
# =========================================
# Statistic (t):   3.2456
# p-value:         0.0012
# Significant:     Yes (alpha = 0.05)
# Effect size (d): 0.6834   (medium)
# 95% CI:          [1.23, 4.56]

# Access individual values
print(result.p_value)          # 0.0012
print(result.effect_size)      # 0.6834
print(result.is_significant)   # True
```

---

## Examples

### One-Sample t-test
```python
# Natural language
result = hx.analyze(df, "Is the average score different from 75?")

# Direct API
result = hx.ttest_1samp(df['scores'].tolist(), mu=75, alternative='two-sided')
```

### Two-Sample t-test
```python
# Natural language — columns detected from schema
result = hx.analyze(df, "Do males have higher income than females?")

# Direct API with full control
males   = df[df['gender'] == 'M']['income'].tolist()
females = df[df['gender'] == 'F']['income'].tolist()
result  = hx.ttest_2samp(males, females, alternative='greater', equal_var=False)
```

### Paired t-test
```python
# Natural language
result = hx.analyze(df, "Did scores improve from pre_score to post_score?")

# Direct API
result = hx.ttest_paired(df['pre_score'].tolist(), df['post_score'].tolist(),
                         alternative='less')
```

### Correlation
```python
# Natural language
result = hx.analyze(df, "Is age correlated with salary?")

# Direct API
result = hx.pearson(df['age'].tolist(), df['salary'].tolist())
```

### Categorical Association
```python
# Natural language
result = hx.analyze(df, "Is there an association between gender and department?")

# Direct API
import hypotestx as hx
table = [[30, 10], [20, 40]]   # 2x2 contingency table
result = hx.chi2_test(table)
```

### Using a Real LLM Backend
```python
import hypotestx as hx
import pandas as pd

df = pd.read_csv('employees.csv')

# Gemini free tier — best out-of-the-box accuracy
result = hx.analyze(
    df,
    "Is there a salary difference between engineering and sales departments?",
    backend="gemini",
    api_key="AIza...",
    model="gemini-2.0-flash",   # or "gemini-2.0-flash-lite" for faster/cheaper
    temperature=0.0,
)
print(result.summary())

# Groq free tier (OpenAI-compatible, very fast)
result = hx.analyze(
    df,
    "Is employee satisfaction correlated with tenure?",
    backend="groq",
    api_key="gsk_...",
    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
)

# Local Ollama (fully offline, no API key)
result = hx.analyze(
    df,
    "Are there differences in performance scores across teams?",
    backend="ollama",           # uses llama3.2 by default
    model="phi4",               # override model
)
```

---

## Natural Language Examples

`analyze()` understands plain English. The built-in regex fallback handles the patterns below with no API key. A real LLM backend handles arbitrarily complex phrasings.

### Two-group comparisons
```python
hx.analyze(df, "Do males spend more than females?")
hx.analyze(df, "Is there a difference between group A and group B?")
hx.analyze(df, "Are premium customers different from basic customers?")
hx.analyze(df, "Test whether method 1 is better than method 2")
```

### One-sample tests
```python
hx.analyze(df, "Is the average score different from 100?")
hx.analyze(df, "Test if the mean equals 50")
hx.analyze(df, "Is the average significantly greater than 75?")
```

### Correlation & relationships
```python
hx.analyze(df, "Is there a correlation between age and income?")
hx.analyze(df, "Is salary related to years of experience?")
hx.analyze(df, "Does age predict salary?")
```

### Categorical associations
```python
hx.analyze(df, "Are gender and department independent?")
hx.analyze(df, "Is there an association between treatment and outcome?")
hx.analyze(df, "Are product preference and region related?")
```

### Multi-group comparisons
```python
hx.analyze(df, "Compare satisfaction scores across all regions")
hx.analyze(df, "Are there differences in performance across three teams?")
```

### Paired / before-after
```python
hx.analyze(df, "Did scores improve from pre_score to post_score?")
hx.analyze(df, "Compare before and after treatment")
```

---

## Supported Tests

### Parametric
| Test | NL phrase examples | Direct function |
|---|---|---|
| One-sample t-test | "Is the mean different from 100?" | `ttest_1samp()` |
| Two-sample t-test | "Do groups differ?" | `ttest_2samp()` |
| Welch's t-test | "Compare (unequal variances)" | `welch_ttest()` |
| Paired t-test | "Did scores change?" | `ttest_paired()` |
| One-way ANOVA | "Compare three or more groups" | `anova_1way()` |

### Non-Parametric
| Test | NL phrase examples | Direct function |
|---|---|---|
| Mann-Whitney U | "Compare (non-normal data)" | `mannwhitney()` |
| Wilcoxon signed-rank | "Paired (non-normal)" | `wilcoxon()` |
| Kruskal-Wallis | "Multiple groups (non-normal)" | `kruskal()` |

### Categorical
| Test | NL phrase examples | Direct function |
|---|---|---|
| Chi-square | "Are the variables independent?" | `chi2_test()` |
| Fisher's exact | "2x2 table, small sample" | `fisher_exact()` |

### Correlation
| Test | NL phrase examples | Direct function |
|---|---|---|
| Pearson | "Linear relationship between X and Y?" | `pearson()` |
| Spearman | "Monotonic / rank correlation?" | `spearman()` |
| Point-biserial | "Continuous vs binary?" | `pointbiserial()` |

---

## Advanced Features

### LLM Backends

All backends require zero extra dependencies except where noted.

| Backend string | Provider | Cost | Default model | Dependencies |
|---|---|---|---|---|
| `None` / `"fallback"` | Built-in regex router | Free, offline | — | None |
| `"ollama"` | Local Ollama | Free, offline | `llama3.2` | Ollama app |
| `"gemini"` | Google Gemini | Free (1500 req/day) | `gemini-2.0-flash` | None |
| `"groq"` | Groq Cloud | Free tier | `llama-3.3-70b-versatile` | None |
| `"openai"` | OpenAI | Paid | `gpt-4o-mini` | None |
| `"azure"` | Azure OpenAI | Paid | *(deployment name)* | None |
| `"together"` | Together AI | Free tier | `meta-llama/Llama-3-70b-chat-hf` | None |
| `"mistral"` | Mistral AI | Free tier | `mistral-small-latest` | None |
| `"perplexity"` | Perplexity AI | Free tier | `llama-3.1-sonar-small-128k-online` | None |
| `"huggingface"` | HF Inference API or local | Free tier / Local | `zephyr-7b-beta` | `transformers` (local only) |

All extra kwargs are passed directly to the backend constructor via `hx.analyze()`:

| kwarg | backends | notes |
|---|---|---|
| `model` | all | override the default model name |
| `temperature` | gemini, openai-compat, huggingface | sampling temperature (0 = deterministic) |
| `max_tokens` | gemini, openai-compat, huggingface | max tokens in the LLM response |
| `timeout` | all | HTTP timeout in seconds (default: 60) |
| `host` | ollama | server URL (default `http://localhost:11434`) |
| `options` | ollama | dict forwarded to Ollama model options |
| `token` | huggingface | HF access token for Inference API |
| `use_local` | huggingface | load model locally via `transformers` |
| `device` | huggingface (local) | `"cpu"` or `"cuda"` |
| `base_url` | openai-compat, azure | override API base URL |
| `api_version` | azure | Azure API version (default: `"2024-02-01"`) |
| `extra_headers` | openai-compat | additional HTTP headers dict |
| `backend_options` | all | dict of extra kwargs passed through to the backend constructor (useful for provider-specific settings not in the table above) |

```python
import hypotestx as hx

# --- Gemini (free tier) ---
result = hx.analyze(df, "Is age correlated with salary?",
                    backend="gemini", api_key="AIza...",
                    model="gemini-2.0-flash",       # or "gemini-2.0-flash-lite"
                    temperature=0.0, max_tokens=512)

# --- Groq (free tier, very fast) ---
result = hx.analyze(df, "Compare departments",
                    backend="groq", api_key="gsk_...",
                    model="llama-3.3-70b-versatile") # or "mixtral-8x7b-32768"

# --- OpenAI ---
result = hx.analyze(df, "Is salary correlated with tenure?",
                    backend="openai", api_key="sk-...",
                    model="gpt-4o-mini",             # or "gpt-4o"
                    temperature=0.0, max_tokens=256)

# --- Together AI / Mistral / Perplexity ---
result = hx.analyze(df, "Do groups differ?",
                    backend="together", api_key="...",
                    model="meta-llama/Llama-3-70b-chat-hf")

# --- Custom OpenAI-compatible endpoint (vLLM, LiteLLM, …) ---
result = hx.analyze(df, "Compare groups",
                    backend="openai", api_key="...",
                    base_url="https://my-self-hosted-llm/v1",
                    model="gpt-4o")

# --- Azure OpenAI ---
# Requires: base_url (your resource endpoint) + model (deployment name)
result = hx.analyze(
    df, "Do departments differ in performance?",
    backend="azure",
    api_key="<azure-api-key>",            # the resource api-key, NOT a Bearer token
    base_url="https://<resource>.openai.azure.com",
    model="<deployment-name>",            # the deployment name, e.g. "gpt-4o"
    api_version="2024-02-01",             # optional, defaults to "2024-02-01"
)

# --- Ollama (local, offline) ---
result = hx.analyze(df, "Do males earn more?",
                    backend="ollama",
                    model="mistral",                 # default: llama3.2
                    host="http://localhost:11434", timeout=120)

# --- HuggingFace Inference API ---
result = hx.analyze(df, "Are departments different?",
                    backend="huggingface", token="hf_...",
                    model="HuggingFaceH4/zephyr-7b-beta")

# --- HuggingFace local (requires: pip install transformers torch) ---
result = hx.analyze(df, "Is income different across regions?",
                    backend="huggingface",
                    model="microsoft/Phi-3.5-mini-instruct",
                    use_local=True, device="cuda")   # or device="cpu"

# --- Custom / plug-in backend ---
class MyCompanyLLM(hx.LLMBackend):
    name = "my_llm"
    def chat(self, messages):
        return my_internal_api.complete(messages[-1]["content"])

result = hx.analyze(df, "Is satisfaction higher in Q4?",
                    backend=MyCompanyLLM())

# --- Wrap any callable ---
result = hx.analyze(df, "...",
                    backend=lambda msgs: my_fn(msgs[-1]["content"]))
```

### Assumption Checking
```python
from hypotestx import check_normality, check_equal_variances

norm = check_normality(data)
if not norm.is_significant:          # Shapiro-Wilk p > 0.05 -> normal
    print("Normality assumption met")
else:
    print("Non-normal — consider Mann-Whitney U")
    result = hx.mannwhitney(group1, group2)
```

### Effect Size Interpretation
```python
result = hx.ttest_2samp(group1, group2)

print(f"Effect size: {result.effect_size:.3f}")
print(f"Magnitude:   {result.effect_magnitude}")  # 'small', 'medium', 'large'

if result.is_significant and result.effect_magnitude in ('medium', 'large'):
    print("Both statistically and practically significant")
```

### Power Analysis
```python
# How many participants do I need?
n = hx.n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Required n per group: {n}")

# Post-hoc power
pow_result = hx.power_ttest_two_sample(
    effect_size=0.4, n1=30, n2=30, alpha=0.05
)
print(f"Achieved power: {pow_result.power:.2f}")
```

### Bootstrap & Permutation Tests
```python
# Bootstrap confidence interval for the difference in means
result = hx.bootstrap_ci(group1, statistic='mean', n_bootstrap=5000)
print(f"95% CI: {result}")

# Permutation test (non-parametric, exact)
result = hx.permutation_test(group1, group2, n_permutations=10000)
```

### Verbose Mode
```python
# See which test was selected and why
result = hx.analyze(
    df, "Is salary different between genders?",
    backend="gemini", api_key="AIza...",
    model="gemini-2.0-flash",
    verbose=True,
)
# [HypoTestX] Schema: 500 rows, columns: ['gender', 'salary', 'age']
# [HypoTestX] Backend: GeminiBackend
# [HypoTestX] Routing -> test='two_sample_ttest', confidence=0.95
# [HypoTestX] Reasoning: Two groups (M/F) compared on a numeric column
```

---

## API Reference

### `analyze()` — Natural Language Entry Point

```python
hx.analyze(df, question, backend=None, alpha=0.05, verbose=False, **kwargs)
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `DataFrame` | pandas or polars DataFrame |
| `question` | `str` | Plain-English hypothesis question |
| `backend` | `str \| LLMBackend \| callable \| None` | LLM to use (default: regex fallback) |
| `alpha` | `float` | Significance level (default `0.05`) |
| `verbose` | `bool` | Print routing info to stdout |
| `api_key` | `str` | API key forwarded to backend constructor |
| `model` | `str` | Model name forwarded to backend constructor |

Returns a `HypoResult` object.

### `get_backend()` — Backend Factory

```python
b = hx.get_backend("groq", api_key="gsk_...")   # by string
b = hx.get_backend(hx.OllamaBackend(model="phi4"))  # pass instance
b = hx.get_backend(my_callable)                 # wrap a callable

routing = b.route("Do males earn more?", hx.build_schema(df))
print(routing.test, routing.group_column, routing.value_column)
```

### `HypoResult` Object

```python
result.test_name            # 'Welch\'s t-test (unequal variances)'
result.statistic            # test statistic value
result.p_value              # p-value
result.effect_size          # Cohen's d / r / eta^2 / Cramer's V
result.effect_size_name     # 'Cohen\'s d', 'Pearson r', ...
result.confidence_interval  # (lower, upper)
result.degrees_of_freedom   # df
result.sample_sizes         # list of group sizes
result.is_significant       # bool — p_value < alpha
result.effect_magnitude     # 'small' | 'medium' | 'large'
result.interpretation       # plain-English interpretation string
result.alpha                # significance level used
result.alternative          # 'two-sided' | 'greater' | 'less'
result.summary()            # formatted multi-line summary string
result.to_dict()            # dict representation
```

### Direct Test Functions

#### t-tests
```python
hx.ttest_1samp(data, mu=0, alpha=0.05, alternative='two-sided')
hx.ttest_2samp(group1, group2, alpha=0.05, alternative='two-sided', equal_var=True)
hx.ttest_paired(before, after, alpha=0.05, alternative='two-sided')
hx.welch_ttest(group1, group2, alpha=0.05, alternative='two-sided')
hx.anova_1way(*groups, alpha=0.05)
```

#### Non-parametric
```python
hx.mannwhitney(group1, group2, alpha=0.05, alternative='two-sided')
hx.wilcoxon(x, y=None, mu=0, alpha=0.05, alternative='two-sided')
hx.kruskal(*groups, alpha=0.05)
```

#### Categorical
```python
hx.chi2_test(observed, alpha=0.05)                          # 2-D table or 1-D GoF
hx.fisher_exact(table, alpha=0.05, alternative='two-sided') # 2x2 only
```

#### Correlation
```python
hx.pearson(x, y, alpha=0.05, alternative='two-sided')
hx.spearman(x, y, alpha=0.05, alternative='two-sided')
hx.pointbiserial(continuous, binary, alpha=0.05)
```

### Backend Classes (for plug-in use)

```python
from hypotestx import (
    LLMBackend,          # Abstract base — subclass to create your own
    CallableBackend,     # Wraps any callable(messages) -> str
    FallbackBackend,     # Built-in regex router (default)
    OllamaBackend,       # Local Ollama
    OpenAICompatBackend, # OpenAI / Groq / Together / Mistral / Azure
    GeminiBackend,       # Google Gemini
    HuggingFaceBackend,  # HuggingFace Inference API or local transformers
)
```

Creating a custom backend:

```python
class MyBackend(hx.LLMBackend):
    name = "my_backend"

    def chat(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system", "content": ...},
                   {"role": "user",   "content": ...}]
        Return a JSON string matching the RoutingResult schema.
        """
        prompt = messages[-1]["content"]
        return call_my_llm_api(prompt)   # must return JSON string

result = hx.analyze(df, "Is salary different by gender?",
                    backend=MyBackend())
```

---

## 🎨 Visualization

Visualization helpers require **matplotlib** (optional):

```bash
pip install matplotlib
# or
pip install hypotestx[visualization]   # matplotlib + plotly
```

### Plot a test result
```python
import hypotestx as hx

result = hx.two_sample_ttest(group1, group2)

# Auto-selects the best chart type (bar for two-group, p-value curve otherwise)
fig = result.plot()          # returns a matplotlib Figure
fig.savefig("result.png")

# Or call directly
fig = hx.plot_result(result, kind="auto")    # "auto" | "p_value" | "bar"
```

### Plot group distributions
```python
fig = hx.plot_distributions(
    [group1, group2],
    labels=["Control", "Treatment"],
    kind="box",    # "box" (default) | "bar" | "violin"
    title="Group Comparison",
)
fig.show()
```

### Visualise a p-value on the null distribution
```python
fig = hx.plot_p_value(
    p_value=0.023,
    alpha=0.05,
    test_statistic=2.41,
    alternative="two-sided",
)
```

### Generate an HTML or text report
```python
# HTML (embedded chart if matplotlib is installed)
html = hx.generate_report(result, fmt="html")

# Save to file
hx.generate_report(result, path="report.html", fmt="html")

# PDF (requires: pip install weasyprint)
hx.generate_report(result, path="report.pdf", fmt="pdf")

# Plain text
text = hx.generate_report(result, fmt="text")

# Or use the reporting helpers directly
from hypotestx.reporting.generator import export_html, export_pdf, export_csv

export_html(result, path="report.html")
export_pdf(result, path="report.pdf")       # requires weasyprint
export_csv([result1, result2], path="results.csv")
```

> **Note:** `plot_effect_size()`, `plot_assumptions()`, and `generate_apa_report()` 
> referenced in older docs are not yet implemented. Use `result.plot()`, 
> `plot_distributions()`, and `apa_report()` instead.

---

## Architecture

### Design Philosophy
- **Zero mandatory dependencies** — pure Python stdlib for math and HTTP
- **Plug-in LLMs** — swap backends without changing test logic
- **Modular** — each component works independently
- **Transparent** — read the source to see exactly how every test works

### Package Layout
```
hypotestx/
├── core/
│   ├── engine.py          # analyze() dispatcher
│   ├── result.py          # HypoResult dataclass
│   ├── parser.py          # Legacy regex NL parser
│   ├── assumptions.py     # Shapiro-Wilk, Levene, Bartlett, ...
│   └── llm/               # LLM sub-package
│       ├── base.py        # LLMBackend ABC, RoutingResult, SchemaInfo
│       ├── prompts.py     # System prompt, schema builder, user prompt
│       └── backends/
│           ├── fallback.py       # Regex router (default, zero deps)
│           ├── ollama.py         # Local Ollama
│           ├── openai_compat.py  # OpenAI / Groq / Together / Mistral
│           ├── gemini.py         # Google Gemini
│           └── huggingface.py    # HF Inference API + local transformers
├── math/           # Pure Python: distributions, statistics, linear algebra
├── tests/          # Statistical test implementations
├── stats/          # Descriptive stats, bootstrap, inference
├── power/          # Power analysis and sample size
├── reporting/      # APA reports, formatters
└── utils/          # Data utilities and validation
```

### How `analyze()` Works
```
analyze(df, question, backend)  ←  user calls this
    │
    ├─ build_schema(df)         → SchemaInfo(columns, dtypes, numerics, categoricals)
    │
    ├─ backend.route(question, schema)
    │       │
    │       ├─ FallbackBackend  → regex pattern matching (instant, offline)
    │       ├─ GeminiBackend    → Gemini REST API (JSON response)
    │       ├─ OllamaBackend    → local HTTP to Ollama server
    │       └─ (any LLMBackend) → JSON parsed into RoutingResult
    │
    └─ _dispatch(routing, df)   → extracts columns, calls test function
            │
            └─ HypoResult       ← returned to caller
```

### Mathematical Implementation
All statistical computations are implemented from scratch using:
- **Newton's method** for square roots and optimization
- **Taylor series** for transcendental functions
- **Lanczos approximation** for gamma function
- **Continued fractions** for special functions
- **Numerical integration** for distribution functions

---

## 🔒 Security & API Key Best Practices

**Never hard-code API keys in source code or commit them to version control.**

### Recommended: Environment Variables

Store keys in your shell environment or a `.env` file:

```bash
# ~/.bashrc / ~/.zshrc / .env (add .env to .gitignore!)
export GEMINI_API_KEY="AIza..."
export GROQ_API_KEY="gsk_..."
export OPENAI_API_KEY="sk-..."
export AZURE_OPENAI_API_KEY="<az-key>"
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
```

Load them in Python:

```python
import os
import hypotestx as hx

# Read from environment — never from a string literal in code
result = hx.analyze(
    df,
    "Is salary different by gender?",
    backend="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
)

# Azure example using env vars
result = hx.analyze(
    df,
    "Do departments differ?",
    backend="azure",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    model="my-deployment",
)
```

With a `.env` file + `python-dotenv`:
```bash
pip install python-dotenv
```
```python
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ

import os, hypotestx as hx
result = hx.analyze(df, "...", backend="groq",
                    api_key=os.environ["GROQ_API_KEY"])
```

### Tips
- Add `.env` to your `.gitignore` to prevent accidental key leaks.
- Use secret managers (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)
  in production environments.
- Rotate keys immediately if they are ever exposed.
- Use the minimum required permission scope for each key.
- The fallback backend (`backend=None`) requires no API key at all.

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
git clone https://github.com/Ankit-Anand123/HypoTestX.git
cd HypoTestX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Contribution Areas
- 🧪 **New statistical tests**: Implement additional tests
- 🗣️ **NLP improvements**: Enhance natural language understanding
- 📊 **Visualizations**: Add new plotting capabilities
- 🎓 **Educational content**: Improve explanations and tutorials
- 🏥 **Domain packages**: Specialized tests for specific fields
- 🌍 **Internationalization**: Support for other languages

### Code Style
- Follow PEP 8
- Type hints required for all public functions
- Comprehensive docstrings with examples
- 95%+ test coverage for new code

---

## 📖 Documentation

### Full Documentation
> Documentation site is not yet available. In the meantime, refer to this README, the inline docstrings, and the example notebooks below.


---

## 📊 Performance

### Benchmarks
```python
# Performance comparison with other libraries
import hypotestx as htx
import scipy.stats as stats
import time

# HypoTestX (pure Python)
start = time.time()
result_htx = htx.ttest_2samp(group1, group2)
time_htx = time.time() - start

# SciPy (compiled)
start = time.time()
result_scipy = stats.ttest_ind(group1, group2)
time_scipy = time.time() - start

print(f"HypoTestX: {time_htx:.4f}s")
print(f"SciPy: {time_scipy:.4f}s")
print(f"Results match: {abs(result_htx.p_value - result_scipy.pvalue) < 1e-10}")
```

**Typical performance:**
- Small datasets (n < 1000): Comparable to SciPy
- Large datasets (n > 10000): 2-3x slower than compiled libraries
- Trade-off: Transparency and educational value vs. raw speed

---

## Roadmap

### Version 0.1.0 (Released)
- Complete parametric test suite (t-tests, ANOVA)
- Non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
- Categorical tests (Chi-square, Fisher's exact)
- Correlation tests (Pearson, Spearman, point-biserial)
- Pure Python math core (distributions, special functions)
- Assumption checking (Shapiro-Wilk, Levene, Bartlett, Jarque-Bera)
- Power analysis and sample size calculation
- Bootstrap and permutation tests
- APA-style reporting
- **LLM-powered `analyze()` interface with plug-in backend system**
  - Built-in regex fallback (zero deps, offline)
  - Ollama backend (local, free)
  - Gemini backend (free tier)
  - Groq / OpenAI / Together / Mistral / Azure backends
  - HuggingFace Inference API + local transformers
  - Custom backend API (`LLMBackend` subclass or callable)

### Version 0.2.0 (Planned)
- Two-way ANOVA and repeated-measures ANOVA
- Regression-based tests (linear, logistic)
- Automatic assumption-driven test selection
- Streaming LLM responses for verbose mode
- `analyze()` result explains *why* a test was chosen

### Version 0.3.0 (Planned)
- Bayesian alternatives (Bayesian t-test, Bayes factor)
- Time series stationarity and change-point tests
- Meta-analysis tools
- Interactive Jupyter widgets for results

### Version 1.0.0 (Released)
- Domain-specific packages (clinical, A/B testing, finance)
- Publication-ready PDF/HTML reporting
- LLM-powered `analyze()` interface with plug-in backend system
- Full test suite (483 tests passing)

### Version 1.0.5 (Released — Current)
- **Visualization** — `result.plot()`, `plot_result()`, `plot_distributions()`,
  `plot_p_value()`, `generate_report()` (HTML/PDF/text); requires optional `matplotlib`
- **Azure OpenAI** — `backend="azure"` with correct deployment URL, `api-key` header,
  and `api_version` parameter
- **HTML & PDF export** — `export_html()` and `export_pdf()` added to reporting module;
  `weasyprint` optional dep for PDF
- **Routing validation** — explicit column checks per test type with actionable error messages
  before dispatch
- **`backend_options` passthrough** — pass provider-specific kwargs via `backend_options={}`
- **Structured logging** — `logging.getLogger("hypotestx")` throughout; zero noise by default
- **Division-by-zero fix** — Welch and Student t-tests guard against zero-variance groups
- **Duck-typed backends** — any object exposing `.route()` accepted by `get_backend()`
- **Expanded test suite** — 532 tests passing (up from 483); new Azure and visualization
  test files plus edge-case tests for parametric functions
- **Security docs** — API key best-practices section in README

---

## Support & Community

### Getting Help
- 💬 [GitHub Discussions](https://github.com/Ankit-Anand123/HypoTestX/discussions)
- 🐛 [Issue Tracker](https://github.com/Ankit-Anand123/HypoTestX/issues)

---

## 📄 License

HypoTestX is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 HypoTestX Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

### Author
- **Ankit** — [Ankit-Anand123](https://github.com/Ankit-Anand123) — sole developer and maintainer

### Inspiration
- **R's** elegant statistical interface
- **spaCy's** intuitive NLP design  
- **pandas'** data manipulation philosophy
- **scikit-learn's** consistent API design

### Dependencies
The mathematical core and all LLM HTTP calls are pure Python stdlib.
Optional extras that unlock additional functionality:
- **Ollama** desktop app — for the local `OllamaBackend`
- **transformers + torch** — for `HuggingFaceBackend` local inference mode
- **matplotlib** — for visualization helpers

---

## 📈 Citation

If you use HypoTestX in your research, please cite:

```bibtex
@software{hypotestx2025,
  author = {Ankit},
  title = {HypoTestX: Natural Language Hypothesis Testing for Python},
  url = {https://github.com/Ankit-Anand123/HypoTestX},
  version = {1.0.0},
  year = {2026}
}
```

---

<div align="center">

**Made with ❤️ for the data science community**

[GitHub](https://github.com/Ankit-Anand123/HypoTestX) • [PyPI](https://pypi.org/project/hypotestx/)

</div>