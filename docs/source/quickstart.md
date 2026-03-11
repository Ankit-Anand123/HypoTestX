# Quick Start

Get up and running with HypoTestX in under five minutes.

## Installation

```bash
pip install hypotestx
```

No mandatory external dependencies — all statistical math and HTTP calls are pure Python stdlib.

Optional extras for additional features:

```bash
pip install hypotestx[visualization]   # matplotlib + plotly for plots
pip install hypotestx[dev]             # testing and linting tools
pip install hypotestx[docs]            # sphinx, furo, myst-parser
pip install hypotestx[all]             # all optional extras
```

---

## Your First Test

```python
import hypotestx as hx
import pandas as pd

df = pd.read_csv("survey.csv")

# Zero config — built-in regex router, no API key needed
result = hx.analyze(df, "Do males earn more than females?")
print(result.summary())
```

### Example output

```
[ Welch's t-test (unequal variances) ]
=======================================
Result: SIGNIFICANT (alpha = 0.05)
Test statistic: 3.2456
p-value: 0.0012
Degrees of freedom: 248.0
Cohen's d: 0.6834 (medium)
95% Confidence Interval: [1.2300, 4.5600]

Interpretation:
There is a statistically significant difference between the two groups
(t = 3.25, df = 248, p = 0.0012, Cohen's d = 0.68).
```

---

## Reading the HypoResult

Every test returns a `HypoResult` object with the following key attributes:

| Attribute | Type | Description |
|---|---|---|
| `result.test_name` | `str` | Human-readable test name |
| `result.statistic` | `float` | Test statistic value (t, F, χ², U, …) |
| `result.p_value` | `float` | p-value |
| `result.is_significant` | `bool` | True if p_value < alpha |
| `result.effect_size` | `float` | Effect size (Cohen's d, r, η², V, …) |
| `result.effect_size_name` | `str` | Name of the effect size measure |
| `result.effect_magnitude` | `str` | `'negligible'`, `'small'`, `'medium'`, `'large'` |
| `result.confidence_interval` | `tuple` | (lower, upper) confidence interval |
| `result.degrees_of_freedom` | `int/float` | Degrees of freedom |
| `result.sample_sizes` | `int/tuple` | Sample size(s) |
| `result.interpretation` | `str` | Plain-English interpretation |
| `result.routing_confidence` | `float` | 1.0 for LLM, 0.6 for regex fallback |
| `result.routing_source` | `str` | `'llm'` or `'fallback'` |
| `result.summary()` | `str` | Formatted multi-line summary |
| `result.to_dict()` | `dict` | All fields as a plain dict |

---

## Using a Real LLM Backend

The default regex fallback is fast and works offline but has limited accuracy on
complex questions. Use a real LLM backend for production:

### Google Gemini (free tier — 1 500 req/day)

```python
import os
import hypotestx as hx

result = hx.analyze(
    df,
    "Is there a salary difference between engineering and sales departments?",
    backend="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-2.0-flash",
    temperature=0.0,
)
print(result.summary())
```

### Groq (free tier, very fast)

```python
result = hx.analyze(
    df,
    "Is employee satisfaction correlated with tenure?",
    backend="groq",
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
)
```

### Ollama (fully offline, no API key)

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull a model
ollama pull llama3.2
```

```python
result = hx.analyze(
    df,
    "Are there differences in performance scores across teams?",
    backend="ollama",
    model="llama3.2",
)
```

---

## Direct API

If you already know which test you want, call it directly with full parameter control:

```python
import hypotestx as hx

# Two-sample t-test
males   = df[df["gender"] == "M"]["salary"].tolist()
females = df[df["gender"] == "F"]["salary"].tolist()
result  = hx.ttest_2samp(males, females, alternative="greater", equal_var=False)

# Pearson correlation
result = hx.pearson(df["age"].tolist(), df["salary"].tolist())

# One-way ANOVA
groups = [df[df["dept"] == d]["score"].tolist() for d in df["dept"].unique()]
result = hx.anova_1way(*groups, alpha=0.01)

print(result.p_value)
print(result.effect_magnitude)
```

See [Direct API](direct_api.md) for a full reference of all 12 test functions.
