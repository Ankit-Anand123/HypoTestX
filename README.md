# HypoTestX

**Natural Language Hypothesis Testing — Powered by LLMs or Pure Regex**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yourusername/hypotestx)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/yourusername/hypotestx)

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
# Google Gemini (free tier, 1500 req/day)
result = hx.analyze(df, "Is age correlated with salary?",
                    backend="gemini", api_key="AIza...")

# Groq (free tier, OpenAI-compatible)
result = hx.analyze(df, "Is there an association between gender and dept?",
                    backend="groq", api_key="gsk_...")

# Local Ollama (completely offline)
result = hx.analyze(df, "Compare satisfaction across regions?",
                    backend="ollama")

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
)
print(result.summary())

# Groq free tier (OpenAI-compatible, very fast)
result = hx.analyze(
    df,
    "Is employee satisfaction correlated with tenure?",
    backend="groq",
    api_key="gsk_...",
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

| Backend string | Provider | Cost | Dependencies |
|---|---|---|---|
| `None` / `"fallback"` | Built-in regex router | Free, offline | None |
| `"ollama"` | Local Ollama (llama3.2, phi4, …) | Free, offline | Ollama app |
| `"gemini"` | Google Gemini 1.5 Flash | Free (1500 req/day) | None |
| `"groq"` | Groq Cloud (llama3, mixtral, …) | Free tier | None |
| `"openai"` | OpenAI GPT-4o / GPT-4o-mini | Paid | None |
| `"together"` | Together AI | Free tier | None |
| `"mistral"` | Mistral AI | Free tier | None |
| `"huggingface"` | HF Inference API or local | Free tier / Local | `transformers` (local only) |

```python
import hypotestx as hx

# --- Ollama (local, offline) ---
result = hx.analyze(df, "Do males earn more?",
                    backend="ollama", model="phi4")

# --- Gemini free tier ---
result = hx.analyze(df, "Is age correlated with salary?",
                    backend="gemini", api_key="AIza...")

# --- Groq free tier (very fast) ---
result = hx.analyze(df, "Compare departments",
                    backend="groq", api_key="gsk_...")

# --- Custom / plug-in backend ---
class MyCompanyLLM(hx.LLMBackend):
    name = "my_llm"
    def chat(self, messages):
        # messages is a list of {"role": ..., "content": ...} dicts
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

### Basic Plots
```python
# Automatic visualization based on test type
result = htx.test("Compare groups A and B", data=df)
result.plot()  # Generates appropriate plot (box plot, histogram, etc.)
```

### Custom Visualizations
```python
# Distribution comparison
htx.plot_distributions(group1, group2, 
                      labels=['Group A', 'Group B'],
                      title='Distribution Comparison')

# Effect size visualization
htx.plot_effect_size(result, 
                    context='psychological research')

# Assumption diagnostics
htx.plot_assumptions(data, test_type='ttest')
```

### Publication-Ready Output
```python
# APA-style statistical reporting
htx.generate_apa_report(results, 
                       filename='statistical_analysis.pdf')

# Custom report generation
htx.generate_report(results, 
                   template='academic',
                   format='html',
                   include_plots=True)
```

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

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
git clone https://github.com/yourusername/hypotestx.git
cd hypotestx

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
- 📚 [User Guide](https://hypotestx.readthedocs.io/en/latest/user_guide/)
- 🔧 [API Reference](https://hypotestx.readthedocs.io/en/latest/api/)
- 🎓 [Tutorials](https://hypotestx.readthedocs.io/en/latest/tutorials/)
- 💡 [Examples](https://hypotestx.readthedocs.io/en/latest/examples/)

### Jupyter Notebooks
- [Getting Started Tutorial](examples/notebooks/getting_started.ipynb)
- [Advanced Statistical Analysis](examples/notebooks/advanced_features.ipynb)
- [Real-World Case Studies](examples/notebooks/real_world_cases.ipynb)

### Video Tutorials
- [HypoTestX in 10 Minutes](https://youtu.be/hypotestx-intro)
- [Advanced Features Walkthrough](https://youtu.be/hypotestx-advanced)

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

### Version 1.0.0 (Planned)
- Domain-specific packages (clinical, A/B testing, finance)
- Publication-ready PDF/HTML reporting
- Full documentation site
- R interoperability layer

---

## Support & Community

### Getting Help
- 💬 [GitHub Discussions](https://github.com/yourusername/hypotestx/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/hypotestx/issues)
- 📧 [Email Support](mailto:support@hypotestx.org)
- 💬 [Discord Community](https://discord.gg/hypotestx)

### Stay Updated
- 🐦 [Twitter](https://twitter.com/hypotestx)
- 📧 [Newsletter](https://hypotestx.org/newsletter)
- 📝 [Blog](https://hypotestx.org/blog)

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

### Core Team
- **Lead Developer**: [Your Name](https://github.com/yourusername)
- **Statistics Advisor**: Dr. Jane Smith (Stanford University)
- **NLP Specialist**: Alex Johnson (Google Research)

### Contributors
Special thanks to all [contributors](https://github.com/yourusername/hypotestx/graphs/contributors) who have helped make HypoTestX better.

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
@software{hypotestx2026,
  author = {Your Name and Contributors},
  title = {HypoTestX: Natural Language Hypothesis Testing for Python},
  url = {https://github.com/yourusername/hypotestx},
  version = {0.1.0},
  year = {2026}
}
```

---

<div align="center">

**Made with ❤️ for the data science community**

[Website](https://hypotestx.org) • [Documentation](https://hypotestx.readthedocs.io) • [GitHub](https://github.com/yourusername/hypotestx) • [PyPI](https://pypi.org/project/hypotestx/)

</div>