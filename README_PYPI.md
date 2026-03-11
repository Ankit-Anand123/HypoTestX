# HypoTestX

**Natural Language Hypothesis Testing for Python**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.6-blue.svg)](https://pypi.org/project/hypotestx/)
[![Tests](https://img.shields.io/badge/tests-532%20passing-brightgreen.svg)](https://github.com/Ankit-Anand123/HypoTestX)

> Ask a statistical question in plain English. Get a structured result back — with the right test chosen automatically.

---

## The problem with scipy

```python
# scipy — you decide the test, extract groups, interpret results yourself
from scipy import stats
males = df[df['gender'] == 'M']['salary'].values
females = df[df['gender'] == 'F']['salary'].values
t, p = stats.ttest_ind(males, females, equal_var=False)
# p = 0.0012 ... now what?
```

```python
# HypoTestX — ask the question, get a full result
import hypotestx as hx
result = hx.analyze(df, "Do males earn more than females?")
print(result.summary())
```

```
[ Welch's t-test (unequal variances) ]
=============================================
Statistic (t):   3.2456
p-value:         0.0012
Significant:     Yes (alpha = 0.05)
Effect size (d): 0.6834   (medium)
95% CI:          [1.23, 4.56]
```

---

## Install

```bash
pip install hypotestx
```

No mandatory dependencies — pure Python stdlib for all math and HTTP calls.

---

## Usage

### Natural language (no API key needed)

```python
import hypotestx as hx
import pandas as pd

df = pd.read_csv('data.csv')

result = hx.analyze(df, "Is there a correlation between age and salary?")
result = hx.analyze(df, "Did scores improve from pre_score to post_score?")
result = hx.analyze(df, "Are gender and department independent?")
result = hx.analyze(df, "Compare satisfaction scores across all regions")
```

### With a real LLM backend (better accuracy on complex questions)

```python
# Google Gemini — free tier, 1500 req/day
result = hx.analyze(df, "...", backend="gemini", api_key="AIza...")

# Groq — free tier, very fast
result = hx.analyze(df, "...", backend="groq", api_key="gsk_...")

# Local Ollama — completely offline, no API key
result = hx.analyze(df, "...", backend="ollama")
```

### Direct API (explicit control)

```python
result = hx.ttest_2samp(group1, group2, equal_var=False, alpha=0.01)
result = hx.pearson(x, y, alternative='greater')
result = hx.chi2_test(contingency_table)
result = hx.mannwhitney(group1, group2)
```

### Result object

```python
result.p_value            # 0.0012
result.is_significant     # True
result.effect_size        # 0.6834
result.effect_magnitude   # 'medium'
result.confidence_interval # (1.23, 4.56)
result.interpretation     # plain-English explanation
result.summary()          # formatted summary string
result.to_dict()          # dict for logging or serialization
```

---

## Supported tests

**Parametric:** one-sample t-test, two-sample t-test, Welch's t-test, paired t-test, one-way ANOVA

**Non-parametric:** Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis

**Categorical:** chi-square (independence + GoF), Fisher's exact

**Correlation:** Pearson, Spearman, point-biserial

**Plus:** assumption checking, effect sizes, power analysis, bootstrap & permutation tests, HTML/PDF reporting

---

## Links

- **GitHub:** https://github.com/Ankit-Anand123/HypoTestX
- **Docs:** https://hypotestx.readthedocs.io
- **Changelog:** https://github.com/Ankit-Anand123/HypoTestX/blob/main/CHANGELOG.md
- **Issues:** https://github.com/Ankit-Anand123/HypoTestX/issues