# Direct API

All 12 statistical test functions are available directly. Use these when you
already know which test you need and want explicit, readable code with full
parameter control.

Every function returns a `HypoResult` — the same structured object as `analyze()`.

---

## Parametric Tests

### One-sample t-test

Test whether the mean of a single sample equals a hypothesised value.

```python
hx.ttest_1samp(data, mu=0, alpha=0.05, alternative="two-sided")
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `list[float]` | Sample data |
| `mu` | `float` | Null hypothesis mean (default `0`) |
| `alpha` | `float` | Significance level (default `0.05`) |
| `alternative` | `str` | `"two-sided"`, `"greater"`, or `"less"` |

```python
import hypotestx as hx

scores = df["test_score"].tolist()
result = hx.ttest_1samp(scores, mu=70, alternative="greater")
print(result.summary())
```

---

### Two-sample t-test (Welch / Student)

Compare means of two independent groups.

```python
hx.ttest_2samp(group1, group2, alpha=0.05, alternative="two-sided", equal_var=False)
```

| Parameter | Type | Description |
|---|---|---|
| `group1` | `list[float]` | First group |
| `group2` | `list[float]` | Second group |
| `alpha` | `float` | Significance level |
| `alternative` | `str` | `"two-sided"`, `"greater"`, or `"less"` |
| `equal_var` | `bool` | `False` = Welch's (default), `True` = Student's |

```python
males   = df[df["gender"] == "M"]["salary"].tolist()
females = df[df["gender"] == "F"]["salary"].tolist()
result  = hx.ttest_2samp(males, females, alternative="greater", equal_var=False)
```

---

### Welch's t-test (explicit)

Identical to `ttest_2samp` with `equal_var=False`. Provided as a convenience alias.

```python
hx.welch_ttest(group1, group2, alpha=0.05, alternative="two-sided")
```

---

### Paired t-test

Compare two measurements from the same subjects (before/after, repeated measures).

```python
hx.ttest_paired(before, after, alpha=0.05, alternative="two-sided")
```

| Parameter | Type | Description |
|---|---|---|
| `before` | `list[float]` | Pre-treatment or first measurement |
| `after` | `list[float]` | Post-treatment or second measurement |

```python
pre  = df["pre_score"].tolist()
post = df["post_score"].tolist()
result = hx.ttest_paired(pre, post, alternative="less")
# alternative="less": testing whether pre < post, i.e. scores improved
```

---

### One-way ANOVA

Test whether means differ across three or more independent groups.

```python
hx.anova_1way(*groups, alpha=0.05)
```

```python
eng   = df[df["dept"] == "engineering"]["salary"].tolist()
sales = df[df["dept"] == "sales"]["salary"].tolist()
mktg  = df[df["dept"] == "marketing"]["salary"].tolist()

result = hx.anova_1way(eng, sales, mktg, alpha=0.05)
print(result.effect_size_name)   # 'eta-squared'
print(result.effect_magnitude)   # 'small' | 'medium' | 'large'
```

---

## Non-parametric Tests

### Mann-Whitney U

Non-parametric alternative to the two-sample t-test. Does not assume normality.

```python
hx.mannwhitney(group1, group2, alpha=0.05, alternative="two-sided")
```

```python
result = hx.mannwhitney(
    df[df["group"] == "A"]["score"].tolist(),
    df[df["group"] == "B"]["score"].tolist(),
)
print(result.effect_size_name)   # 'rank-biserial r'
```

---

### Wilcoxon Signed-Rank

Non-parametric alternative to the paired t-test.

```python
hx.wilcoxon(x, y=None, mu=0, alpha=0.05, alternative="two-sided")
```

```python
# Paired: compare pre and post
result = hx.wilcoxon(df["pre"].tolist(), y=df["post"].tolist())

# One-sample: test if median equals mu
result = hx.wilcoxon(df["score"].tolist(), mu=50)
```

---

### Kruskal-Wallis

Non-parametric alternative to one-way ANOVA. Does not assume normality.

```python
hx.kruskal(*groups, alpha=0.05)
```

```python
groups = [df[df["region"] == r]["sales"].tolist() for r in df["region"].unique()]
result = hx.kruskal(*groups)
```

---

## Categorical Tests

### Chi-square test

Test independence between two categorical variables (contingency table), or
goodness-of-fit for a 1-D observed distribution.

```python
hx.chi2_test(observed, alpha=0.05)
```

| Parameter | Type | Description |
|---|---|---|
| `observed` | `list[list[int]]` or `list[int]` | 2-D contingency table, or 1-D observed counts |

```python
# 2×2 contingency table
table = [[30, 10], [20, 40]]
result = hx.chi2_test(table)

# Build table from DataFrame columns
from hypotestx.core.engine import _build_contingency_table
table = _build_contingency_table(df, "gender", "department")
result = hx.chi2_test(table)
```

---

### Fisher's exact test

Exact test for 2×2 contingency tables. Preferred when expected cell counts are small.

```python
hx.fisher_exact(table, alpha=0.05, alternative="two-sided")
```

```python
table = [[8, 2], [1, 5]]
result = hx.fisher_exact(table, alternative="greater")
print(result.effect_size_name)   # 'odds ratio'
```

---

## Correlation Tests

### Pearson correlation

Test for linear association between two continuous numeric variables.

```python
hx.pearson(x, y, alpha=0.05, alternative="two-sided")
```

```python
result = hx.pearson(df["age"].tolist(), df["salary"].tolist())
print(result.effect_size_name)   # 'Pearson r'
```

---

### Spearman correlation

Rank-based test for monotonic association. More robust than Pearson.

```python
hx.spearman(x, y, alpha=0.05, alternative="two-sided")
```

```python
result = hx.spearman(df["rank"].tolist(), df["performance"].tolist())
```

---

### Point-biserial correlation

Correlation between one continuous and one binary (0/1) variable.

```python
hx.pointbiserial(continuous, binary, alpha=0.05)
```

```python
result = hx.pointbiserial(
    df["exam_score"].tolist(),
    df["passed"].tolist(),   # binary: 0 or 1
)
```
