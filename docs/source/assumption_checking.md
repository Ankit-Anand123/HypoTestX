# Assumption Checking

Before running a parametric test, it is good practice to verify that the data
meets the test's assumptions.  HypoTestX provides standalone functions for the
most common checks.

---

## `check_normality()`

Test whether a sample comes from a normally distributed population using the
Shapiro-Wilk test (n ≤ 5000) or Jarque-Bera (n > 5000).

```python
from hypotestx import check_normality

norm = check_normality(data, alpha=0.05)
print(norm.summary())
```

### Interpreting the result

- `norm.is_significant == False` → **fail to reject** H₀ (data is consistent with normality) → assumption **met**
- `norm.is_significant == True`  → **reject** H₀ → normality assumption **violated**

```python
if not norm.is_significant:
    print("Normality assumption met — proceed with t-test")
    result = hx.ttest_2samp(group1, group2)
else:
    print("Non-normal data — use Mann-Whitney U instead")
    result = hx.mannwhitney(group1, group2)
```

### Available tests

| n | Test used | Description |
|---|---|---|
| ≤ 5 000 | Shapiro-Wilk | Sensitive power test; exact for small n |
| > 5 000 | Jarque-Bera | Tests skewness and excess kurtosis |

---

## `check_equal_variances()`

Test whether two or more groups have equal variances.

```python
from hypotestx import check_equal_variances

ev = check_equal_variances(group1, group2, method="levene", alpha=0.05)
print(ev.summary())
```

| Parameter | Type | Description |
|---|---|---|
| `*groups` | `list[float]` | Two or more data groups |
| `method` | `str` | `"levene"` (default) or `"bartlett"` |
| `alpha` | `float` | Significance level |

### Levene vs Bartlett

| Test | When to use |
|---|---|
| **Levene** | Recommended when normality cannot be assumed; more robust |
| **Bartlett** | More powerful when normality holds |

### Interpreting the result

- `ev.is_significant == False` → **fail to reject** H₀ → variances are **equal** → use Student's t-test
- `ev.is_significant == True`  → **reject** H₀ → variances are **unequal** → use Welch's t-test

```python
ev = check_equal_variances(group1, group2)

if not ev.is_significant:
    result = hx.ttest_2samp(group1, group2, equal_var=True)   # Student's t
else:
    result = hx.ttest_2samp(group1, group2, equal_var=False)  # Welch's t
```

---

## Individual Tests

### Shapiro-Wilk

```python
from hypotestx.core.assumptions import shapiro_wilk

result = shapiro_wilk(data, alpha=0.05)
print(result.statistic)   # W statistic
print(result.p_value)
```

### Levene's Test

```python
from hypotestx.core.assumptions import levene_test

result = levene_test(group1, group2, group3, alpha=0.05)
```

### Bartlett's Test

```python
from hypotestx.core.assumptions import bartlett_test

result = bartlett_test(group1, group2, alpha=0.05)
```

### Jarque-Bera

```python
from hypotestx.core.assumptions import jarque_bera

result = jarque_bera(data, alpha=0.05)
print(result.statistic)   # χ² statistic
print(result.data_summary["skewness"])
print(result.data_summary["kurtosis"])
```

---

## Full Decision Workflow

```python
import hypotestx as hx
from hypotestx import check_normality, check_equal_variances

# Step 1: check normality for each group
n1 = check_normality(group1, alpha=0.05)
n2 = check_normality(group2, alpha=0.05)

both_normal = (not n1.is_significant) and (not n2.is_significant)

if both_normal:
    # Step 2: check equal variances
    ev = check_equal_variances(group1, group2)
    equal = not ev.is_significant
    result = hx.ttest_2samp(group1, group2, equal_var=equal)
    print(f"Used {'Student' if equal else 'Welch'}'s t-test")
else:
    result = hx.mannwhitney(group1, group2)
    print("Used Mann-Whitney U (non-parametric)")

print(result.summary())
```
