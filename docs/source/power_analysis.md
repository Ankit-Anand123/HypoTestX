# Power Analysis

Statistical power is the probability that a test correctly rejects a false null
hypothesis.  HypoTestX provides functions to calculate required sample sizes
before a study and to compute post-hoc power after data collection.

---

## Required sample size — two-sample t-test

How many participants per group do you need to detect an effect of a given size?

```python
import hypotestx as hx

n = hx.n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Required n per group: {n}")   # 64
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `effect_size` | `float` | — | Cohen's d (standardised mean difference) |
| `alpha` | `float` | `0.05` | Significance level (Type I error rate) |
| `power` | `float` | `0.80` | Desired power (1 − Type II error rate) |
| `alternative` | `str` | `"two-sided"` | `"two-sided"`, `"greater"`, or `"less"` |

### Common effect size guidelines (Cohen's d)

| d | Magnitude |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

```python
# Need to detect a small effect with 90% power
n_small = hx.n_ttest_two_sample(effect_size=0.2, alpha=0.05, power=0.9)
print(f"Small effect, 90% power: n = {n_small} per group")

# One-tailed test (directional hypothesis)
n_one = hx.n_ttest_two_sample(
    effect_size=0.5, alpha=0.05, power=0.8, alternative="greater"
)
```

---

## Post-hoc power — two-sample t-test

After collecting data, compute the power actually achieved:

```python
pow_result = hx.power_ttest_two_sample(
    effect_size=0.4,
    n1=30,
    n2=30,
    alpha=0.05,
    alternative="two-sided",
)
print(f"Achieved power: {pow_result.power:.2f}")
```

| Parameter | Type | Description |
|---|---|---|
| `effect_size` | `float` | Cohen's d |
| `n1` | `int` | Size of group 1 |
| `n2` | `int` | Size of group 2 |
| `alpha` | `float` | Significance level used in the test |
| `alternative` | `str` | `"two-sided"`, `"greater"`, or `"less"` |

The returned object has a `.power` attribute (float 0–1).

---

## Worked Example: Planning a Study

```python
import hypotestx as hx

# Before the study: power calculation
# Expecting a medium effect (d = 0.5), standard alpha, 80% power
n_required = hx.n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Collect at least {n_required} participants per group")

# After the study: you collected 45 per group
# Effect observed was d = 0.42 (slightly smaller than expected)
achieved = hx.power_ttest_two_sample(
    effect_size=0.42,
    n1=45,
    n2=45,
    alpha=0.05,
)
print(f"Post-hoc power: {achieved.power:.1%}")

# Run the actual test
result = hx.ttest_2samp(group1, group2, alpha=0.05)
print(result.summary())
```

---

## Power Curves

To visualise how power changes with sample size:

```python
import hypotestx as hx

effect = 0.5     # medium effect
alpha  = 0.05

ns     = list(range(10, 200, 5))
powers = [
    hx.power_ttest_two_sample(effect, n1=n, n2=n, alpha=alpha).power
    for n in ns
]

# Find minimum n for 80% power
adequate = [(n, p) for n, p in zip(ns, powers) if p >= 0.80]
if adequate:
    print(f"First n with ≥80% power: {adequate[0][0]}")
```
