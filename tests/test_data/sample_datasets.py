"""
Shared sample datasets for use in tests.
"""

from __future__ import annotations


def two_group_numeric(n=30):
    """Two groups with clear mean difference."""
    group_a = [70.0 + i % 10 for i in range(n)]
    group_b = [60.0 + i % 10 for i in range(n)]
    return {
        "group": ["A"] * n + ["B"] * n,
        "value": group_a + group_b,
    }


def three_group_numeric(n=20):
    """Three groups for ANOVA / Kruskal-Wallis."""
    ga = [5.0 + (i % 5) * 0.5 for i in range(n)]
    gb = [10.0 + (i % 5) * 0.5 for i in range(n)]
    gc = [3.0 + (i % 5) * 0.5 for i in range(n)]
    return {
        "group": ["A"] * n + ["B"] * n + ["C"] * n,
        "value": ga + gb + gc,
    }


def paired_numeric(n=20, delta=2.0):
    """Paired before/after measurements."""
    before = [10.0 + i * 0.5 for i in range(n)]
    after = [v + delta for v in before]
    return {"before": before, "after": after}


def bivariate_linear(n=30, slope=2.0, intercept=1.0):
    """Linearly correlated x and y."""
    x = [float(i) for i in range(n)]
    y = [slope * xi + intercept for xi in x]
    return {"x": x, "y": y}


def contingency_2x2(n=50):
    """2x2 contingency table data."""
    treatment = ["A"] * n + ["B"] * n
    outcome = (["Y"] * int(n * 0.8) + ["N"] * int(n * 0.2)) + (
        ["Y"] * int(n * 0.4) + ["N"] * int(n * 0.6)
    )
    # trim to equal length
    min_len = min(len(treatment), len(outcome))
    return {"treatment": treatment[:min_len], "outcome": outcome[:min_len]}


def one_sample_normal(n=50, mu=0.0, sigma=1.0):
    """Simple one-sample dataset drawn from N(mu, sigma^2)."""
    import math

    # Deterministic pseudo-random using lcg
    data = []
    seed = 42
    for _ in range(n):
        seed = (seed * 1664525 + 1013904223) % (2**32)
        u1 = seed / 2**32
        seed = (seed * 1664525 + 1013904223) % (2**32)
        u2 = seed / 2**32
        z = math.sqrt(-2 * math.log(max(u1, 1e-15))) * math.cos(2 * math.pi * u2)
        data.append(mu + sigma * z)
    return {"value": data}


def categorical_dataset(n=100):
    """Dataset with categorical variables."""
    gender = (["M", "F"] * (n // 2))[:n]
    dept = (["Eng", "Sales", "HR"] * ((n // 3) + 1))[:n]
    salary = [
        70.0 + (5 if g == "M" else 0) + (3 if d == "Eng" else 0)
        for g, d in zip(gender, dept)
    ]
    return {"gender": gender, "dept": dept, "salary": salary}
