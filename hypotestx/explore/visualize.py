"""
Visualization helpers for HypoTestX.

All plotting functions gracefully degrade when matplotlib is not installed:
they raise ``ImportError`` with a helpful install message.

Install the optional visualization dependencies with::

    pip install hypotestx[visualization]   # matplotlib + plotly
    # or just
    pip install matplotlib

Public API
----------
plot_result(result)                   -> matplotlib Figure
plot_distributions(groups, labels)    -> matplotlib Figure
plot_p_value(p_value, alpha, df)      -> matplotlib Figure
generate_report(result, path, fmt)    -> saves HTML / PNG report
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "plot_result",
    "plot_distributions",
    "plot_p_value",
    "generate_report",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_matplotlib():
    """Return (plt, patches) or raise a descriptive ImportError."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with:  pip install matplotlib  "
            "or  pip install hypotestx[visualization]"
        ) from exc


def _stderr_bar_chart(ax, group_labels, means, stds, title=""):
    """Draw a simple bar chart with ±1 SD error bars."""
    x = list(range(len(group_labels)))
    ax.bar(x, means, width=0.5, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean ± SD")
    if title:
        ax.set_title(title)


def _normal_pdf(x_vals, mu, sigma):
    """Compute normal PDF values, returning zeros if sigma == 0."""
    import math
    if sigma == 0:
        return [0.0] * len(x_vals)
    return [
        (1 / (sigma * math.sqrt(2 * math.pi)))
        * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
        for x in x_vals
    ]


# ---------------------------------------------------------------------------
# plot_p_value
# ---------------------------------------------------------------------------

def plot_p_value(
    p_value: float,
    alpha: float = 0.05,
    degrees_of_freedom: Optional[float] = None,
    test_statistic: Optional[float] = None,
    alternative: str = "two-sided",
    title: str = "",
) -> Any:
    """
    Visualise the p-value on a standard-normal (or t) distribution curve.

    Hatches the rejection region(s) and marks the observed p-value.

    Parameters
    ----------
    p_value : float
    alpha : float
        Significance level (default 0.05).
    degrees_of_freedom : float, optional
        If provided, a t-distribution tail is shown instead of normal.
    test_statistic : float, optional
        If provided, marks the observed statistic on the x-axis.
    alternative : str
        ``"two-sided"``, ``"greater"``, or ``"less"``.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, mpatches = _require_matplotlib()
    import math

    fig, ax = plt.subplots(figsize=(8, 4))
    n_pts = 400

    if degrees_of_freedom is not None:
        # Approximate t-distribution via scaled normal for visualisation
        scale = math.sqrt(degrees_of_freedom / (degrees_of_freedom - 2)) if degrees_of_freedom > 2 else 1.0
        x_range = (-4 * scale, 4 * scale)
    else:
        x_range = (-4.0, 4.0)

    step = (x_range[1] - x_range[0]) / n_pts
    xs = [x_range[0] + i * step for i in range(n_pts + 1)]
    ys = _normal_pdf(xs, 0, 1)

    ax.plot(xs, ys, color="#2d6a9f", linewidth=2)
    ax.fill_between(xs, ys, 0, alpha=0.08, color="#2d6a9f")

    # Shade rejection region(s)
    if alternative in ("two-sided", "less"):
        # left tail: x < critical_low
        crit = _normal_ppf(alpha / 2 if alternative == "two-sided" else alpha)
        xs_rej = [x for x in xs if x <= crit]
        ys_rej = _normal_pdf(xs_rej, 0, 1)
        ax.fill_between(xs_rej, ys_rej, 0, alpha=0.45, color="#d62728", label="Rejection region")
    if alternative in ("two-sided", "greater"):
        crit = _normal_ppf(1 - (alpha / 2 if alternative == "two-sided" else alpha))
        xs_rej = [x for x in xs if x >= crit]
        ys_rej = _normal_pdf(xs_rej, 0, 1)
        # Only add label here if the left tail didn't already claim it
        _right_label = "Rejection region" if alternative == "greater" else None
        ax.fill_between(xs_rej, ys_rej, 0, alpha=0.45, color="#d62728",
                        **(dict(label=_right_label) if _right_label else {}))

    # Mark observed statistic
    if test_statistic is not None:
        ax.axvline(x=test_statistic, color="#e67e22", linewidth=2,
                   linestyle="--", label=f"Test statistic = {test_statistic:.3f}")

    sig_label = "significant" if p_value < alpha else "not significant"
    ax.set_xlabel("Standard units")
    ax.set_ylabel("Density")
    ax.set_title(title or f"p = {p_value:.4f}  (alpha = {alpha})  →  {sig_label}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def _normal_ppf(p: float) -> float:
    """Approximate normal inverse CDF (rational approximation)."""
    import math
    # Rational approximation (Beasley-Springer-Moro)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator   = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    result = t - numerator / denominator
    return result if p < 0.5 else -result


# ---------------------------------------------------------------------------
# plot_distributions
# ---------------------------------------------------------------------------

def plot_distributions(
    groups: List[Sequence[float]],
    labels: Optional[List[str]] = None,
    title: str = "",
    kind: str = "box",
) -> Any:
    """
    Plot the distribution of one or more groups side-by-side.

    Parameters
    ----------
    groups : list of sequences
        Each element is a numeric sequence (one per group).
    labels : list of str, optional
        Group labels.  Defaults to ``["Group 1", "Group 2", ...]``.
    title : str
        Plot title.
    kind : str
        ``"box"`` (default), ``"violin"``, or ``"bar"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()

    if labels is None:
        labels = [f"Group {i + 1}" for i in range(len(groups))]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.8), 5))

    if kind == "violin":
        parts = ax.violinplot(groups, showmedians=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(labels)
    elif kind == "bar":
        import math
        means = [sum(g) / len(g) if g else 0.0 for g in groups]
        stds  = [
            math.sqrt(sum((v - m) ** 2 for v in g) / len(g)) if len(g) > 1 else 0.0
            for g, m in zip(groups, means)
        ]
        _stderr_bar_chart(ax, labels, means, stds)
    else:  # box (default)
        import matplotlib as _mpl
        _mpl_ver = tuple(int(x) for x in _mpl.__version__.split(".")[:2])
        _bp_kw = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
        ax.boxplot(groups, **{_bp_kw: labels}, patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                   medianprops=dict(color="white", linewidth=2))

    ax.set_title(title or "Group Distribution Comparison")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plot_result
# ---------------------------------------------------------------------------

def plot_result(result: Any, kind: str = "auto") -> Any:
    """
    Generate a figure summarising a ``HypoResult``.

    Automatically picks the best chart type based on the test:
    - ``"p_value"``       - p-value distribution curve
    - ``"bar"``           - bar chart (when group means are in data_summary)
    - ``"auto"``          - pick automatically

    Parameters
    ----------
    result : HypoResult
    kind : str
        ``"auto"``, ``"p_value"``, ``"bar"``, ``"box"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()

    test_name  = (result.test_name or "").lower()
    p_value    = result.p_value
    alpha      = result.alpha
    stat       = result.statistic
    df_stat    = result.degrees_of_freedom
    alt        = getattr(result, "alternative", "two-sided")
    d_summary  = result.data_summary or {}

    # ── auto strategy ────────────────────────────────────────────────────
    if kind == "auto":
        if "group1_mean" in d_summary and "group2_mean" in d_summary:
            kind = "comparison_bar"
        else:
            kind = "p_value"

    # ── comparison bar (two-group t-test) ────────────────────────────────
    if kind == "comparison_bar":
        import math
        means  = [d_summary.get("group1_mean", 0), d_summary.get("group2_mean", 0)]
        stds   = [d_summary.get("group1_std",  0), d_summary.get("group2_std",  0)]
        labels = ["Group 1", "Group 2"]
        n1     = d_summary.get("group1_size", 1)
        n2     = d_summary.get("group2_size", 1)
        if n1 and n2:
            labels = [f"Group 1 (n={n1})", f"Group 2 (n={n2})"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        _stderr_bar_chart(axes[0], labels, means, stds, title="Group Means ± SD")

        # p-value panel
        df_val = df_stat if isinstance(df_stat, (int, float)) else None
        _draw_p_panel(axes[1], p_value, alpha, stat, df_val, alt)

        sig = "Significant" if p_value < alpha else "Not significant"
        fig.suptitle(
            f"{result.test_name}  |  {sig}  (p = {p_value:.4f})",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    # ── p-value only ─────────────────────────────────────────────────────
    df_val = df_stat if isinstance(df_stat, (int, float)) else None
    return plot_p_value(
        p_value, alpha=alpha, degrees_of_freedom=df_val,
        test_statistic=stat, alternative=alt,
        title=f"{result.test_name}  (p = {p_value:.4f})",
    )


def _draw_p_panel(ax, p_value, alpha, test_stat, df_val, alternative):
    """Draw the p-value distribution panel on an existing Axes object."""
    import math

    n_pts = 300
    xs = [-4.0 + 8.0 * i / n_pts for i in range(n_pts + 1)]
    ys = _normal_pdf(xs, 0, 1)

    ax.plot(xs, ys, color="#2d6a9f", linewidth=2)
    ax.fill_between(xs, ys, 0, alpha=0.08, color="#2d6a9f")

    if alternative in ("two-sided", "less"):
        crit = _normal_ppf(alpha / 2 if alternative == "two-sided" else alpha)
        xs_r = [x for x in xs if x <= crit]
        ax.fill_between(xs_r, _normal_pdf(xs_r, 0, 1), 0,
                        alpha=0.45, color="#d62728")
    if alternative in ("two-sided", "greater"):
        crit = _normal_ppf(1 - (alpha / 2 if alternative == "two-sided" else alpha))
        xs_r = [x for x in xs if x >= crit]
        ax.fill_between(xs_r, _normal_pdf(xs_r, 0, 1), 0,
                        alpha=0.45, color="#d62728")

    if test_stat is not None:
        ax.axvline(x=test_stat, color="#e67e22", linewidth=2, linestyle="--")

    sig = "significant" if p_value < alpha else "not significant"
    ax.set_title(f"p = {p_value:.4f}  →  {sig}")
    ax.set_xlabel("Standard units")
    ax.set_ylabel("Density")


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def generate_report(
    result: Any,
    path: Optional[str] = None,
    fmt: str = "html",
) -> str:
    """
    Generate a self-contained HTML (or plain-text) report and optionally
    save it to *path*.

    Parameters
    ----------
    result : HypoResult
    path : str, optional
        File path to save the report.  If None, the report string is
        returned without saving.
    fmt : str
        ``"html"`` (default) or ``"text"``.

    Returns
    -------
    str : report content (HTML or plain text)

    Notes
    -----
    * PDF export requires the optional ``weasyprint`` package.
      Install with:  ``pip install weasyprint``
    * For ``fmt="html"`` matplotlib is embedded as a base-64 PNG if
      available; otherwise a text summary is embedded.
    """
    if fmt == "text":
        from ..reporting.generator import text_report
        content = text_report(result, verbose=True)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        return content

    if fmt == "pdf":
        html_content = generate_report(result, path=None, fmt="html")
        try:
            import weasyprint  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PDF export requires weasyprint. "
                "Install with:  pip install weasyprint"
            ) from exc
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        if path:
            with open(path, "wb") as fh:
                fh.write(pdf_bytes)
        return f"<PDF: {len(pdf_bytes)} bytes>"

    # Default: HTML
    from ..reporting.generator import apa_report, text_report

    # Try to embed a base64-encoded plot
    img_tag = ""
    try:
        import base64
        import io
        fig = plot_result(result)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        img_tag = (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="max-width:100%;margin:1em 0;" alt="test result plot"/>'
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass  # plotting is optional

    apa = apa_report(result)
    significance = "Significant" if result.is_significant else "Not Significant"
    sig_color = "#27ae60" if result.is_significant else "#e74c3c"

    # Build simple key-value stats table
    rows = [
        ("Test", result.test_name),
        ("Statistic", f"{result.statistic:.4f}"),
        ("p-value", f"{result.p_value:.6f}"),
        ("Significant", significance),
        ("Alpha", str(result.alpha)),
        ("Alternative", result.alternative),
    ]
    if result.degrees_of_freedom is not None:
        rows.append(("df", str(result.degrees_of_freedom)))
    if result.effect_size is not None:
        rows.append((result.effect_size_name or "Effect size",
                     f"{result.effect_size:.4f} ({result.effect_magnitude})"))
    if result.confidence_interval is not None:
        ci_level = int((1 - result.alpha) * 100)
        ci = result.confidence_interval
        rows.append((f"{ci_level}% CI",
                     f"[{ci[0]:.4f}, {ci[1]:.4f}]"))

    table_rows_html = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{result.test_name} — HypoTestX Report</title>
<style>
  body {{font-family: system-ui, sans-serif; max-width: 860px; margin: 2em auto;
         padding: 0 1em; color: #222;}}
  h1   {{color: #2d6a9f; border-bottom: 2px solid #2d6a9f; padding-bottom:.3em;}}
  h2   {{color: #444; font-size: 1.1em; margin-top:1.8em;}}
  .badge {{display:inline-block; padding:.25em .7em; border-radius:4px;
            color:#fff; font-weight:bold; background:{sig_color};}}
  table {{border-collapse: collapse; width:100%; margin:.5em 0;}}
  th    {{text-align:left; width:40%; background:#f0f4f8;
           padding:.4em .7em; border:1px solid #dde;}}
  td    {{padding:.4em .7em; border:1px solid #dde;}}
  pre   {{background:#f8f8f8; padding:1em; overflow-x:auto; font-size:.88em;
           border-left:4px solid #2d6a9f;}}
  footer{{font-size:.8em; color:#888; margin-top:2em;}}
</style>
</head>
<body>
<h1>{result.test_name}</h1>
<p><span class="badge">{significance}</span></p>

<h2>Test Statistics</h2>
<table>{table_rows_html}</table>

{img_tag}

<h2>APA Citation</h2>
<pre>{apa}</pre>

{"<h2>Interpretation</h2><p>" + result.interpretation + "</p>" if result.interpretation else ""}

<footer>Generated by HypoTestX — https://github.com/Ankit-Anand123/hypotestx</footer>
</body>
</html>
"""
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_content)
    return html_content

