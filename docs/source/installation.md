# Installation

## Requirements

- **Python 3.8 or later**
- No mandatory external dependencies — all statistical math and LLM HTTP calls use only Python stdlib

## Basic Installation

```bash
pip install hypotestx
```

This installs HypoTestX with zero runtime dependencies. All test functions, the
regex fallback router, and all LLM backends work out of the box.

## Optional Extras

### Visualization

Enables `result.plot()`, `plot_result()`, `plot_distributions()`, `plot_p_value()`,
and the embedded-chart path in `generate_report()` / `export_html()`:

```bash
pip install hypotestx[visualization]
# installs: matplotlib>=3.5.0, plotly>=5.0.0
```

### PDF Reporting

Enables `export_pdf()` and `generate_report(fmt="pdf")` via WeasyPrint (HTML → PDF):

```bash
pip install hypotestx[reporting]
# installs: weasyprint>=53.0
```

### Development

Installs testing, linting, and formatting tools:

```bash
pip install hypotestx[dev]
# installs: pytest, pytest-cov, black, isort, flake8, mypy, pre-commit, pandas
```

### Documentation

Installs tools needed to build this documentation:

```bash
pip install hypotestx[docs]
# installs: sphinx>=7.0, furo>=2024.1.29, myst-parser>=2.0
```

### All Extras

```bash
pip install hypotestx[all]
# installs: matplotlib, plotly, weasyprint, pandas
```

---

## Installing from Source

```bash
git clone https://github.com/Ankit-Anand123/HypoTestX.git
cd HypoTestX
pip install -e ".[dev]"
```

---

## Ollama (Local LLM Backend)

To use the `"ollama"` backend for fully offline, free LLM routing:

1. **Install Ollama** from [https://ollama.com](https://ollama.com) (macOS, Linux, Windows)
2. **Pull a model:**

```bash
ollama pull llama3.2        # recommended default (~2 GB)
ollama pull mistral         # good alternative (~4 GB)
ollama pull phi4            # smaller, fast (~2.5 GB)
```

3. **Use in HypoTestX:**

```python
import hypotestx as hx

result = hx.analyze(
    df,
    "Is age correlated with salary?",
    backend="ollama",
    model="llama3.2",
)
```

Ollama runs a local server at `http://localhost:11434` by default. Override with
`host="http://your-host:port"`.

---

## HuggingFace Local Inference

To run models locally via the `"huggingface"` backend:

```bash
pip install transformers torch
```

Usage:

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

> **Note:** Downloading large transformer models requires significant disk space
> and RAM/VRAM. For most use cases the HuggingFace Inference API (cloud, free tier)
> is a simpler option — just pass a `token` instead of `use_local=True`.
