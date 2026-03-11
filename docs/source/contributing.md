# Contributing

Contributions of all kinds are welcome — new statistical tests, NLP improvements,
better documentation, bug fixes, and example notebooks.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/HypoTestX.git
cd HypoTestX

# 2. Create a virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks (black, isort, flake8)
pre-commit install

# 5. Run the test suite to verify everything works
pytest
```

---

## Running Tests

```bash
# Full suite
pytest

# With coverage report
pytest --cov=hypotestx --cov-report=term-missing

# Specific test file
pytest tests/test_tests/test_parametric.py

# Specific test function
pytest tests/test_tests/test_parametric.py::test_two_sample_ttest_basic
```

---

## Contribution Areas

| Area | Details |
|---|---|
| **New statistical tests** | Add to `hypotestx/tests/` and expose through `hypotestx/__init__.py` |
| **NLP / routing improvements** | Extend `_TESTS_BY_KEYWORD` in `fallback.py`, or improve the LLM prompts in `prompts.py` |
| **Visualizations** | Add new chart types to `hypotestx/explore/visualize.py` |
| **Educational content** | Improve docstrings with examples, add entries to `hypotestx/education/` |
| **Domain packages** | Add specialist tests to `hypotestx/domains/` |
| **Documentation** | Update or add pages in `docs/source/` |
| **Bug fixes** | Check open issues on GitHub |

---

## Code Style

### PEP 8

The project uses **black** (line length 88) for formatting and **isort** for
import order. Pre-commit hooks enforce these automatically.

```bash
# Format manually
black hypotestx/
isort hypotestx/
```

### Type hints

All public functions must have complete type annotations:

```python
def ttest_2samp(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    equal_var: bool = False,
) -> HypoResult:
```

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def my_test(data: list[float], mu: float = 0.0) -> HypoResult:
    """
    One-line summary.

    Longer description if needed.

    Args:
        data: The sample data as a list of floats.
        mu: The null hypothesis mean. Defaults to 0.

    Returns:
        A HypoResult object.

    Example:
        >>> result = my_test([1.2, 3.4, 2.1], mu=2.0)
        >>> result.is_significant
        False
    """
```

### Test coverage

New code must have **95 %+ test coverage**. Place tests in the corresponding
`tests/` subdirectory:

- `tests/test_tests/` — for `hypotestx/tests/`
- `tests/test_core/` — for `hypotestx/core/`
- `tests/test_stats/` — for `hypotestx/stats/`

---

## Pull Request Process

1. **Branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feat/two-way-anova
   git checkout -b fix/fallback-column-matching
   ```
2. **Write tests first** — confirm the test fails, then implement the feature.
3. **Keep commits atomic** — one logical change per commit.
4. **Run the full test suite** before opening a PR:
   ```bash
   pytest --cov=hypotestx
   flake8 hypotestx/
   mypy hypotestx/
   ```
5. **Open a Pull Request** against `main` and fill in the PR template.
6. At least one review approval is required before merging.

---

## Filing a Bug Report

Please use the [GitHub issue tracker](https://github.com/Ankit-Anand123/HypoTestX/issues)
and include:

- HypoTestX version (`pip show hypotestx`)
- Python version
- Minimal reproducible example
- Full traceback
