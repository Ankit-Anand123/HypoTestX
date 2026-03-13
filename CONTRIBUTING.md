# Contributing to HypoTestX

Thank you for your interest in contributing! Please read this guide carefully before opening a PR.

---

## Project Scope

HypoTestX is a **statistical hypothesis testing library** for Python.

**In scope:**
- Parametric and non-parametric hypothesis tests
- Bayesian testing methods
- Power analysis and sample size calculations
- Assumption checking and diagnostics
- Statistical reporting and visualisation
- The NLP/LLM routing layer for natural-language test selection
- EDA and correlation utilities that directly support hypothesis testing

**Out of scope (PRs will be closed):**
- General-purpose data manipulation utilities
- Machine learning model training
- Data engineering / ETL tools
- Anything unrelated to statistical inference

If you are unsure whether your idea fits, open a **Feature Request** issue first.

---

## Module Map

| What you want to add | Where it goes |
|---|---|
| Parametric test (t-test, ANOVA, …) | `hypotestx/tests/parametric.py` |
| Non-parametric test (Mann-Whitney, …) | `hypotestx/tests/nonparametric.py` |
| Categorical test (chi-square, Fisher, …) | `hypotestx/tests/categorical.py` |
| Correlation test | `hypotestx/tests/correlation.py` |
| Bayesian method | `hypotestx/bayesian/` |
| Power / sample size | `hypotestx/power/` |
| Robust estimator | `hypotestx/robust/` |
| Time series test | `hypotestx/timeseries/` |
| Core engine logic | `hypotestx/core/` — discuss in an issue first |
| NLP / LLM routing | `hypotestx/core/llm/` — discuss in an issue first |
| Reporting / export | `hypotestx/reporting/` |

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/hypotestx.git
cd hypotestx

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev,visualization]"

# 4. Install pre-commit hooks (runs black, isort, flake8 on every commit)
pre-commit install
```

---

## Adding a New Statistical Test — Step by Step

1. **Implement** the function in the appropriate module (see Module Map above).
2. **Docstring** — every public function must have:
   - A one-line summary
   - `Parameters` and `Returns` sections (NumPy style)
   - A `References` section citing the paper or textbook that defines the method
3. **Export** — add the function to the module's `__init__.py`.
4. **Tests** — add a test file (or extend an existing one) under `tests/`. Tests must:
   - Cover the happy path with known expected values
   - Cover edge cases (empty input, single sample, etc.)
   - Not duplicate tests that already exist in `scipy.stats`
5. **Run the full suite** locally before pushing:
   ```bash
   pytest tests/ --cov=hypotestx -v
   ```

---

## Code Style

This project enforces consistent style automatically.

| Tool | Purpose | Config |
|---|---|---|
| `black` | Formatting | `pyproject.toml` |
| `isort` | Import ordering | `pyproject.toml` |
| `flake8` | Linting | `.flake8` |
| `mypy` | Type checking | `pyproject.toml` |

Run all checks manually:
```bash
black hypotestx/ tests/
isort hypotestx/ tests/
flake8 hypotestx/ tests/
mypy hypotestx/ --ignore-missing-imports
```

Pre-commit hooks run these automatically on `git commit`.

---

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Brunner-Munzel test to nonparametric module
fix: correct p-value calculation in two-sample t-test
docs: add docstring references for Wilcoxon test
test: add edge case tests for chi-square with zero cells
refactor: simplify assumption checker logic
ci: increase coverage threshold to 80%
```

---

## Pull Request Process

1. Branch off `main`: `git checkout -b feat/brunner-munzel-test`
2. Make your changes following the steps above
3. Push and open a PR — fill in the PR template completely
4. CI must pass (tests, lint, type check, coverage ≥ 75%)
5. A maintainer will review and may request changes
6. Once approved, the maintainer merges — **do not merge your own PR**

---

## What Gets a PR Closed Without Review

- Adding code with no tests
- Dropping test coverage below 75%
- Adding unrelated utilities outside the project scope
- Reformatting entire files that are unrelated to the PR
- PRs that do not fill in the PR template
