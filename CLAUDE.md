# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`polars_reg` is a Python package implementing econometric regression methods using Polars DataFrames. It recreates Stata's `reghdfe`, `ivregress`, and related estimators with robust/clustered standard errors throughout. **All estimators are validated against Stata output** (94 tests, 27 Stata parity tests passing).

### Implemented estimators:
- **`ols()`** — OLS with HC0-HC3 robust and multi-way clustered SEs
- **`ols()` with FE** — reghdfe-style absorption via iterative demeaning (formula: `y ~ x1 | fe1 + fe2`)
- **`iv2sls()`** — two-stage least squares with first-stage F-stat
- **`liml()`** — limited information maximum likelihood
- **`gmm_iv()`** — two-step efficient GMM with Hansen J test
- **`panel_fe()`** — within estimator for panel data

### Design principles:
- Polars-native: accept `pl.DataFrame` / `pl.LazyFrame`, never convert to pandas
- Formula API: `ols("y ~ x1 + x2 | fe1 + fe2", data=df, cluster=["fe1"])`
- NumPy/SciPy for linear algebra, Polars for data handling

## Build & Development Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run all tests (includes Stata parity if Stata available)
pytest

# Run unit tests only (no Stata needed)
pytest tests/ -k "not Parity"

# Run Stata parity tests (requires Stata BE/SE/MP on Windows via WSL2)
pytest tests/test_stata_parity.py -v

# Lint and format
ruff check .
ruff format .

# Type checking
mypy polars_reg/

# Generate API reference docs (output: docs/api/)
uv run pdoc polars_reg -o docs/api --docformat google

# Serve API docs locally
uv run pdoc polars_reg --docformat google
```

## Architecture

```
polars_reg/
    __init__.py      # Public API exports: ols, iv2sls, liml, gmm_iv, panel_fe, RegressionResult
    _formula.py      # Formula parser → FormulaSpec
    _utils.py        # Polars-to-NumPy extraction → ExtractedArrays
    _demean.py       # Symmetric Kaczmarz + CG acceleration (Correia 2016)
    _ols.py          # OLS estimator (with FE absorption, nesting detection)
    _iv.py           # 2SLS/IV estimator
    _gmm.py          # LIML and GMM-IV estimators
    _panel.py        # Panel FE (within) estimator
    _se.py           # VCV: iid, HC0-3, one-way/multi-way clustered
    _results.py      # RegressionResult dataclass with .summary()
tests/
    stata_compare.py # WSL2 batch-mode Stata execution and result comparison
    test_stata_parity.py  # 27 Stata parity tests (OLS, reghdfe, IV)
    test_*.py        # Unit tests for each module
```

### Key internal flow:
1. **Formula parsing** (`_formula.py`): Parse formula string → `FormulaSpec` (depvar, regressors, FE, endogenous/instruments)
2. **Data extraction** (`_utils.py`): Polars columns → NumPy arrays (`ExtractedArrays`)
3. **Demeaning** (`_demean.py`): Multi-way FE absorption via symmetric Kaczmarz + conjugate gradient
4. **Estimation** (`_ols.py`, `_iv.py`, `_gmm.py`): Solve normal equations or IV moment conditions
5. **Standard errors** (`_se.py`): VCV matrix with appropriate correction
6. **Results** (`_results.py`): Package into `RegressionResult`

### Critical dfc formulas (must match Stata exactly):
- **`reg, cluster()`**: `dfc = G/(G-1) * (N-1)/(N-k)`
- **`reghdfe, cluster()`**: `dfc = G/(G-1) * N/(N-d-k)` where `d` = FE DoF not nested in any cluster dim
- **`reghdfe` multi-way cluster**: single dfc using `min(G)` across cluster dimensions
- **Nesting**: FE is nested in cluster if every FE group maps to exactly one cluster group
- **GMM VCV**: `V = n * (X'Z S⁻¹ Z'X)⁻¹` where `S = (1/n) Z'diag(e²)Z`

## Conventions

- All internal modules prefixed with underscore (`_ols.py`, not `ols.py`)
- Public API lives in `__init__.py` — users import from `polars_reg` directly
- Demeaning convergence tolerance: default 1e-8
- Stata parity tests use `REGHDFE = 1e-5` tolerance (demeaning algorithm differences)

## Feature Showcase Notebooks

After implementing a new feature or significant change, create a Jupyter notebook that demonstrates the feature for human review.

**Location:** `notebooks/new_features/YYYY-MM-DD-<descriptive-name>.ipynb`

**Structure:**
1. Title + brief description of what's new
2. Imports
3. Data simulation (with known true parameters when possible)
4. Feature demonstrations — one section per feature, each showing inputs and outputs
5. Cross-package comparison using `compare()` where applicable

**Example:** `notebooks/new_features/2026-03-15-package-comparison.ipynb`

**Guidelines:**
- Use `report.summary()` (renders GT in Jupyter) rather than `print()` for comparison reports
- Include `report.code()` output so readers can trace results
- Show both simple and complex use cases
- Use `np.random.default_rng(seed)` for reproducible data

## Principles

Detailed development principles are in `docs/principles/`:
- [`code-organization.md`](docs/principles/code-organization.md) — module placement, data flow, Rust contract, test structure
- [`api-consistency.md`](docs/principles/api-consistency.md) — return types, parameter naming/ordering, vcov vocabulary, error handling
- [`statistical-rigor.md`](docs/principles/statistical-rigor.md) — citation standards, numerical engineering, validation expectations
