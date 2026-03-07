# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`polars_reg` is a Python package implementing econometric regression methods using Polars DataFrames. The goal is to recreate Stata's `reghdfe` (linear regression with multi-way fixed effects via iterative demeaning) and various instrumental variable (IV) regression estimators, with robust/clustered standard errors throughout.

### Target regression methods:
- **OLS** with robust (HC0-HC3) and multi-way clustered standard errors (Cameron-Gelbach-Miller inclusion-exclusion for arbitrary N clustering dimensions)
- **High-dimensional fixed effects** (reghdfe-style absorption via iterative demeaning/Frisch-Waugh-Lovell)
- **2SLS / IV regression** (two-stage least squares)
- **LIML** (limited information maximum likelihood)
- **GMM-IV** (generalized method of moments)
- **Panel data**: fixed effects, random effects, first-difference estimators

### Design principles:
- Polars-native: accept `pl.DataFrame` / `pl.LazyFrame` as inputs, never convert to pandas internally
- Stata-like formula API (e.g., `reg("y ~ x1 + x2 | fe1 + fe2", data=df, cluster=["fe1", "fe2"])`)
- Return result objects with `.summary()`, coefficient tables, diagnostics (F-stat, weak instrument tests, Sargan/Hansen J)
- NumPy/SciPy for linear algebra internals (Polars for data handling, not matrix ops)
- Sparse matrices where beneficial (fixed effects dummies, instrument matrices)

## Build & Development Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_ols.py

# Run a specific test
pytest tests/test_ols.py::test_robust_se -v

# Lint and format
ruff check .
ruff format .

# Type checking
mypy polars_reg/
```

## Architecture

```
polars_reg/
    __init__.py          # Public API: reg(), iv_reg(), etc.
    _formula.py          # Formula parser (y ~ x1 + x2 | fe1 + fe2, endog = (x3 ~ z1 + z2))
    _demean.py           # Iterative demeaning for high-dimensional FE absorption
    _ols.py              # OLS estimator
    _iv.py               # IV/2SLS/LIML estimators
    _gmm.py              # GMM estimator
    _panel.py            # Panel data estimators (FE, RE, FD)
    _se.py               # Standard error computations (robust, multi-way clustered, HAC)
    _results.py          # Regression result objects, summary tables, diagnostics
    _utils.py            # Polars-to-numpy conversion, dof adjustments, sparse helpers
tests/
    conftest.py          # Shared fixtures (sample DataFrames, known Stata results)
    test_ols.py
    test_iv.py
    test_demean.py
    test_se.py
    test_formula.py
```

### Key internal flow:
1. **Formula parsing** (`_formula.py`): Parse user formula string into dependent var, regressors, fixed effects, endogenous/instrument specs
2. **Data extraction** (`_utils.py`): Convert relevant Polars columns to NumPy arrays (or sparse matrices for FE dummies)
3. **Demeaning** (`_demean.py`): For multi-way FE models, iteratively demean y and X within groups using the alternating projections algorithm (Gauss-Seidel acceleration)
4. **Estimation** (`_ols.py`, `_iv.py`, etc.): Solve normal equations or IV moment conditions
5. **Standard errors** (`_se.py`): Compute variance-covariance matrix with requested correction (HC0-3, multi-way clustering via Cameron-Gelbach-Miller inclusion-exclusion, Driscoll-Kraay)
6. **Results** (`_results.py`): Package estimates, SEs, test statistics, diagnostics into result object

### Stata parity targets:
- `reghdfe`: OLS + absorbed multi-way FE + clustered SEs (match coefficients and SEs to 6+ decimal places)
- `ivregress 2sls` / `ivreg2`: 2SLS with robust/clustered SEs
- `xtivreg`: Panel IV estimators
- First-stage F-statistics, weak instrument tests (Stock-Yogo, Kleibergen-Paap)
- Sargan/Hansen over-identification tests

## Conventions

- All internal modules prefixed with underscore (`_ols.py`, not `ols.py`)
- Public API lives in `__init__.py` — users import from `polars_reg` directly
- Test fixtures should include known results from Stata for validation
- Use `scipy.sparse` for fixed-effect dummy matrices when number of groups > 1000
- Demeaning convergence tolerance: default 1e-8, configurable
