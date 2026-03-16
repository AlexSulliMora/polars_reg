# Code Organization Principles

How code is structured, where new code goes, and how the data pipeline works.

## Module Placement

Estimators live in `_<name>.py`, one estimator family per module. A "family" groups functions that share data extraction, formula parsing, or mathematical framework:

| Module              | Functions                                        |
|---------------------|--------------------------------------------------|
| `_ols.py`           | `ols()`                                          |
| `_iv.py`            | `iv2sls()`                                       |
| `_gmm.py`           | `liml()`, `gmm_iv()`                             |
| `_panel.py`         | `panel_fe()`, `panel_re()`, `panel_fd()`         |
| `_arellano_bond.py` | `panel_ab()`, `panel_sys_gmm()`                  |
| `_binary.py`        | `probit()`, `logit()`, `marginal_effects()`, `odds_ratios()` |
| `_ppml.py`          | `ppml()`                                         |
| `_quantile.py`      | `quantreg()`                                     |

**Singleton modules are valid** when an estimator has a distinct mathematical framework (e.g., PPML is IRLS-based Poisson pseudo-likelihood, quantile regression is IRLS + bootstrap).

**Heuristic for new code:** if the new function shares its mathematical framework with an existing module, it belongs there. If it introduces a fundamentally different estimation approach, it gets its own module.

**Diagnostics** live in `_diagnostics.py`. Split a diagnostic into its own module only when the function + result class + supporting helpers exceed ~300 lines of self-contained code.

**Known non-conformance:** `quantreg()` lives in `_quantile.py` -- the function name and module name differ. All other estimators match their module name.

## Public API Surface

Everything users import comes from `__init__.py`:

```python
import polars_reg as pr
result = pr.ols("y ~ x1 + x2", df)
```

All internal modules are prefixed with `_` (`_ols.py`, `_se.py`, `_utils.py`). Users should never import from these directly.

**Naming exceptions:** `stata.py` and `r_equiv.py` lack the underscore prefix because they wrap external tools with optional dependencies. The un-prefixed name signals "external integration layer, not internal implementation."

## Data Flow Contract

### Input Requirement

All public functions accept `pl.DataFrame | pl.LazyFrame` only. No pandas auto-conversion. Users with pandas DataFrames call `pl.from_pandas()` themselves. This is a Polars-native package -- pushing conversion to the caller simplifies every estimator's entry point.

**Current state:** `ensure_polars()` validates that input is `pl.DataFrame | pl.LazyFrame` and raises `TypeError` with a helpful message if not. It is called in estimator entry points intentionally — the validation is cheap and catches a common mistake early.

### Single Data Cleaning Step

`extract_arrays()` in `_utils.py` handles ALL data cleaning in one place:

1. NaN → null conversion (IEEE NaN passes through Polars `drop_nulls()`, so must be converted first)
2. inf → null conversion
3. Null dropping
4. Polars columns → NumPy arrays

After `extract_arrays()`, row count is fixed and all values are finite. No downstream code should independently sanitize data.

**Current state:** `extract_arrays()` handles inf/NaN sanitization for all estimators that use it. The Rust fast-paths in `_ols.py` and `_iv.py` call `sanitize_inf()` independently because they bypass `extract_arrays()` for performance. This is an accepted exception — the Rust paths need pre-cleaned data before passing to native code.

**Why this matters:** the seam between data cleaning steps (singleton drop, code reindexing, NaN-to-null conversion) was the source of the project's most critical bug cluster. See `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md`.

### Pipeline Variants

**Canonical pipeline** (OLS, IV, panel FE/RE/FD):

```
Formula → column selection (LazyFrame) → extract_arrays() [cleans + extracts]
  → (demean if FE) → estimate → _se.py VCV → RegressionResult
```

**MLE variant** (probit, logit, PPML):

```
Formula → extract_arrays() → Newton-Raphson IRLS
  → Hessian-based VCV (uses _clustered_meat directly) → RegressionResult
```

MLE estimators bypass `vcov_robust()` / `vcov_clustered()` and construct their sandwich VCV internally.

**Multi-equation variant** (GRS test):

```
Formula → parse_formula() → custom array extraction
  → per-asset OLS → cross-equation test → typed result dataclass
```

## Rust Extension

**Single implementation:** all compute functions exist in Rust only. The native extension (`polars_reg._native`) is a build requirement, not optional. No Python fallback, no `_HAS_NATIVE` flag, no dual-path branching. Wheels are distributed via CI for all platforms, so most users never compile from source.

**One codebase, one truth:** when a bug is fixed, it is fixed once. No risk of Rust and Python paths diverging.

This principle exists because of a real incident: the Rust OLS path once returned zeros for fitted values while the Python path worked correctly (C6 in comprehensive code review). Dual paths create dual maintenance burden and dual failure modes.

**Status:** All Python fallback paths removed. Every module imports directly from `polars_reg._native` with no `try/except` guards.

**Adding new Rust functions:** justify with benchmarks showing meaningful speedup on representative workloads. Functions called inside iterative convergence loops (e.g., demeaning) have their effective speedup multiplied by iteration count, so even modest per-call improvements are worthwhile.

## Test Organization

**File naming:** one test file per internal module, named `test_<module>.py`:

| Test file                | Tests                          |
|--------------------------|--------------------------------|
| `test_ols.py`            | `_ols.py`                      |
| `test_iv.py`             | `_iv.py`                       |
| `test_panel.py`          | `_panel.py`                    |
| `test_diagnostics.py`    | `_diagnostics.py`              |
| `test_stata_parity.py`   | Stata parity (requires Stata)  |
| `test_r_equiv.py`        | R equivalence (requires R)     |

**Cross-cutting tests** use descriptive names: `test_bootstrap.py`, `test_hac.py`, `test_pandas_compat.py`.

**Function naming:** `test_<feature>_<scenario>` (e.g., `test_ols_fe_python_vs_rust`). Prefer the full `test_<module>_<feature>_<scenario>` form for grep-ability.

**Split threshold:** a function gets its own test file only when it has ~200+ lines of test cases.

### Testing Philosophy

From `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md`:

> "Tests that pass are not tests that protect."

- Avoid tautological assertions that test algebraic identities
- Test the formulas that differentiate this package: dfc corrections, demeaning convergence, Stata parity
- Use deterministic RNG seeds: `np.random.default_rng(seed)`, not `np.random.randn()`
- For bootstrap tests: either fix the seed or test weak properties that hold with overwhelming probability

### Pre-PR Checklist

Before creating a pull request for a new feature or significant change:

1. **Tests pass:** `pytest` with no failures
2. **Lint clean:** `ruff check .` and `ruff format --check .`
3. **Showcase notebook:** create `notebooks/new_features/YYYY-MM-DD-<name>.ipynb` demonstrating the feature with imports, data simulation, and worked examples. Use `compare()` to show cross-package parity where applicable. See CLAUDE.md for the full convention.
