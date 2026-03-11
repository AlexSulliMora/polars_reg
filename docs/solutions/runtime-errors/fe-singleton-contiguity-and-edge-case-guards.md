---
title: FE Singleton Contiguity and Edge Case Guards
type: runtime-error
date: 2026-03-11
component: _demean, _ols, _iv, _panel, _utils, _se
severity: critical
tags: [singleton, fixed-effects, demeaning, NaN, CG-overflow, contiguity, edge-case]
related_files:
  - polars_reg/_demean.py
  - polars_reg/_ols.py
  - polars_reg/_iv.py
  - polars_reg/_panel.py
  - polars_reg/_utils.py
  - polars_reg/_se.py
  - src/lib.rs
related_tests:
  - tests/test_demean.py
  - tests/test_dual_path.py
  - tests/conftest.py
related_docs:
  - docs/plans/2026-03-11-test-edge-case-robustness-plan.md
  - docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md
  - .claude/logs/2026-03-11_code-review/summary.md
---

# FE Singleton Contiguity and Edge Case Guards

## Problem Symptom

Three runtime warnings during normal package usage (not tests):
1. `matmul` warnings in `_ols.py:498` and `_se.py:27`
2. Demeaning non-convergence warning in `_demean.py:198`
3. NaN coefficients and Inf standard errors in regression output

No `LinAlgError` was raised — results were silently corrupted.

## Root Cause

**Non-contiguous FE codes after `drop_singletons()` caused CG demeaning overflow and NaN propagation.**

Causal chain:
1. `drop_singletons()` filters rows, leaving FE codes like `[0, 3, 7]` instead of `[0, 1, 2]`
2. Demeaning uses `n_groups = max(codes) + 1`, creating phantom zero-count groups (bins 1, 2, 4, 5, 6)
3. CG acceleration computes `alpha = ssr / uv` — phantom groups make `uv` very small
4. `alpha` overflows or becomes NaN → `x += alpha * u` contaminates the demeaned output
5. `np.linalg.solve(X'X, X'y)` accepts NaN input without raising an exception (LU decomposition doesn't check finiteness)
6. NaN coefficients and Inf standard errors propagate to `RegressionResult`

### Why It Was Subtle

- NumPy's `linalg.solve` and `linalg.inv` silently accept NaN matrices
- IEEE NaN passes through Polars `drop_nulls()` (NaN != Polars null)
- The "seam" between `drop_singletons` and `demean` was untested — unit tests for each function passed individually

## Investigation Steps

1. **Symptom observation:** OLS with FE produced NaN coefficients on real-world panel data
2. **Traced to demeaning:** NaN appeared in demeaned output, not in the linear solve
3. **Identified phantom groups:** After singleton drop, `np.bincount(codes)` showed zero-count bins
4. **CG breakdown:** `uv = sum(u * v)` became near-zero from phantom group contributions → `alpha = ssr/uv` overflow
5. **Confirmed with minimal repro:** Created dataset with forced singletons → non-contiguous codes → NaN after demean
6. **Audited all estimators:** `_ols.py`, `_iv.py`, `_panel.py` all had the same pattern; `_ppml.py`, `_binary.py`, `_gmm.py`, `_arellano_bond.py`, `_quantile.py` were unaffected (no FE absorption or use first-differencing)

## Solution

Four layers of defense implemented across Python and Rust codepaths:

### Layer 1: Data Cleaning (`_utils.py`)

```python
# Convert IEEE NaN to Polars null before drop_nulls
float_cols = [c for c in numeric_cols if df[c].dtype.is_float()]
if float_cols:
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    ])

# Early rejection of empty DataFrames
if row_count == 0:
    raise ValueError("DataFrame has no observations")
```

### Layer 2: Contiguity Enforcement (`_demean.py`)

```python
def reindex_fe_codes(fe_dict: dict[str, NDArray]) -> dict[str, NDArray]:
    """Re-index FE codes to be contiguous (0, 1, 2, ...) after filtering."""
    return {k: np.unique(v, return_inverse=True)[1] for k, v in fe_dict.items()}
```

Called immediately after `drop_singletons()` in `_ols.py`, `_iv.py`, and `_panel.py`:

```python
keep = drop_singletons(fe_dict)
if not keep.all():
    if keep.sum() == 0:
        raise ValueError("All observations dropped as singletons")
    fe_dict = {k: v[keep] for k, v in fe_dict.items()}
    fe_dict = reindex_fe_codes(fe_dict)  # CRITICAL
```

### Layer 3: CG Overflow Detection (`_demean.py`)

```python
# Inside CG iteration loop
if abs(uv) < 1e-30:
    break  # CG coefficient denominator too small

ssr_new = np.sum(r * r)
if not np.isfinite(ssr_new):
    raise ValueError("Demeaning diverged (numerical overflow).")

# After loop
if not np.all(np.isfinite(x)):
    raise ValueError("Demeaning produced non-finite values.")
```

### Layer 4: Estimator-Level Guards (`_se.py`)

```python
# Clustered SEs
if G < 2:
    raise ValueError("Clustered SEs require at least 2 cluster groups")

# Driscoll-Kraay SEs
if T < 2:
    raise ValueError("Driscoll-Kraay SEs require at least 2 time periods")
```

### Rust Native Extension (`src/lib.rs`)

Equivalent fixes applied:
- `reindex_codes()` function using mapping array
- CG overflow detection: `if !ssr_new.is_finite() { break; }`
- Applied at all 3 `drop_singletons_mask` call sites

## Testing Gap Analysis

**Why this slipped through:**
- `test_drop_singletons` tested the boolean mask but never demeaned the filtered result
- All demeaning tests used perfectly contiguous codes (no gaps)
- The "seam" between `drop_singletons` → `demean` was untested
- No test injected NaN into float columns

**"Seam testing" lesson:** Unit tests for function A and function B can both pass while A→B fails. The seam between data transformation and numerical computation needs explicit tests.

## Tests Added

77 new tests across 14 test files covering:

| Category | Count | Examples |
|----------|-------|---------|
| Input validation (NaN, empty, types) | ~16 | NaN-to-null conversion, empty DataFrame rejection |
| Singleton handling | ~8 | Cascading singletons, all-singletons error, reindex |
| Demean edge cases | ~10 | Single obs, constant columns, extreme values, many FE levels |
| Public API robustness | ~25 | OLS/IV/panel with messy data, G=1 cluster, LazyFrame |
| Dual-path (Rust vs Python) | 4 | Coefficient and SE parity via monkeypatch |
| Seam tests | ~14 | drop_singletons→reindex→demean pipeline |

Key fixture: `messy_data` in `conftest.py` (500 rows with NaN, singletons, mixed types).

## Prevention Strategies

### Code Review Checklist

- [ ] After any `drop_singletons()`, is `reindex_fe_codes()` called immediately?
- [ ] Are all associated arrays (weights, clusters, time) filtered in lock-step?
- [ ] Does the numerical algorithm check `np.isfinite()` inside iteration loops?
- [ ] Are degenerate cases guarded (G<2, T<2, N=0, all singletons)?
- [ ] At library boundaries (Polars↔NumPy), are missing values converted explicitly?

### Seam Test Template

```python
def test_seam_singleton_drop_then_demean():
    """Seam: drop_singletons -> reindex_fe_codes -> demean must produce finite output."""
    rng = np.random.default_rng(42)
    n = 200
    fe_dict = {"a": rng.integers(0, 10, size=n), "b": rng.integers(0, 8, size=n)}
    fe_dict["a"][0] = 99  # force singleton

    keep = drop_singletons(fe_dict)
    fe_filtered = {k: v[keep] for k, v in fe_dict.items()}
    fe_reindexed = reindex_fe_codes(fe_filtered)

    X = rng.standard_normal((keep.sum(), 3))
    result = demean(X, fe_reindexed)
    assert np.all(np.isfinite(result))
```

### Invariant Documentation Pattern

Every transformation function should document output invariants; every computation function should validate input invariants:

```python
def reindex_fe_codes(fe_dict):
    """Output: all code arrays are contiguous [0, G-1] with no gaps."""

def demean(X, fe_dict):
    """Input: fe_dict codes must be contiguous [0, G-1]. Use reindex_fe_codes() after filtering."""
```

## Cross-References

- **Comprehensive code review:** `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md` (82 issues, 16 critical)
- **Test robustness plan:** `docs/plans/2026-03-11-test-edge-case-robustness-plan.md` (5 phases, ~110 tests)
- **Review logs:** `.claude/logs/2026-03-11_code-review/review_demean.md` (C1, C2, C3, C16)
- **Design plans:** `docs/plans/2026-03-08-hac-ivfe-re-parity-design.md` (SE types with FE)
- **Progress log:** `docs/plans/progress-log.md` (Phase 6: demeaning pipeline)
