# Reduce Python/Rust Boundary Crossings

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Minimize Python<->Rust round-trips by consolidating hot paths into single Rust calls, eliminating NumPy from performance-critical codepaths.

**Architecture:** Extend the existing `rust_ols_from_arrays` pattern to cover all OLS variants (no-FE, robust SE, HC0-HC3) and IV/panel estimators. The end state: one Rust call per regression, returning (beta, vcov, residuals, stats) directly.

**Tech Stack:** PyO3, numpy crate, rayon

---

## Current Boundary Map (N=1M profiling)

```
Python path (no FE, HC1):
  extract_arrays   ~54ms  ← Polars→numpy conversion, column_stack, astype
  np.linalg.solve  ~25ms  ← BLAS (already fast)
  vcov_robust HC1  ~37ms  ← sandwich X' diag(e²) X in NumPy
  ─────────────────────
  Total:          ~106ms

Rust direct path (2-way FE + cluster):
  _to_codes_fast    ~5ms  ← Python loop over columns
  rust_ols_from_arrays    ← single Rust call does demean+solve+SE
  Python result wrap ~2ms
  ─────────────────────
  Total:          ~160ms  (dominated by demeaning, not boundaries)
```

**Key insight:** The FE+cluster path already has minimal boundaries. The big wins are:
1. **No-FE OLS** — currently 100% Python, should be one Rust call
2. **Robust SEs (HC0-HC3)** — sandwich matrix computed in NumPy after Rust returns
3. **IV/LIML/GMM** — entirely Python, including extract_arrays overhead
4. **Panel estimators** — entirely Python

## Task 1: `rust_ols_nofe` — Single Rust call for plain OLS (no FE)

Currently the no-FE path goes: `extract_arrays` (Python) → `np.linalg.solve` (NumPy) → `vcov_*` (Python/NumPy). This is the most common regression call and should be a single Rust entry point.

**Files:**
- Modify: `src/lib.rs` — add `rust_ols_nofe` function
- Modify: `polars_reg/_ols.py` — route no-FE calls through Rust

**What `rust_ols_nofe` does (all in one call):**
1. Accept: y_col, x_cols (individual arrays), cl_cols (optional), vcov_type string
2. Build X row-major from individual columns (no Python column_stack)
3. Add intercept column (ones) in Rust
4. Compute X'X, X'y, solve, residuals
5. Compute VCV based on vcov_type:
   - "iid": sigma² (X'X)⁻¹
   - "HC0": (X'X)⁻¹ X'diag(e²)X (X'X)⁻¹
   - "HC1": n/(n-k) * HC0
   - "HC2": uses hat matrix diagonal h_ii = x_i'(X'X)⁻¹x_i
   - "HC3": e²/(1-h_ii)² weights
   - "cluster": clustered meat + CGM (already implemented in Rust, reuse `clustered_meat_raw`)
6. Return: (beta, vcov, residuals, r2, r2_adj, n, k, cluster_n_groups)

**Rust implementation sketch:**

```rust
#[pyfunction]
fn rust_ols_nofe<'py>(
    py: Python<'py>,
    y_col: PyReadonlyArray1<'py, f64>,
    x_cols: Vec<PyReadonlyArray1<'py, f64>>,
    x_names: Vec<String>,
    add_intercept: bool,
    cl_cols: Vec<PyReadonlyArray1<'py, i32>>,
    cl_names: Vec<String>,
    vcov_type: String,  // "iid", "HC0", "HC1", "HC2", "HC3"
) -> PyResult<(...)>
```

**For HC2/HC3**, compute hat matrix diagonal in Rust:
```rust
// h_ii = x_i' (X'X)^{-1} x_i — O(n*k²) via precomputed (X'X)^{-1}
for i in 0..n {
    let row = &x_flat[i*k..(i+1)*k];
    let mut h = 0.0;
    for j in 0..k {
        for l in 0..k {
            h += row[j] * xtx_inv[j*k+l] * row[l];
        }
    }
    hat[i] = h;
}
```

**Python routing in `ols()`:**
```python
# Before extract_arrays, check if eligible for Rust no-FE path
use_nofe_rust = (
    _HAS_NATIVE
    and not spec.fe
    and not weights and not fweights
    and not spec.endog
    and not spec.indicators
    and not any(":" in c for c in spec.exog)
    and vcov in ("iid", "HC0", "HC1", "HC2", "HC3")
    or (cluster and vcov not in ("bootstrap", "wildboot", "NW", "DK"))
)
if use_nofe_rust:
    return _ols_nofe_rust(data, spec, cluster, vcov)
```

**Expected speedup at N=1M:**
- OLS iid: 60ms → ~15ms (eliminate extract_arrays + NumPy solve overhead)
- OLS HC1: 107ms → ~25ms (eliminate extract_arrays + NumPy sandwich)
- OLS cluster no FE: 106ms → ~20ms (eliminate extract_arrays, reuse Rust clustered meat)

**Step 1:** Add `rust_ols_nofe` to `src/lib.rs`. Reuse existing helpers: `solve_kxk`, `invert_kxk`, `clustered_meat_raw`, `recode_vec`, `combinations`, `matmul`.

**Step 2:** Add `_ols_nofe_rust()` wrapper in `polars_reg/_ols.py` and route eligible calls.

**Step 3:** Run tests: `pytest tests/test_ols.py tests/test_pandas_compat.py -v`

**Step 4:** Benchmark: `python benchmarks/bench.py`

---

## Task 2: Extend `rust_ols_from_arrays` with robust SE support

Currently the FE path only supports "iid" and "cluster" vcov. When HC0-HC3 is requested with FE, it falls back to the slow Python path. Add HC0-HC3 sandwich computation inside the existing Rust FE pipeline.

**Files:**
- Modify: `src/lib.rs` — add vcov_type parameter to `rust_ols_from_arrays` and `rust_ols_core`
- Modify: `polars_reg/_ols.py` — widen `use_direct` eligibility to include HC0-HC3

**What changes in Rust:**
- Add `vcov_type: String` parameter to both `rust_ols_from_arrays` and `rust_ols_core`
- After computing residuals and `xtx_inv`, branch on vcov_type:
  - "iid": existing sigma² * (X'X)⁻¹
  - "HC0"/"HC1": sandwich with e² weights
  - "HC2"/"HC3": sandwich with hat-adjusted weights
  - "cluster": existing CGM code

**Python changes:**
- Remove `vcov not in ("HC2", "HC3")` from `use_direct` eligibility
- Pass vcov string to Rust

**Step 1:** Add vcov_type parameter to Rust functions, implement HC0-HC3 sandwich.

**Step 2:** Update Python routing to pass vcov_type and widen eligibility.

**Step 3:** Run tests: `pytest tests/test_ols.py -v -k "robust or hc"`

---

## Task 3: `rust_iv2sls` — Single Rust call for 2SLS

The IV path is 100% Python: extract_arrays → column_stack → inv → solve → SE. At N=1M this is ~200ms+. A single Rust call eliminates all boundary crossings.

**Files:**
- Modify: `src/lib.rs` — add `rust_iv2sls`
- Modify: `polars_reg/_iv.py` — route eligible calls through Rust

**What `rust_iv2sls` does:**
1. Accept: y, x_exog_cols, x_endog_cols, z_excl_cols, fe_cols, cl_cols, vcov_type
2. If FE: demean all arrays (y, X_exog, X_endog, Z_excl) — reuse demean_cg_slices
3. Build Z = [X_exog, Z_excl]
4. Stage 1: X_endog_hat = Z (Z'Z)⁻¹ Z' X_endog
5. Stage 2: X_hat = [X_exog, X_endog_hat], beta = (X_hat'X)⁻¹ X_hat'y
6. First-stage F-stat
7. VCV (iid, HC0/HC1, cluster)
8. Return: (beta, vcov, residuals, r2, r2_adj, first_stage_f, n, k, df_abs, cluster_n_groups)

**Key implementation detail:** For IV, the bread is (X_hat'X)⁻¹ (not (X'X)⁻¹), and the meat uses X_hat (not X). The existing `clustered_meat_raw` works — just pass X_hat as the "X" matrix.

**Expected speedup at N=1M:** ~200ms → ~40ms for 2SLS with FE+cluster

**Step 1:** Add `rust_iv2sls` to `src/lib.rs`.

**Step 2:** Add `_iv2sls_rust()` wrapper in `polars_reg/_iv.py` and route eligible calls.

**Step 3:** Run tests: `pytest tests/test_iv.py -v`

---

## Task 4: `rust_extract_arrays` — Move data extraction to Rust

`extract_arrays` is ~54ms at N=1M and is called by every estimator. The main cost is:
- `df[col].to_numpy().astype(np.float64)` per column (~5ms each)
- `np.column_stack` (~8ms for 3 columns at N=1M)
- `_to_codes` for FE/cluster encoding

For estimators that can't use the `from_arrays` pattern (weighted, HAC, bootstrap, interactions), moving extraction to Rust still saves time.

**Files:**
- Modify: `src/lib.rs` — add `rust_extract_columns`
- Modify: `polars_reg/_utils.py` — use Rust extraction when available

**What `rust_extract_columns` does:**
1. Accept: list of (col_array, dtype) pairs as individual numpy arrays
2. Build the X matrix in Rust (row-major), handling:
   - Float columns: direct copy
   - Integer FE columns: recode to contiguous codes
3. Return: (X_matrix, fe_codes_list, n_groups_list)

This is a lighter-weight optimization — only worth doing if Tasks 1-3 don't cover enough cases.

**Expected speedup:** extract_arrays ~54ms → ~15ms

---

## Task 5: Move remaining Python-only estimators to Rust entry points

After Tasks 1-3, the remaining Python-heavy paths are:
- `liml()` — LIML estimator (generalized eigenvalue problem)
- `gmm_iv()` — two-step GMM
- `panel_fe()` / `panel_re()` / `panel_fd()` — panel estimators

These are lower priority because:
- They're less frequently called than OLS/2SLS
- LIML requires scipy.linalg.eig (eigenvalue solver) — hard to replicate in Rust without a dependency
- Panel estimators are mostly thin wrappers around OLS/demeaning that's already in Rust

**Approach:** For each estimator, create a Rust entry point that does as much as possible (extract + demean + solve + SE), falling back to scipy only for the eigenvalue step in LIML.

**Files:**
- Modify: `src/lib.rs`
- Modify: `polars_reg/_gmm.py`, `polars_reg/_panel.py`

**Priority order:**
1. `panel_fe` — almost identical to OLS+FE, easy win
2. `gmm_iv` — two-step iterative, moderate complexity
3. `liml` — needs eigenvalue solver, lowest ROI

---

## Task 6: Eliminate `np.linalg` for small systems

Currently even the Rust path creates numpy arrays to return results to Python, which then does `np.linalg.solve` for some post-processing. The Rust `solve_kxk` and `invert_kxk` already handle this — ensure ALL linear algebra on k×k matrices (where k is typically 2-10) stays in Rust.

Audit: After Tasks 1-3, check if any Python code still calls `np.linalg.solve` or `np.linalg.inv` on data that came from Rust. If so, move that computation into the Rust call.

**Files:** Audit `_ols.py`, `_iv.py`, `_gmm.py`, `_panel.py`

---

## Priority and Expected Impact

| Task | Effort | Speedup (N=1M) | Coverage |
|------|--------|-----------------|----------|
| 1. rust_ols_nofe | Medium | OLS iid 60→15ms, HC1 107→25ms | ~60% of calls |
| 2. FE + robust SE | Small | Unlocks HC2/HC3 for Rust FE path | ~10% of calls |
| 3. rust_iv2sls | Medium | IV 200→40ms | ~15% of calls |
| 4. rust_extract | Small | 54→15ms for fallback paths | All remaining |
| 5. Other estimators | Large | Moderate | ~10% of calls |
| 6. Audit np.linalg | Trivial | Minor | Cleanup |

**Recommended order:** 1 → 2 → 3 → 6 → 4 → 5

Tasks 1+2 together cover ~70% of real-world usage and deliver the biggest absolute speedup. Task 3 handles IV. Task 4 is a safety net for edge cases. Task 5 is only worth doing if users report panel/GMM being slow.

## Success Criteria

After Tasks 1-3, the benchmark at N=1M should show:
- OLS iid: <20ms (from 60ms)
- OLS HC1: <30ms (from 107ms)
- OLS cluster no FE: <25ms (from 106ms)
- OLS + 2FE + cluster: <170ms (unchanged, already Rust)
- All 362+ tests still passing
