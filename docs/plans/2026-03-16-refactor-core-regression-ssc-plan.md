---
title: "refactor: Core regression consolidation + configurable SSC"
type: refactor
date: 2026-03-16
version: 0.3.0
breaking: true
brainstorm: docs/brainstorms/2026-03-16-core-refactor-ssc-brainstorm.md
deepened: 2026-03-16
---

## Enhancement Summary

**Deepened on:** 2026-03-16
**Sections enhanced:** 6 phases + edge cases
**Research agents used:** architecture-strategist, code-simplicity-reviewer, kieran-python-reviewer, performance-oracle, pattern-recognition-specialist, framework-docs-researcher, best-practices-researcher, learnings-explorer

### Key Improvements
1. **model_type override gap** — panel wrappers must override `model_type` on the returned RegressionResult
2. **SSC parameter ordering** — `ssc` goes after `cluster` in the canonical parameter order, must update `api-consistency.md`
3. **HC2/HC3 + SSC interaction** — HC2/HC3 have their own correction `1/(1-h_ii)^p`; `k_adj` only controls the residual df factor, not the leverage correction
4. **Driscoll-Kraay G_adj** — DK uses `T/(T-1)` as its time-dimension adjustment; `G_adj` should NOT control this (it's not a cluster adjustment, it's intrinsic to the DK estimator)
5. **Rust fast path SSC** — Rust paths compute their own VCV; SSC must be plumbed through to Rust or the Rust paths must delegate back to Python for SSC-aware VCV
6. **pyfixest returns dict, we use frozen dataclass** — deliberate design choice for type safety and IDE support; document the difference

### New Considerations Discovered
- `panel_fe()` wrapper calling `ols()` produces `model_type="OLS"` not `"Panel FE"` — must override
- `panel_fd()` drops first observation per entity — no FE contiguity concern (panel_fd doesn't use FE absorption)
- `_gmm_solve()` should stay in `_arellano_bond.py` (follows "one estimator family per module" principle)
- `_newton_raphson()` should stay in `_binary.py` since PPML is the only other consumer
- `compute_vcov()` with `bread` override is the right abstraction — avoids duplicating the entire VCV dispatch

# ♻️ Core Regression Consolidation + Configurable SSC

## Overview

Refactor polars_reg to (1) add pyfixest-style `ssc()` for user-configurable small-sample corrections, (2) collapse specialty estimators into thin wrappers around core functions, and (3) extract shared VCV dispatch, Newton-Raphson, and GMM solve logic. Version bump to 0.3.0 (breaking: IV default dfc changes).

## Problem Statement

**Duplicated estimation logic:** 7 VCV dispatch chains across estimator files, each with subtly different dfc formulas. GMM solve copy-pasted between `panel_ab` and `panel_sys_gmm`. Newton-Raphson reimplemented in `_ppml.py` when `_binary.py` already has an abstracted version.

**Hardcoded dfc formulas:** Each estimator bakes in its own degrees-of-freedom correction matching a specific Stata command. Users cannot choose between conventions. `compare()` shows diffs that are purely dfc convention differences, not bugs.

**Existing bug:** `panel_fe()` does not pass `df_a_non_nested` to `vcov_clustered()`, so it uses `reg`-style dfc (`(N-1)/(N-k) * G/(G-1)`) even with absorbed FE, while `ols()` with the same FE formula uses `reghdfe`-style dfc (`N/(N-d-k) * G/(G-1)`). This is silently wrong.

## Proposed Solution

### New `ssc()` API (matches pyfixest)

```python
# polars_reg/_ssc.py
@dataclass(frozen=True)
class SSC:
    k_adj: bool = True           # (N-1)/(N-k) residual df scaling
    k_fixef: str = "nonnested"   # FE in k: "nonnested", "none", "full"
    G_adj: bool = True           # G/(G-1) cluster scaling
    G_df: str = "min"            # multiway: "min" or "conventional"

def ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min") -> SSC:
    """Configure small-sample corrections. Matches pyfixest conventions."""
    ...
```

Usage:
```python
import polars_reg as pr

# Default (pyfixest / Stata reghdfe convention)
pr.ols("y ~ x1 | fe1", df, cluster=["fe1"])

# Match Stata ivregress (asymptotic)
pr.iv2sls("y ~ x1 | z1", df, ssc=pr.ssc(k_adj=False, G_adj=False))

# Exclude FE from k (non-default)
pr.ols("y ~ x1 | fe1", df, cluster=["fe1"],
       ssc=pr.ssc(k_fixef="none"))
```

### Core + Sugar Architecture

```
panel_fe(formula, data, entity, ...)  →  ols(fe_formula, data, cluster=[entity], ...)
panel_fd(formula, data, entity, ...)  →  first_diff(data) → ols(formula, diffed_data, ...)
probit/logit                          →  already share _binary_model()
ppml                                  →  share _newton_raphson() with binary
panel_ab/panel_sys_gmm                →  share _gmm_solve()
```

## Technical Approach

### Implementation Phases

#### Phase 1: SSC Dataclass + Plumbing

Create `_ssc.py` and wire the `ssc` parameter through all public functions without changing any formulas yet.

**Files to create:**
- `polars_reg/_ssc.py` — `SSC` frozen dataclass, `ssc()` constructor, `_default_ssc()`, validation

**Files to modify:**
- `polars_reg/__init__.py` — export `ssc`, `SSC`
- `polars_reg/_ols.py` — add `ssc: SSC | None = None` parameter to `ols()`
- `polars_reg/_iv.py` — add `ssc` parameter to `iv2sls()`
- `polars_reg/_gmm.py` — add `ssc` parameter to `liml()`, `gmm_iv()`
- `polars_reg/_panel.py` — add `ssc` parameter to `panel_fe()`, `panel_re()`, `panel_fd()`
- `polars_reg/_binary.py` — add `ssc` parameter to `probit()`, `logit()`
- `polars_reg/_ppml.py` — add `ssc` parameter to `ppml()`
- `polars_reg/_quantile.py` — add `ssc` parameter to `quantreg()`
- `polars_reg/_arellano_bond.py` — add `ssc` parameter to `panel_ab()`, `panel_sys_gmm()`

**Acceptance criteria:**
- [ ] `SSC` dataclass with 4 fields, frozen, validated on construction
- [ ] `ssc()` constructor validates `k_fixef` in `{"none", "nonnested", "full"}` and `G_df` in `{"min", "conventional"}`
- [ ] Every public estimator function accepts `ssc: SSC | None = None`
- [ ] `ssc=None` resolves to default `SSC()` inside each function
- [ ] All existing tests pass unchanged (no behavioral change yet)
- [ ] New tests: `test_ssc.py` for SSC construction, validation, repr

**Research insights:**

- **Frozen dataclass vs dict:** pyfixest's `ssc()` returns `dict[str, str | bool]`. We use `@dataclass(frozen=True)` deliberately — it provides: type checking via `SSC.k_adj` instead of `ssc["k_adj"]`, IDE autocomplete, immutability guarantee, and `__repr__` for debugging. Negligible performance difference (dataclass creation ~100ns vs dict ~50ns, called once per estimator invocation).
- **SSC parameter position in signature:** Per `api-consistency.md`, `ssc` should go after `cluster` in the canonical ordering: `(formula, data, [entity, time], vcov, cluster, ssc, [time, bandwidth], ...)`. Update `api-consistency.md` to document this.
- **`__post_init__` validation:** Use `__post_init__` on the frozen dataclass for validation. Use `object.__setattr__(self, ...)` pattern if needed for normalization on frozen dataclass, or validate-and-raise only.

```python
@dataclass(frozen=True)
class SSC:
    k_adj: bool = True
    k_fixef: str = "nonnested"
    G_adj: bool = True
    G_df: str = "min"

    def __post_init__(self):
        if self.k_fixef not in ("none", "nonnested", "full"):
            raise ValueError(
                f"k_fixef must be 'none', 'nonnested', or 'full', got {self.k_fixef!r}"
            )
        if self.G_df not in ("min", "conventional"):
            raise ValueError(
                f"G_df must be 'min' or 'conventional', got {self.G_df!r}"
            )
```

- **Store SSC on RegressionResult:** Add `ssc: SSC | None` field to `RegressionResult` so downstream tools (compare, regtable) can inspect which corrections were used. This is important for `match_ssc` in Phase 6.

- **SSC cheat sheet in docstring:** Include a "common presets" table in the `ssc()` docstring for discoverability:

```python
def ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min"):
    """Configure small-sample corrections. Matches pyfixest conventions.

    Common presets:
        Default (pyfixest):    ssc()  # k_fixef="nonnested", G_df="min"
        Stata reghdfe:         ssc()  # same as default
        Stata ivregress:       ssc(k_adj=False, G_adj=False)
        R fixest:              ssc()  # same as default
        Exclude FE from k:     ssc(k_fixef="none")
        Per-term G correction: ssc(G_df="conventional")
        No corrections:        ssc(k_adj=False, G_adj=False)
    """
```

- **Test strategy for SSC × VCV:** Test switches independently (not all 192 combinations): `k_adj` True/False × {iid, HC1, clustered}; `G_adj` True/False × {clustered}; `k_fixef` × {iid, clustered with FE}; `G_df` × {multiway}. ~20 targeted tests covering each switch's effect.

#### Phase 2: SSC-Driven VCV in `_se.py`

Replace hardcoded dfc formulas with ssc-driven logic. This is the core behavioral change.

**Current _se.py interface:**
```python
vcov_iid(X, resid, df_abs=0)
vcov_robust(X, resid, kind="HC1", df_abs=0)
vcov_clustered(X, resid, clusters, df_correction=True, df_a_non_nested=-1)
vcov_multiway_clustered(X, resid, cluster_list, df_a_non_nested=-1)
vcov_hac(X, resid, time_ids, bandwidth=None)
```

**New _se.py interface:**
```python
vcov_iid(X, resid, ssc: SSC, df_abs: int = 0)
vcov_robust(X, resid, kind: str, ssc: SSC, df_abs: int = 0)
vcov_clustered(X, resid, clusters, ssc: SSC, df_a_non_nested: int = 0)
vcov_multiway_clustered(X, resid, cluster_list, ssc: SSC, df_a_non_nested: int = 0)
vcov_hac(X, resid, time_ids, ssc: SSC, bandwidth=None)
```

**dfc formula mapping:**

| Function | Current formula | SSC-driven formula |
|---|---|---|
| `vcov_iid` | `e'e/(N-k-df_abs)` | `k_adj=T`: `e'e/(N-k_eff)` where `k_eff` depends on `k_fixef`. `k_adj=F`: `e'e/N` |
| `vcov_robust` HC1 | `N/(N-k-df_abs)` | `k_adj=T`: `N/(N-k_eff)`. `k_adj=F`: 1 (no scaling, same as HC0) |
| `vcov_clustered` | `G/(G-1) * (N-1)/(N-k)` or `G/(G-1) * N/(N-d-k)` | `G_adj * k_adj_factor` where each is independently toggled |
| `vcov_multiway` | `G_min/(G_min-1) * N/(N-d-k)` or per-term `G_i/(G_i-1)` | `G_df="min"` vs `"conventional"` controls which formula |
| `vcov_hac` | `N/(N-k)` | `k_adj=T`: `N/(N-k_eff)`. `k_adj=F`: 1 |

**`k_eff` computation (centralized helper):**
```python
def _compute_k_eff(k: int, k_fixef: str, df_abs: int, df_a_non_nested: int) -> int:
    if k_fixef == "none":
        return k  # FE excluded from k
    elif k_fixef == "nonnested":
        return k + max(df_a_non_nested, 0)  # non-nested FE counted
    elif k_fixef == "full":
        return k + df_abs  # all FE counted
```

**Files to modify:**
- `polars_reg/_se.py` — all VCV functions get `ssc` parameter, replace hardcoded formulas
- `polars_reg/_ols.py` — pass `ssc` to VCV calls
- `polars_reg/_iv.py` — pass `ssc` to VCV calls (replaces hardcoded asymptotic convention)
- `polars_reg/_gmm.py` — pass `ssc` to VCV calls
- `polars_reg/_panel.py` — pass `ssc` to VCV calls
- `polars_reg/_binary.py` — pass `ssc` to VCV calls (MLE sandwich)
- `polars_reg/_ppml.py` — pass `ssc` to VCV calls

**Acceptance criteria:**
- [ ] All VCV functions in `_se.py` accept `SSC` object
- [ ] `k_adj=True/False` toggles residual df scaling
- [ ] `G_adj=True/False` toggles cluster G/(G-1) scaling
- [ ] `k_fixef` controls how FE count in k for all VCV types
- [ ] `G_df` controls min vs conventional in multiway clustering
- [ ] SSC has NO effect on `vcov="bootstrap"` or `vcov="wildboot"` (resampling-based)
- [ ] OLS with default SSC matches current pyfixest output (not current polars_reg output for IV)
- [ ] New tests: parametrized tests for each SSC switch × VCV type combination
- [ ] Stata parity tests updated: each test passes explicit `ssc` matching the Stata command

**Research insights:**

- **HC2/HC3 + SSC interaction:** HC2/HC3 use leverage-based corrections `e_i^2/(1-h_ii)^p`. These are NOT controlled by `k_adj`. The `k_adj` factor only applies to the overall HC1-style `N/(N-k)` scaling. HC0 has no correction at all. HC2/HC3 corrections are intrinsic to the estimator — SSC does not affect them. Match pyfixest: HC2/HC3 ignore `k_adj`.
- **Driscoll-Kraay and G_adj:** DK uses `T/(T-1)` as its time-dimension correction (line 229 of `_se.py`). This is NOT a cluster adjustment — it's the Newey-West HAC correction applied to time-averaged scores. `G_adj` should NOT control this. Only `k_adj` applies to DK (for the `N/(N-k)` residual factor). Match pyfixest behavior.
- **vcov_iid + k_adj=False:** When `k_adj=False`, `sigma2 = e'e/N` (not `e'e/(N-k)`). This is the ML estimator of sigma^2. Some users may want this for consistency with MLE-based inference.
- **Rust fast paths:** The Rust OLS paths (`rust_ols_core`, `rust_ols_nofe`) compute VCV internally. Two options: (a) pass SSC fields as arguments to Rust functions, or (b) have Rust return raw `(X'X)^{-1}`, `resid`, etc. and compute SSC-adjusted VCV in Python. Option (b) is simpler and avoids Rust API changes, but means Rust can no longer return a complete result in one call. **Recommendation:** Option (a) — pass `k_adj`, `G_adj` as booleans to Rust. The Rust functions already take `df_abs` etc., adding 2-3 more parameters is minimal.

#### Phase 2b: Rust SSC Plumbing

The Rust fast paths compute VCV internally with hardcoded dfc. Each must be updated.

**Rust functions to modify in `src/lib.rs`:**

| Function | Current dfc | SSC params to add |
|---|---|---|
| `rust_ols_core` | Hardcoded iid/HC/cluster dfc | `k_adj: bool`, `G_adj: bool` |
| `rust_ols_nofe` | Hardcoded iid/HC/cluster dfc | `k_adj: bool`, `G_adj: bool` |
| `rust_ols_from_arrays` | Hardcoded iid/HC/cluster dfc | `k_adj: bool`, `G_adj: bool` |
| `rust_iv2sls` | Hardcoded asymptotic (no dfc) | `k_adj: bool`, `G_adj: bool` |

**Note:** `k_fixef` and `G_df` do not need Rust plumbing — `k_fixef` affects `k_eff` which is computed in Python and passed as `k` to Rust. `G_df` affects multiway clustering which is Python-only.

**Acceptance criteria:**
- [ ] All 4 Rust functions accept `k_adj` and `G_adj` boolean parameters
- [ ] Rust iid VCV uses `e'e/(N-k)` when `k_adj=True`, `e'e/N` when `k_adj=False`
- [ ] Rust clustered VCV applies `G/(G-1)` only when `G_adj=True`
- [ ] Rust HC1 uses `N/(N-k)` when `k_adj=True`, 1.0 when `k_adj=False`
- [ ] Python callers in `_ols.py` and `_iv.py` pass SSC fields to Rust functions
- [ ] `maturin develop --release` builds successfully
- [ ] Tests pass with both Rust and Python paths producing identical results

#### Phase 3: Shared VCV Dispatch

Extract the repeated if/elif VCV dispatch into a shared function.

**New function in `_se.py`:**
```python
def compute_vcov(
    X: NDArray,
    resid: NDArray,
    vcov: str,
    ssc: SSC,
    *,
    cluster_arrays: list[NDArray] | None = None,
    time_array: NDArray | None = None,
    entity_array: NDArray | None = None,
    bandwidth: int | None = None,
    df_abs: int = 0,
    df_a_non_nested: int = 0,
    n_boot: int = 999,
    seed: int | None = None,
    y: NDArray | None = None,
    bread: NDArray | None = None,       # override (X'X)^{-1} for IV/MLE
    score_X: NDArray | None = None,     # override X for score computation (IV: X_hat)
) -> NDArray:
    """Unified VCV dispatch. Replaces 7 copy-pasted if/elif chains."""
```

**Key design:** `bread` and `score_X` overrides enable IV (bread=`(X_hat'X)^{-1}`, score=`X_hat*resid`) and MLE (bread=`H_inv`, score=`per-obs scores`) to use the same dispatch without private VCV functions.

**Files to modify:**
- `polars_reg/_se.py` — add `compute_vcov()` dispatcher
- `polars_reg/_ols.py` — replace VCV if/elif with `compute_vcov()` call
- `polars_reg/_panel.py` — replace VCV if/elif in `panel_fe()`, `panel_re()`, `panel_fd()`
- `polars_reg/_binary.py` — replace VCV dispatch with `compute_vcov(bread=H_inv, ...)`
- `polars_reg/_ppml.py` — replace VCV dispatch with `compute_vcov(bread=H_inv, ...)`

**Not unified (remains separate):**
- `polars_reg/_iv.py` — IV bootstrap re-estimates entire 2SLS, not pairs bootstrap. Keep separate dispatch but use `compute_vcov()` for non-bootstrap cases.
- `polars_reg/_gmm.py` — GMM has fundamentally different effective scores. Keep custom VCV but could use `compute_vcov()` for LIML's simpler cases.

**Acceptance criteria:**
- [ ] `compute_vcov()` handles iid, HC0-HC3, clustered, multiway, NW, DK, bootstrap, wildboot
- [ ] `bread` override works for IV (`X_hat`-based) and MLE (`H_inv`-based)
- [ ] OLS, panel_fe, panel_re, panel_fd, binary, ppml all use `compute_vcov()`
- [ ] Delete `_iv_vcov_iid`, `_iv_vcov_robust`, `_iv_vcov_clustered` private functions where possible
- [ ] All existing tests pass (no behavioral change — ssc defaults produce same output)

**Research insights:**

- **compute_vcov() parameter count concern:** 13+ parameters is a lot. However, most are keyword-only with sensible defaults. The alternative (keeping 7 separate dispatch chains) is worse. The `bread` and `score_X` overrides are the key design innovation — they enable IV and MLE to share the dispatch without duplicating it. This is the standard sandwich VCV pattern: `V = bread @ meat @ bread` where meat varies by vcov type.
- **Validation in compute_vcov():** Centralize ALL feasibility checks here:
  - `G >= 2` for clustered (already exists but scattered)
  - `T >= 2` for DK
  - `N > k_eff` for k_adj (prevent negative denominator)
  - `vcov in supported_set` (delegate to `validate_vcov()`)
  - GMM multiway → `NotImplementedError`

#### Phase 4: Panel Wrappers

Convert `panel_fe()` and `panel_fd()` into thin wrappers around `ols()`.

**`panel_fe()` → ols() wrapper:**
```python
def panel_fe(formula, data, entity, time=None, vcov="iid",
             cluster=None, ssc=None, **kwargs):
    """Panel FE (within estimator). Sugar for ols() with absorbed FE."""
    if isinstance(cluster, str):
        cluster = [cluster]
    elif cluster is None:
        cluster = [entity]
    # Inject entity/time into formula FE slot
    fe_terms = entity + (f" + {time}" if time else "")
    fe_formula = _inject_fe(formula, fe_terms)
    return ols(fe_formula, data, vcov=vcov, cluster=cluster, ssc=ssc, **kwargs)
```

This fixes the current bug where `panel_fe()` doesn't compute `df_a_non_nested`.

**`panel_fd()` keeps data transform, delegates estimation:**
```python
def panel_fd(formula, data, entity, time, vcov="iid",
             cluster=None, ssc=None, **kwargs):
    """First-difference estimator. Differences data, then runs OLS."""
    diffed_data = _first_difference(data, entity, time, formula)
    return ols(formula, diffed_data, vcov=vcov, cluster=cluster, ssc=ssc, **kwargs)
```

**`_inject_fe()` helper:** Parses existing formula, appends FE terms. If formula already has `|`, raises ValueError.

**`_first_difference()` helper:** Sorts by entity+time, computes within-entity diffs, drops first obs per entity. Returns a new DataFrame with diffed columns.

**Files to modify:**
- `polars_reg/_panel.py` — rewrite `panel_fe()` (~130 lines → ~15 lines), rewrite `panel_fd()` (~140 lines → ~25 lines + `_first_difference()` helper). `panel_re()` stays as-is.
- `polars_reg/_ols.py` — ensure `ols()` handles all cases panel_fe currently handles (should already work)

**Acceptance criteria:**
- [ ] `panel_fe()` is ≤20 lines of sugar around `ols()`
- [ ] `panel_fe()` produces identical results to `ols()` with equivalent FE formula and clustering
- [ ] `panel_fd()` data transform is preserved; estimation delegates to `ols()`
- [ ] `panel_fe()` now gets Rust fast path (inherited from `ols()`)
- [ ] `panel_fe()` now supports HC0-HC3 (inherited from `ols()`)
- [ ] `panel_fe()` now supports weights (inherited from `ols()`)
- [ ] `panel_re()` unchanged
- [ ] Existing panel tests pass (with explicit `ssc` where needed)
- [ ] New test: `panel_fe()` result == `ols()` result for same FE/cluster spec
- [ ] `hausman_test()` still works correctly after panel_fe refactor (may produce different results — validate against Stata `hausman`)

**Research insights:**

- **model_type override (CRITICAL GAP):** When `panel_fe()` calls `ols()`, the result has `model_type="OLS"`. Must override to `"Panel FE"` after the call:

```python
def panel_fe(formula, data, entity, time=None, vcov="iid",
             cluster=None, ssc=None, **kwargs):
    # ... setup cluster, formula ...
    result = ols(fe_formula, data, vcov=vcov, cluster=cluster, ssc=ssc, **kwargs)
    result.model_type = "Panel FE"  # Override for regtable/display
    return result
```

Same for `panel_fd()` → `model_type = "Panel FD"`.

- **`cluster=[]` edge case:** Current `panel_fe()` explicitly handles `cluster=[]` (empty list) for Hausman test compatibility — it means "no clustering, use iid SEs." The wrapper must preserve this. When delegating to `ols()`, `cluster=[]` should be converted to `cluster=None` since `ols()` doesn't have the empty-list convention.
- **`_inject_fe()` approach:** Rather than string manipulation, consider using `parse_formula()` to parse the existing formula and programmatically set `spec.fe`. This avoids regex/string fragility. However, `ols()` expects a formula string, not a `FormulaSpec`. Two options: (a) build a formula string with `| entity + time` appended, or (b) add an internal `_ols_from_spec()` that accepts a pre-built FormulaSpec. Option (a) is simpler for the wrapper refactor.
- **panel_fd() has no FE:** First-differencing eliminates entity fixed effects by construction. The `_first_difference()` helper drops the first observation per entity but does NOT need FE contiguity re-indexing since no FE absorption follows. This is safe.
- **`**kwargs` and validate_vcov():** When panel_fe delegates to ols(), the ols() vcov validation (`validate_vcov`) will now accept HC0-HC3 that panel_fe previously rejected. This is intentional and documented as a non-breaking additive change. But existing panel_fe tests that assert `ValueError` on HC1 will need updating.

#### Phase 5: Shared NR + GMM

**5A: Shared Newton-Raphson for MLE**

Extract `_newton_raphson()` from `_binary.py` into a shared module (or keep in `_binary.py` since it's already abstracted). Make `_ppml.py` call it instead of reimplementing.

```python
# Already in _binary.py, make it importable
def _newton_raphson(score_hess_fn, beta0, X, y, max_iter=100, tol=1e-8):
    """Generic Newton-Raphson solver for MLE.

    score_hess_fn(beta, X, y) -> (ll, score, H, *aux)
    """
```

`_ppml.py` needs: clipping `mu = np.clip(np.exp(X @ beta), 1e-10, 1e10)`, separation detection post-convergence. These become the score/hessian function and a post-convergence hook.

**5B: Shared GMM solve**

Extract ~30 lines of identical GMM estimation code.

```python
# polars_reg/_gmm_core.py or within _arellano_bond.py
def _gmm_solve(X, y, Z, twostep=True):
    """One-step or two-step GMM estimation.

    Returns: (beta, resid, V, A_inv, W)
    """
```

**Files to modify:**
- `polars_reg/_binary.py` — export `_newton_raphson()` (may already be importable)
- `polars_reg/_ppml.py` — replace inline NR loop with `_newton_raphson()` call + PPML score/hessian function
- `polars_reg/_arellano_bond.py` — extract `_gmm_solve()`, both `panel_ab()` and `panel_sys_gmm()` call it

**Acceptance criteria:**
- [ ] `_ppml.py` NR loop replaced with call to shared `_newton_raphson()`
- [ ] PPML separation detection preserved as post-convergence check
- [ ] `_gmm_solve()` used by both `panel_ab()` and `panel_sys_gmm()`
- [ ] All existing tests pass

**Research insights:**

- **Module placement:** Per code organization principles, `_newton_raphson()` should stay in `_binary.py` (it's the MLE family module). PPML is the only other consumer. If more MLE estimators are added later, extract to `_mle.py` then. `_gmm_solve()` should stay in `_arellano_bond.py` (Arellano-Bond family) — both panel_ab and panel_sys_gmm are in this family.
- **PPML score/hessian callable:** The PPML score function must include `np.clip(mu, 1e-10, 1e10)` inside the callable, not outside. This ensures the clipping happens every iteration, preventing `exp()` overflow.

```python
def _ppml_score_hess(beta, X, y):
    mu = np.clip(np.exp(X @ beta), 1e-10, 1e10)
    score_resid = y - mu
    score = X.T @ score_resid
    H = -X.T @ (X * mu[:, None])
    ll = float(np.sum(y * np.log(mu) - mu))  # Poisson log-likelihood
    return ll, score, H, mu, score_resid
```

- **Separation detection:** Keep as a post-convergence check in `ppml()` after calling `_newton_raphson()`. Don't move it into the shared solver — it's PPML-specific.

#### Phase 6: compare() match_ssc + Cleanup

**compare() enhancement:**

Add `match_ssc: bool = False` parameter. When True, for each backend, determine the appropriate ssc and run polars_reg with that ssc. The report shows one polars_reg column per backend.

```python
# Backend → SSC mapping (all major backends now match our defaults)
_BACKEND_SSC = {
    "pyfixest": SSC(),  # same defaults
    "statsmodels": SSC(),  # OLS: same; needs validation
    "r": SSC(),  # R fixest: same defaults
    "stata": SSC(),  # reghdfe: same defaults
    # IV-specific: stata_iv → SSC(k_adj=False, G_adj=False)
}
```

When `match_ssc=True`, `ComparisonReport` may have multiple polars_reg results (one per backend's ssc). The `summary()` GT table shows columns like "polars_reg (pyfixest ssc)" and "pyfixest".

**Files to modify:**
- `polars_reg/_compare.py` — add `match_ssc` parameter, backend-to-ssc mapping
- `polars_reg/_ssc.py` — add `_backend_ssc(backend, estimator)` helper
- `tests/test_compare.py` — tests for match_ssc behavior

**Version bump + cleanup:**
- `pyproject.toml` — version 0.3.0
- `polars_reg/__init__.py` — export `ssc`, `SSC`
- Update Stata parity tests with explicit `ssc`
- Delete dead code from old VCV dispatch chains
- Delete `_iv_vcov_*` private functions (replaced by `compute_vcov` with bread override)

**Acceptance criteria:**
- [ ] `compare(..., match_ssc=True)` shows polars_reg columns with per-backend ssc
- [ ] `compare(..., match_ssc=False)` (default) uses user's ssc unchanged
- [ ] Backend-to-ssc mapping documented and tested
- [ ] Version bumped to 0.3.0
- [ ] `ssc` and `SSC` exported from `polars_reg`
- [ ] All Stata parity tests pass with explicit `ssc` matching each Stata command
- [ ] `ruff check .` and `ruff format .` clean
- [ ] Feature showcase notebook: `notebooks/new_features/2026-03-16-ssc-configuration.ipynb`

**Research insights:**

- **Backend SSC mapping is estimator-dependent:** Stata uses different dfc for `reg` vs `ivregress` vs `ppmlhdfe`. The `_BACKEND_SSC` mapping should be `_backend_ssc(backend: str, estimator: str) -> SSC` not a flat dict. Example:

```python
def _backend_ssc(backend: str, estimator: str) -> SSC:
    if backend == "stata":
        if estimator in ("iv2sls", "liml"):
            return SSC(k_adj=False, G_adj=False)  # Stata ivregress: asymptotic
        return SSC()  # Stata reghdfe: same as default
    elif backend == "r":
        return SSC()  # R fixest: same as default
    elif backend == "pyfixest":
        return SSC()  # Same defaults
    elif backend == "statsmodels":
        return SSC()  # Needs validation per model type
    return SSC()
```

- **match_ssc column naming:** When `match_ssc=True`, the polars_reg columns should clearly indicate which SSC was used. Format: `"polars_reg (stata ssc)"`, `"polars_reg (R ssc)"`. If user's ssc already matches the backend, show single column.
- **Update api-consistency.md:** Add `ssc` to the parameter table and ordering convention. Add SSC to the "Type Annotations" section.

## Edge Cases + Safety

### From institutional learnings (docs/solutions/)

1. **FE singleton contiguity** (`fe-singleton-contiguity-and-edge-case-guards.md`): After any row filtering (singleton removal, first-differencing), FE codes must be re-indexed before demeaning. The panel_fe wrapper delegates to `ols()` which already handles this. The `panel_fd()` `_first_difference()` helper must also handle this if it filters rows.

2. **Validate VCV feasibility** (`polars-reg-comprehensive-code-review.md`): Centralize guards like G≥2 for clustering, T≥2 for DK. The `compute_vcov()` dispatcher is the natural place for these checks.

3. **GMM multiway cluster** — currently `NotImplementedError`. Maintain this loud-fail pattern in `compute_vcov()`.

### SSC edge cases

| Scenario | Behavior |
|---|---|
| `ssc` with `vcov="bootstrap"` or `"wildboot"` | SSC ignored (resampling-based SEs) |
| `ssc` with `vcov="iid"` | Only `k_adj` and `k_fixef` apply; `G_adj` and `G_df` ignored |
| `G_adj=True` with `G=1` | Raise `ValueError("Cannot compute clustered SEs with only 1 cluster group")` — already guarded |
| `k_fixef="nonnested"` with no FE | `df_a_non_nested=0`, so `k_eff = k + 0 = k` — no change |
| `k_fixef="full"` with many FE groups | `k_eff` could exceed `N`, making `N-k_eff` negative → raise `ValueError` |
| `k_adj=False, G_adj=False` | No corrections at all (asymptotic VCV) |
| Multiway cluster with `G_df="min"` when one dim has `G=1` | Same G=1 guard applies to `min(G)` |
| HC2/HC3 + `k_adj` | `k_adj` does NOT affect HC2/HC3 leverage corrections `1/(1-h_ii)^p` — only the overall `N/(N-k)` HC1 factor is toggled |
| DK + `G_adj` | Driscoll-Kraay's `T/(T-1)` correction is intrinsic to the estimator, NOT controlled by `G_adj`. Only `k_adj` applies to DK. |
| `panel_fe()` with `cluster=[]` (empty list) | Convert to `cluster=None` when delegating to `ols()` for iid SEs (Hausman test pattern) |
| `panel_fe()` → `model_type` | Override `result.model_type = "Panel FE"` after calling `ols()` |
| `panel_fd()` → `model_type` | Override `result.model_type = "Panel FD"` after calling `ols()` |
| `ssc` stored on `RegressionResult` | Add `ssc` field to `RegressionResult` so `compare()` can inspect which corrections were used |
| `ssc` passed to `quantreg()` | Silently ignored — quantreg uses bootstrap-only inference, SSC does not apply to resampling |

### Breaking changes

| Change | Previous behavior | New behavior | Migration |
|---|---|---|---|
| `iv2sls()` default dfc | Asymptotic (no corrections) | `ssc()` defaults (k_adj=True, G_adj=True) | Pass `ssc=ssc(k_adj=False, G_adj=False)` |
| `gmm_iv()` clustered dfc | Custom `G/(G-1)*(N-1)/(N-k)` | Controlled by `ssc` | Check if default ssc matches |
| `panel_fe()` clustered dfc | Wrong (reg-style, missing df_a_non_nested) | Correct (inherits ols behavior) | Results improve (bug fix) |
| `panel_fe()` allowed vcov | No HC0-HC3 | HC0-HC3 now available (via ols) | Non-breaking (additive) |
| `hausman_test()` results | Uses panel_fe with reg-style dfc (wrong) | Uses panel_fe→ols with reghdfe-style dfc (correct) | Results change — validate against Stata `hausman` |

## Dependencies & Risks

**Risk: Regression in dfc formulas.** The most critical risk. Mitigations:
- Stata parity tests with explicit `ssc` matching each Stata command
- pyfixest parity tests verifying default `ssc` matches pyfixest output
- Parametrized tests for every `ssc` switch × VCV type combination

**Risk: Panel wrapper breaks edge cases.** panel_fe/fd currently handle some edge cases inline. Mitigations:
- Test panel_fe wrapper against direct ols() call for identical results
- Test with `messy_data` fixture (singletons, NaN, mixed types)
- Test `cluster=[]` for Hausman test compatibility

**Risk: Shared VCV dispatch subtle bugs.** The `bread` and `score_X` overrides for IV/MLE add complexity. Mitigations:
- IV and GMM keep their own dispatch for complex cases (bootstrap)
- Only unify the straightforward cases (iid, robust, clustered, HAC)

## Additional Files to Update

These were discovered during deepening and are not in the original phase list:

- `polars_reg/_results.py` — add `ssc: SSC | None = None` field to `RegressionResult`
- `docs/principles/api-consistency.md` — add `ssc` to parameter table and ordering convention
- `src/lib.rs` — add SSC-related parameters to Rust fast path functions (k_adj, G_adj booleans)
- `polars_reg/_groupby.py` — `groupby_reg()` passes `**kwargs` to estimators, so `ssc` flows through automatically. Verify this works.
- `polars_reg/_diagnostics.py` — `hausman_test()` calls `panel_fe()` and `panel_re()` internally. Verify the panel_fe wrapper change doesn't break it.

## What This Does NOT Change

- Formula parsing (`_formula.py`)
- Data extraction pipeline (`_utils.py`, `extract_arrays()`)
- Demeaning algorithm (`_demean.py`)
- `regtable()`, `groupby_reg()` (ssc flows through kwargs automatically)
- `panel_re()` (genuinely distinct GLS estimator)

## What Changes Slightly

- `RegressionResult` dataclass — gains `ssc` field
- Rust native functions — gain SSC boolean parameters for VCV computation
- `api-consistency.md` — updated with `ssc` parameter convention

## References

- Brainstorm: `docs/brainstorms/2026-03-16-core-refactor-ssc-brainstorm.md`
- pyfixest ssc docs: https://pyfixest.org/ssc.html
- R fixest ssc docs: https://search.r-project.org/CRAN/refmans/fixest/html/ssc.html
- FE singleton learnings: `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md`
- Comprehensive code review: `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md`
- Code organization principles: `docs/principles/code-organization.md`
- API consistency principles: `docs/principles/api-consistency.md`
