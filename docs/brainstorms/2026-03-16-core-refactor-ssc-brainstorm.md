---
topic: Core regression refactor + configurable small-sample corrections
date: 2026-03-16
status: decided
---

# Core Regression Refactor + Configurable SSC

## What We're Building

A two-part refactor that (1) collapses specialty estimators into thin wrappers around core regression functions, and (2) adds a fixest-style `ssc()` object that gives users explicit control over small-sample corrections across all estimators.

### Part 1: Core + Sugar Architecture

Many estimators are special cases of more fundamental computations:

| Wrapper (sugar) | Core function | What the wrapper sets |
|---|---|---|
| `panel_fe()` | `ols()` | Injects entity/time into FE, defaults cluster to entity |
| `panel_fd()` | `ols()` | First-differences data, then runs OLS on differenced arrays |
| `panel_re()` | Stays distinct | Swamy-Arora GLS is genuinely different (quasi-demeaning + two-stage variance components) |
| `probit()` / `logit()` | Already share `_binary_model()` | Just pass different link functions |
| `ppml()` | Share Newton-Raphson with binary family | Same NR loop, different score/Hessian |
| `panel_ab()` / `panel_sys_gmm()` | Share `_gmm_solve()` | Same GMM estimation, different instrument construction |

**Guiding principle:** core functions do the heavy computational lifting. Wrappers are syntactic sugar that set defaults and transform inputs. Wrappers never introduce their own estimation logic.

### Part 2: Configurable SSC (Small Sample Corrections)

Match pyfixest's `ssc()` exactly — 4 parameters, same names, same defaults:

```python
pr.ssc(
    k_adj=True,            # (N-1)/(N-k) residual df scaling
    k_fixef="none",        # how FE count in k: "none", "nonnested", "full"
    G_adj=True,            # G/(G-1) cluster scaling
    G_df="conventional",   # multiway: "min" or "conventional"
)
```

**Parameter details (from pyfixest source):**

- `k_adj`: when True, multiplies VCV by `(N-1)/(N-k)`. For heteroskedastic errors: `N/(N-k)`. When False, no residual df scaling.
- `k_fixef`: controls whether absorbed FE parameters count in `k`.
  - `"nonnested"` (default): FE not nested in any cluster dimension count in `k`
  - `"none"`: FE parameters excluded from `k`
  - `"full"`: all FE parameters count in `k`
- `G_adj`: when True and clustering, multiplies VCV by `G/(G-1)`. When False, no cluster scaling.
- `G_df`: for multiway clustering only. `"min"` (default): all summands use `min(G)/(min(G)-1)`. `"conventional"`: each summand `V_i` gets its own `G_i/(G_i-1)`.

**Key decisions:**

- **Uniform defaults across ALL estimators.** Every estimator (OLS, IV, GMM, probit, logit, ppml, quantreg) uses the same default `ssc()`. No estimator-specific defaults.
- **Default = pyfixest convention:** `ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")`.
- **Breaking change for IV:** Currently iv2sls uses asymptotic (no dfc). After refactor, it gets the uniform default. Users who want Stata ivregress behavior pass `ssc=pr.ssc(k_adj=False, G_adj=False)`.
- **SSC applies to MLE estimators too** (probit, logit, ppml), following pyfixest. The `k_adj` and `G_adj` switches affect the sandwich VCV regardless of whether the bread is `(X'X)^{-1}` or `H^{-1}`.
- **Wrappers inherit core defaults.** `panel_fe()` does NOT set its own ssc — it passes through whatever the user provides (or the uniform default). Wrappers only set FE/cluster/data-transform defaults.
- **No `t.df`, `fixef.force_exact`, or global default setter.** These are R fixest-only features not in pyfixest. Can be added later if users request them.

## Why This Approach

1. **String presets ("stata", "r") don't work** because even within Stata, different commands use different dfc conventions (reg vs reghdfe vs ivregress vs ppmlhdfe). A preset "stata" is ambiguous.

2. **Decomposed switches map to the math.** Each switch controls one independent component of the correction formula. Users can reason about what each toggle does.

3. **Uniform defaults simplify the mental model.** Users learn one set of defaults. Documentation says "by default, polars_reg uses fixest conventions. Here's how to match Stata's X command: ..."

4. **Following fixest/pyfixest** means users familiar with those packages get the same behavior, and we can link to their documentation for explanation.

## Key Decisions

1. **ssc API:** pyfixest-style decomposed switches (`k_adj`, `k_fixef`, `G_adj`, `G_df`), not string presets
2. **Defaults match pyfixest:** `ssc(k_adj=True, k_fixef="none", G_adj=True, G_df="conventional")`
3. **Uniform defaults:** same ssc for all estimators — no estimator-specific defaults
4. **Wrappers inherit:** thin wrappers don't override ssc defaults
5. **MLE included:** ssc applies to probit/logit/ppml, same as pyfixest
6. **Scope cut:** no `t.df`, no `fixef.force_exact`, no global `set_default_ssc()`
7. **Three refactor tiers:**
   - Tier 1: panel_fe/panel_fd become thin wrappers around ols()
   - Tier 2: Shared VCV dispatch extracted from 6+ estimator files
   - Tier 3: Shared NR loop for MLE, shared GMM solve for dynamic panel
8. **Version 0.3.0** — breaking change (IV default dfc changes)
9. **Backward compat:** document in changelog, no runtime warning

## Resolved: Parity Tests + compare()

**Stata parity tests:** Each parity test explicitly passes the ssc that matches the Stata command being tested. e.g. `iv2sls(..., ssc=ssc(k_adj=False, G_adj=False))` for tests comparing against `ivregress`. This documents which ssc matches which Stata command and keeps tight tolerances.

**`compare()` integration:** Default behavior uses the user's ssc unchanged — diffs reveal convention differences. New `match_ssc=True` option runs polars_reg once per backend with ssc settings matching that backend's conventions. This produces multiple polars_reg columns in the report (one per backend), making the comparison "fair" while keeping dfc differences transparent. Requires mapping each backend to its ssc convention:

| Backend | Matching ssc |
|---|---|
| pyfixest | `ssc()` (same defaults) |
| R fixest | `ssc()` (same defaults) |
| statsmodels OLS | `ssc()` (needs validation) |
| Stata `reg`/`reghdfe` | `ssc()` (same defaults) |
| Stata `ivregress` | `ssc(k_adj=False, G_adj=False)` |
| linearmodels | TBD (varies by estimator) |

## SSC Applicability by VCV Type

Not all ssc switches are relevant for all vcov types:

| VCV type | `k_adj` | `k_fixef` | `G_adj` | `G_df` |
|---|---|---|---|---|
| `"iid"` | Yes — controls `e'e/(N-k)` vs `e'e/N` | Yes — affects `k` | N/A | N/A |
| `"HC0"`-`"HC3"` | Yes — controls `N/(N-k)` scaling | Yes — affects `k` | N/A | N/A |
| Clustered (single) | Yes | Yes | Yes | N/A |
| Clustered (multiway) | Yes | Yes | Yes | Yes |
| `"NW"` / `"DK"` | Yes | Yes | N/A (HAC, not clustered) | N/A |
| `"bootstrap"` / `"wildboot"` | N/A (SEs from resampling) | N/A | N/A | N/A |

Bootstrap/wildboot SEs are computed from resampling distributions, not sandwich formulas. SSC does not apply to them — they already account for sample size through the resampling process.

## What This Does NOT Change

- Formula parsing (`_formula.py`)
- Data extraction pipeline (`_utils.py`, `extract_arrays()`)
- Demeaning algorithm (`_demean.py`)
- Rust native functions
- The `RegressionResult` dataclass
- `regtable()`, `groupby_reg()`
- `compare()` API shape (adds `match_ssc` parameter, internal changes only)
