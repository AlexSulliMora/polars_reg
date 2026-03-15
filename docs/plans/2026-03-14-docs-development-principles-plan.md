---
title: Development Principles Documentation
type: docs
date: 2026-03-14
deepened: 2026-03-15
brainstorm: docs/brainstorms/2026-03-13-development-principles-brainstorm.md
---

# Development Principles Documentation

## Enhancement Summary

**Deepened on:** 2026-03-15
**Research sources:** architecture review, pattern analysis, scikit-learn developer guide, statsmodels/SciPy/fixest conventions, codebase learnings (`docs/solutions/`)

### Key Improvements
1. Expanded data flow contract to cover MLE and multi-equation variant pipelines
2. Resolved "robust" vcov ambiguity — document as user-facing alias for HC1
3. Added composability contract (what makes an estimator work with groupby_reg/regtable)
4. Added error boundary for all exception types (ValueError, TypeError, NotImplementedError, RuntimeError)
5. Added "silent wrong answers" philosophy from project learnings
6. Added vcov support matrix and model_type naming convention
7. Added parameter vocabulary table pattern (from scikit-learn glossary)
8. Added tolerance table with rationale column (from SciPy conventions)
9. Added error message pattern: include problematic value, what's wrong, and available options

### Architectural Simplifications (from user feedback)
- **Polars-only**: remove `ensure_polars()` — users call `pl.from_pandas()` themselves
- **Single cleaning step**: consolidate NaN/inf/null handling into `extract_arrays()`
- **Rust-only**: remove all Python fallback paths and `_HAS_NATIVE` branching — one codebase, one truth
- These are documented as target principles; code changes are a separate refactor task

### New Considerations Discovered
- `quantreg()` returns `RegressionResult | list[RegressionResult]` — union return type not addressed by original plan
- `time` parameter has dual semantics: panel dimension vs HAC ordering column
- RegressionResult uses ad-hoc `setattr()` for estimator-specific fields (`_X`, `_iv_X_exog`) — needs a policy
- Citation coverage is near zero in current codebase — plan should set expectations

## Overview

Create three focused development principles documents in `docs/principles/` and link them from CLAUDE.md. These codify the design philosophy and standards for `polars_reg`, guiding the maintainer and Claude Code when adding or modifying code.

**Scope:** Documentation only — no code changes. New code must follow principles going forward; existing code aligned opportunistically.

**Documents (in writing order):**
1. `docs/principles/code-organization.md`
2. `docs/principles/api-consistency.md`
3. `docs/principles/statistical-rigor.md`

**Target length:** 150-300 lines each. Enough to be comprehensive, short enough to stay readable.

## Acceptance Criteria

- [x] Three principles docs exist in `docs/principles/`
- [x] CLAUDE.md has a "Principles" section with one-line descriptions and relative links
- [x] Each doc is actionable, with at least one concrete example (file path + pattern) per section
- [x] Known non-conformance is noted briefly (not exhaustive audit, just prominent cases)

## Proposed Solution

### 1. `code-organization.md`

**Sections:**

1. **Module Placement**
   - Estimators in `_<name>.py`, one estimator family per module
   - "Family" defined by example: `_panel.py` = {panel_fe, panel_re, panel_fd}; `_binary.py` = {probit, logit, marginal_effects, odds_ratios}; `_gmm.py` = {liml, gmm_iv}; `_arellano_bond.py` = {panel_ab, panel_sys_gmm}
   - Singleton modules are valid when an estimator has a distinct mathematical framework (e.g., `_ppml.py`, `_quantile.py`)
   - Heuristic for new code: if the new function shares data extraction, formula parsing, or mathematical framework with an existing module, it belongs there
   - Diagnostics in `_diagnostics.py`; split only when function + result class + helpers exceed ~300 lines of self-contained code
   - **Known non-conformance:** `quantreg()` lives in `_quantile.py` — function name and module name differ substantively

   *Reference: scikit-learn's developer guide similarly groups estimators by mathematical family (linear models, ensemble, etc.) with one module per family.*

2. **Public API Surface**
   - Everything exported from `__init__.py`
   - Internal modules prefixed with `_`
   - Naming exceptions: `stata.py` and `r_equiv.py` lack underscore because they wrap external tools with optional dependencies — the un-prefixed name signals "external integration layer"

3. **Data Flow Contract**
   - **Input requirement:** all public functions accept `pl.DataFrame | pl.LazyFrame` only. No pandas auto-conversion — users call `pl.from_pandas()` themselves. This is a Polars-native package; pushing conversion to the caller simplifies every estimator's entry point
   - **Single data cleaning step:** `extract_arrays()` handles ALL cleaning in one place: NaN→null conversion, inf→null conversion, null dropping. No downstream code should independently sanitize data. After `extract_arrays()`, row count is fixed and all values are finite
   - **Canonical pipeline** (OLS, IV, panel FE/RE/FD):
     Formula → column selection (LazyFrame) → `extract_arrays()` [cleans + extracts] → (demean if FE) → estimate → `_se.py` VCV → RegressionResult
   - **MLE variant** (probit, logit, PPML):
     Formula → `extract_arrays()` → Newton-Raphson IRLS → Hessian-based VCV (uses `_clustered_meat` directly, not `vcov_robust()`) → RegressionResult
   - **Multi-equation variant** (GRS test):
     Formula → `parse_formula()` → custom array extraction → per-asset OLS → cross-equation test → typed result dataclass
   - **Current state:** `ensure_polars()` still called in most estimators (pandas gateway), inf sanitization scattered across `_ols.py` and `_iv.py` only. Target: consolidate into `extract_arrays()` and remove `ensure_polars()` calls (separate refactor task)

   *Insight from `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md`: the seam between data cleaning steps (singleton drop, code reindexing, NaN-to-null conversion) was the source of the project's most critical bug cluster. Consolidating all cleaning into `extract_arrays()` eliminates these seams.*

4. **Rust Extension**
   - **Single implementation:** all compute functions exist in Rust only — no Python fallback, no `_HAS_NATIVE` flag, no dual-path branching. The native extension is a build requirement, not optional. Wheels are distributed via CI for all platforms, so most users never compile from source
   - **One codebase, one truth:** when a bug is fixed, it is fixed once. No risk of Rust and Python paths diverging (real incident: Rust OLS path once returned zeros for fitted values — C6 in comprehensive code review)
   - **Current state:** five modules have `try/except ImportError` with `_HAS_NATIVE` flags and Python fallbacks, `_to_codes_fast()` duplicated in `_ols.py:76` and `_iv.py:27`. Target: remove all Python fallback paths and `_HAS_NATIVE` branching (separate refactor task)
   - **Adding new Rust functions:** justify with benchmarks showing meaningful speedup on representative workloads. Functions called inside iterative convergence loops (e.g., demeaning) have their effective speedup multiplied by iteration count

5. **Test Organization**
   - Unit tests: `test_<module>.py` (one per internal module)
   - Parity tests: `test_stata_parity.py`, `test_r_equiv.py` (separate, require external software)
   - Cross-cutting: descriptive names (`test_dual_path.py`, `test_bootstrap.py`, `test_hac.py`)
   - Function naming: `test_<feature>_<scenario>` (e.g., `test_ols_fe_python_vs_rust`); full `test_<module>_<feature>_<scenario>` form preferred for grep-ability
   - A function gets its own test file only when it has enough test cases to warrant ~200+ lines
   - **Testing philosophy** (from `docs/solutions/`): "Tests that pass are not tests that protect." Avoid tautological assertions testing algebraic identities. Test the formulas that differentiate this package (dfc corrections, demeaning convergence, Stata parity)

**Key files to reference:** `__init__.py`, `_ols.py` (Rust patterns), `_diagnostics.py` (module sizing), `test_dual_path.py`, `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md`

### 2. `api-consistency.md`

**Sections:**

1. **Return Type Contract**
   - All estimator functions return `RegressionResult`
   - New diagnostic functions: typed dataclass with `.summary()` method (following `GRSTestResult`)
   - Existing dict-based diagnostics (`hausman_test`, `weak_instrument_test`, `kleibergen_paap_test`): left as-is — noted as tech debt
   - `wald_test()` on `RegressionResult`: returns dict (method, not standalone function — different pattern)
   - **Known non-conformance:** `quantreg()` returns `RegressionResult | list[RegressionResult]` depending on whether `tau` is scalar or list. Union return types should be avoided in new estimators — prefer always returning a list or having separate functions
   - **RegressionResult extension fields:** estimators like `iv2sls()` attach private fields via `setattr()` (e.g., `result._X`, `result._iv_X_exog`). New estimator-specific fields should be declared on the dataclass with `Optional` typing rather than using ad-hoc `setattr()`

   *Reference: scikit-learn enforces a strict return type contract — `fit()` always returns `self`, `predict()` always returns an array. This consistency is what enables their composition framework (`Pipeline`, `GridSearchCV`).*

2. **Parameter Naming**
   - Canonical names: `formula`, `data`, `entity`, `time`, `vcov`, `cluster`, `bandwidth`, `weights`, `fweights`, `n_boot`, `seed`
   - Never: `df`, `fml`, `cl`, `se_type`, `B`, `panel_id`
   - **Dual semantics of `time`:** in panel estimators, `time` is a structural panel dimension (column name). In cross-sectional estimators with HAC/DK, `time` is a vcov-related ordering column. These are semantically different despite sharing a name. Document this distinction; consider renaming the HAC variant to `time_col` in future
   - **Parameter vocabulary table** (include in the doc, following scikit-learn's glossary pattern):

     | Parameter  | Type                          | Semantics                                          |
     |------------|-------------------------------|-----------------------------------------------------|
     | `formula`  | `str`                         | Wilkinson notation: `"y ~ x1 + x2 \| fe1 + fe2"`   |
     | `data`     | `pl.DataFrame \| pl.LazyFrame`| Input data (auto-converts pandas via ensure_polars) |
     | `entity`   | `str`                         | Panel entity column name                            |
     | `time`     | `str \| None`                 | Panel time column / HAC ordering column             |
     | `vcov`     | `str`                         | One of the vcov vocabulary strings                  |
     | `cluster`  | `str \| list[str] \| None`    | Column name(s) for clustered SEs                    |
     | `weights`  | `str \| None`                 | Column name for analytic weights                    |
     | `fweights` | `str \| None`                 | Column name for frequency weights                   |
     | `n_boot`   | `int`                         | Number of bootstrap replications                    |
     | `seed`     | `int \| None`                 | RNG seed for reproducibility                        |

3. **Parameter Ordering** (estimator functions only)
   - `(formula, data, [entity, time], vcov, cluster, [time_col, bandwidth], [weights, fweights], [n_boot, seed])`
   - Estimator-specific required params (entity, time for panel; tau for quantile; lags for Arellano-Bond) come after `data` but before `vcov`
   - Non-estimator functions (diagnostics, utilities) follow "most important argument first, keyword-only for optionals" — no strict template
   - **Exceptions:** `panel_ab()` and `panel_sys_gmm()` have no `vcov`/`cluster` parameters — VCV is determined by the estimation method (robust by construction)

4. **vcov Vocabulary**
   - Input vocabulary: `{"iid", "HC0", "HC1", "HC2", "HC3", "robust", "NW", "DK", "bootstrap", "wildboot"}`
   - `"robust"` is a user-facing alias for `"HC1"` — already exposed in `probit()`, `logit()`, and `ppml()` docstrings
   - Minimum *vcov string* set for new estimators: `{iid, HC1}`; minimum *functionality* set: `{iid, HC1, one-way clustered}` (clustering is a separate `cluster` parameter, not a vcov string)
   - Bootstrap and HAC are optional
   - Unsupported types: raise `ValueError` with message listing available options
   - Default: `"iid"` for most estimators; MLE-based estimators may default to `"HC1"` when the statistical theory requires robust SEs (e.g., PPML is quasi-MLE, so sandwich VCV is standard)
   - **vcov support matrix** (include in the doc):

     | Estimator | iid | HC0-3 | NW | DK | bootstrap | wildboot | cluster |
     |-----------|-----|-------|----|----|-----------|----------|---------|
     | ols       | Y   | Y     | Y  | Y  | Y         | Y        | Y       |
     | iv2sls    | Y   | HC0-1 | Y  | Y  | Y         | Y        | Y       |
     | panel_fe  | Y   | -     | Y  | Y  | Y         | Y        | Y       |
     | probit    | Y   | HC1   | -  | -  | -         | -        | Y       |
     | panel_ab  | -   | -     | -  | -  | -         | -        | -       |
     | ...       |     |       |    |    |           |          |         |

5. **model_type Vocabulary**
   - Enumerate current set: `"OLS"`, `"WLS"`, `"OLS (fweight)"`, `"2SLS"`, `"LIML"`, `"GMM"`, `"Panel FE"`, `"Panel RE"`, `"Panel FD"`, `"Arellano-Bond"`, `"System GMM"`, `"Probit"`, `"Logit"`, `"Quantile(τ)"` (parameterized), `"PPML"`
   - Naming convention: all-caps for acronym methods (`"OLS"`, `"2SLS"`, `"PPML"`), title-case for descriptive names (`"Panel FE"`, `"Arellano-Bond"`), title-case for proper-noun methods (`"Probit"`, `"Logit"`)
   - `"Quantile(τ)"` is a parameterized template, not a fixed string — the only model_type that varies at runtime. Document this as a sanctioned exception

6. **Error Handling Boundaries**
   - `ValueError`: data-integrity issues — singular matrix, no observations, wrong column types, invalid formula syntax, unsupported vcov type
   - `TypeError`: wrong argument types (e.g., passing a string where a DataFrame is expected)
   - `NotImplementedError`: feature combinations that could be supported but are not yet (e.g., FE absorption in LIML, multi-way clustering in GMM)
   - `RuntimeError`: infrastructure failures (e.g., missing native extension when required)
   - `warnings.warn()`: statistical judgment calls — low power, high condition number, non-convergence, PPML separation, multiple endogenous with single-endogenous F-stat
   - **LinAlgError policy:** linear algebra exceptions from NumPy/SciPy (`np.linalg.LinAlgError`) should be caught at the estimator layer and re-raised as `ValueError` with a descriptive message, not allowed to propagate raw
   - **Philosophy** (from `docs/solutions/`): "Silent wrong answers are worse than crashes." A crash is a gift — it tells you something is wrong. Silent corruption (wrong DoF, misaligned weights, NaN propagation) is the real enemy. Prefer raising over silently computing wrong results
   - **Error message pattern** (from SciPy/scikit-learn conventions): include (1) the problematic value, (2) what's wrong, (3) available options. Example: `f"vcov={vcov!r} is not supported for {model_type}. Available: {', '.join(sorted(supported))}"`
   - Worked examples from existing code for each category

7. **Composability Contract** (groupby_reg + regtable)
   - Any estimator that meets these four conditions works automatically with `groupby_reg()` and `regtable()`:
     1. Accepts `formula` as the first positional argument
     2. Accepts `data` as a keyword argument
     3. Returns `RegressionResult`
     4. Populates the standard fields: `names`, `coefficients`, `vcov`, `n_obs`, `r_squared`, `model_type`
   - `regtable()` renders from `RegressionResult` attributes: `params`, `se`, `pvalues`, `nobs`, `r_squared`, `model_type`
   - Cross-reference: diagnostic return types (when to use dataclass vs dict) — see also `code-organization.md` module placement rules for diagnostics

8. **Output Formatting**
   - `.summary()` format: coefficient table with Coef/SE/t/P>|t|/[CI] columns
   - `regtable()` handles significance stars and multi-model comparison
   - `GRSTestResult.summary()` is the template for new diagnostic summaries

9. **Type Annotations**
   - Public functions: full annotations on all parameters and return type
   - Internal functions: annotate parameters and return types
   - Stable dict returns should use TypedDict where the key set is fixed

**Key files to reference:** `_ols.py:383` (parameter signature), `_ppml.py:46-73` (gold-standard docstring with Reference line, full Args/Returns), `_results.py` (RegressionResult attributes), `_diagnostics.py` (GRSTestResult pattern vs legacy dicts), `_groupby.py` (composability mechanism)

### 3. `statistical-rigor.md`

**Sections:**

1. **Citation Standards**
   - **Threshold**: cite formulas where the implementation choice affects numerical results versus a reference implementation (Stata/R). Standard linear algebra identities don't need citations
   - **Current baseline**: near zero equation-level citations exist in the codebase. Only `Kamstra & Shi (2021), eq. 7` in `_diagnostics.py:395` and `Stock & Yogo (2005), Table 5.2` in `_diagnostics.py:14` have equation/table-level refs. This is the biggest gap relative to the proposed principles. New code must follow the standard; retroactive coverage is a separate, lower-priority task
   - **Format variants:**
     - Equations: `# Cameron & Trivedi (2005), eq 11.23`
     - Tables: `# Stock & Yogo (2005), Table 5.2`
     - Algorithms: `# Symmetric Kaczmarz (Correia 2016, §3.2)`
     - Stata conventions: `# Matches Stata reghdfe (Correia 2016)`
   - **In docstrings**: author-year reference, brief method description. Gold standard: `_ppml.py:53-55` — `Reference: Santos Silva and Tenreyro (2006), 'The Log of Gravity', Review of Economics and Statistics`
   - **In code comments**: equation/table/section-level reference next to the implementing line

   *Reference: fixest (R) provides the gold standard — its "On standard-errors" vignette documents every dfc formula with paper-level citations and explicit Stata parity notes. linearmodels uses Notes sections with LaTeX equations. statsmodels uses numpydoc `[1]_` numbered references. For polars_reg, use Google-style docstrings (matching pdoc) with author-year inline references rather than numpydoc numbered references.*

2. **Conflicting Sources**
   - Cite both sources
   - State which is implemented and why
   - Worked example: LIML σ² uses `1/(n-k)` (textbook) vs `e'e/n` (Stata `ivregress`, asymptotic) — we implement the Stata convention for parity

3. **Numerical Engineering**
   - Categories that warrant documentation:
     - Eigenvalue clamping (e.g., `np.maximum(eigvals, 0.0)` in matrix power)
     - Probability clipping (e.g., `np.clip(Phi, 1e-15, 1-1e-15)` in probit)
     - Pseudoinverse fallback (e.g., `np.linalg.pinv` when `solve` fails)
     - Convergence tolerances (e.g., demeaning tol=1e-8, Newton-Raphson tol=1e-8)
     - Separation detection thresholds (PPML)
     - CG overflow detection (e.g., `abs(uv) < 1e-30` guard in `_demean.py`)
     - NaN-to-null conversion boundary (IEEE NaN passes through Polars `drop_nulls()` — NaN != Polars null)
   - Format: inline code comment explaining the rationale, no citation needed

   *Insight from `docs/solutions/`: the non-contiguous FE codes bug was a numerical engineering failure — `codes.max()` on an empty array after singleton removal, leading to phantom zero-count groups and CG overflow. Multi-layer defense (data cleaning → contiguity enforcement → overflow detection → estimator guards) is the pattern to follow.*

4. **Validation Expectations**
   - New estimators should have Stata or R parity tests where feasible
   - **Tolerance table** (include in the doc with rationale column, following SciPy convention):

     | Context               | Default | Rationale                              | Adjustable? |
     |-----------------------|---------|----------------------------------------|-------------|
     | Demeaning convergence | 1e-8    | Matches reghdfe default                | Yes (`tol=`)|
     | Stata parity tests    | 2e-5    | Demeaning algorithm differences        | Per-test    |
     | LIML parity tests     | 2e-3    | Near-singular eigenvalue sensitivity   | Per-test    |
     | Panel parity tests    | 5e-2    | RE theta estimation differences        | Per-test    |
     | Eigenvalue clamping   | max(0,λ)| Prevent negative variance from roundoff| No          |
     | Prob clipping (probit) | 1e-15  | Prevent log(0) in log-likelihood       | No          |

   - Cross-path parity: Rust and Python paths tested in `test_dual_path.py`
   - **Deterministic test seeding**: replace all `np.random.randn(...)` with explicit `np.random.default_rng(seed)`. For bootstrap tests, either fix the seed or test weak properties that hold with overwhelming probability
   - **Document convergence metrics, not just thresholds**: e.g., "We check `max(abs(x_new - x_old)) < tol`" rather than just "tol=1e-8"

5. **Key Formulas Requiring Citations** (author-year refs; exact equation lookups are a separate task)
   - Degree-of-freedom corrections — Cameron, Gelbach & Miller (2011)
   - Demeaning algorithm — Correia (2016)
   - LIML eigenvalue approach — Anderson & Rubin (1949), Stata ivregress docs
   - GMM VCV formula — Hansen (1982)
   - GRS F-statistic — Gibbons, Ross & Shanken (1989), Kamstra & Shi (2021)
   - Probit/Logit MLE — Greene (2018) or Cameron & Trivedi (2005)
   - Newey-West / Bartlett kernel — Newey & West (1987)
   - Driscoll-Kraay — Driscoll & Kraay (1998)
   - Arellano-Bond GMM — Arellano & Bond (1991)
   - Swamy-Arora GLS — Swamy & Arora (1972)

6. **Bibliography**
   - Full citations (author, title, journal/publisher, year) for all referenced texts
   - Organized alphabetically by first author

**Key files to reference:** `_se.py` (dfc formulas, HAC), `_demean.py` (Correia), `_diagnostics.py` (GRS, Stock-Yogo), `_gmm.py` (LIML), `_ppml.py` (gold-standard docstring), `docs/solutions/runtime-errors/` (numerical engineering lessons)

### 4. CLAUDE.md Update

Add a "Principles" section to CLAUDE.md after the existing "Conventions" section:

```markdown
## Principles

Detailed development principles are in `docs/principles/`:
- [`code-organization.md`](docs/principles/code-organization.md) — module placement, data flow, Rust contract, test structure
- [`api-consistency.md`](docs/principles/api-consistency.md) — return types, parameter naming/ordering, vcov vocabulary, error handling
- [`statistical-rigor.md`](docs/principles/statistical-rigor.md) — citation standards, numerical engineering, validation expectations
```

## Context

### Design Decisions

1. **Polars-only input**: no pandas auto-conversion — `ensure_polars()` removed from pipeline. Users call `pl.from_pandas()` themselves
2. **Single data cleaning step**: `extract_arrays()` handles NaN/inf/null consolidation and dropping in one place — no scattered sanitization
3. **Rust-only compute**: no Python fallback paths, no `_HAS_NATIVE` branching. Native extension is a build requirement. One codebase, one truth
4. **"robust" is a user-facing alias for HC1**: already exposed in probit/logit/ppml docstrings; include in vocabulary and document as alias
5. **Parameter ordering scoped to estimators**: non-estimator functions have different signatures that don't fit the template
6. **`cluster=[]` semantics**: documented as panel_fe-specific exception, not a general pattern
7. **Citation inventory lists author-year only**: exact equation lookups are a separate implementation task
8. **All four exception types documented**: ValueError, TypeError, NotImplementedError, RuntimeError — not just ValueError vs warnings

### Files to Create

- `docs/principles/code-organization.md`
- `docs/principles/api-consistency.md`
- `docs/principles/statistical-rigor.md`

### Files to Modify

- `CLAUDE.md` — add "Principles" section with links

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-03-13-development-principles-brainstorm.md`
- Gold-standard docstring: `polars_reg/_ppml.py:46-73` (summary, extended description, full journal citation in `Reference:` line, complete Args with types/defaults, Returns block)
- Gold-standard diagnostic dataclass: `polars_reg/_diagnostics.py` (`GRSTestResult`)
- Parameter ordering reference: `polars_reg/_ols.py:383` (ols signature)
- Rust fallback pattern: `polars_reg/_demean.py:10-20`
- Test parity: `tests/test_dual_path.py`
- Learnings: `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md`
- Learnings: `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md`

### External
- [scikit-learn Developer's Guide](https://scikit-learn.org/stable/developers/develop.html) — API conventions, parameter naming, return type contracts, `check_estimator()` compliance model
- [scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html) — standardized parameter vocabulary, attribute naming conventions
- [statsmodels Naming Conventions](https://www.statsmodels.org/dev/dev/naming_conventions.html) — `endog`/`exog`, `k_`/`n_` prefixes
- [statsmodels Exceptions and Warnings](https://www.statsmodels.org/dev/dev/warnings-and-exceptions.html) — custom warning classes taxonomy
- [SciPy nan_policy Design Spec](https://docs.scipy.org/doc/scipy/dev/api-dev/nan_policy.html) — model for vcov parameter specification
- [fixest Standard Errors Vignette](https://lrberge.github.io/fixest/articles/standard_errors.html) — gold standard for dfc formula documentation with Stata parity notes
- [Scientific Python Development Guide: Design](https://learn.scientific-python.org/development/principles/design/) — return type stability, keyword-only arguments
