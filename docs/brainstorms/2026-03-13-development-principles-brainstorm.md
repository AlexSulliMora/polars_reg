# Brainstorm: Development Principles Documentation

**Date:** 2026-03-13
**Status:** Complete

## What We're Building

A set of focused development principles documents in `docs/principles/` that codify the design philosophy and standards for `polars_reg`. These guide both the maintainer and Claude Code when adding or modifying code, ensuring the project stays coherent as it scales.

**Three documents:**
1. **`statistical-rigor.md`** — Traceability of every calculation to statistical sources
2. **`api-consistency.md`** — Common output styles, parameter naming, return types
3. **`code-organization.md`** — Module structure, internal/public boundaries, where new code lives

**Audience:** Maintainer + Claude Code (not onboarding docs for external contributors).

**Scope:** Documentation only — no code changes in this pass. New code must follow principles going forward; existing code aligned opportunistically later.

## Why This Approach

- **Multiple focused docs** over a single monolith: each topic is self-contained, can grow independently, and stays readable. CLAUDE.md links to them but stays lean.
- **Equation-level tracing** for citations: formulas get comments like `# Cameron & Trivedi (2005), eq 11.23` pointing to the exact source equation, not just author-year. This is the highest-rigor option and matches the project's Stata-parity ethos.
- **Principles-only (no code changes)** keeps this task focused and avoids scope creep into refactoring.

## Key Decisions

### 1. Statistical Rigor (`statistical-rigor.md`)
- **Equation-level source tracing**: every key formula in code gets a comment citing author, year, and equation/page number
- **Docstring citations**: public functions reference the method's statistical basis (author-year in docstring, equation ref in code comments)
- **Citation threshold**: cite formulas where the implementation choice affects numerical results vs a reference implementation. Standard linear algebra identities (e.g., `β = (X'X)⁻¹X'y`) don't need citations
- **Conflicting sources**: when two references disagree (e.g., LIML σ² divisor), cite both and state which is implemented and why (typically: match Stata)
- **Numerical engineering**: document stability choices (clipping, eigenvalue clamping, tolerance thresholds) with rationale in code comments, even when no textbook citation applies
- **Validation expectations**: new estimators should have Stata or R parity tests where possible; tolerances documented per estimator class
- **Central bibliography**: embedded in `statistical-rigor.md` itself — full citations for all referenced texts. Extract to separate file only if it exceeds ~30 entries

### 2. API Consistency (`api-consistency.md`)
- **Return type contract**: all estimators return `RegressionResult`; new diagnostic functions should use typed dataclasses with `.summary()` (following `GRSTestResult` pattern); existing dict-based diagnostics left as-is
- **Parameter naming conventions**: `data`, `formula`, `cluster`, `vcov`, `weights` — consistent across all estimators
- **Parameter ordering**: prescriptive — `(formula, data, [entity, time], vcov, cluster, [time, bandwidth], [weights, fweights], [n_boot, seed])`; estimator-specific required params (entity, time) come after `data` but before `vcov`
- **vcov vocabulary**: full set is `{"iid", "HC0", "HC1", "HC2", "HC3", "robust", "NW", "DK", "bootstrap", "wildboot"}`; minimum for new estimators is `{iid, HC1, cluster}`; unsupported types raise `ValueError` listing available options
- **Output formatting**: `.summary()` table format, significance stars, coefficient display
- **GroupBy/regtable integration**: new estimators must work with `groupby_reg()` and `regtable()` out of the box
- **Error handling boundaries**: raise `ValueError` on data-integrity issues (singular matrix, no observations, wrong types); `warnings.warn()` on statistical judgment calls (low power, high condition number)
- **model_type vocabulary**: document the valid set (`"OLS"`, `"WLS"`, `"2SLS"`, etc.) and naming convention for new ones

### 3. Code Organization (`code-organization.md`)
- **Module placement**: estimators in `_<name>.py`, one estimator family per module; diagnostics in `_diagnostics.py` unless they have their own result dataclass
- **Public API surface**: everything exported from `__init__.py`, internal modules prefixed with `_`
- **Naming exceptions**: `stata.py` and `r_equiv.py` intentionally lack underscore prefix — modules representing external tool integrations may omit it
- **Data flow contract**: Formula → ExtractedArrays → (demean) → estimate → SE → RegressionResult; column selection pushed into LazyFrame, then collect, then sanitize inf, then drop nulls
- **Rust extension contract**: Rust paths must produce identical results to Python within machine epsilon; `test_dual_path.py` verifies this; add new Rust paths only when benchmarks show meaningful speedup
- **Type annotations**: all public functions fully annotated; internal functions annotated on params and return types; stable dict returns should use TypedDict
- **Test organization**: unit tests per module (`test_<name>.py`), parity tests separate (`test_stata_parity.py`, `test_r_equiv.py`), integration/cross-cutting tests named descriptively

## Resolved Open Questions

1. **CLAUDE.md integration**: Add a "Principles" section with a one-line description of each doc and a relative path link. Claude Code reads the relevant principles doc when doing work in that area.
2. **Bibliography location**: Embedded in `statistical-rigor.md`. A separate file adds indirection without clear benefit at this scale.
3. **Parameter ordering**: Prescriptive — codify the existing pattern (see API Consistency above).

## Remaining Open Questions

1. Should there be a brief "performance principles" subsection in `code-organization.md`? (e.g., when Rust paths are warranted, benchmarking expectations)
2. Should `_to_codes_fast()` duplication in `_ols.py`/`_iv.py` be flagged as tech debt to consolidate into `_utils.py`?
3. How should `cluster=[]` (force iid) semantics in `panel_fe()` be documented — as a pattern to follow or an exception?
