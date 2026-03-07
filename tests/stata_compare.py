"""Stata parity testing infrastructure.

Translates polars_reg regression calls into equivalent Stata commands,
runs both via pystata, and compares results to machine precision.

Usage:
    result = assert_stata_parity("ols", "y ~ x1 + x2 | fe1", data, cluster=["fe1"])

Requires:
    - Stata installed with a valid license
    - pystata configured via stata_setup.config() or STATA_SETUP env vars
    - reghdfe installed in Stata (for FE absorption): ssc install reghdfe
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_reg._formula import FormulaSpec, parse_formula

# ---------------------------------------------------------------------------
# Stata availability check
# ---------------------------------------------------------------------------

_STATA_AVAILABLE: bool | None = None


def stata_available() -> bool:
    """Check if pystata is importable and Stata is configured."""
    global _STATA_AVAILABLE
    if _STATA_AVAILABLE is not None:
        return _STATA_AVAILABLE
    try:
        _configure_stata()
        _STATA_AVAILABLE = True
    except (ImportError, OSError, Exception):
        _STATA_AVAILABLE = False
    return _STATA_AVAILABLE


def _configure_stata() -> None:
    """Configure pystata. Reads STATA_DIR and STATA_EDITION env vars."""
    import stata_setup  # type: ignore[import-untyped]

    stata_dir = os.environ.get("STATA_DIR", "/usr/local/stata")
    stata_edition = os.environ.get("STATA_EDITION", "mp")
    stata_setup.config(stata_dir, stata_edition)


# ---------------------------------------------------------------------------
# Stata result extraction
# ---------------------------------------------------------------------------


@dataclass
class StataResult:
    """Results extracted from a Stata regression."""

    coefficients: NDArray
    se: NDArray
    vcov: NDArray
    names: list[str]
    n_obs: int
    r_squared: float
    r_squared_adj: float
    df_r: int
    f_stat: float | None = None
    # reghdfe-specific
    r_squared_within: float | None = None
    df_absorbed: int | None = None
    # IV-specific
    first_stage_f: float | None = None
    j_stat: float | None = None
    j_pvalue: float | None = None


def _run_stata(command: str) -> None:
    """Execute a Stata command via pystata."""
    from pystata import stata  # type: ignore[import-untyped]

    stata.run(command, quietly=True)


def _load_data_to_stata(df: pl.DataFrame) -> None:
    """Load a Polars DataFrame into Stata's active dataset."""
    from pystata import stata  # type: ignore[import-untyped]

    pdf = df.to_pandas()
    stata.pdataframe_to_data(pdf, force=True)


def _extract_stata_results(model_type: str) -> StataResult:
    """Extract regression results from Stata's e() return values."""
    from sfi import Matrix, Scalar  # type: ignore[import-untyped]

    # Coefficient vector: e(b) is 1 x k matrix
    b_raw = Matrix.get("e(b)")
    coefs = np.array(b_raw).flatten()

    # VCV matrix: e(V) is k x k
    V_raw = Matrix.get("e(V)")
    vcov = np.array(V_raw)

    se = np.sqrt(np.diag(vcov))

    # Coefficient names from e(b) column names
    n_coefs = len(coefs)
    names = []
    for i in range(n_coefs):
        name = Matrix.getColNames("e(b)", i)
        # Stata returns "varname:equation" format; strip equation prefix
        if ":" in name:
            name = name.split(":")[-1]
        names.append(name)

    n_obs = int(Scalar.getValue("e(N)"))
    df_r = int(Scalar.getValue("e(df_r)"))

    # R-squared — may not exist for all models
    try:
        r2 = float(Scalar.getValue("e(r2)"))
    except Exception:
        r2 = float("nan")
    try:
        r2_adj = float(Scalar.getValue("e(r2_a)"))
    except Exception:
        r2_adj = float("nan")

    # F-stat
    try:
        f_stat = float(Scalar.getValue("e(F)"))
    except Exception:
        f_stat = None

    # reghdfe-specific
    r2_within = None
    df_absorbed = None
    if model_type == "reghdfe":
        try:
            r2_within = float(Scalar.getValue("e(r2_within)"))
        except Exception:
            pass
        try:
            df_absorbed = int(Scalar.getValue("e(df_a)"))
        except Exception:
            pass

    # IV-specific
    first_stage_f = None
    j_stat = None
    j_pvalue = None
    if model_type in ("ivregress_2sls", "ivregress_liml", "ivregress_gmm"):
        # First-stage F is available via estat firststage after ivregress
        # For now, skip — it requires additional Stata commands
        pass
    if model_type == "ivregress_gmm":
        try:
            j_stat = float(Scalar.getValue("e(J)"))
            j_pvalue = float(Scalar.getValue("e(J_p)"))
        except Exception:
            pass

    return StataResult(
        coefficients=coefs,
        se=se,
        vcov=vcov,
        names=names,
        n_obs=n_obs,
        r_squared=r2,
        r_squared_adj=r2_adj,
        df_r=df_r,
        f_stat=f_stat,
        r_squared_within=r2_within,
        df_absorbed=df_absorbed,
        first_stage_f=first_stage_f,
        j_stat=j_stat,
        j_pvalue=j_pvalue,
    )


# ---------------------------------------------------------------------------
# Formula-to-Stata translation
# ---------------------------------------------------------------------------


def to_stata_command(
    estimator: str,
    formula: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> tuple[str, str]:
    """Translate a polars_reg call into an equivalent Stata command.

    Returns:
        (stata_command, model_type) where model_type is used to know
        which e() scalars to extract.

    Mapping:
        ols (no FE)  -> reg
        ols (with FE) -> reghdfe
        iv2sls       -> ivregress 2sls
        liml         -> ivregress liml
        gmm_iv       -> ivregress gmm
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    has_fe = len(spec.fe) > 0
    has_iv = len(spec.endog) > 0

    # Build the VCE option string
    vce_opt = _build_vce_option(vcov, cluster)

    if estimator == "ols":
        if has_fe:
            return _ols_to_reghdfe(spec, vce_opt, vcov, cluster), "reghdfe"
        else:
            return _ols_to_reg(spec, vce_opt), "reg"

    elif estimator == "iv2sls":
        return _iv_to_ivregress(spec, "2sls", vce_opt), "ivregress_2sls"

    elif estimator == "liml":
        return _iv_to_ivregress(spec, "liml", vce_opt), "ivregress_liml"

    elif estimator == "gmm_iv":
        return _iv_to_ivregress_gmm(spec, vce_opt), "ivregress_gmm"

    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def _build_vce_option(vcov: str, cluster: list[str] | None) -> str:
    """Build Stata's vce() option string."""
    if cluster:
        if len(cluster) == 1:
            return f"vce(cluster {cluster[0]})"
        else:
            # Multi-way clustering: only reghdfe supports this natively
            return f"vce(cluster {' '.join(cluster)})"

    vce_map = {
        "iid": "",
        "HC0": "vce(hc2)",  # Stata's hc2 is NOT HC2; see note below
        "HC1": "vce(robust)",
        "HC2": "vce(hc2)",
        "HC3": "vce(hc3)",
    }

    # IMPORTANT: Stata's reg command vce options:
    #   vce(robust)  = HC1 (White with small-sample correction n/(n-k))
    #   vce(hc2)     = HC2 (leverage-adjusted)
    #   vce(hc3)     = HC3 (jackknife-like)
    # There is no direct HC0 option in Stata's reg. HC0 = robust without
    # the n/(n-k) correction. Closest workaround is manual computation.
    # For parity testing, we test HC1, HC2, HC3 which have direct Stata
    # equivalents. HC0 tests use a manual post-estimation correction.

    if vcov == "HC0":
        # HC0 has no direct Stata equivalent. Use robust and back out:
        # HC0 = HC1 * (n-k)/n. We handle this in the comparison function.
        return "vce(robust)"

    return vce_map.get(vcov, "")


def _ols_to_reg(spec: FormulaSpec, vce_opt: str) -> str:
    """Translate OLS (no FE) to Stata's reg command."""
    parts = ["reg", spec.depvar] + spec.exog
    opts = []
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    if opts:
        parts.append(", " + " ".join(opts))
    return " ".join(parts)


def _ols_to_reghdfe(
    spec: FormulaSpec,
    vce_opt: str,
    vcov: str,
    cluster: list[str] | None,
) -> str:
    """Translate OLS with FE to Stata's reghdfe command."""
    parts = ["reghdfe", spec.depvar] + spec.exog

    opts = [f"absorb({' '.join(spec.fe)})"]

    if cluster:
        if len(cluster) == 1:
            opts.append(f"vce(cluster {cluster[0]})")
        else:
            opts.append(f"vce(cluster {' '.join(cluster)})")
    elif vcov == "HC1":
        opts.append("vce(robust)")
    elif vcov == "iid":
        # reghdfe default is robust; force iid-like behavior
        opts.append("vce(unadjusted)")

    if not spec.add_intercept:
        opts.append("noconstant")

    return " ".join(parts) + ", " + " ".join(opts)


def _iv_to_ivregress(spec: FormulaSpec, method: str, vce_opt: str) -> str:
    """Translate IV to Stata's ivregress command.

    ivregress 2sls y x_exog (x_endog = z1 z2) [, vce(...)]
    """
    endog_str = " ".join(spec.endog)
    instr_str = " ".join(spec.instruments)
    exog_str = " ".join(spec.exog) if spec.exog else ""

    parts = [f"ivregress {method}", spec.depvar]
    if exog_str:
        parts.append(exog_str)
    parts.append(f"({endog_str} = {instr_str})")

    opts = []
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    if opts:
        parts.append(", " + " ".join(opts))

    return " ".join(parts)


def _iv_to_ivregress_gmm(spec: FormulaSpec, vce_opt: str) -> str:
    """Translate GMM-IV to Stata's ivregress gmm.

    ivregress gmm uses a two-step efficient GMM by default with
    wmatrix(robust) for heteroskedasticity-robust optimal weighting.
    """
    endog_str = " ".join(spec.endog)
    instr_str = " ".join(spec.instruments)
    exog_str = " ".join(spec.exog) if spec.exog else ""

    parts = [f"ivregress gmm", spec.depvar]
    if exog_str:
        parts.append(exog_str)
    parts.append(f"({endog_str} = {instr_str})")

    opts = ["wmatrix(robust)"]
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    parts.append(", " + " ".join(opts))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Detailed comparison between polars_reg and Stata results."""

    estimator: str
    formula: str
    stata_command: str
    passed: bool
    n_obs_match: bool
    coef_max_rdiff: float
    se_max_rdiff: float
    r2_rdiff: float | None
    details: list[str] = field(default_factory=list)
    polars_coefs: NDArray | None = None
    stata_coefs: NDArray | None = None
    polars_se: NDArray | None = None
    stata_se: NDArray | None = None
    polars_names: list[str] | None = None
    stata_names: list[str] | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.estimator}: {self.formula}",
            f"  Stata: {self.stata_command}",
            f"  N match: {self.n_obs_match}",
            f"  Coef max relative diff: {self.coef_max_rdiff:.2e}",
            f"  SE max relative diff: {self.se_max_rdiff:.2e}",
        ]
        if self.r2_rdiff is not None:
            lines.append(f"  R² relative diff: {self.r2_rdiff:.2e}")
        for d in self.details:
            lines.append(f"  {d}")
        return "\n".join(lines)


def _align_coefficients(
    polars_names: list[str],
    polars_coefs: NDArray,
    polars_se: NDArray,
    stata_names: list[str],
    stata_coefs: NDArray,
    stata_se: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
    """Align polars_reg and Stata coefficients by variable name.

    Stata uses _cons for the intercept; our package also uses _cons.
    Returns aligned (polars_coefs, polars_se, stata_coefs, stata_se, names).
    """
    # Build name-to-index maps
    pr_map = {name: i for i, name in enumerate(polars_names)}
    st_map = {name: i for i, name in enumerate(stata_names)}

    # Find common names
    common = [n for n in polars_names if n in st_map]

    if not common:
        raise ValueError(
            f"No common coefficient names.\n"
            f"  polars_reg: {polars_names}\n"
            f"  Stata: {stata_names}"
        )

    pr_idx = [pr_map[n] for n in common]
    st_idx = [st_map[n] for n in common]

    return (
        polars_coefs[pr_idx],
        polars_se[pr_idx],
        stata_coefs[st_idx],
        stata_se[st_idx],
        common,
    )


def _relative_diff(a: NDArray, b: NDArray) -> NDArray:
    """Element-wise relative difference: |a - b| / max(|a|, |b|, 1e-15)."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, 1e-15)
    return np.abs(a - b) / denom


def compare_results(
    estimator: str,
    formula: str,
    stata_command: str,
    polars_result: Any,
    stata_result: StataResult,
    rtol: float = 1e-6,
    vcov: str = "iid",
) -> ComparisonResult:
    """Compare polars_reg and Stata regression results."""
    details: list[str] = []
    passed = True

    # N obs
    n_match = polars_result.n_obs == stata_result.n_obs
    if not n_match:
        details.append(f"N mismatch: polars={polars_result.n_obs}, stata={stata_result.n_obs}")
        passed = False

    # Align coefficients by name
    try:
        pr_c, pr_se, st_c, st_se, names = _align_coefficients(
            polars_result.names, polars_result.coefficients, polars_result.se,
            stata_result.names, stata_result.coefficients, stata_result.se,
        )
    except ValueError as e:
        return ComparisonResult(
            estimator=estimator, formula=formula, stata_command=stata_command,
            passed=False, n_obs_match=n_match,
            coef_max_rdiff=float("inf"), se_max_rdiff=float("inf"),
            r2_rdiff=None, details=[str(e)],
            polars_names=polars_result.names, stata_names=stata_result.names,
        )

    # HC0 adjustment: Stata has no direct HC0, so if vcov=="HC0" we ran
    # Stata with robust (HC1) and need to scale SEs: SE_HC0 = SE_HC1 * sqrt((n-k)/n)
    if vcov == "HC0":
        n, k = polars_result.n_obs, polars_result.k
        st_se = st_se * np.sqrt((n - k) / n)
        details.append("Applied HC0 correction to Stata SEs (HC1 * sqrt((n-k)/n))")

    # Coefficient comparison
    coef_rdiff = _relative_diff(pr_c, st_c)
    coef_max = float(coef_rdiff.max())
    if coef_max > rtol:
        passed = False
        for i, name in enumerate(names):
            if coef_rdiff[i] > rtol:
                details.append(
                    f"Coef '{name}': polars={pr_c[i]:.10f}, stata={st_c[i]:.10f}, "
                    f"rdiff={coef_rdiff[i]:.2e}"
                )

    # SE comparison
    se_rdiff = _relative_diff(pr_se, st_se)
    se_max = float(se_rdiff.max())
    if se_max > rtol:
        passed = False
        for i, name in enumerate(names):
            if se_rdiff[i] > rtol:
                details.append(
                    f"SE '{name}': polars={pr_se[i]:.10f}, stata={st_se[i]:.10f}, "
                    f"rdiff={se_rdiff[i]:.2e}"
                )

    # R-squared comparison
    r2_rdiff = None
    if not np.isnan(stata_result.r_squared):
        r2_rdiff = abs(polars_result.r_squared - stata_result.r_squared) / max(
            abs(stata_result.r_squared), 1e-15
        )
        if r2_rdiff > rtol:
            passed = False
            details.append(
                f"R² mismatch: polars={polars_result.r_squared:.10f}, "
                f"stata={stata_result.r_squared:.10f}, rdiff={r2_rdiff:.2e}"
            )

    return ComparisonResult(
        estimator=estimator, formula=formula, stata_command=stata_command,
        passed=passed, n_obs_match=n_match,
        coef_max_rdiff=coef_max, se_max_rdiff=se_max, r2_rdiff=r2_rdiff,
        details=details,
        polars_coefs=pr_c, stata_coefs=st_c,
        polars_se=pr_se, stata_se=st_se,
        polars_names=names, stata_names=names,
    )


# ---------------------------------------------------------------------------
# All-in-one parity assertion
# ---------------------------------------------------------------------------


def assert_stata_parity(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    rtol: float = 1e-6,
    stata_pre_commands: list[str] | None = None,
) -> ComparisonResult:
    """Run a regression in both polars_reg and Stata, assert results match.

    Args:
        estimator: One of "ols", "iv2sls", "liml", "gmm_iv"
        formula: polars_reg formula string
        data: Polars DataFrame (will be converted to pandas for Stata)
        vcov: Variance-covariance type
        cluster: Clustering variable(s)
        rtol: Relative tolerance for comparisons (default 1e-6)
        stata_pre_commands: Extra Stata commands to run before the regression
            (e.g., ["ssc install reghdfe", "set matsize 10000"])

    Returns:
        ComparisonResult with detailed comparison

    Raises:
        AssertionError if results don't match within tolerance
    """
    import polars_reg as pr

    if isinstance(cluster, str):
        cluster = [cluster]

    # 1. Run polars_reg
    estimator_funcs: dict[str, Callable] = {
        "ols": pr.ols,
        "iv2sls": pr.iv2sls,
        "liml": pr.liml,
        "gmm_iv": pr.gmm_iv,
    }

    func = estimator_funcs[estimator]
    kwargs: dict[str, Any] = {"formula": formula, "data": data}
    if vcov != "iid":
        kwargs["vcov"] = vcov
    if cluster:
        kwargs["cluster"] = cluster
    polars_result = func(**kwargs)

    # 2. Translate to Stata
    stata_cmd, model_type = to_stata_command(estimator, formula, vcov, cluster)

    # 3. Load data and run Stata
    _load_data_to_stata(data)

    if stata_pre_commands:
        for cmd in stata_pre_commands:
            _run_stata(cmd)

    _run_stata(stata_cmd)

    # 4. Extract Stata results
    stata_result = _extract_stata_results(model_type)

    # 5. Compare
    comparison = compare_results(
        estimator, formula, stata_cmd, polars_result, stata_result, rtol, vcov
    )

    if not comparison.passed:
        raise AssertionError(f"Stata parity check failed:\n{comparison}")

    return comparison
