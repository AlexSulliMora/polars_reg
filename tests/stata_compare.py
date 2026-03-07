"""Stata parity testing infrastructure.

Translates polars_reg regression calls into equivalent Stata commands,
runs both (polars_reg in Python, Stata via batch mode over WSL2),
and compares results to machine precision.

Usage:
    result = assert_stata_parity("ols", "y ~ x1 + x2 | fe1", data, cluster=["fe1"])

Requires:
    - Stata installed (Windows side, accessible via WSL2)
    - Set STATA_EXE env var to the Stata executable path if not at default location
    - reghdfe installed in Stata (for FE absorption): ssc install reghdfe
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_reg._formula import FormulaSpec, parse_formula

# ---------------------------------------------------------------------------
# Stata availability check
# ---------------------------------------------------------------------------

# Default Stata paths to try (Windows paths accessed via WSL)
_DEFAULT_STATA_PATHS = [
    "/mnt/c/Program Files/Stata18/StataBE-64.exe",
    "/mnt/c/Program Files/Stata18/StataMP-64.exe",
    "/mnt/c/Program Files/Stata18/StataSE-64.exe",
    "/mnt/c/Program Files/Stata17/StataBE-64.exe",
    "/mnt/c/Program Files/Stata17/StataMP-64.exe",
    "/mnt/c/Program Files/Stata17/StataSE-64.exe",
]

_STATA_EXE: str | None = None
_STATA_AVAILABLE: bool | None = None


def _find_stata_exe() -> str | None:
    """Find the Stata executable."""
    # Check env var first
    env_exe = os.environ.get("STATA_EXE")
    if env_exe and os.path.isfile(env_exe):
        return env_exe

    # Try default paths
    for path in _DEFAULT_STATA_PATHS:
        if os.path.isfile(path):
            return path

    return None


def stata_available() -> bool:
    """Check if Stata is available via batch mode."""
    global _STATA_AVAILABLE, _STATA_EXE
    if _STATA_AVAILABLE is not None:
        return _STATA_AVAILABLE

    _STATA_EXE = _find_stata_exe()
    _STATA_AVAILABLE = _STATA_EXE is not None
    return _STATA_AVAILABLE


# ---------------------------------------------------------------------------
# Batch mode Stata execution
# ---------------------------------------------------------------------------

# Use a persistent temp directory on the Windows side for Stata I/O
_STATA_WORKDIR = Path("/mnt/c/tmp/polars_reg_stata")


def _ensure_workdir() -> Path:
    """Create the working directory on the Windows side."""
    _STATA_WORKDIR.mkdir(parents=True, exist_ok=True)
    return _STATA_WORKDIR


def _wsl_to_win(path: str | Path) -> str:
    """Convert a WSL path to a Windows path."""
    result = subprocess.run(
        ["wslpath", "-w", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _load_data_to_stata(df: pl.DataFrame) -> str:
    """Save a Polars DataFrame as CSV for Stata to import.

    Returns the Windows path to the CSV file.
    """
    workdir = _ensure_workdir()
    csv_path = workdir / "data.csv"
    df.to_pandas().to_csv(csv_path, index=False)
    return _wsl_to_win(csv_path)


def _run_stata_do(do_content: str, timeout: int = 120) -> None:
    """Write a .do file and execute it in Stata batch mode.

    Stata is run with /e flag which executes the do-file and exits.
    We poll for a sentinel file to know when execution is complete.
    """
    global _STATA_EXE
    if _STATA_EXE is None:
        raise RuntimeError("Stata executable not found")

    workdir = _ensure_workdir()
    do_path = workdir / "run.do"
    sentinel = workdir / "done.txt"
    error_file = workdir / "error.txt"

    # Clean up previous sentinel/error files
    sentinel.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)

    # Wrap do content: add error handling and sentinel file
    win_sentinel = _wsl_to_win(sentinel)
    win_error = _wsl_to_win(error_file)

    wrapped = f"""capture noisily {{
{do_content}
}}
if _rc != 0 {{
    file open errfh using "{win_error}", write replace
    file write errfh "ERROR: " (_rc) _n
    file close errfh
}}
file open donefh using "{win_sentinel}", write replace
file write donefh "done" _n
file close donefh
"""
    do_path.write_text(wrapped)

    win_do = _wsl_to_win(do_path)

    # Launch Stata in background
    subprocess.Popen(
        [_STATA_EXE, "/e", "do", win_do],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll for sentinel file
    start = time.monotonic()
    while not sentinel.exists():
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"Stata did not complete within {timeout}s. Check {do_path} for issues."
            )
        time.sleep(0.5)

    # Check for errors
    if error_file.exists():
        error_msg = error_file.read_text().strip()
        raise RuntimeError(f"Stata returned an error: {error_msg}")


# ---------------------------------------------------------------------------
# Stata result extraction (from CSV written by .do file)
# ---------------------------------------------------------------------------


@dataclass
class StataResult:
    """Results extracted from a Stata regression."""

    coefficients: NDArray
    se: NDArray
    names: list[str]
    n_obs: int
    r_squared: float
    r_squared_adj: float
    df_r: int
    # reghdfe-specific
    r_squared_within: float | None = None
    df_absorbed: int | None = None
    # IV-specific
    j_stat: float | None = None
    j_pvalue: float | None = None


def _build_results_do(stata_cmd: str, model_type: str, win_csv_path: str) -> str:
    """Build the full .do file content: import data, run regression, save results."""
    workdir = _ensure_workdir()
    win_results = _wsl_to_win(workdir / "results.csv")

    # Build the results extraction block
    extract_lines = [
        f'import delimited "{win_csv_path}", clear',
        stata_cmd,
        "",
        f'file open fh using "{win_results}", write replace',
        'file write fh "param,coef,se" _n',
        "",
        "matrix b = e(b)",
        "matrix V = e(V)",
        "local names : colnames b",
        "local k = colsof(b)",
        "forvalues i = 1/`k' {",
        "    local name : word `i' of `names'",
        "    local coef = b[1,`i']",
        "    local se = sqrt(V[`i',`i'])",
        "    file write fh \"`name',`coef',`se'\" _n",
        "}",
        "",
        "* Write scalar metadata",
        'file write fh "___N___," %20.0f (e(N)) ",0" _n',
        'file write fh "___df_r___," %20.0f (e(df_r)) ",0" _n',
    ]

    # R-squared (may not exist for all models)
    extract_lines += [
        "capture local r2_val = e(r2)",
        "if _rc == 0 {",
        '    file write fh "___r2___," %20.12f (e(r2)) ",0" _n',
        "}",
        "capture local r2a_val = e(r2_a)",
        "if _rc == 0 {",
        '    file write fh "___r2_a___," %20.12f (e(r2_a)) ",0" _n',
        "}",
    ]

    # reghdfe-specific
    if model_type == "reghdfe":
        extract_lines += [
            "capture {",
            '    file write fh "___r2_within___," %20.12f (e(r2_within)) ",0" _n',
            '    file write fh "___df_a___," %20.0f (e(df_a)) ",0" _n',
            "}",
        ]

    # GMM J-test
    if model_type == "ivregress_gmm":
        extract_lines += [
            "capture {",
            '    file write fh "___J___," %20.12f (e(J)) ",0" _n',
            '    file write fh "___J_p___," %20.12f (e(J_p)) ",0" _n',
            "}",
        ]

    extract_lines.append("file close fh")

    return "\n".join(extract_lines)


def _extract_stata_results(model_type: str) -> StataResult:
    """Parse the results CSV written by Stata."""
    workdir = _ensure_workdir()
    results_path = workdir / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"Stata results file not found: {results_path}")

    lines = results_path.read_text().strip().split("\n")
    if not lines or lines[0].strip() != "param,coef,se":
        raise ValueError(f"Unexpected results file format: {lines[:3]}")

    names = []
    coefs = []
    ses = []
    metadata: dict[str, float] = {}

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        name, coef_str, se_str = parts

        if name.startswith("___") and name.endswith("___"):
            val = coef_str.strip()
            # Stata uses "." for missing values
            if val == ".":
                metadata[name] = float("nan")
            else:
                metadata[name] = float(val)
        else:
            # Strip Stata's equation prefix (e.g., "y1:x1" -> "x1")
            if ":" in name:
                name = name.split(":")[-1]
            names.append(name)
            coefs.append(float(coef_str.strip()))
            ses.append(float(se_str.strip()))

    n_obs = int(metadata.get("___N___", 0))
    df_r_val = metadata.get("___df_r___", 0)
    df_r = 0 if np.isnan(df_r_val) else int(df_r_val)
    r2 = metadata.get("___r2___", float("nan"))
    r2_adj = metadata.get("___r2_a___", float("nan"))
    r2_within = metadata.get("___r2_within___")
    df_absorbed = int(metadata["___df_a___"]) if "___df_a___" in metadata else None
    j_stat_val = metadata.get("___J___")
    j_stat = None if j_stat_val is None or np.isnan(j_stat_val) else j_stat_val
    j_pvalue_val = metadata.get("___J_p___")
    j_pvalue = None if j_pvalue_val is None or np.isnan(j_pvalue_val) else j_pvalue_val

    return StataResult(
        coefficients=np.array(coefs),
        se=np.array(ses),
        names=names,
        n_obs=n_obs,
        r_squared=r2,
        r_squared_adj=r2_adj,
        df_r=df_r,
        r_squared_within=r2_within,
        df_absorbed=df_absorbed,
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

    # IMPORTANT: Stata's reg command vce options:
    #   vce(robust)  = HC1 (White with small-sample correction n/(n-k))
    #   vce(hc2)     = HC2 (leverage-adjusted)
    #   vce(hc3)     = HC3 (jackknife-like)
    # There is no direct HC0 option in Stata's reg. HC0 = robust without
    # the n/(n-k) correction. We run HC1 and correct in comparison.

    if vcov == "HC0":
        return "vce(robust)"

    vce_map = {
        "iid": "",
        "HC1": "vce(robust)",
        "HC2": "vce(hc2)",
        "HC3": "vce(hc3)",
    }

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

    opts = ["small"]  # small-sample correction to match our n-k dof
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
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

    parts = ["ivregress gmm", spec.depvar]
    if exog_str:
        parts.append(exog_str)
    parts.append(f"({endog_str} = {instr_str})")

    opts = ["wmatrix(robust)"]  # GMM uses large-sample VCV by default
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
            f"No common coefficient names.\n  polars_reg: {polars_names}\n  Stata: {stata_names}"
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
            polars_result.names,
            polars_result.coefficients,
            polars_result.se,
            stata_result.names,
            stata_result.coefficients,
            stata_result.se,
        )
    except ValueError as e:
        return ComparisonResult(
            estimator=estimator,
            formula=formula,
            stata_command=stata_command,
            passed=False,
            n_obs_match=n_match,
            coef_max_rdiff=float("inf"),
            se_max_rdiff=float("inf"),
            r2_rdiff=None,
            details=[str(e)],
            polars_names=polars_result.names,
            stata_names=stata_result.names,
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
    # Skip for no-intercept models: Stata uses uncentered R², we use centered
    has_intercept = "_cons" in names
    r2_rdiff = None
    if has_intercept and not np.isnan(stata_result.r_squared):
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
        estimator=estimator,
        formula=formula,
        stata_command=stata_command,
        passed=passed,
        n_obs_match=n_match,
        coef_max_rdiff=coef_max,
        se_max_rdiff=se_max,
        r2_rdiff=r2_rdiff,
        details=details,
        polars_coefs=pr_c,
        stata_coefs=st_c,
        polars_se=pr_se,
        stata_se=st_se,
        polars_names=names,
        stata_names=names,
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
        data: Polars DataFrame (will be saved as CSV for Stata)
        vcov: Variance-covariance type
        cluster: Clustering variable(s)
        rtol: Relative tolerance for comparisons (default 1e-6)
        stata_pre_commands: Extra Stata commands to run before the regression
            (e.g., ["set matsize 10000"])

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

    # 3. Load data (save as CSV) and build do-file
    win_csv = _load_data_to_stata(data)

    pre_cmds = ""
    if stata_pre_commands:
        pre_cmds = "\n".join(stata_pre_commands) + "\n"

    do_content = pre_cmds + _build_results_do(stata_cmd, model_type, win_csv)

    # 4. Run Stata
    _run_stata_do(do_content)

    # 5. Extract Stata results
    stata_result = _extract_stata_results(model_type)

    # 6. Compare
    comparison = compare_results(
        estimator, formula, stata_cmd, polars_result, stata_result, rtol, vcov
    )

    if not comparison.passed:
        raise AssertionError(f"Stata parity check failed:\n{comparison}")

    return comparison
