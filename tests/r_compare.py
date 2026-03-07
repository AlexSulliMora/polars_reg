"""R parity testing infrastructure.

Translates polars_reg regression calls into equivalent R scripts,
runs both (polars_reg in Python, R via Rscript),
and compares results to machine precision.

Usage:
    result = assert_r_parity("ols", "y ~ x1 + x2", data, cluster=["firm"])

Requires:
    - Rscript on PATH
    - R packages: fixest, sandwich, lmtest, AER, plm
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_reg._formula import parse_formula

# ---------------------------------------------------------------------------
# R availability check
# ---------------------------------------------------------------------------

_R_AVAILABLE: bool | None = None
_R_PACKAGES: set[str] | None = None

_REQUIRED_PACKAGES = {"fixest", "sandwich", "lmtest", "AER", "plm"}


def _check_r_packages() -> set[str]:
    """Return the set of required R packages that are installed."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", 'cat(paste(installed.packages()[,"Package"], collapse="\\n"))'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return set()
        installed = set(result.stdout.strip().split("\n"))
        return _REQUIRED_PACKAGES & installed
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def r_available() -> bool:
    """Check if Rscript is available and required packages are installed."""
    global _R_AVAILABLE, _R_PACKAGES
    if _R_AVAILABLE is not None:
        return _R_AVAILABLE

    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _R_AVAILABLE = False
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _R_AVAILABLE = False
        return False

    _R_PACKAGES = _check_r_packages()
    # Need at least fixest for the core tests
    _R_AVAILABLE = "fixest" in _R_PACKAGES
    return _R_AVAILABLE


def r_has_package(pkg: str) -> bool:
    """Check if a specific R package is available."""
    global _R_PACKAGES
    if _R_PACKAGES is None:
        r_available()
    return _R_PACKAGES is not None and pkg in _R_PACKAGES


# ---------------------------------------------------------------------------
# R script execution
# ---------------------------------------------------------------------------


def _run_r_script(script: str, timeout: int = 60) -> str:
    """Execute an R script via Rscript and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["Rscript", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}\n\nScript:\n{script}")
        return result.stdout
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# R result extraction
# ---------------------------------------------------------------------------


@dataclass
class RResult:
    """Results extracted from an R regression."""

    coefficients: NDArray
    se: NDArray
    names: list[str]
    n_obs: int
    r_squared: float


def _parse_r_csv(csv_path: str) -> RResult:
    """Parse the CSV results written by an R script."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"R results file not found: {csv_path}")

    lines = path.read_text().strip().split("\n")
    if not lines or lines[0].strip() != "param,coef,se":
        raise ValueError(f"Unexpected R results format: {lines[:3]}")

    names = []
    coefs = []
    ses = []
    metadata: dict[str, float] = {}

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        name, coef_str, se_str = parts
        name = name.strip().strip('"')

        if name.startswith("___") and name.endswith("___"):
            metadata[name] = float(coef_str.strip())
        else:
            # R uses (Intercept) for the constant
            if name == "(Intercept)":
                name = "_cons"
            names.append(name)
            coefs.append(float(coef_str.strip()))
            ses.append(float(se_str.strip()))

    n_obs = int(metadata.get("___N___", 0))
    r2 = metadata.get("___r2___", float("nan"))

    return RResult(
        coefficients=np.array(coefs),
        se=np.array(ses),
        names=names,
        n_obs=n_obs,
        r_squared=r2,
    )


# ---------------------------------------------------------------------------
# Formula-to-R translation
# ---------------------------------------------------------------------------


def to_r_script(
    estimator: str,
    formula: str,
    csv_path: str,
    results_path: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    entity: str | None = None,
    time: str | None = None,
) -> str:
    """Build a complete R script that reads CSV data, runs regression, writes results CSV."""
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)

    lines = [
        "# Auto-generated R parity test script",
        f'df <- read.csv("{csv_path}")',
        "",
    ]

    if estimator == "ols":
        lines += _ols_r_script(spec, formula, vcov, cluster)
    elif estimator == "iv2sls":
        lines += _iv2sls_r_script(spec, formula, vcov, cluster)
    elif estimator == "liml":
        lines += _liml_r_script(spec, formula, vcov, cluster)
    elif estimator == "gmm_iv":
        # GMM doesn't have a clean R equivalent; use 2SLS as approximate comparison
        lines += _iv2sls_r_script(spec, formula, vcov, cluster)
    elif estimator == "panel_fe":
        lines += _panel_r_script(spec, formula, "within", entity, time, cluster)
    elif estimator == "panel_re":
        lines += _panel_r_script(spec, formula, "random", entity, time, cluster)
    elif estimator == "panel_fd":
        lines += _panel_r_script(spec, formula, "fd", entity, time, cluster)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Write results CSV
    lines += [
        "",
        "# Extract results",
        "b <- coef(model)",
        "s <- sqrt(diag(vcov_mat))",
        "nms <- names(b)",
        f'fh <- file("{results_path}", "w")',
        'writeLines("param,coef,se", fh)',
        "for (i in seq_along(b)) {",
        '  writeLines(paste0(nms[i], ",", format(b[i], digits=15),'
        ' ",", format(s[i], digits=15)), fh)',
        "}",
        'writeLines(paste0("___N___,", nobs(model), ",0"), fh)',
        "tryCatch({",
        "  r2 <- summary(model)$r.squared",
        "  if (is.null(r2)) r2 <- summary(model)$r.sq",
        '  if (!is.null(r2)) writeLines(paste0("___r2___,", format(r2, digits=15), ",0"), fh)',
        "}, error=function(e) {})",
        "close(fh)",
    ]

    return "\n".join(lines)


def _ols_r_script(spec: Any, formula: str, vcov: str, cluster: list[str] | None) -> list[str]:
    """Generate R script lines for OLS regression."""
    has_fe = len(spec.fe) > 0

    if has_fe or cluster:
        # Use fixest::feols
        lines = ["library(fixest)"]
        r_formula = _build_feols_formula(spec)
        vcov_arg = _feols_vcov_arg(vcov, cluster)
        lines.append(f"model <- feols({r_formula}, data=df, vcov={vcov_arg})")
        lines.append("vcov_mat <- vcov(model)")
        return lines
    else:
        # Use base lm()
        r_formula = _build_lm_formula(spec)
        lines = [f"model <- lm({r_formula}, data=df)"]
        if vcov in ("HC1", "HC2", "HC3", "HC0"):
            lines = ["library(sandwich)", "library(lmtest)"] + lines
            lines.append(f'vcov_mat <- vcovHC(model, type="{vcov}")')
        else:
            lines.append("vcov_mat <- vcov(model)")
        return lines


def _iv2sls_r_script(spec: Any, formula: str, vcov: str, cluster: list[str] | None) -> list[str]:
    """Generate R script lines for 2SLS regression."""
    lines = ["library(fixest)"]
    r_formula = _build_feols_iv_formula(spec)
    vcov_arg = _feols_vcov_arg(vcov, cluster)
    lines.append(f"model <- feols({r_formula}, data=df, vcov={vcov_arg})")
    lines.append("vcov_mat <- vcov(model)")
    return lines


def _liml_r_script(spec: Any, formula: str, vcov: str, cluster: list[str] | None) -> list[str]:
    """Generate R script lines for LIML regression."""
    lines = ["library(AER)"]
    # AER::ivreg formula: y ~ exog + endog | exog + instruments
    exog = " + ".join(spec.exog) if spec.exog else "1"
    endog = " + ".join(spec.endog)
    instr = " + ".join(spec.instruments)

    lhs = f"{spec.depvar} ~ {exog} + {endog}"
    rhs = f"{exog} + {instr}"
    r_formula = f"{lhs} | {rhs}"

    noconstant = "" if spec.add_intercept else " - 1"
    if noconstant:
        r_formula = f"{spec.depvar} ~ {exog} + {endog} - 1 | {exog} + {instr} - 1"

    lines.append(f'model <- ivreg({r_formula}, data=df, model=TRUE, method="LIML")')
    if vcov in ("HC1", "HC2", "HC3", "HC0"):
        lines.append("library(sandwich)")
        lines.append(f'vcov_mat <- vcovHC(model, type="{vcov}")')
    else:
        lines.append("vcov_mat <- vcov(model)")
    return lines


def _panel_r_script(
    spec: Any,
    formula: str,
    model_type: str,
    entity: str | None,
    time: str | None,
    cluster: list[str] | None,
) -> list[str]:
    """Generate R script lines for panel regression."""
    lines = ["library(plm)"]
    r_formula = _build_lm_formula(spec)

    index_parts = []
    if entity:
        index_parts.append(f'"{entity}"')
    if time:
        index_parts.append(f'"{time}"')
    index_str = ", ".join(index_parts)

    lines.append(f'model <- plm({r_formula}, data=df, model="{model_type}", index=c({index_str}))')

    if cluster and len(cluster) == 1:
        lines.append("library(lmtest)")
        lines.append("library(sandwich)")
        lines.append('vcov_mat <- vcovHC(model, method="arellano", type="HC1", cluster="group")')
    else:
        lines.append("vcov_mat <- vcov(model)")

    return lines


# ---------------------------------------------------------------------------
# Formula builders
# ---------------------------------------------------------------------------


def _build_lm_formula(spec: Any) -> str:
    """Build an R formula for lm(): y ~ x1 + x2"""
    rhs = " + ".join(spec.exog) if spec.exog else "1"
    noconstant = " - 1" if not spec.add_intercept else ""
    return f"{spec.depvar} ~ {rhs}{noconstant}"


def _build_feols_formula(spec: Any) -> str:
    """Build an R formula for fixest::feols(): y ~ x1 + x2 | fe1 + fe2"""
    rhs = " + ".join(spec.exog) if spec.exog else "1"
    noconstant = " - 1" if not spec.add_intercept else ""
    fe_part = " + ".join(spec.fe) if spec.fe else ""
    formula = f"{spec.depvar} ~ {rhs}{noconstant}"
    if fe_part:
        formula += f" | {fe_part}"
    return formula


def _build_feols_iv_formula(spec: Any) -> str:
    """Build an R formula for fixest::feols() with IV: y ~ exog | fe | endog ~ instr"""
    exog = " + ".join(spec.exog) if spec.exog else "1"
    endog = " + ".join(spec.endog)
    instr = " + ".join(spec.instruments)
    fe_part = " + ".join(spec.fe) if spec.fe else "0"

    return f"{spec.depvar} ~ {exog} | {fe_part} | {endog} ~ {instr}"


def _feols_vcov_arg(vcov: str, cluster: list[str] | None) -> str:
    """Build the vcov argument for fixest::feols()."""
    if cluster:
        if len(cluster) == 1:
            return f"~{cluster[0]}"
        return "~" + " + ".join(cluster)
    if vcov in ("HC1", "HC0"):
        return '"hetero"'
    if vcov == "iid":
        return '"iid"'
    return '"hetero"'


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------


@dataclass
class RComparisonResult:
    """Detailed comparison between polars_reg and R results."""

    estimator: str
    formula: str
    r_command: str
    passed: bool
    n_obs_match: bool
    coef_max_rdiff: float
    se_max_rdiff: float
    r2_rdiff: float | None
    details: list[str] = field(default_factory=list)
    polars_coefs: NDArray | None = None
    r_coefs: NDArray | None = None
    polars_se: NDArray | None = None
    r_se: NDArray | None = None
    polars_names: list[str] | None = None
    r_names: list[str] | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.estimator}: {self.formula}",
            f"  R: {self.r_command}",
            f"  N match: {self.n_obs_match}",
            f"  Coef max relative diff: {self.coef_max_rdiff:.2e}",
            f"  SE max relative diff: {self.se_max_rdiff:.2e}",
        ]
        if self.r2_rdiff is not None:
            lines.append(f"  R2 relative diff: {self.r2_rdiff:.2e}")
        for d in self.details:
            lines.append(f"  {d}")
        return "\n".join(lines)


def _align_coefficients(
    polars_names: list[str],
    polars_coefs: NDArray,
    polars_se: NDArray,
    r_names: list[str],
    r_coefs: NDArray,
    r_se: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
    """Align polars_reg and R coefficients by variable name."""
    pr_map = {name: i for i, name in enumerate(polars_names)}
    r_map = {name: i for i, name in enumerate(r_names)}

    common = [n for n in polars_names if n in r_map]

    if not common:
        raise ValueError(
            f"No common coefficient names.\n  polars_reg: {polars_names}\n  R: {r_names}"
        )

    pr_idx = [pr_map[n] for n in common]
    r_idx = [r_map[n] for n in common]

    return (
        polars_coefs[pr_idx],
        polars_se[pr_idx],
        r_coefs[r_idx],
        r_se[r_idx],
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
    r_command: str,
    polars_result: Any,
    r_result: RResult,
    rtol: float = 1e-6,
) -> RComparisonResult:
    """Compare polars_reg and R regression results."""
    details: list[str] = []
    passed = True

    n_match = polars_result.n_obs == r_result.n_obs
    if not n_match:
        details.append(f"N mismatch: polars={polars_result.n_obs}, R={r_result.n_obs}")
        passed = False

    try:
        pr_c, pr_se, r_c, r_se, names = _align_coefficients(
            polars_result.names,
            polars_result.coefficients,
            polars_result.se,
            r_result.names,
            r_result.coefficients,
            r_result.se,
        )
    except ValueError as e:
        return RComparisonResult(
            estimator=estimator,
            formula=formula,
            r_command=r_command,
            passed=False,
            n_obs_match=n_match,
            coef_max_rdiff=float("inf"),
            se_max_rdiff=float("inf"),
            r2_rdiff=None,
            details=[str(e)],
            polars_names=polars_result.names,
            r_names=r_result.names,
        )

    coef_rdiff = _relative_diff(pr_c, r_c)
    coef_max = float(coef_rdiff.max())
    if coef_max > rtol:
        passed = False
        for i, name in enumerate(names):
            if coef_rdiff[i] > rtol:
                details.append(
                    f"Coef '{name}': polars={pr_c[i]:.10f}, R={r_c[i]:.10f}, "
                    f"rdiff={coef_rdiff[i]:.2e}"
                )

    se_rdiff = _relative_diff(pr_se, r_se)
    se_max = float(se_rdiff.max())
    if se_max > rtol:
        passed = False
        for i, name in enumerate(names):
            if se_rdiff[i] > rtol:
                details.append(
                    f"SE '{name}': polars={pr_se[i]:.10f}, R={r_se[i]:.10f}, "
                    f"rdiff={se_rdiff[i]:.2e}"
                )

    has_intercept = "_cons" in names
    r2_rdiff = None
    if has_intercept and not np.isnan(r_result.r_squared):
        r2_rdiff = abs(polars_result.r_squared - r_result.r_squared) / max(
            abs(r_result.r_squared), 1e-15
        )
        if r2_rdiff > rtol:
            passed = False
            details.append(
                f"R2 mismatch: polars={polars_result.r_squared:.10f}, "
                f"R={r_result.r_squared:.10f}, rdiff={r2_rdiff:.2e}"
            )

    return RComparisonResult(
        estimator=estimator,
        formula=formula,
        r_command=r_command,
        passed=passed,
        n_obs_match=n_match,
        coef_max_rdiff=coef_max,
        se_max_rdiff=se_max,
        r2_rdiff=r2_rdiff,
        details=details,
        polars_coefs=pr_c,
        r_coefs=r_c,
        polars_se=pr_se,
        r_se=r_se,
        polars_names=names,
        r_names=names,
    )


# ---------------------------------------------------------------------------
# All-in-one parity assertion
# ---------------------------------------------------------------------------


def assert_r_parity(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    entity: str | None = None,
    time: str | None = None,
    rtol: float = 1e-6,
) -> RComparisonResult:
    """Run a regression in both polars_reg and R, assert results match.

    Args:
        estimator: One of "ols", "iv2sls", "liml", "gmm_iv", "panel_fe", "panel_re", "panel_fd"
        formula: polars_reg formula string
        data: Polars DataFrame (will be saved as CSV for R)
        vcov: Variance-covariance type
        cluster: Clustering variable(s)
        entity: Panel entity column
        time: Panel time column
        rtol: Relative tolerance for comparisons (default 1e-6)

    Returns:
        RComparisonResult with detailed comparison

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
        "panel_fe": pr.panel_fe,
        "panel_re": pr.panel_re,
        "panel_fd": pr.panel_fd,
    }

    func = estimator_funcs[estimator]
    kwargs: dict[str, Any] = {"formula": formula, "data": data}
    if vcov != "iid":
        kwargs["vcov"] = vcov
    if cluster:
        kwargs["cluster"] = cluster
    if entity:
        kwargs["entity"] = entity
    if time:
        kwargs["time"] = time
    polars_result = func(**kwargs)

    # 2. Save data as CSV for R
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_pandas().to_csv(f, index=False)
        csv_path = f.name

    results_path = csv_path.replace(".csv", "_results.csv")

    try:
        # 3. Generate R script
        script = to_r_script(
            estimator,
            formula,
            csv_path,
            results_path,
            vcov=vcov,
            cluster=cluster,
            entity=entity,
            time=time,
        )

        # 4. Run R
        _run_r_script(script)

        # 5. Parse R results
        r_result = _parse_r_csv(results_path)

        # 6. Compare
        comparison = compare_results(
            estimator,
            formula,
            script.split("\n")[3],  # The model <- ... line
            polars_result,
            r_result,
            rtol,
        )

        if not comparison.passed:
            raise AssertionError(f"R parity check failed:\n{comparison}")

        return comparison

    finally:
        os.unlink(csv_path)
        if os.path.exists(results_path):
            os.unlink(results_path)
