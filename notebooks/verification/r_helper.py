"""Helper utilities for R-equivalence verification notebooks.

Provides functions to:
- Load standard R datasets as Polars DataFrames
- Run R regression scripts and parse results
- Compare polars_reg results with R output

Requires:
    - Rscript on PATH (or ~/micromamba/envs/renv/bin/Rscript)
    - R packages: fixest, sandwich, lmtest, AER, plm, quantreg
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Try common Rscript locations
_RSCRIPT_CANDIDATES = [
    os.environ.get("RSCRIPT_PATH", ""),
    os.path.expanduser("~/micromamba/envs/renv/bin/Rscript"),
    "Rscript",
]
RSCRIPT = next((p for p in _RSCRIPT_CANDIDATES if p and os.path.isfile(p)), "Rscript")


# ---------------------------------------------------------------------------
# R execution
# ---------------------------------------------------------------------------


def run_r(script: str, timeout: int = 120) -> str:
    """Execute an R script via Rscript and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(
            [RSCRIPT, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"R script failed (exit {result.returncode}):\n"
                f"--- stderr ---\n{result.stderr}\n"
                f"--- script ---\n{script}"
            )
        return result.stdout
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_r_dataset(
    name: str,
    package: str | None = None,
    extra_code: str = "",
) -> pl.DataFrame:
    """Load an R dataset and return as a Polars DataFrame.

    Args:
        name: Dataset name (e.g. "Grunfeld", "mtcars", "engel")
        package: R package to load (e.g. "plm", "AER", "quantreg")
        extra_code: Additional R code to transform the data before export

    Returns:
        Polars DataFrame with the dataset
    """
    lines = []
    if package:
        lines.append(f"library({package})")
    lines.append(f"data({name})")
    lines.append(f"df <- {name}")
    if extra_code:
        lines.append(extra_code)
    lines.append('write.csv(df, file=stdout(), row.names=FALSE)')

    stdout = run_r("\n".join(lines))
    return pl.read_csv(io.StringIO(stdout))


# ---------------------------------------------------------------------------
# R regression results
# ---------------------------------------------------------------------------


@dataclass
class RResult:
    """Parsed R regression output."""

    coef: dict[str, float]
    se: dict[str, float]
    n_obs: int | None = None
    r_squared: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.coef.keys())

    @property
    def coef_array(self) -> np.ndarray:
        return np.array(list(self.coef.values()))

    @property
    def se_array(self) -> np.ndarray:
        return np.array(list(self.se.values()))


def run_r_regression(script: str, timeout: int = 120) -> RResult:
    """Run an R regression script and parse results.

    The R script MUST end by printing results in this format:
        ===RESULTS===
        param,coef,se
        varname1,0.123,0.045
        varname2,0.456,0.078
        ===META===
        N,100
        r2,0.85
        extra_key,value

    Use the helper function `r_extract_block()` to generate this output in R.
    """
    stdout = run_r(script, timeout=timeout)

    # Parse results section
    if "===RESULTS===" not in stdout:
        raise ValueError(
            f"R script output missing ===RESULTS=== marker.\nOutput:\n{stdout[:2000]}"
        )

    parts = stdout.split("===RESULTS===")[1]
    if "===META===" in parts:
        results_text, meta_text = parts.split("===META===")
    else:
        results_text = parts
        meta_text = ""

    coef = {}
    se = {}
    for line in results_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("param,"):
            continue
        fields = line.split(",")
        if len(fields) >= 3:
            name = fields[0].strip().strip('"')
            if name == "(Intercept)":
                name = "_cons"
            coef[name] = float(fields[1])
            se[name] = float(fields[2])

    # Parse metadata
    n_obs = None
    r_squared = None
    extra = {}
    for line in meta_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) >= 2:
            key = fields[0].strip()
            val = float(fields[1])
            if key == "N":
                n_obs = int(val)
            elif key == "r2":
                r_squared = val
            else:
                extra[key] = val

    return RResult(coef=coef, se=se, n_obs=n_obs, r_squared=r_squared, extra=extra)


# Reusable R code snippet for extracting regression results
R_EXTRACT = '''
# --- Extract results ---
cat("===RESULTS===\\n")
cat("param,coef,se\\n")
b <- coef(model)
s <- sqrt(diag(vcov_mat))
for (i in seq_along(b)) {
  cat(sprintf("%s,%.15e,%.15e\\n", names(b)[i], b[i], s[i]))
}
cat("===META===\\n")
cat(sprintf("N,%d\\n", nobs(model)))
tryCatch({
  r2 <- summary(model)$r.squared
  if (is.null(r2)) r2 <- summary(model)$r.sq
  if (!is.null(r2)) cat(sprintf("r2,%.15e\\n", r2))
}, error=function(e) {})
'''


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare(
    pr_result,
    r_result: RResult,
    rtol: float = 1e-6,
    se_rtol: float | None = None,
    label: str = "",
) -> pl.DataFrame:
    """Compare polars_reg result with R result.

    Returns a Polars DataFrame with side-by-side comparison and relative errors.
    Prints a pass/fail summary.
    """
    if se_rtol is None:
        se_rtol = rtol

    # Build name mapping: polars_reg name -> R name
    pr_names = pr_result.names
    r_names = r_result.names

    # Find common variables
    common = [n for n in pr_names if n in r_result.coef]

    if not common:
        print(f"WARNING: No matching variable names!")
        print(f"  polars_reg: {pr_names}")
        print(f"  R:          {r_names}")
        return pl.DataFrame()

    rows = []
    all_pass = True
    for name in common:
        pr_idx = pr_names.index(name)
        pr_c = float(pr_result.coefficients[pr_idx])
        pr_s = float(pr_result.se[pr_idx])
        r_c = r_result.coef[name]
        r_s = r_result.se[name]

        coef_rd = abs(pr_c - r_c) / max(abs(r_c), 1e-15)
        se_rd = abs(pr_s - r_s) / max(abs(r_s), 1e-15)

        coef_ok = coef_rd <= rtol
        se_ok = se_rd <= se_rtol
        if not (coef_ok and se_ok):
            all_pass = False

        rows.append({
            "variable": name,
            "coef_polars": round(pr_c, 8),
            "coef_R": round(r_c, 8),
            "coef_rdiff": f"{coef_rd:.2e}",
            "se_polars": round(pr_s, 8),
            "se_R": round(r_s, 8),
            "se_rdiff": f"{se_rd:.2e}",
            "pass": "ok" if (coef_ok and se_ok) else "FAIL",
        })

    df = pl.DataFrame(rows)

    tag = f" [{label}]" if label else ""
    if all_pass:
        n_match = f"N: polars={pr_result.n_obs}, R={r_result.n_obs}"
        status = "PASS" if pr_result.n_obs == r_result.n_obs else "PASS (N mismatch)"
        print(f"  {status}{tag}  (rtol={rtol:.0e})  {n_match}")
    else:
        print(f"  FAIL{tag}  (rtol={rtol:.0e})")

    return df


def compare_scalar(
    pr_value: float,
    r_value: float,
    name: str,
    rtol: float = 1e-6,
) -> bool:
    """Compare a single scalar value between polars_reg and R."""
    rdiff = abs(pr_value - r_value) / max(abs(r_value), 1e-15)
    ok = rdiff <= rtol
    status = "ok" if ok else "FAIL"
    print(f"  {name}: polars={pr_value:.8f}, R={r_value:.8f}, rdiff={rdiff:.2e} [{status}]")
    return ok
