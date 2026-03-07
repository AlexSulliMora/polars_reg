"""Generate equivalent Stata code and optionally compare results.

Public API:
    to_stata(estimator, formula, ...) -> str
        Returns Stata code as a string for copy-paste or pystata execution.

    compare_stata(estimator, formula, data, ...) -> ComparisonReport
        Runs both polars_reg and Stata (via pystata), prints side-by-side comparison.

Usage:
    >>> import polars_reg as pr
    >>> print(pr.to_stata("ols", "y ~ x1 + x2 | fe1", cluster=["fe1"]))
    reghdfe y x1 x2, absorb(fe1) vce(cluster fe1)

    >>> report = pr.compare_stata("ols", "y ~ x1 + x2", data=df)
    # prints comparison table and returns report object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_reg._formula import parse_formula

# ---------------------------------------------------------------------------
# Formula-to-Stata translation
# ---------------------------------------------------------------------------


def to_stata(
    estimator: str,
    formula: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    pystata: bool = False,
) -> str:
    """Generate equivalent Stata code for a polars_reg regression call.

    Args:
        estimator: One of "ols", "iv2sls", "liml", "gmm_iv",
                   "panel_fe", "panel_re", "panel_fd"
        formula: polars_reg formula string
        vcov: Variance-covariance type ("iid", "HC1", etc.)
        cluster: Clustering variable(s)
        entity: Entity variable (panel estimators)
        time: Time variable (panel_fe, panel_fd)
        pystata: If True, wrap in pystata-compatible Python code

    Returns:
        Stata command string, or pystata Python code if pystata=True
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    stata_cmd = _translate(estimator, formula, vcov, cluster, entity, time)

    if pystata:
        return _wrap_pystata(stata_cmd, estimator)
    return stata_cmd


def _translate(
    estimator: str,
    formula: str,
    vcov: str,
    cluster: list[str] | None,
    entity: str | None,
    time: str | None,
) -> str:
    """Core translation logic."""
    spec = parse_formula(formula)
    vce_opt = _build_vce(vcov, cluster)

    if estimator == "ols":
        if spec.fe:
            return _to_reghdfe(spec, vcov, cluster)
        return _to_reg(spec, vce_opt)

    elif estimator == "iv2sls":
        if spec.fe:
            return _to_ivreghdfe(spec, "2sls", vcov, cluster)
        return _to_ivregress(spec, "2sls", vce_opt)

    elif estimator == "liml":
        return _to_ivregress(spec, "liml", vce_opt)

    elif estimator == "gmm_iv":
        return _to_ivregress_gmm(spec, vce_opt)

    elif estimator == "panel_fe":
        return _to_xtreg_fe(spec, vcov, cluster, entity, time)

    elif estimator == "panel_re":
        return _to_xtreg_re(spec, entity)

    elif estimator == "panel_fd":
        return _to_panel_fd(spec, vcov, cluster, entity, time)

    else:
        raise ValueError(
            f"Unknown estimator: {estimator!r}. "
            "Use one of: ols, iv2sls, liml, gmm_iv, panel_fe, panel_re, panel_fd"
        )


# ---------------------------------------------------------------------------
# VCE option builder
# ---------------------------------------------------------------------------


def _build_vce(vcov: str, cluster: list[str] | None) -> str:
    if cluster:
        if len(cluster) == 1:
            return f"vce(cluster {cluster[0]})"
        return f"vce(cluster {' '.join(cluster)})"

    vce_map = {
        "iid": "",
        "HC0": "vce(robust)",
        "HC1": "vce(robust)",
        "HC2": "vce(hc2)",
        "HC3": "vce(hc3)",
    }
    return vce_map.get(vcov, "")


def _opts_str(opts: list[str]) -> str:
    opts = [o for o in opts if o]
    return ", " + " ".join(opts) if opts else ""


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def _to_reg(spec: Any, vce_opt: str) -> str:
    parts = ["reg", spec.depvar] + spec.exog
    opts = []
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    return " ".join(parts) + _opts_str(opts)


def _to_reghdfe(spec: Any, vcov: str, cluster: list[str] | None) -> str:
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
        opts.append("vce(unadjusted)")
    if not spec.add_intercept:
        opts.append("noconstant")
    return " ".join(parts) + _opts_str(opts)


# ---------------------------------------------------------------------------
# IV
# ---------------------------------------------------------------------------


def _to_ivregress(spec: Any, method: str, vce_opt: str) -> str:
    endog = " ".join(spec.endog)
    instr = " ".join(spec.instruments)
    exog = " ".join(spec.exog) if spec.exog else ""
    parts = [f"ivregress {method}", spec.depvar]
    if exog:
        parts.append(exog)
    parts.append(f"({endog} = {instr})")
    opts = ["small"]
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    return " ".join(parts) + _opts_str(opts)


def _to_ivreghdfe(spec: Any, method: str, vcov: str, cluster: list[str] | None) -> str:
    endog = " ".join(spec.endog)
    instr = " ".join(spec.instruments)
    exog = " ".join(spec.exog) if spec.exog else ""
    parts = ["ivreghdfe", spec.depvar]
    if exog:
        parts.append(exog)
    parts.append(f"({endog} = {instr})")
    opts = [f"absorb({' '.join(spec.fe)})"]
    if cluster:
        if len(cluster) == 1:
            opts.append(f"vce(cluster {cluster[0]})")
        else:
            opts.append(f"vce(cluster {' '.join(cluster)})")
    elif vcov == "HC1":
        opts.append("vce(robust)")
    return " ".join(parts) + _opts_str(opts)


def _to_ivregress_gmm(spec: Any, vce_opt: str) -> str:
    endog = " ".join(spec.endog)
    instr = " ".join(spec.instruments)
    exog = " ".join(spec.exog) if spec.exog else ""
    parts = ["ivregress gmm", spec.depvar]
    if exog:
        parts.append(exog)
    parts.append(f"({endog} = {instr})")
    opts = ["wmatrix(robust)"]
    if not spec.add_intercept:
        opts.append("noconstant")
    if vce_opt:
        opts.append(vce_opt)
    return " ".join(parts) + _opts_str(opts)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def _to_xtreg_fe(
    spec: Any, vcov: str, cluster: list[str] | None, entity: str | None, time: str | None
) -> str:
    if not entity:
        raise ValueError("panel_fe requires entity=")
    lines = [f"xtset {entity}" + (f" {time}" if time else "")]
    parts = ["xtreg", spec.depvar] + spec.exog
    opts = ["fe"]
    if cluster:
        opts.append(f"vce(cluster {cluster[0]})")
    elif vcov == "HC1":
        opts.append("vce(robust)")
    lines.append(" ".join(parts) + _opts_str(opts))
    return "\n".join(lines)


def _to_xtreg_re(spec: Any, entity: str | None) -> str:
    if not entity:
        raise ValueError("panel_re requires entity=")
    lines = [f"xtset {entity}"]
    parts = ["xtreg", spec.depvar] + spec.exog
    opts = ["re"]
    lines.append(" ".join(parts) + _opts_str(opts))
    return "\n".join(lines)


def _to_panel_fd(
    spec: Any, vcov: str, cluster: list[str] | None, entity: str | None, time: str | None
) -> str:
    if not entity or not time:
        raise ValueError("panel_fd requires entity= and time=")
    lines = [f"xtset {entity} {time}"]
    # Stata: reg D.y D.x1 D.x2, vce(cluster entity)
    d_depvar = f"D.{spec.depvar}"
    d_exog = [f"D.{v}" for v in spec.exog]
    parts = ["reg", d_depvar] + d_exog
    opts = []
    if cluster:
        opts.append(f"vce(cluster {cluster[0]})")
    elif vcov == "HC1":
        opts.append("vce(robust)")
    lines.append(" ".join(parts) + _opts_str(opts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# pystata wrapper
# ---------------------------------------------------------------------------


def _wrap_pystata(stata_cmd: str, estimator: str) -> str:
    """Wrap Stata command in pystata-compatible Python code."""
    lines = [
        "import stata_setup",
        "stata_setup.config('/path/to/stata', 'mp')  # adjust path and edition",
        "from pystata import stata",
        "",
        "# Load your data first:",
        "# stata.pdataframe_to_data(df.to_pandas(), force=True)",
        "",
        f"# Equivalent to polars_reg.{estimator}():",
    ]
    for cmd_line in stata_cmd.split("\n"):
        lines.append(f'stata.run("{cmd_line}")')
    lines.extend(
        [
            "",
            "# Extract results:",
            'stata.run("matrix list e(b)")',
            'stata.run("matrix list e(V)")',
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# compare_stata: run both and compare
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Side-by-side comparison of polars_reg vs Stata results."""

    estimator: str
    formula: str
    stata_command: str
    polars_coefs: NDArray
    polars_se: NDArray
    stata_coefs: NDArray | None
    stata_se: NDArray | None
    names: list[str]
    match: bool | None  # None if Stata wasn't run
    max_coef_rdiff: float | None
    max_se_rdiff: float | None
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        w = 70
        lines = [
            f"{'=' * w}",
            f"  polars_reg vs Stata: {self.estimator}",
            f"{'=' * w}",
            f"  Stata command: {self.stata_command}",
            f"{'=' * w}",
        ]
        if self.stata_coefs is None:
            lines.append("  Stata results: not available (pystata not configured)")
            lines.append("  Showing polars_reg results only:")
            lines.append(f"{'=' * w}")
            hdr = f"  {'':>12} {'polars_reg':>12} {'Stata':>12}"
            lines.append(hdr)
            lines.append(f"  {'-' * (w - 4)}")
            for i, name in enumerate(self.names):
                lines.append(f"  {name:>12} {self.polars_coefs[i]:>12.6f} {'—':>12}")
                lines.append(f"  {'(SE)':>12} ({self.polars_se[i]:.6f}){'':>7} {'—':>12}")
        else:
            status = "PASS" if self.match else "FAIL"
            lines.append(f"  Status: {status}")
            if self.max_coef_rdiff is not None:
                lines.append(f"  Max coef relative diff: {self.max_coef_rdiff:.2e}")
            if self.max_se_rdiff is not None:
                lines.append(f"  Max SE relative diff:   {self.max_se_rdiff:.2e}")
            lines.append(f"{'=' * w}")
            hdr = f"  {'':>12} {'polars_reg':>12} {'Stata':>12} {'rel.diff':>12}"
            lines.append(hdr)
            lines.append(f"  {'-' * (w - 4)}")
            for i, name in enumerate(self.names):
                c_rd = _rdiff_scalar(self.polars_coefs[i], self.stata_coefs[i])
                lines.append(
                    f"  {name:>12} {self.polars_coefs[i]:>12.6f} "
                    f"{self.stata_coefs[i]:>12.6f} {c_rd:>12.2e}"
                )
                se_rd = _rdiff_scalar(self.polars_se[i], self.stata_se[i])
                lines.append(
                    f"  {'(SE)':>12} ({self.polars_se[i]:.6f}) "
                    f"({self.stata_se[i]:.6f}) {se_rd:>12.2e}"
                )
        for d in self.details:
            lines.append(f"  {d}")
        lines.append(f"{'=' * w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "PASS" if self.match else ("FAIL" if self.match is False else "no Stata")
        return f"<ComparisonReport {self.estimator} [{status}]>"


def _rdiff_scalar(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom


def compare_stata(
    estimator: str,
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
    rtol: float = 1e-6,
) -> ComparisonReport:
    """Run regression in polars_reg and Stata (via pystata), compare results.

    Requires pystata to be configured. If pystata is not available, returns
    the polars_reg results alongside the generated Stata code so users can
    run it manually.

    Args:
        estimator: "ols", "iv2sls", "liml", "gmm_iv", "panel_fe", "panel_re", "panel_fd"
        formula: polars_reg formula string
        data: Polars DataFrame or LazyFrame
        vcov: Variance-covariance type
        cluster: Clustering variable(s)
        entity: Entity variable (panel estimators)
        time: Time variable (panel estimators)
        rtol: Relative tolerance for comparison (default 1e-6)

    Returns:
        ComparisonReport with side-by-side results
    """

    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(cluster, str):
        cluster = [cluster]

    # 1. Run polars_reg
    polars_result = _run_polars(estimator, formula, data, vcov, cluster, entity, time)

    # 2. Generate Stata command
    stata_cmd = to_stata(estimator, formula, vcov, cluster, entity=entity, time=time)

    # 3. Try pystata
    stata_coefs, stata_se, stata_names, details = _try_pystata(
        stata_cmd, estimator, data, entity, time
    )

    # 4. Align and compare
    if stata_coefs is not None and stata_names is not None and stata_se is not None:
        pr_c, pr_se, st_c, st_se, names = _align(
            polars_result.names,
            polars_result.coefficients,
            polars_result.se,
            stata_names,
            stata_coefs,
            stata_se,
        )
        coef_rdiff = np.abs(pr_c - st_c) / np.maximum(np.maximum(np.abs(pr_c), np.abs(st_c)), 1e-15)
        se_denom = np.maximum(np.maximum(np.abs(pr_se), np.abs(st_se)), 1e-15)
        se_rdiff = np.abs(pr_se - st_se) / se_denom
        max_c = float(coef_rdiff.max())
        max_s = float(se_rdiff.max())
        match = max_c <= rtol and max_s <= rtol

        report = ComparisonReport(
            estimator=estimator,
            formula=formula,
            stata_command=stata_cmd,
            polars_coefs=pr_c,
            polars_se=pr_se,
            stata_coefs=st_c,
            stata_se=st_se,
            names=names,
            match=match,
            max_coef_rdiff=max_c,
            max_se_rdiff=max_s,
            details=details,
        )
    else:
        report = ComparisonReport(
            estimator=estimator,
            formula=formula,
            stata_command=stata_cmd,
            polars_coefs=polars_result.coefficients,
            polars_se=polars_result.se,
            stata_coefs=None,
            stata_se=None,
            names=polars_result.names,
            match=None,
            max_coef_rdiff=None,
            max_se_rdiff=None,
            details=details,
        )

    print(report.summary())
    return report


def _run_polars(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str,
    cluster: list[str] | None,
    entity: str | None,
    time: str | None,
) -> Any:
    """Run the polars_reg estimator."""
    import polars_reg as pr

    kwargs: dict[str, Any] = {"formula": formula, "data": data}

    if estimator in ("ols", "iv2sls", "liml", "gmm_iv"):
        if vcov != "iid":
            kwargs["vcov"] = vcov
        if cluster:
            kwargs["cluster"] = cluster
        func = {"ols": pr.ols, "iv2sls": pr.iv2sls, "liml": pr.liml, "gmm_iv": pr.gmm_iv}
        return func[estimator](**kwargs)

    elif estimator == "panel_fe":
        if not entity:
            raise ValueError("panel_fe requires entity=")
        kwargs["entity"] = entity
        if time:
            kwargs["time"] = time
        if vcov != "iid":
            kwargs["vcov"] = vcov
        if cluster:
            kwargs["cluster"] = cluster
        return pr.panel_fe(**kwargs)

    elif estimator == "panel_re":
        if not entity:
            raise ValueError("panel_re requires entity=")
        kwargs["entity"] = entity
        if vcov != "iid":
            kwargs["vcov"] = vcov
        return pr.panel_re(**kwargs)

    elif estimator == "panel_fd":
        if not entity or not time:
            raise ValueError("panel_fd requires entity= and time=")
        kwargs["entity"] = entity
        kwargs["time"] = time
        if vcov != "iid":
            kwargs["vcov"] = vcov
        if cluster:
            kwargs["cluster"] = cluster
        return pr.panel_fd(**kwargs)

    else:
        raise ValueError(f"Unknown estimator: {estimator!r}")


def _try_pystata(
    stata_cmd: str,
    estimator: str,
    data: pl.DataFrame,
    entity: str | None,
    time: str | None,
) -> tuple[NDArray | None, NDArray | None, list[str] | None, list[str]]:
    """Try to run the Stata command via pystata. Returns (coefs, se, names, details)."""
    details: list[str] = []
    try:
        from pystata import stata  # type: ignore[import-untyped]
    except ImportError:
        details.append(
            "pystata not available. Install and configure pystata to enable "
            "automatic comparison. You can run the Stata command manually."
        )
        return None, None, None, details

    try:
        # Load data into Stata
        pdf = data.to_pandas()
        stata.pdataframe_to_data(pdf, force=True)

        # Run the command(s)
        for line in stata_cmd.split("\n"):
            line = line.strip()
            if line:
                stata.run(line, quietly=True)

        # Extract e(b) and e(V)
        stata.run("matrix __b = e(b)", quietly=True)
        stata.run("matrix __V = e(V)", quietly=True)

        b_df = stata.matrix_to_pdataframe("__b")
        v_df = stata.matrix_to_pdataframe("__V")

        names = list(b_df.columns)
        # Strip equation prefixes (e.g. "y:x1" -> "x1", "D.x1" -> "x1")
        clean_names = []
        for n in names:
            if ":" in n:
                n = n.split(":")[-1]
            if n.startswith("D."):
                n = n[2:]
            clean_names.append(n)

        coefs = b_df.values.flatten().astype(np.float64)
        se = np.sqrt(np.diag(v_df.values.astype(np.float64)))

        return coefs, se, clean_names, details

    except Exception as e:
        details.append(f"pystata execution failed: {e}")
        return None, None, None, details


def _align(
    pr_names: list[str],
    pr_coefs: NDArray,
    pr_se: NDArray,
    st_names: list[str],
    st_coefs: NDArray,
    st_se: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
    """Align coefficients by variable name."""
    pr_map = {n: i for i, n in enumerate(pr_names)}
    st_map = {n: i for i, n in enumerate(st_names)}
    common = [n for n in pr_names if n in st_map]
    if not common:
        raise ValueError(f"No common names: polars_reg={pr_names}, Stata={st_names}")
    pi = [pr_map[n] for n in common]
    si = [st_map[n] for n in common]
    return pr_coefs[pi], pr_se[pi], st_coefs[si], st_se[si], common
