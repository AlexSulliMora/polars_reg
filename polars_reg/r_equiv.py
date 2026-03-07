"""Generate equivalent R code and optionally compare results.

Public API:
    to_r(estimator, formula, ...) -> str
        Returns R code as a string for copy-paste or rpy2 execution.

    compare_r(estimator, formula, data, ...) -> ComparisonReport
        Runs both polars_reg and R (via rpy2), prints side-by-side comparison.

Usage:
    >>> import polars_reg as pr
    >>> print(pr.to_r("ols", "y ~ x1 + x2 | fe1", cluster=["fe1"]))
    library(fixest)
    model <- feols(y ~ x1 + x2 | fe1, data=df, vcov=~fe1)
    summary(model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_reg._formula import parse_formula

# ---------------------------------------------------------------------------
# Formula-to-R translation
# ---------------------------------------------------------------------------


def to_r(
    estimator: str,
    formula: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    *,
    entity: str | None = None,
    time: str | None = None,
) -> str:
    """Generate equivalent R code for a polars_reg regression call.

    Args:
        estimator: One of "ols", "iv2sls", "liml", "gmm_iv",
                   "panel_fe", "panel_re", "panel_fd"
        formula: polars_reg formula string
        vcov: Variance-covariance type ("iid", "HC1", etc.)
        cluster: Clustering variable(s)
        entity: Entity variable (panel estimators)
        time: Time variable (panel_fe, panel_fd)

    Returns:
        R code string.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    return _translate(estimator, formula, vcov, cluster, entity, time)


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

    if estimator == "ols":
        if spec.fe or cluster:
            return _to_feols(spec, vcov, cluster)
        return _to_lm(spec, vcov)

    elif estimator == "iv2sls":
        return _to_feols_iv(spec, vcov, cluster)

    elif estimator == "liml":
        return _to_ivreg(spec, vcov, method="liml")

    elif estimator == "gmm_iv":
        return _to_gmm(spec)

    elif estimator == "panel_fe":
        return _to_plm(spec, "within", entity, time, vcov, cluster)

    elif estimator == "panel_re":
        return _to_plm(spec, "random", entity, time, vcov, cluster=None)

    elif estimator == "panel_fd":
        return _to_plm(spec, "fd", entity, time, vcov, cluster)

    else:
        raise ValueError(
            f"Unknown estimator: {estimator!r}. "
            "Use one of: ols, iv2sls, liml, gmm_iv, panel_fe, panel_re, panel_fd"
        )


# ---------------------------------------------------------------------------
# OLS: lm() or fixest::feols()
# ---------------------------------------------------------------------------


def _to_lm(spec: Any, vcov: str) -> str:
    """OLS without FE or clustering → base R lm()."""
    r_formula = f"{spec.depvar} ~ {' + '.join(spec.exog)}"
    if not spec.add_intercept:
        r_formula += " - 1"

    lines = [f"model <- lm({r_formula}, data=df)"]

    if vcov == "iid":
        lines.append("summary(model)")
    else:
        lines = ["library(sandwich)", "library(lmtest)", ""] + lines
        hc_type = _r_hc_type(vcov)
        lines.append(f'coeftest(model, vcov=vcovHC(model, type="{hc_type}"))')

    return "\n".join(lines)


def _to_feols(spec: Any, vcov: str, cluster: list[str] | None) -> str:
    """OLS with FE and/or clustering → fixest::feols()."""
    lines = ["library(fixest)", ""]

    r_formula = f"{spec.depvar} ~ {' + '.join(spec.exog)}"
    if not spec.add_intercept:
        r_formula += " - 1"
    if spec.fe:
        r_formula += f" | {' + '.join(spec.fe)}"

    vcov_arg = _feols_vcov(vcov, cluster)
    if vcov_arg:
        lines.append(f"model <- feols({r_formula}, data=df, vcov={vcov_arg})")
    else:
        lines.append(f"model <- feols({r_formula}, data=df)")

    lines.append("summary(model)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IV: fixest::feols() or AER::ivreg()
# ---------------------------------------------------------------------------


def _to_feols_iv(spec: Any, vcov: str, cluster: list[str] | None) -> str:
    """2SLS with optional FE → fixest::feols() with IV syntax."""
    lines = ["library(fixest)", ""]

    exog_str = " + ".join(spec.exog) if spec.exog else "1"
    endog_str = " + ".join(spec.endog)
    instr_str = " + ".join(spec.instruments)

    # fixest IV syntax: y ~ exog | FE | endog ~ instruments
    if spec.fe:
        fe_str = " + ".join(spec.fe)
        r_formula = f"{spec.depvar} ~ {exog_str} | {fe_str} | {endog_str} ~ {instr_str}"
    else:
        r_formula = f"{spec.depvar} ~ {exog_str} | {endog_str} ~ {instr_str}"

    vcov_arg = _feols_vcov(vcov, cluster)
    if vcov_arg:
        lines.append(f"model <- feols({r_formula}, data=df, vcov={vcov_arg})")
    else:
        lines.append(f"model <- feols({r_formula}, data=df)")

    lines.append("summary(model)")
    return "\n".join(lines)


def _to_ivreg(spec: Any, vcov: str, method: str = "liml") -> str:
    """LIML → AER::ivreg()."""
    lines = ["library(AER)", ""]

    exog_str = " + ".join(spec.exog) if spec.exog else "1"
    endog_str = " + ".join(spec.endog)
    instr_str = " + ".join(spec.instruments)

    # AER ivreg syntax: y ~ exog + endog | exog + instruments
    lhs_vars = f"{exog_str} + {endog_str}" if spec.exog else endog_str
    rhs_vars = f"{exog_str} + {instr_str}" if spec.exog else instr_str

    r_formula = f"{spec.depvar} ~ {lhs_vars} | {rhs_vars}"
    if not spec.add_intercept:
        r_formula += " - 1"

    lines.append(f'model <- ivreg({r_formula}, data=df, model="{method}")')
    lines.append("summary(model)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def _to_gmm(spec: Any) -> str:
    """GMM-IV — no direct single-line R equivalent."""
    lines = [
        "# No direct single-function R equivalent for two-step efficient GMM.",
        "# Options:",
        "#   1. gmm::gmm() with custom moment conditions",
        "#   2. fixest::feols() for 2SLS (not identical to GMM)",
        "#",
        "# For approximate comparison, 2SLS with robust SEs:",
        "",
    ]

    exog_str = " + ".join(spec.exog) if spec.exog else "1"
    endog_str = " + ".join(spec.endog)
    instr_str = " + ".join(spec.instruments)
    r_formula = f"{spec.depvar} ~ {exog_str} | {endog_str} ~ {instr_str}"

    lines.append("library(fixest)")
    lines.append(f'model <- feols({r_formula}, data=df, vcov="HC1")')
    lines.append("summary(model)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Panel: plm
# ---------------------------------------------------------------------------


def _to_plm(
    spec: Any,
    model: str,
    entity: str | None,
    time: str | None,
    vcov: str,
    cluster: list[str] | None,
) -> str:
    """Panel estimators → plm::plm()."""
    if not entity:
        raise ValueError(f"panel_{model} requires entity=")
    if model == "fd" and not time:
        raise ValueError("panel_fd requires time=")

    lines = ["library(plm)", ""]

    r_formula = f"{spec.depvar} ~ {' + '.join(spec.exog)}"

    index_parts = [f'"{entity}"']
    if time:
        index_parts.append(f'"{time}"')
    index_str = f"c({', '.join(index_parts)})"

    lines.append(f'model <- plm({r_formula}, data=df, model="{model}", index={index_str})')

    if cluster:
        lines.append('coeftest(model, vcov=vcovHC(model, type="HC1", cluster="group"))')
    else:
        lines.append("summary(model)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_hc_type(vcov: str) -> str:
    """Map polars_reg vcov names to R sandwich HC type strings."""
    return {"HC0": "HC0", "HC1": "HC1", "HC2": "HC2", "HC3": "HC3"}.get(vcov, "HC1")


def _feols_vcov(vcov: str, cluster: list[str] | None) -> str:
    """Build fixest vcov argument string."""
    if cluster:
        if len(cluster) == 1:
            return f"~{cluster[0]}"
        return f"~{' + '.join(cluster)}"
    if vcov == "iid":
        return '"iid"'
    if vcov in ("HC0", "HC1"):
        return '"HC1"'
    if vcov == "HC2":
        return '"HC2"'
    if vcov == "HC3":
        return '"HC3"'
    return ""


# ---------------------------------------------------------------------------
# compare_r: run both and compare
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Side-by-side comparison of polars_reg vs R results."""

    estimator: str
    formula: str
    r_code: str
    polars_coefs: NDArray
    polars_se: NDArray
    r_coefs: NDArray | None
    r_se: NDArray | None
    names: list[str]
    match: bool | None  # None if R wasn't run
    max_coef_rdiff: float | None
    max_se_rdiff: float | None
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        w = 70
        lines = [
            f"{'=' * w}",
            f"  polars_reg vs R: {self.estimator}",
            f"{'=' * w}",
            "  R code:",
        ]
        for code_line in self.r_code.split("\n"):
            lines.append(f"    {code_line}")
        lines.append(f"{'=' * w}")

        if self.r_coefs is None:
            lines.append("  R results: not available (rpy2 not configured)")
            lines.append("  Showing polars_reg results only:")
            lines.append(f"{'=' * w}")
            hdr = f"  {'':>12} {'polars_reg':>12} {'R':>12}"
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
            hdr = f"  {'':>12} {'polars_reg':>12} {'R':>12} {'rel.diff':>12}"
            lines.append(hdr)
            lines.append(f"  {'-' * (w - 4)}")
            for i, name in enumerate(self.names):
                c_rd = _rdiff_scalar(self.polars_coefs[i], self.r_coefs[i])
                lines.append(
                    f"  {name:>12} {self.polars_coefs[i]:>12.6f} "
                    f"{self.r_coefs[i]:>12.6f} {c_rd:>12.2e}"
                )
                se_rd = _rdiff_scalar(self.polars_se[i], self.r_se[i])
                lines.append(
                    f"  {'(SE)':>12} ({self.polars_se[i]:.6f}) ({self.r_se[i]:.6f}) {se_rd:>12.2e}"
                )
        for d in self.details:
            lines.append(f"  {d}")
        lines.append(f"{'=' * w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "PASS" if self.match else ("FAIL" if self.match is False else "no R")
        return f"<ComparisonReport {self.estimator} [{status}]>"


def _rdiff_scalar(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom


def compare_r(
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
    """Run regression in polars_reg and R (via rpy2), compare results.

    Requires rpy2 and the relevant R packages (fixest, AER, plm, sandwich,
    lmtest). If rpy2 is not available, returns the polars_reg results
    alongside the generated R code so users can run it manually.

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

    # 2. Generate R code
    r_code = to_r(estimator, formula, vcov, cluster, entity=entity, time=time)

    # 3. Try rpy2
    r_coefs, r_se, r_names, details = _try_rpy2(r_code, data)

    # 4. Align and compare
    if r_coefs is not None and r_names is not None and r_se is not None:
        pr_c, pr_se, rc, rse, names = _align(
            polars_result.names,
            polars_result.coefficients,
            polars_result.se,
            r_names,
            r_coefs,
            r_se,
        )
        coef_rdiff = np.abs(pr_c - rc) / np.maximum(np.maximum(np.abs(pr_c), np.abs(rc)), 1e-15)
        se_denom = np.maximum(np.maximum(np.abs(pr_se), np.abs(rse)), 1e-15)
        se_rdiff = np.abs(pr_se - rse) / se_denom
        max_c = float(coef_rdiff.max())
        max_s = float(se_rdiff.max())
        match = max_c <= rtol and max_s <= rtol

        report = ComparisonReport(
            estimator=estimator,
            formula=formula,
            r_code=r_code,
            polars_coefs=pr_c,
            polars_se=pr_se,
            r_coefs=rc,
            r_se=rse,
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
            r_code=r_code,
            polars_coefs=polars_result.coefficients,
            polars_se=polars_result.se,
            r_coefs=None,
            r_se=None,
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
        func = {
            "ols": pr.ols,
            "iv2sls": pr.iv2sls,
            "liml": pr.liml,
            "gmm_iv": pr.gmm_iv,
        }
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


def _try_rpy2(
    r_code: str,
    data: pl.DataFrame,
) -> tuple[NDArray | None, NDArray | None, list[str] | None, list[str]]:
    """Try to run R code via rpy2. Returns (coefs, se, names, details)."""
    details: list[str] = []
    try:
        import rpy2.robjects as ro  # type: ignore[import-untyped]
        from rpy2.robjects import pandas2ri  # type: ignore[import-untyped]
    except ImportError:
        details.append(
            "rpy2 not available. Install rpy2 and the relevant R packages "
            "(fixest, AER, plm, sandwich, lmtest) to enable automatic comparison. "
            "You can run the R code manually."
        )
        return None, None, None, details

    try:
        pandas2ri.activate()

        # Load data into R
        pdf = data.to_pandas()
        ro.globalenv["df"] = pandas2ri.py2rpy(pdf)

        # Run the R code
        ro.r(r_code)

        # Extract coefficients and SEs
        coefs = np.array(ro.r("coef(model)"))
        names = list(ro.r("names(coef(model))"))

        # Get SEs from vcov
        se = np.array(ro.r("sqrt(diag(vcov(model)))"))

        # Clean names: "(Intercept)" -> "_cons"
        clean_names = []
        for n in names:
            if n == "(Intercept)":
                n = "_cons"
            clean_names.append(n)

        return coefs, se, clean_names, details

    except Exception as e:
        details.append(f"rpy2 execution failed: {e}")
        return None, None, None, details


def _align(
    pr_names: list[str],
    pr_coefs: NDArray,
    pr_se: NDArray,
    r_names: list[str],
    r_coefs: NDArray,
    r_se: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[str]]:
    """Align coefficients by variable name."""
    pr_map = {n: i for i, n in enumerate(pr_names)}
    r_map = {n: i for i, n in enumerate(r_names)}
    common = [n for n in pr_names if n in r_map]
    if not common:
        raise ValueError(f"No common names: polars_reg={pr_names}, R={r_names}")
    pi = [pr_map[n] for n in common]
    ri = [r_map[n] for n in common]
    return pr_coefs[pi], pr_se[pi], r_coefs[ri], r_se[ri], common
