"""Unified cross-package comparison: run the same regression in multiple backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

# ── Result dataclasses ────────────────────────────────────────────


@dataclass
class BackendResult:
    """Result from a single backend."""

    name: str
    coefs: NDArray
    se: NDArray
    names: list[str]
    n_obs: int
    r_squared: float | None
    code: str
    max_coef_rdiff: float = 0.0
    max_se_rdiff: float = 0.0
    match: bool = True


@dataclass
class ComparisonReport:
    """Multi-backend comparison report."""

    estimator: str
    formula: str
    polars_coefs: NDArray
    polars_se: NDArray
    polars_names: list[str]
    polars_n_obs: int
    polars_r_squared: float
    backends: dict[str, BackendResult] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    rtol: float = 1e-6

    def summary(self) -> str:
        """Formatted side-by-side comparison table."""
        lines: list[str] = []
        lines.append(f'compare("{self.estimator}", "{self.formula}")')
        lines.append("")

        # Column headers
        backend_names = ["polars_reg"] + list(self.backends.keys())
        col_w = max(14, max(len(n) for n in backend_names) + 2)
        name_w = max(14, max((len(n) for n in self.polars_names), default=4) + 2)
        total_w = name_w + len(backend_names) * (col_w + 1)
        sep = "=" * total_w

        lines.append(sep)
        hdr = f"{'':>{name_w}}"
        for bn in backend_names:
            hdr += f" {bn:>{col_w}}"
        lines.append(hdr)
        lines.append("-" * total_w)

        # Coefficient rows
        for i, var in enumerate(self.polars_names):
            # Coef row
            row = f"{var:<{name_w}}"
            row += f" {self.polars_coefs[i]:>{col_w}.6g}"
            for br in self.backends.values():
                idx = _find_name(br.names, var)
                if idx is not None:
                    row += f" {br.coefs[idx]:>{col_w}.6g}"
                else:
                    row += f" {'':>{col_w}}"
            lines.append(row)

            # SE row
            row = f"{'':>{name_w}}"
            row += f" {'(' + f'{self.polars_se[i]:.6g}' + ')':>{col_w}}"
            for br in self.backends.values():
                idx = _find_name(br.names, var)
                if idx is not None:
                    row += f" {'(' + f'{br.se[idx]:.6g}' + ')':>{col_w}}"
                else:
                    row += f" {'':>{col_w}}"
            lines.append(row)

        lines.append("-" * total_w)

        # Summary stats
        row = f"{'N':<{name_w}}"
        row += f" {self.polars_n_obs:>{col_w}}"
        for br in self.backends.values():
            row += f" {br.n_obs:>{col_w}}"
        lines.append(row)

        row = f"{'R²':<{name_w}}"
        row += f" {self.polars_r_squared:>{col_w}.4f}"
        for br in self.backends.values():
            r2_str = f"{br.r_squared:.4f}" if br.r_squared is not None else ""
            row += f" {r2_str:>{col_w}}"
        lines.append(row)

        lines.append("-" * total_w)

        # Diff summary
        row = f"{'Max |Δcoef|':<{name_w}}"
        row += f" {'':>{col_w}}"
        for br in self.backends.values():
            row += f" {br.max_coef_rdiff:>{col_w}.2e}"
        lines.append(row)

        row = f"{'Max |Δse|':<{name_w}}"
        row += f" {'':>{col_w}}"
        for br in self.backends.values():
            row += f" {br.max_se_rdiff:>{col_w}.2e}"
        lines.append(row)

        match_label = f"Match (rtol={self.rtol:.0e})"
        row = f"{match_label:<{name_w}}"
        row += f" {'':>{col_w}}"
        for br in self.backends.values():
            symbol = "✓" if br.match else "✗"
            row += f" {symbol:>{col_w}}"
        lines.append(row)

        lines.append(sep)

        # Skipped
        if self.skipped:
            skipped_str = ", ".join(f"{k} ({v})" for k, v in self.skipped.items())
            lines.append(f"Skipped: {skipped_str}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_match = sum(1 for br in self.backends.values() if br.match)
        n_total = len(self.backends)
        return (
            f"<ComparisonReport {self.estimator} "
            f"{n_match}/{n_total} backends match, "
            f"{len(self.skipped)} skipped>"
        )


# ── Name alignment ───────────────────────────────────────────────

_INTERCEPT_NAMES = {"_cons", "(Intercept)", "Intercept", "const"}


def _normalize_name(name: str) -> str:
    """Normalize coefficient name for cross-package matching."""
    if name in _INTERCEPT_NAMES:
        return "_cons"
    return name


def _find_name(names: list[str], target: str) -> int | None:
    """Find a coefficient name in a list, handling intercept aliases."""
    target_norm = _normalize_name(target)
    for i, n in enumerate(names):
        if _normalize_name(n) == target_norm:
            return i
    return None


def _compute_diffs(
    polars_coefs: NDArray,
    polars_se: NDArray,
    polars_names: list[str],
    backend_result: BackendResult,
    rtol: float,
) -> None:
    """Compute max relative diffs and set match flag on backend_result."""
    max_coef_rdiff = 0.0
    max_se_rdiff = 0.0
    all_match = True

    for i, var in enumerate(polars_names):
        idx = _find_name(backend_result.names, var)
        if idx is None:
            continue
        pc, ps = polars_coefs[i], polars_se[i]
        bc, bs = backend_result.coefs[idx], backend_result.se[idx]

        if abs(pc) > 1e-15:
            rdiff = abs(pc - bc) / abs(pc)
            max_coef_rdiff = max(max_coef_rdiff, rdiff)
            if rdiff > rtol:
                all_match = False
        if abs(ps) > 1e-15:
            rdiff = abs(ps - bs) / abs(ps)
            max_se_rdiff = max(max_se_rdiff, rdiff)
            if rdiff > rtol:
                all_match = False

    backend_result.max_coef_rdiff = max_coef_rdiff
    backend_result.max_se_rdiff = max_se_rdiff
    backend_result.match = all_match


# ── polars_reg runner ─────────────────────────────────────────────

_ESTIMATOR_MAP: dict[str, Any] = {}


def _get_estimator_map() -> dict[str, Any]:
    """Lazy-load estimator functions to avoid circular imports."""
    if not _ESTIMATOR_MAP:
        from polars_reg._arellano_bond import panel_ab, panel_sys_gmm
        from polars_reg._binary import logit, probit
        from polars_reg._gmm import gmm_iv, liml
        from polars_reg._iv import iv2sls
        from polars_reg._ols import ols
        from polars_reg._panel import panel_fd, panel_fe, panel_re
        from polars_reg._ppml import ppml
        from polars_reg._quantile import quantreg

        _ESTIMATOR_MAP.update(
            {
                "ols": ols,
                "iv2sls": iv2sls,
                "liml": liml,
                "gmm_iv": gmm_iv,
                "panel_fe": panel_fe,
                "panel_re": panel_re,
                "panel_fd": panel_fd,
                "panel_ab": panel_ab,
                "panel_sys_gmm": panel_sys_gmm,
                "probit": probit,
                "logit": logit,
                "ppml": ppml,
                "quantreg": quantreg,
            }
        )
    return _ESTIMATOR_MAP


def _run_polars_reg(
    estimator: str, formula: str, data: pl.DataFrame, **kwargs: Any
) -> tuple[NDArray, NDArray, list[str], int, float]:
    """Run a polars_reg estimator and extract results."""
    emap = _get_estimator_map()
    if estimator not in emap:
        raise ValueError(f"Unknown estimator: {estimator!r}. Available: {sorted(emap)}")

    fn = emap[estimator]
    result = fn(formula, data=data, **kwargs)
    return (
        result.coefficients,
        result.se,
        list(result.names),
        result.n_obs,
        result.r_squared,
    )


# ── Python backend adapters ──────────────────────────────────────


def _run_pyfixest(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> BackendResult | None:
    """Run regression in pyfixest."""
    try:
        import pyfixest as pf
    except ImportError:
        return None

    df_pd = data.to_pandas()
    spec_from = __import__("polars_reg._formula", fromlist=["parse_formula"]).parse_formula
    spec = spec_from(formula)

    # Build pyfixest formula
    pf_formula = f"{spec.depvar} ~ " + (" + ".join(spec.exog) if spec.exog else "1")
    if spec.add_intercept and spec.exog:
        pass  # pyfixest adds intercept by default
    elif not spec.add_intercept:
        pf_formula += " - 1" if spec.exog else "- 1"

    # FE part
    fe_list = list(spec.fe)
    if estimator == "panel_fe" and entity and entity not in fe_list:
        fe_list.insert(0, entity)
    if fe_list:
        pf_formula += " | " + " + ".join(fe_list)

    # IV part
    if spec.endog and spec.instruments:
        iv_part = " + ".join(spec.endog) + " ~ " + " + ".join(spec.instruments)
        if fe_list:
            pf_formula += " | " + iv_part
        else:
            pf_formula += " | " + iv_part

    # vcov mapping
    pf_vcov: Any
    if cluster:
        pf_vcov = {"CRV1": " + ".join(cluster)}
    elif vcov in ("HC1", "hetero"):
        pf_vcov = "hetero"
    elif vcov in ("HC2", "HC3"):
        pf_vcov = vcov
    elif vcov == "iid":
        pf_vcov = "iid"
    else:
        pf_vcov = "iid"

    code = f'pf.feols("{pf_formula}", data=df, vcov={pf_vcov!r})'

    try:
        if estimator in ("probit", "logit"):
            family = "probit" if estimator == "probit" else "logit"
            model = pf.feglm(pf_formula, data=df_pd, family=family, vcov=pf_vcov)
            code = f'pf.feglm("{pf_formula}", data=df, family="{family}", vcov={pf_vcov!r})'
        elif estimator == "ppml":
            model = pf.fepois(pf_formula, data=df_pd, vcov=pf_vcov)
            code = f'pf.fepois("{pf_formula}", data=df, vcov={pf_vcov!r})'
        elif estimator == "quantreg":
            tau = kwargs.get("tau", 0.5)
            model = pf.quantreg(pf_formula, data=df_pd, quantile=tau)
            code = f'pf.quantreg("{pf_formula}", data=df, quantile={tau})'
        else:
            wt = kwargs.get("weights")
            wt_kw = {"weights": wt} if wt else {}
            model = pf.feols(pf_formula, data=df_pd, vcov=pf_vcov, **wt_kw)
    except Exception:
        return None

    try:
        coefs = np.array(model.coef())
        se = np.array(model.se())
        names = list(model.coef().index)
        n_obs = int(model._N)
        r2 = float(model._r2) if hasattr(model, "_r2") else None
    except Exception:
        return None

    return BackendResult(
        name="pyfixest",
        coefs=coefs,
        se=se,
        names=names,
        n_obs=n_obs,
        r_squared=r2,
        code=code,
    )


def _run_statsmodels(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | None = None,
    **kwargs: Any,
) -> BackendResult | None:
    """Run regression in statsmodels."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    spec_from = __import__("polars_reg._formula", fromlist=["parse_formula"]).parse_formula
    spec = spec_from(formula)
    df_pd = data.to_pandas()

    # Extract arrays
    y = df_pd[spec.depvar].values.astype(float)
    x_cols = [c for c in spec.exog if c not in spec.indicators]
    X = df_pd[x_cols].values.astype(float) if x_cols else np.empty((len(y), 0))
    if spec.add_intercept:
        X = sm.add_constant(X)
        col_names = ["const"] + x_cols  # add_constant prepends
    else:
        col_names = list(x_cols)

    # vcov kwargs
    fit_kwargs: dict[str, Any] = {}
    if cluster:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": df_pd[cluster[0]].values}
    elif vcov in ("HC0", "HC1", "HC2", "HC3"):
        fit_kwargs["cov_type"] = vcov
    # else iid (default)

    code = ""
    try:
        if estimator in ("ols", "panel_fe", "panel_fd"):
            model = sm.OLS(y, X)
            result = model.fit(**fit_kwargs)
            code = f"sm.OLS(y, X).fit(cov_type={fit_kwargs.get('cov_type', 'nonrobust')!r})"
        elif estimator == "probit":
            model = sm.Probit(y, X)
            result = model.fit(disp=0, **fit_kwargs)
            code = f"sm.Probit(y, X).fit(cov_type={fit_kwargs.get('cov_type', 'nonrobust')!r})"
        elif estimator == "logit":
            model = sm.Logit(y, X)
            result = model.fit(disp=0, **fit_kwargs)
            code = f"sm.Logit(y, X).fit(cov_type={fit_kwargs.get('cov_type', 'nonrobust')!r})"
        elif estimator == "ppml":
            model = sm.GLM(y, X, family=sm.families.Poisson())
            result = model.fit(**fit_kwargs)
            code = "sm.GLM(y, X, family=Poisson()).fit()"
        elif estimator == "quantreg":
            tau = kwargs.get("tau", 0.5)
            model = sm.QuantReg(y, X)
            result = model.fit(q=tau)
            code = f"sm.QuantReg(y, X).fit(q={tau})"
        else:
            return None  # unsupported estimator
    except Exception:
        return None

    coefs = np.array(result.params)
    se = np.array(result.bse)
    n_obs = int(result.nobs)
    r2 = float(result.rsquared) if hasattr(result, "rsquared") else None

    return BackendResult(
        name="statsmodels",
        coefs=coefs,
        se=se,
        names=col_names,
        n_obs=n_obs,
        r_squared=r2,
        code=code,
    )


def _run_linearmodels(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> BackendResult | None:
    """Run regression in linearmodels."""
    try:
        import linearmodels  # noqa: F401
    except ImportError:
        return None

    spec_from = __import__("polars_reg._formula", fromlist=["parse_formula"]).parse_formula
    spec = spec_from(formula)
    df_pd = data.to_pandas()

    # vcov mapping for linearmodels
    lm_cov_type = "unadjusted"
    lm_cov_kw: dict[str, Any] = {}
    if cluster:
        lm_cov_type = "clustered"
        if entity and cluster == [entity]:
            lm_cov_kw["cluster_entity"] = True
        else:
            lm_cov_kw["clusters"] = df_pd[cluster[0]].values
    elif vcov in ("HC0", "HC1"):
        lm_cov_type = "robust"
    elif vcov == "iid":
        lm_cov_type = "unadjusted"

    code = ""
    try:
        if estimator in ("panel_fe", "panel_re", "panel_fd"):
            if not entity or not time:
                return None
            df_panel = df_pd.set_index([entity, time])

            import statsmodels.api as sm

            y = df_panel[spec.depvar]
            x_cols = [c for c in spec.exog if c not in spec.indicators]
            X = sm.add_constant(df_panel[x_cols]) if spec.add_intercept else df_panel[x_cols]

            if estimator == "panel_fe":
                from linearmodels.panel import PanelOLS

                model = PanelOLS(y, X, entity_effects=True)
                code = "PanelOLS(y, X, entity_effects=True).fit()"
            elif estimator == "panel_re":
                from linearmodels.panel import RandomEffects

                model = RandomEffects(y, X)
                code = "RandomEffects(y, X).fit()"
            elif estimator == "panel_fd":
                from linearmodels.panel import FirstDifferenceOLS

                # FD doesn't use intercept
                X_fd = df_panel[x_cols] if x_cols else df_panel[[]]
                model = FirstDifferenceOLS(y, X_fd)
                code = "FirstDifferenceOLS(y, X).fit()"

            result = model.fit(cov_type=lm_cov_type, **lm_cov_kw)

        elif estimator in ("iv2sls", "liml", "gmm_iv"):
            import statsmodels.api as sm
            from linearmodels.iv import IV2SLS, IVGMM, IVLIML

            y = df_pd[spec.depvar].values.astype(float)
            x_cols = [c for c in spec.exog if c not in spec.indicators]
            X_exog = sm.add_constant(df_pd[x_cols]) if spec.add_intercept else df_pd[x_cols]
            X_endog = df_pd[spec.endog]
            Z = df_pd[spec.instruments]

            cls_map = {"iv2sls": IV2SLS, "liml": IVLIML, "gmm_iv": IVGMM}
            cls = cls_map[estimator]
            model = cls(dependent=y, exog=X_exog, endog=X_endog, instruments=Z)
            result = model.fit(cov_type=lm_cov_type)
            code = f"{cls.__name__}(y, X_exog, X_endog, Z).fit(cov_type={lm_cov_type!r})"

        elif estimator == "ols" and spec.fe:
            # AbsorbingLS for FE absorption
            import pandas as pd
            import statsmodels.api as sm
            from linearmodels.iv.absorbing import AbsorbingLS

            y = df_pd[spec.depvar].values.astype(float)
            x_cols = [c for c in spec.exog if c not in spec.indicators]
            X = sm.add_constant(df_pd[x_cols]) if spec.add_intercept else df_pd[x_cols]
            cats = pd.DataFrame({fe: pd.Categorical(df_pd[fe]) for fe in spec.fe})
            model = AbsorbingLS(y, X, absorb=cats)
            result = model.fit(cov_type=lm_cov_type)
            code = "AbsorbingLS(y, X, absorb=cats).fit()"

        else:
            return None
    except Exception:
        return None

    coefs = np.array(result.params)
    se = np.array(result.std_errors)
    names = list(result.params.index)
    n_obs = int(result.nobs)
    r2 = float(result.rsquared) if hasattr(result, "rsquared") else None

    return BackendResult(
        name="linearmodels",
        coefs=coefs,
        se=se,
        names=names,
        n_obs=n_obs,
        r_squared=r2,
        code=code,
    )


def _run_r(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> BackendResult | None:
    """Run regression in R via Rscript subprocess."""
    try:
        from tests.r_compare import RResult, _parse_r_csv, _run_r_script, r_available, to_r_script
    except ImportError:
        return None

    if not r_available():
        return None

    import tempfile

    from polars_reg.r_equiv import to_r

    # Generate R code string for the report
    try:
        code = to_r(estimator, formula, vcov=vcov, cluster=cluster, entity=entity, time=time)
    except (ValueError, NotImplementedError):
        return None

    # Write data to CSV, generate R script, run it
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_pandas().to_csv(f, index=False)
            csv_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            results_path = f.name

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
        _run_r_script(script)
        r_result: RResult = _parse_r_csv(results_path)
    except Exception:
        return None
    finally:
        import os

        for p in [csv_path, results_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    return BackendResult(
        name="R",
        coefs=r_result.coefficients,
        se=r_result.se,
        names=r_result.names,
        n_obs=r_result.n_obs,
        r_squared=r_result.r_squared,
        code=code,
    )


def _run_stata(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str = "iid",
    cluster: list[str] | None = None,
    entity: str | None = None,
    time: str | None = None,
    **kwargs: Any,
) -> BackendResult | None:
    """Run regression in Stata via batch mode subprocess."""
    try:
        from tests.stata_compare import StataResult, stata_available
    except ImportError:
        return None

    if not stata_available():
        return None

    from polars_reg.stata import to_stata

    try:
        code = to_stata(estimator, formula, vcov=vcov, cluster=cluster, entity=entity, time=time)
    except (ValueError, NotImplementedError):
        return None

    # Use the test infrastructure to run Stata and get results
    try:
        from tests.stata_compare import _run_stata_regression

        stata_result: StataResult = _run_stata_regression(
            estimator,
            formula,
            data,
            vcov=vcov,
            cluster=cluster,
            entity=entity,
            time=time,
        )
    except Exception:
        return None

    return BackendResult(
        name="Stata",
        coefs=stata_result.coefficients,
        se=stata_result.se,
        names=stata_result.names,
        n_obs=stata_result.n_obs,
        r_squared=stata_result.r_squared,
        code=code,
    )


# ── Main compare function ────────────────────────────────────────

_ALL_BACKENDS = ["pyfixest", "statsmodels", "linearmodels", "r", "stata"]

_BACKEND_RUNNERS = {
    "pyfixest": _run_pyfixest,
    "statsmodels": _run_statsmodels,
    "linearmodels": _run_linearmodels,
    "r": _run_r,
    "stata": _run_stata,
}


def compare(
    estimator: str,
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    entity: str | None = None,
    time: str | None = None,
    backend: str | list[str] = "all",
    rtol: float = 1e-6,
    **kwargs: Any,
) -> ComparisonReport:
    """Compare a regression across multiple backends.

    Runs the same regression in polars_reg and one or more external
    packages, then compares coefficients and standard errors.

    Args:
        estimator: Estimator name (e.g., "ols", "iv2sls", "probit").
        formula: Formula string in polars_reg syntax.
        data: Polars DataFrame or LazyFrame.
        vcov: Variance-covariance type.
        cluster: Clustering variable(s).
        entity: Panel entity column (for panel estimators).
        time: Panel time column.
        backend: Backend(s) to compare against. "all" runs every available
            backend. Can also be a single name ("pyfixest") or a list
            (["pyfixest", "statsmodels"]).
        rtol: Relative tolerance for match check (default 1e-6).
        **kwargs: Additional arguments forwarded to the polars_reg estimator
            (e.g., tau= for quantreg, weights= for WLS).

    Returns:
        ComparisonReport with results from all backends that succeeded.
        Unavailable backends are listed in .skipped with a reason.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    # Run polars_reg — only pass params the estimator accepts
    _no_vcov = {"panel_ab", "panel_sys_gmm", "quantreg"}
    pr_kwargs: dict[str, Any] = {}
    if estimator not in _no_vcov:
        pr_kwargs["vcov"] = vcov
    if cluster and estimator not in _no_vcov:
        pr_kwargs["cluster"] = cluster
    if entity:
        pr_kwargs["entity"] = entity
    if time:
        pr_kwargs["time"] = time
    pr_kwargs.update(kwargs)

    polars_coefs, polars_se, polars_names, polars_n, polars_r2 = _run_polars_reg(
        estimator, formula, data, **pr_kwargs
    )

    # Determine backends to run
    if backend == "all":
        backends_to_run = list(_ALL_BACKENDS)
    elif isinstance(backend, str):
        backends_to_run = [backend]
    else:
        backends_to_run = list(backend)

    report = ComparisonReport(
        estimator=estimator,
        formula=formula,
        polars_coefs=polars_coefs,
        polars_se=polars_se,
        polars_names=polars_names,
        polars_n_obs=polars_n,
        polars_r_squared=polars_r2,
        rtol=rtol,
    )

    # Run each backend
    for bn in backends_to_run:
        if bn not in _BACKEND_RUNNERS:
            report.skipped[bn] = "unknown backend"
            continue

        runner = _BACKEND_RUNNERS[bn]
        result = runner(
            estimator=estimator,
            formula=formula,
            data=data,
            vcov=vcov,
            cluster=cluster,
            entity=entity,
            time=time,
            **kwargs,
        )

        if result is None:
            report.skipped[bn] = "not available or estimator unsupported"
        else:
            _compute_diffs(polars_coefs, polars_se, polars_names, result, rtol)
            report.backends[bn] = result

    return report
