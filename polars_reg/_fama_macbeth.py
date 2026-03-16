"""Fama-MacBeth (1973) two-pass cross-sectional regression."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from polars_reg._results import RegressionResult
from polars_reg._utils import ensure_polars

if TYPE_CHECKING:
    from polars_reg._group_by import GroupRegressionResult
    from polars_reg._rolling import RollingRegressionResult


@dataclass
class FamaMacBethResult:
    """Result of a Fama-MacBeth (1973) two-pass regression.

    The first pass estimates factor loadings (betas) per entity via
    time-series regressions.  The second pass runs cross-sectional
    regressions of returns on estimated betas for each period, yielding
    per-period risk premia (lambdas).  Standard errors are computed from
    the time-series variation of lambda estimates.

    References:
        Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and
        Equilibrium: Empirical Tests." *Journal of Political Economy*,
        81(3), 607-636.

        Shanken, J. (1992). "On the Estimation of Beta-Pricing Models."
        *Review of Financial Studies*, 5(1), 1-33.
    """

    # Stored fields (8 fields -- derived values are @property)
    lambdas: NDArray
    """(T, k+1) per-period risk premia (intercept LAST, ``_cons`` convention)."""
    fm_se: NDArray
    """(k+1,) Fama-MacBeth standard errors: sigma(lambda_t) / sqrt(T_eff)."""
    shanken_se: NDArray | None
    """(k+1,) Shanken-corrected SEs, or ``None`` if ``shanken=False``."""
    first_pass: GroupRegressionResult | RollingRegressionResult
    """First-pass regression results (per-entity time-series regressions)."""
    names: list[str]
    """Factor names including ``_cons`` LAST."""
    n_periods: int
    """T_eff (number of valid cross-sections used for inference)."""
    n_assets: int
    """N (number of entities)."""
    avg_r_squared: float
    """Average R-squared across valid cross-sections."""
    model_type: str = "Fama-MacBeth"

    # ── Computed properties ──────────────────────────────────────────

    @property
    def mean_lambda(self) -> NDArray:
        """Time-averaged risk premia."""
        return np.nanmean(self.lambdas, axis=0)

    @property
    def fm_tstat(self) -> NDArray:
        """Fama-MacBeth t-statistics."""
        return self.mean_lambda / self.fm_se

    @property
    def fm_pvalue(self) -> NDArray:
        """Fama-MacBeth p-values (two-sided, t-distribution)."""
        return 2 * (1 - scipy_stats.t.cdf(np.abs(self.fm_tstat), df=self.n_periods - 1))

    @property
    def shanken_tstat(self) -> NDArray | None:
        """Shanken-corrected t-statistics, or ``None`` if unavailable."""
        if self.shanken_se is None:
            return None
        return self.mean_lambda / self.shanken_se

    @property
    def shanken_pvalue(self) -> NDArray | None:
        """Shanken-corrected p-values, or ``None`` if unavailable."""
        if self.shanken_tstat is None:
            return None
        return 2 * (1 - scipy_stats.t.cdf(np.abs(self.shanken_tstat), df=self.n_periods - 1))

    # ── Duck-typing properties for regtable() compatibility ──────────

    @property
    def coefficients(self) -> NDArray:
        """Mean lambda estimates (for regtable compatibility)."""
        return self.mean_lambda

    @property
    def se(self) -> NDArray:
        """Standard errors: Shanken-corrected if available, else FM."""
        return self.shanken_se if self.shanken_se is not None else self.fm_se

    @property
    def tstat(self) -> NDArray:
        """t-statistics using the preferred SE (Shanken if available)."""
        return self.coefficients / self.se

    @property
    def pvalue(self) -> NDArray:
        """p-values using the preferred SE (Shanken if available)."""
        return 2 * (1 - scipy_stats.t.cdf(np.abs(self.tstat), df=self.n_periods - 1))

    @property
    def n_obs(self) -> int:
        """Total observations (N * T_eff)."""
        return self.n_assets * self.n_periods

    @property
    def r_squared(self) -> float:
        """Average cross-sectional R-squared."""
        return self.avg_r_squared

    @property
    def r_squared_adj(self) -> float:
        """Adjusted R-squared (same as R-squared for FM -- no natural dof adjustment)."""
        return self.avg_r_squared

    @property
    def fe_absorbed(self) -> list[str] | None:
        """No absorbed FE in Fama-MacBeth (regtable compatibility)."""
        return None

    @property
    def n_clusters(self) -> dict[str, int] | None:
        """No clustering in Fama-MacBeth (regtable compatibility)."""
        return None

    # ── Output methods ───────────────────────────────────────────────

    def summary(self, precision: int = 4) -> str:
        """Formatted summary of Fama-MacBeth regression results.

        Args:
            precision: Number of decimal places to display (default 4).

        Returns:
            Multi-line string with coefficient table and diagnostics.
        """
        w = 90
        lines = [
            "=" * w,
            "  Fama-MacBeth (1973) Two-Pass Regression",
            "=" * w,
            f"  N assets:   {self.n_assets:>6}        N periods (T_eff):  {self.n_periods:>6}",
            f"  N obs:      {self.n_obs:>6}        "
            f"Avg R-squared:      {self.avg_r_squared:.{precision}f}",
            "=" * w,
        ]

        # Build header
        if self.shanken_se is not None:
            hdr = (
                f"{'':>14} {'Mean Lam':>10} {'FM SE':>10} {'FM t':>8} {'FM p':>8}"
                f"   {'Sh SE':>10} {'Sh t':>8} {'Sh p':>8}"
            )
        else:
            hdr = f"{'':>14} {'Mean Lam':>10} {'FM SE':>10} {'FM t':>8} {'FM p':>8}"
        lines.append(hdr)
        lines.append("-" * w)

        # Coefficient rows
        mean_lam = self.mean_lambda
        fm_t = self.fm_tstat
        fm_p = self.fm_pvalue

        for i, name in enumerate(self.names):
            row = (
                f"{name:<14} {mean_lam[i]:>10.{precision}f} "
                f"{self.fm_se[i]:>10.{precision}f} "
                f"{fm_t[i]:>8.2f} "
                f"{self._fmt_p(fm_p[i]):>8}"
            )
            if self.shanken_se is not None:
                sh_t = self.shanken_tstat
                sh_p = self.shanken_pvalue
                row += (
                    f"   {self.shanken_se[i]:>10.{precision}f} "
                    f"{sh_t[i]:>8.2f} "  # type: ignore[index]
                    f"{self._fmt_p(sh_p[i]):>8}"  # type: ignore[index]
                )
            lines.append(row)

        lines.append("=" * w)

        if self.shanken_se is not None:
            lines.append(
                "  Shanken (1992) correction applied. "
                "'Sh' columns adjust for generated-regressor bias."
            )

        return "\n".join(lines)

    @staticmethod
    def _fmt_p(p: float) -> str:
        """Format a p-value."""
        if p < 0.01:
            return "<0.01"
        return f"{p:.3f}"

    def coef_table(self) -> pl.DataFrame:
        """Coefficient table as a Polars DataFrame.

        Returns:
            DataFrame with columns: ``name``, ``mean_lambda``, ``fm_se``,
            ``fm_t``, ``fm_p``, and optionally ``shanken_se``, ``shanken_t``,
            ``shanken_p``.
        """
        data: dict[str, Any] = {
            "name": self.names,
            "mean_lambda": self.mean_lambda,
            "fm_se": self.fm_se,
            "fm_t": self.fm_tstat,
            "fm_p": self.fm_pvalue,
        }
        if self.shanken_se is not None:
            data["shanken_se"] = self.shanken_se
            data["shanken_t"] = self.shanken_tstat
            data["shanken_p"] = self.shanken_pvalue
        return pl.DataFrame(data)

    def lambda_series(self) -> pl.DataFrame:
        """Per-period lambda estimates in wide format.

        Returns:
            DataFrame with columns ``[time_index, <name1>_lambda, ...]``
            where each row is one cross-sectional period.
        """
        cols: dict[str, Any] = {"time_index": np.arange(self.lambdas.shape[0])}
        for j, name in enumerate(self.names):
            cols[f"{name}_lambda"] = self.lambdas[:, j]
        return pl.DataFrame(cols)

    def plot_lambdas(
        self,
        variables: list[str] | None = None,
        alpha: float = 0.05,
    ) -> Any:
        """Altair chart of per-period risk premia with confidence bands.

        Plots the time series of cross-sectional lambda estimates with
        mean +/- critical-value * SE bands, analogous to
        ``RollingRegressionResult.plot_coefs()``.

        Args:
            variables: Subset of factor names to plot. If ``None``,
                all factors (excluding ``_cons``) are plotted.
            alpha: Significance level for confidence bands (default 0.05).

        Returns:
            An Altair ``LayerChart`` object.

        Raises:
            ImportError: If ``altair`` is not installed.
        """
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "altair is required for plot_lambdas(). Install it with: pip install altair"
            )

        from scipy import stats as scipy_stats

        z = scipy_stats.norm.ppf(1 - alpha / 2)

        rows = []
        for t in range(self.lambdas.shape[0]):
            for j, name in enumerate(self.names):
                val = self.lambdas[t, j]
                se_j = self.fm_se[j]
                rows.append(
                    {
                        "time": t,
                        "variable": name,
                        "lambda": float(val),
                        "ci_lower": float(self.mean_lambda[j] - z * se_j),
                        "ci_upper": float(self.mean_lambda[j] + z * se_j),
                        "mean": float(self.mean_lambda[j]),
                    }
                )

        df = pl.DataFrame(rows)

        if variables is None:
            # Exclude _cons by default
            variables = [n for n in self.names if n != "_cons"]
        df = df.filter(pl.col("variable").is_in(variables))

        df_pd = df.to_pandas()

        scatter = (
            alt.Chart(df_pd)
            .mark_point(opacity=0.3, size=10)
            .encode(
                x=alt.X("time:Q", title="Period"),
                y=alt.Y("lambda:Q", title="Risk Premium (λ)"),
                color="variable:N",
            )
        )
        mean_line = (
            alt.Chart(df_pd)
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y="mean:Q",
                color="variable:N",
            )
        )
        band = (
            alt.Chart(df_pd)
            .mark_area(opacity=0.15)
            .encode(
                x="time:Q",
                y="ci_lower:Q",
                y2="ci_upper:Q",
                color="variable:N",
            )
        )

        return band + scatter + mean_line

    def __repr__(self) -> str:
        return (
            f"<FamaMacBethResult N={self.n_assets} T={self.n_periods} "
            f"k={len(self.names)} avg_R2={self.avg_r_squared:.4f}>"
        )


# ═══════════════════════════════════════════════════════════════════════
# Public function
# ═══════════════════════════════════════════════════════════════════════


def fama_macbeth(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    estimator_fn: Callable[..., RegressionResult] | None = None,
    window: int | None = None,
    stride: int = 1,
    shanken: bool = True,
    **kwargs: Any,
) -> FamaMacBethResult:
    """Fama-MacBeth (1973) two-pass cross-sectional regression.

    Estimates risk premia (lambdas) via two-pass procedure:

    1. **First pass (time-series):** For each entity, regress returns on
       factors to estimate factor loadings (betas).
    2. **Second pass (cross-sectional):** For each period, regress
       entity returns on estimated betas to estimate risk premia.

    Standard errors are computed from the time-series variation of lambda
    estimates.  Optionally applies the Shanken (1992) correction for
    generated-regressor bias.

    References:
        Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and
        Equilibrium: Empirical Tests." *Journal of Political Economy*,
        81(3), 607-636.

        Shanken, J. (1992). "On the Estimation of Beta-Pricing Models."
        *Review of Financial Studies*, 5(1), 1-33.

        Cochrane, J.H. (2005). *Asset Pricing*, Revised Edition.
        Princeton University Press. Chapter 12.

    Args:
        formula: Regression formula for the first-pass time-series
            regression (e.g. ``"ret ~ mkt + smb + hml"``).
        data: Panel data as a Polars DataFrame or LazyFrame with entity
            and time identifiers.
        entity: Column name identifying cross-sectional entities
            (e.g. ``"stock_id"``).
        time: Column name identifying time periods (e.g. ``"month"``).
        estimator_fn: Estimator for the first-pass regressions.  Any
            polars_reg estimator that returns ``RegressionResult``
            (default: ``ols``).
        window: If provided, use rolling-window first-pass regressions
            with this window size.  When ``None`` (default), full-sample
            first-pass regressions are used.
        stride: Step size for rolling windows (default 1).  Ignored
            when ``window is None``.
        shanken: Whether to compute Shanken (1992) corrected standard
            errors (default ``True``).
        **kwargs: Additional arguments passed to the first-pass estimator
            (``vcov``, ``cluster``, etc.).

    Returns:
        FamaMacBethResult with lambda estimates, standard errors, and
        access to first-pass results.

    Raises:
        ValueError: If fewer than 2 entities have successful first-pass
            regressions, or fewer than 2 valid cross-sections are
            available for inference.
        TypeError: If data is not a Polars DataFrame or LazyFrame.

    Example:
        >>> import polars as pl
        >>> import polars_reg as pr
        >>> # Three-factor model
        >>> result = pr.fama_macbeth(
        ...     "ret ~ mkt + smb + hml",
        ...     data=panel_df,
        ...     entity="stock_id",
        ...     time="month",
        ... )
        >>> print(result.summary())
        >>> result.coef_table()  # Polars DataFrame
    """
    data = ensure_polars(data)

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    if entity not in data.columns:
        raise ValueError(f"Entity column '{entity}' not found in data")
    if time not in data.columns:
        raise ValueError(f"Time column '{time}' not found in data")

    # Default estimator
    if estimator_fn is None:
        from polars_reg._ols import ols

        estimator_fn = ols

    # Parse formula to identify factor columns
    from polars_reg._formula import parse_formula

    spec = parse_formula(formula)
    depvar = spec.depvar
    # Factor columns = exogenous regressors (slope coefficients only, no intercept)
    factor_cols = list(spec.exog)

    if not factor_cols:
        raise ValueError("Formula must include at least one regressor (factor).")

    # ── First pass ───────────────────────────────────────────────────
    if window is None:
        from polars_reg._group_by import group_by_reg

        first_pass_result: GroupRegressionResult | RollingRegressionResult = group_by_reg(
            estimator_fn, formula, data, group_by=entity, **kwargs
        )
        return _fm_from_group(
            first_pass_result,
            data,
            depvar,
            factor_cols,
            entity,
            time,
            shanken,
        )
    else:
        from polars_reg._rolling import rolling_reg

        first_pass_result = rolling_reg(
            estimator_fn,
            formula,
            data,
            time=time,
            window=window,
            stride=stride,
            group_by=entity,
            **kwargs,
        )
        return _fm_from_rolling(
            first_pass_result,
            data,
            depvar,
            factor_cols,
            entity,
            time,
            shanken,
        )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _fm_from_group(
    first_pass: GroupRegressionResult,
    data: pl.DataFrame,
    depvar: str,
    factor_cols: list[str],
    entity: str,
    time: str,
    shanken: bool,
) -> FamaMacBethResult:
    """Second pass for full-sample (GroupBy) first pass."""
    # Collect entities with successful first-pass results
    entities = []
    beta_rows = []

    for ent_key, result in first_pass.results.items():
        # Extract slope coefficients (exclude _cons)
        slope_betas = []
        for fc in factor_cols:
            if fc in result.names:
                idx = result.names.index(fc)
                slope_betas.append(result.coefficients[idx])
            else:
                # Factor not in this entity's result -- skip entity
                break
        else:
            entities.append(ent_key)
            beta_rows.append(slope_betas)

    if len(entities) < 2:
        raise ValueError(
            f"Need at least 2 entities with successful first-pass regressions, got {len(entities)}"
        )

    N = len(entities)
    k = len(factor_cols)
    B = np.array(beta_rows, dtype=np.float64)  # (N, k)

    # Build cross-sectional design matrix: [betas, ones] -- intercept LAST
    X_cs = np.column_stack([B, np.ones(N)])  # (N, k+1)

    # Get unique sorted time periods
    time_periods = data.select(time).unique().sort(time).to_series().to_list()
    T_total = len(time_periods)

    # Build return matrix Y: (T, N)
    # Y[t, i] = return of entity i at time t
    Y = np.full((T_total, N), np.nan)

    # Create a lookup: entity -> column index
    entity_to_idx = {e: i for i, e in enumerate(entities)}

    # Pivot the data efficiently
    for t_idx, t_val in enumerate(time_periods):
        period_data = data.filter(pl.col(time) == t_val)
        ent_vals = period_data[entity].to_list()
        ret_vals = period_data[depvar].to_numpy().astype(np.float64)
        for j, ev in enumerate(ent_vals):
            if ev in entity_to_idx:
                Y[t_idx, entity_to_idx[ev]] = ret_vals[j]

    # Vectorized cross-sectional OLS for all periods at once
    # lambdas[t] = (X'X)^{-1} X' Y[t]'
    XtX = X_cs.T @ X_cs
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Cross-sectional design matrix is singular. "
            "Too few assets or perfectly collinear betas."
        )

    # For each period, we need to handle missing entities (NaN in Y)
    # Fall back to per-period when there are NaNs
    lambdas = np.full((T_total, k + 1), np.nan)
    r_squared_arr = np.full(T_total, np.nan)

    # Check if panel is balanced (no NaNs in Y)
    has_nans = np.any(np.isnan(Y))

    if not has_nans:
        # Fully vectorized path: all periods at once
        # lambdas = (XtX_inv @ X_cs.T @ Y.T).T  =>  (T, k+1)
        lambdas = (XtX_inv @ X_cs.T @ Y.T).T

        # Per-period R-squared
        fitted = lambdas @ X_cs.T  # (T, N)
        residuals = Y - fitted
        ss_res = (residuals**2).sum(axis=1)
        y_centered = Y - Y.mean(axis=1, keepdims=True)
        ss_tot = (y_centered**2).sum(axis=1)
        # Avoid division by zero
        nonzero = ss_tot > 0
        r_squared_arr[nonzero] = 1 - ss_res[nonzero] / ss_tot[nonzero]
    else:
        # Per-period path handling missing data
        for t_idx in range(T_total):
            y_t = Y[t_idx]
            valid = np.isfinite(y_t)
            n_valid = valid.sum()
            if n_valid < k + 2:
                # Not enough assets this period
                continue
            y_valid = y_t[valid]
            X_valid = X_cs[valid]
            try:
                lam_t = np.linalg.solve(X_valid.T @ X_valid, X_valid.T @ y_valid)
                lambdas[t_idx] = lam_t
                # R-squared
                fitted_t = X_valid @ lam_t
                resid_t = y_valid - fitted_t
                ss_res_t = (resid_t**2).sum()
                y_bar = y_valid.mean()
                ss_tot_t = ((y_valid - y_bar) ** 2).sum()
                if ss_tot_t > 0:
                    r_squared_arr[t_idx] = 1 - ss_res_t / ss_tot_t
            except np.linalg.LinAlgError:
                continue

    # FM inference on valid periods
    valid_mask = np.all(np.isfinite(lambdas), axis=1)
    valid_lambdas = lambdas[valid_mask]
    T_eff = len(valid_lambdas)

    if T_eff < 2:
        raise ValueError(f"Need at least 2 valid cross-sections for FM inference, got {T_eff}")

    mean_lambda = valid_lambdas.mean(axis=0)
    # ddof=1: finite-sample correction for time-series variance of lambda estimates
    fm_se = valid_lambdas.std(axis=0, ddof=1) / np.sqrt(T_eff)

    # Shanken correction
    shanken_se = None
    if shanken:
        shanken_se = _shanken_correction(
            mean_lambda, fm_se, data, factor_cols, time, time_periods, k, T_eff
        )

    # Second-pass names: factor names + "_cons" LAST
    second_pass_names = list(factor_cols) + ["_cons"]

    avg_r2 = float(np.nanmean(r_squared_arr[valid_mask]))

    return FamaMacBethResult(
        lambdas=lambdas,
        fm_se=fm_se,
        shanken_se=shanken_se,
        first_pass=first_pass,
        names=second_pass_names,
        n_periods=T_eff,
        n_assets=N,
        avg_r_squared=avg_r2,
    )


def _fm_from_rolling(
    first_pass: RollingRegressionResult,
    data: pl.DataFrame,
    depvar: str,
    factor_cols: list[str],
    entity: str,
    time: str,
    shanken: bool,
) -> FamaMacBethResult:
    """Second pass for rolling-window first pass.

    For each period t, use betas from the most recent window ending at
    or before t-1 for each entity.  This ensures no look-ahead bias.
    """
    # Split rolling results by entity
    entity_results = first_pass.by_entity()

    # Identify all entities that have at least one successful window
    valid_entities = [e for e, rr in entity_results.items() if len(rr.results) > 0]
    if len(valid_entities) < 2:
        raise ValueError(
            f"Need at least 2 entities with successful rolling regressions, "
            f"got {len(valid_entities)}"
        )

    k = len(factor_cols)

    # For each entity, build a mapping: window_end -> slope betas
    entity_betas: dict[Any, list[tuple[Any, NDArray]]] = {}
    for ent in valid_entities:
        rr = entity_results[ent]
        beta_list: list[tuple[Any, NDArray]] = []
        for window_end, result in rr.results.items():
            slope_betas = []
            for fc in factor_cols:
                if fc in result.names:
                    idx = result.names.index(fc)
                    slope_betas.append(result.coefficients[idx])
                else:
                    break
            else:
                beta_list.append((window_end, np.array(slope_betas)))
        if beta_list:
            # Sort by window_end
            beta_list.sort(key=lambda x: x[0])
            entity_betas[ent] = beta_list

    # Filter to entities that have betas
    valid_entities = [e for e in valid_entities if e in entity_betas]
    if len(valid_entities) < 2:
        raise ValueError(
            f"Need at least 2 entities with extractable betas, got {len(valid_entities)}"
        )

    N = len(valid_entities)
    entity_to_idx = {e: i for i, e in enumerate(valid_entities)}

    # Get unique sorted time periods
    time_periods = data.select(time).unique().sort(time).to_series().to_list()
    T_total = len(time_periods)

    # Build return matrix Y: (T, N)
    Y = np.full((T_total, N), np.nan)
    for t_idx, t_val in enumerate(time_periods):
        period_data = data.filter(pl.col(time) == t_val)
        ent_vals = period_data[entity].to_list()
        ret_vals = period_data[depvar].to_numpy().astype(np.float64)
        for j, ev in enumerate(ent_vals):
            if ev in entity_to_idx:
                Y[t_idx, entity_to_idx[ev]] = ret_vals[j]

    # Per-period cross-sectional OLS with time-varying betas
    # For each period t, find the most recent window ending STRICTLY before t
    # for each entity, and use those betas.
    lambdas = np.full((T_total, k + 1), np.nan)
    r_squared_arr = np.full(T_total, np.nan)

    for t_idx, t_val in enumerate(time_periods):
        # Build beta matrix for this period (per entity, most recent window
        # ending before t_val -- no look-ahead)
        B_t = np.full((N, k), np.nan)
        for ent in valid_entities:
            i = entity_to_idx[ent]
            beta_list = entity_betas[ent]
            # Find most recent window_end < t_val
            best_beta = None
            for w_end, betas in beta_list:
                if w_end < t_val:
                    best_beta = betas
                else:
                    break
            if best_beta is not None:
                B_t[i] = best_beta

        # Check which entities have both a return and valid betas this period
        y_t = Y[t_idx]
        has_return = np.isfinite(y_t)
        has_beta = np.all(np.isfinite(B_t), axis=1)
        usable = has_return & has_beta
        n_usable = usable.sum()

        if n_usable < k + 2:
            continue

        y_valid = y_t[usable]
        X_valid = np.column_stack([B_t[usable], np.ones(n_usable)])  # intercept LAST

        try:
            lam_t = np.linalg.solve(X_valid.T @ X_valid, X_valid.T @ y_valid)
            lambdas[t_idx] = lam_t
            # R-squared
            fitted_t = X_valid @ lam_t
            resid_t = y_valid - fitted_t
            ss_res_t = (resid_t**2).sum()
            y_bar = y_valid.mean()
            ss_tot_t = ((y_valid - y_bar) ** 2).sum()
            if ss_tot_t > 0:
                r_squared_arr[t_idx] = 1 - ss_res_t / ss_tot_t
        except np.linalg.LinAlgError:
            continue

    # FM inference on valid periods
    valid_mask = np.all(np.isfinite(lambdas), axis=1)
    valid_lambdas = lambdas[valid_mask]
    T_eff = len(valid_lambdas)

    if T_eff < 2:
        raise ValueError(f"Need at least 2 valid cross-sections for FM inference, got {T_eff}")

    mean_lambda = valid_lambdas.mean(axis=0)
    fm_se = valid_lambdas.std(axis=0, ddof=1) / np.sqrt(T_eff)

    # Shanken correction
    shanken_se = None
    if shanken:
        shanken_se = _shanken_correction(
            mean_lambda, fm_se, data, factor_cols, time, time_periods, k, T_eff
        )

    second_pass_names = list(factor_cols) + ["_cons"]

    avg_r2 = float(np.nanmean(r_squared_arr[valid_mask]))

    return FamaMacBethResult(
        lambdas=lambdas,
        fm_se=fm_se,
        shanken_se=shanken_se,
        first_pass=first_pass,
        names=second_pass_names,
        n_periods=T_eff,
        n_assets=N,
        avg_r_squared=avg_r2,
    )


def _shanken_correction(
    mean_lambda: NDArray,
    fm_se: NDArray,
    data: pl.DataFrame,
    factor_cols: list[str],
    time: str,
    time_periods: list[Any],
    k: int,
    T_eff: int,
) -> NDArray | None:
    """Compute Shanken (1992) corrected standard errors.

    The correction adjusts for generated-regressor bias arising from
    using estimated betas in the second-pass cross-sectional regressions.

    Implements Cochrane (2005), eq. 12.19-12.21.
    """
    # Factor covariance matrix (MLE: 1/T, NOT np.cov which uses 1/(T-1))
    # MLE is the correct estimator for the Shanken correction formula
    # because the correction factor c is defined using the population
    # covariance -- Cochrane (2005), eq. 12.19.
    factor_data = np.full((len(time_periods), k), np.nan)

    # Average factor values per period
    for t_idx, t_val in enumerate(time_periods):
        period = data.filter(pl.col(time) == t_val)
        for j, fc in enumerate(factor_cols):
            if fc in period.columns:
                vals = period[fc].drop_nulls().to_numpy().astype(np.float64)
                if len(vals) > 0:
                    factor_data[t_idx, j] = vals.mean()

    # Use only periods with valid factor data
    valid_factor = np.all(np.isfinite(factor_data), axis=1)
    factor_valid = factor_data[valid_factor]

    if len(factor_valid) < 2:
        warnings.warn(
            "Too few periods with valid factor data for Shanken correction.",
            stacklevel=3,
        )
        return None

    f_bar = factor_valid.mean(axis=0)
    centered = factor_valid - f_bar
    Sigma_f = centered.T @ centered / len(factor_valid)  # MLE estimator

    # Shanken correction factor -- Cochrane (2005), eq. 12.21
    # _cons is LAST, so slopes are [:-1]
    lambda_slopes = mean_lambda[:-1]

    try:
        c = float(lambda_slopes @ np.linalg.solve(Sigma_f, lambda_slopes))
    except np.linalg.LinAlgError:
        warnings.warn(
            "Factor covariance matrix is singular. Shanken correction unavailable.",
            stacklevel=3,
        )
        return None

    # Full Shanken variance -- Cochrane (2005), eq. 12.20
    # Var_shanken = (1 + c) * Var_FM + Sigma_f_aug / T_eff
    Sigma_f_aug = np.zeros((k + 1, k + 1))
    Sigma_f_aug[:-1, :-1] = Sigma_f  # intercept row/col stays zero

    # fm_se are already SE = std/sqrt(T), so fm_se**2 = var of mean
    shanken_cov = (1 + c) * np.diag(fm_se**2) * T_eff + Sigma_f_aug
    shanken_cov = shanken_cov / T_eff
    shanken_se = np.sqrt(np.diag(shanken_cov))

    return shanken_se
