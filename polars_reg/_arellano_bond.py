"""Arellano-Bond dynamic panel GMM estimator."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._utils import ensure_polars


def panel_ab(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    lags: int = 2,
    maxlags: int | None = None,
    twostep: bool = False,
) -> RegressionResult:
    """Arellano-Bond dynamic panel GMM estimator.

    Estimates y_it = rho*y_{i,t-1} + x_it'beta + alpha_i + e_it
    by first-differencing and using lagged levels as instruments.

    Uses collapsed instruments: y_{i,t-2}, ..., y_{i,t-maxlag} as
    a fixed set of instrument columns.

    Args:
        formula: "y ~ x1 + x2" (lagged y is added automatically)
        data: Panel DataFrame with entity and time columns
        entity: Column name for entity identifier
        time: Column name for time identifier
        lags: Minimum instrument lag depth (default 2)
        maxlags: Maximum instrument lag depth (None = all available)
        twostep: If True, use two-step efficient GMM
    """
    data = ensure_polars(data)
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    spec = parse_formula(formula)
    depvar = spec.depvar
    # Filter out "0" from exog (used for no-intercept, not a real variable)
    exog = [v for v in spec.exog if v != "0"]

    # Determine columns needed
    cols_needed = list(dict.fromkeys([depvar] + exog + [entity, time]))
    df = data.select(cols_needed).sort([entity, time])

    # Determine max available lag for instruments
    n_times = df[time].n_unique()
    if maxlags is None:
        maxlags = n_times - 1  # use all available
    maxlags = min(maxlags, n_times - 1)

    # Create lagged dependent variable and differences
    lag_exprs = [
        pl.col(depvar).shift(1).over(entity).alias(f"L1_{depvar}"),
        (pl.col(depvar) - pl.col(depvar).shift(1)).over(entity).alias(f"D_{depvar}"),
        (pl.col(depvar).shift(1) - pl.col(depvar).shift(2)).over(entity).alias(f"DL1_{depvar}"),
    ]

    # Instrument lags: y_{t-2}, y_{t-3}, ..., y_{t-maxlag}
    for lag_depth in range(lags, maxlags + 1):
        lag_exprs.append(
            pl.col(depvar).shift(lag_depth).over(entity).alias(f"L{lag_depth}_{depvar}")
        )

    # First-difference exogenous vars
    for col in exog:
        lag_exprs.append((pl.col(col) - pl.col(col).shift(1)).over(entity).alias(f"D_{col}"))

    df = df.with_columns(lag_exprs)

    # Determine which instrument columns exist
    iv_col_names = [f"L{d}_{depvar}" for d in range(lags, maxlags + 1)]
    diff_exog_names = [f"D_{col}" for col in exog]

    # Only require core columns to be non-null; fill deeper instrument lags with 0
    core_required = [f"D_{depvar}", f"DL1_{depvar}", f"L{lags}_{depvar}"]
    core_required += diff_exog_names
    df_clean = df.drop_nulls(subset=core_required)

    # Fill remaining instrument lags with 0 (unavailable instruments)
    for iv_name in iv_col_names:
        if iv_name not in core_required:
            df_clean = df_clean.with_columns(pl.col(iv_name).fill_null(0.0))

    n = len(df_clean)
    if n == 0:
        raise ValueError("No valid observations after differencing and lagging")

    # Extract arrays
    y = df_clean[f"D_{depvar}"].to_numpy().astype(np.float64)

    # Endogenous regressor: Δy_{t-1}
    x_cols = [df_clean[f"DL1_{depvar}"].to_numpy().astype(np.float64)]
    names = [f"L.{depvar}"]

    # Exogenous regressors (differenced)
    for col in exog:
        x_cols.append(df_clean[f"D_{col}"].to_numpy().astype(np.float64))
        names.append(col)

    X = np.column_stack(x_cols) if len(x_cols) > 0 else np.empty((n, 0))
    k = X.shape[1]

    # Instrument matrix: lagged levels + differenced exogenous
    z_cols = []
    for iv_name in iv_col_names:
        z_cols.append(df_clean[iv_name].to_numpy().astype(np.float64))
    for col in exog:
        z_cols.append(df_clean[f"D_{col}"].to_numpy().astype(np.float64))

    Z = np.column_stack(z_cols)
    n_iv = Z.shape[1]

    if n_iv < k:
        raise ValueError(
            f"Under-identified: {n_iv} instruments < {k} regressors. "
            "Increase maxlags or add exogenous variables."
        )

    # One-step GMM: W = (Z'Z)^{-1}
    ZtZ = Z.T @ Z
    W = np.linalg.inv(ZtZ)
    ZtX = Z.T @ X
    Zty = Z.T @ y

    # β = (X'Z W Z'X)^{-1} X'Z W Z'y
    A = ZtX.T @ W @ ZtX
    b = ZtX.T @ W @ Zty
    beta = np.linalg.solve(A, b)
    resid = y - X @ beta

    if twostep:
        # Two-step: W = inv(Z' ê ê' Z) using first-step residuals
        scores = Z * resid[:, None]  # (n x n_iv)
        S = scores.T @ scores  # (n_iv x n_iv)
        W = np.linalg.inv(S)
        A = ZtX.T @ W @ ZtX
        b = ZtX.T @ W @ Zty
        beta = np.linalg.solve(A, b)
        resid = y - X @ beta

    # VCV: (X'Z W Z'X)^{-1} * (X'Z W Z'ê ê'Z W Z'X) * (X'Z W Z'X)^{-1}
    # For one-step with homoskedastic W: simplifies to sigma² (X'Z W Z'X)^{-1}
    # For robust: full sandwich
    A_inv = np.linalg.inv(A)
    scores = Z * resid[:, None]
    meat = ZtX.T @ W @ (scores.T @ scores) @ W @ ZtX
    V = A_inv @ meat @ A_inv

    # Sargan/Hansen J test
    Zte = Z.T @ resid
    j_stat = float(Zte @ W @ Zte)
    j_df = n_iv - k
    j_pvalue = float(1.0 - stats.chi2.cdf(j_stat, j_df)) if j_df > 0 else np.nan

    # AR(1) and AR(2) tests (Arellano-Bond)
    # AR(m): test for m-th order serial correlation in first-differenced residuals
    entity_arr = df_clean[entity].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    ar1_stat, ar1_p = _ar_test(resid, entity_arr, Z, W, ZtX, A_inv, order=1)
    ar2_stat, ar2_p = _ar_test(resid, entity_arr, Z, W, ZtX, A_inv, order=2)

    # R² (not standard for GMM, but useful)
    ss_res = resid @ resid
    y_dm = y - y.mean()
    ss_tot = y_dm @ y_dm
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=n - k,
        r_squared=r2,
        r_squared_adj=r2,
        model_type="Arellano-Bond",
        vcov_type="twostep" if twostep else "onestep",
        j_stat=j_stat,
        j_pvalue=j_pvalue,
    )
    result._ar1 = (ar1_stat, ar1_p)
    result._ar2 = (ar2_stat, ar2_p)
    result._n_instruments = n_iv
    return result


def _ar_test(resid, entity_codes, Z, W, ZtX, A_inv, order=1):
    """Arellano-Bond test for serial correlation of order m in Δe.

    Under H0 of no serial correlation of order m in levels e_it,
    the first-differenced residuals Δe_it should have no correlation
    of order m+1 (but do have order 1 correlation by construction).

    Returns (z_stat, p_value).
    """
    n = len(resid)

    # Create lagged residuals (within entity, shifted by order)
    _, codes = np.unique(entity_codes, return_inverse=True)
    n_entities = codes.max() + 1

    # Build lagged residuals
    resid_lag = np.full(n, np.nan)
    # Sort indices by entity
    for g in range(n_entities):
        mask = codes == g
        idx = np.where(mask)[0]
        if len(idx) > order:
            resid_lag[idx[order:]] = resid[idx[:-order]]

    valid = ~np.isnan(resid_lag)
    if valid.sum() == 0:
        return np.nan, np.nan

    # AR statistic: e_{t-m}' Δe / sqrt(e_{t-m}' V_e e_{t-m})
    # Simplified: z = (Σ e_{t-m} * Δe_t) / se
    e_lag = resid_lag[valid]
    e_cur = resid[valid]

    numerator = e_lag @ e_cur

    # Variance: approximate as e_lag' (I - X(X'ZWZ'X)^{-1}X'ZWZ') Ω (same) e_lag
    # Simplified approximation: Var ≈ e_lag' diag(e²) e_lag
    var_approx = np.sum(e_lag**2 * e_cur**2)
    if var_approx <= 0:
        return np.nan, np.nan

    z_stat = numerator / np.sqrt(var_approx)
    p_value = 2.0 * stats.norm.sf(np.abs(z_stat))
    return float(z_stat), float(p_value)
