"""Arellano-Bond and Blundell-Bond dynamic panel GMM estimators.

Arellano & Bond (1991), "Some Tests of Specification for Panel Data",
Review of Economic Studies 58(2).
Blundell & Bond (1998), "Initial Conditions and Moment Restrictions in
Dynamic Panel Data Models", Journal of Econometrics 87(1).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._ssc import SSC, _default_ssc
from polars_reg._utils import ensure_polars, sanitize_inf


def _gmm_solve(
    X: NDArray,
    y: NDArray,
    Z: NDArray,
    twostep: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """One-step or two-step GMM estimation.

    Args:
        X: Regressor matrix (n x k)
        y: Dependent variable (n,)
        Z: Instrument matrix (n x q)
        twostep: If True, compute two-step efficient GMM

    Returns:
        beta: Coefficient estimates (k,)
        resid: Residuals (n,)
        V: Variance-covariance matrix (k x k)
        A_inv: Inverse of X'Z W Z'X
        W: Weight matrix (q x q)
    """
    ZtZ = Z.T @ Z
    W = np.linalg.inv(ZtZ)
    ZtX = Z.T @ X
    Zty = Z.T @ y

    A = ZtX.T @ W @ ZtX
    b = ZtX.T @ W @ Zty
    beta = np.linalg.solve(A, b)
    resid = y - X @ beta

    if twostep:
        scores = Z * resid[:, None]
        S = scores.T @ scores
        W = np.linalg.inv(S)
        A = ZtX.T @ W @ ZtX
        b = ZtX.T @ W @ Zty
        beta = np.linalg.solve(A, b)
        resid = y - X @ beta

    # Robust VCV: sandwich
    A_inv = np.linalg.inv(A)
    scores = Z * resid[:, None]
    meat = ZtX.T @ W @ (scores.T @ scores) @ W @ ZtX
    V = A_inv @ meat @ A_inv

    return beta, resid, V, A_inv, W


def _j_test(Z: NDArray, resid: NDArray, W: NDArray, n_iv: int, k: int) -> tuple[float, float]:
    """Sargan/Hansen J test for overidentifying restrictions.

    Args:
        Z: Instrument matrix (n x q)
        resid: Residuals (n,)
        W: Weight matrix (q x q)
        n_iv: Number of instruments
        k: Number of regressors

    Returns:
        (j_stat, j_pvalue)
    """
    Zte = Z.T @ resid
    j_stat = float(Zte @ W @ Zte)
    j_df = n_iv - k
    j_pvalue = float(1.0 - stats.chi2.cdf(j_stat, j_df)) if j_df > 0 else np.nan
    return j_stat, j_pvalue


def panel_ab(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    lags: int = 2,
    maxlags: int | None = None,
    twostep: bool = False,
    ssc: SSC | None = None,
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
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
    """
    if ssc is None:
        ssc = _default_ssc()
    data = ensure_polars(data)

    spec = parse_formula(formula)
    depvar = spec.depvar
    # Filter out "0" from exog (used for no-intercept, not a real variable)
    exog = [v for v in spec.exog if v != "0"]

    # Determine columns needed and push selection into LazyFrame
    cols_needed = list(dict.fromkeys([depvar] + exog + [entity, time]))
    if isinstance(data, pl.LazyFrame):
        data = data.select(cols_needed).collect()
    df = data.select(cols_needed).sort([entity, time])

    # Sanitize inf/NaN before any computation
    df = sanitize_inf(df, cols_needed)
    df = df.drop_nulls()
    if len(df) == 0:
        raise ValueError("No observations remaining after dropping nulls/inf.")

    # Determine max available lag for instruments
    n_times = df[time].n_unique()
    if maxlags is None:
        maxlags = n_times - 1  # use all available
    maxlags = min(maxlags, n_times - 1)

    # Check for column name collisions with generated names
    generated_names = {f"L1_{depvar}", f"D_{depvar}", f"DL1_{depvar}"}
    generated_names |= {f"L{d}_{depvar}" for d in range(lags, maxlags + 1)}
    generated_names |= {f"D_{col}" for col in exog}
    collisions = generated_names & set(df.columns)
    if collisions:
        raise ValueError(
            f"Generated column names collide with existing data columns: {collisions}. "
            "Rename those columns before calling this function."
        )

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

    # GMM estimation (shared solver)
    beta, resid, V, A_inv, W = _gmm_solve(X, y, Z, twostep=twostep)

    # Sargan/Hansen J test
    j_stat, j_pvalue = _j_test(Z, resid, W, n_iv, k)

    # AR(1) and AR(2) tests (Arellano-Bond)
    # AR(m): test for m-th order serial correlation in first-differenced residuals
    entity_arr = df_clean[entity].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    ZtX = Z.T @ X
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
    result.ssc = ssc
    return result


def panel_sys_gmm(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    lags: int = 2,
    maxlags: int | None = None,
    twostep: bool = False,
    ssc: SSC | None = None,
) -> RegressionResult:
    """Blundell-Bond system GMM estimator.

    Stacks the difference equations (AB) with level equations.
    Instruments:
      - Difference eq: lagged levels y_{t-2}, ..., y_{t-maxlag}
      - Level eq: lagged differences Δy_{t-1}

    Args:
        formula: "y ~ x1 + x2" (lagged y is added automatically)
        data: Panel DataFrame with entity and time columns
        entity: Column name for entity identifier
        time: Column name for time identifier
        lags: Minimum instrument lag depth (default 2)
        maxlags: Maximum instrument lag depth (None = all available)
        twostep: If True, use two-step efficient GMM
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
    """
    if ssc is None:
        ssc = _default_ssc()
    data = ensure_polars(data)

    spec = parse_formula(formula)
    depvar = spec.depvar
    exog = [v for v in spec.exog if v != "0"]

    cols_needed = list(dict.fromkeys([depvar] + exog + [entity, time]))
    if isinstance(data, pl.LazyFrame):
        data = data.select(cols_needed).collect()
    df = data.select(cols_needed).sort([entity, time])

    # Sanitize inf/NaN before any computation
    df = sanitize_inf(df, cols_needed)
    df = df.drop_nulls()
    if len(df) == 0:
        raise ValueError("No observations remaining after dropping nulls/inf.")

    n_times = df[time].n_unique()
    if maxlags is None:
        maxlags = n_times - 1
    maxlags = min(maxlags, n_times - 1)

    # Check for column name collisions with generated names
    generated_names = {f"L1_{depvar}", f"D_{depvar}", f"DL1_{depvar}", f"DL1_iv_{depvar}"}
    generated_names |= {f"L{d}_{depvar}" for d in range(lags, maxlags + 1)}
    generated_names |= {f"D_{col}" for col in exog}
    collisions = generated_names & set(df.columns)
    if collisions:
        raise ValueError(
            f"Generated column names collide with existing data columns: {collisions}. "
            "Rename those columns before calling this function."
        )

    # Create lags, differences, and lagged differences
    lag_exprs = [
        pl.col(depvar).shift(1).over(entity).alias(f"L1_{depvar}"),
        (pl.col(depvar) - pl.col(depvar).shift(1)).over(entity).alias(f"D_{depvar}"),
        (pl.col(depvar).shift(1) - pl.col(depvar).shift(2)).over(entity).alias(f"DL1_{depvar}"),
    ]

    # Instruments for difference equation: lagged levels
    for lag_depth in range(lags, maxlags + 1):
        lag_exprs.append(
            pl.col(depvar).shift(lag_depth).over(entity).alias(f"L{lag_depth}_{depvar}")
        )

    # Instrument for level equation: lagged first difference Δy_{t-1}
    lag_exprs.append(
        (pl.col(depvar).shift(1) - pl.col(depvar).shift(2)).over(entity).alias(f"DL1_iv_{depvar}")
    )

    # First-difference exogenous vars
    for col in exog:
        lag_exprs.append((pl.col(col) - pl.col(col).shift(1)).over(entity).alias(f"D_{col}"))

    df = df.with_columns(lag_exprs)

    # --- Difference equation data ---
    iv_col_names = [f"L{d}_{depvar}" for d in range(lags, maxlags + 1)]
    diff_exog_names = [f"D_{col}" for col in exog]

    core_diff = [f"D_{depvar}", f"DL1_{depvar}", f"L{lags}_{depvar}"]
    core_diff += diff_exog_names
    df_diff = df.drop_nulls(subset=core_diff)
    for iv_name in iv_col_names:
        if iv_name not in core_diff:
            df_diff = df_diff.with_columns(pl.col(iv_name).fill_null(0.0))

    n_diff = len(df_diff)

    # --- Level equation data ---
    core_lev = [depvar, f"L1_{depvar}", f"DL1_iv_{depvar}"]
    core_lev += exog
    df_lev = df.drop_nulls(subset=core_lev)
    n_lev = len(df_lev)

    n_total = n_diff + n_lev
    if n_total == 0:
        raise ValueError("No valid observations after differencing and lagging")

    # --- Build stacked system ---

    # y vector: [Δy_diff; y_lev]
    y_diff = df_diff[f"D_{depvar}"].to_numpy().astype(np.float64)
    y_lev = df_lev[depvar].to_numpy().astype(np.float64)
    y = np.concatenate([y_diff, y_lev])

    # X matrix: [ΔL.y_diff, Δx_diff; L.y_lev, x_lev]
    names = [f"L.{depvar}"]
    x_diff_cols = [df_diff[f"DL1_{depvar}"].to_numpy().astype(np.float64)]
    x_lev_cols = [df_lev[f"L1_{depvar}"].to_numpy().astype(np.float64)]
    for col in exog:
        x_diff_cols.append(df_diff[f"D_{col}"].to_numpy().astype(np.float64))
        x_lev_cols.append(df_lev[col].to_numpy().astype(np.float64))
        names.append(col)

    X_diff = np.column_stack(x_diff_cols) if x_diff_cols else np.empty((n_diff, 0))
    X_lev = np.column_stack(x_lev_cols) if x_lev_cols else np.empty((n_lev, 0))
    X = np.vstack([X_diff, X_lev])
    k = X.shape[1]

    # Z matrix (block-diagonal): diff instruments for diff eq, level instruments for level eq
    # Diff instruments: lagged levels + differenced exogenous
    z_diff_cols = []
    for iv_name in iv_col_names:
        z_diff_cols.append(df_diff[iv_name].to_numpy().astype(np.float64))
    for col in exog:
        z_diff_cols.append(df_diff[f"D_{col}"].to_numpy().astype(np.float64))
    n_iv_diff = len(z_diff_cols)

    # Level instruments: lagged differences + exogenous levels
    z_lev_cols = [df_lev[f"DL1_iv_{depvar}"].to_numpy().astype(np.float64)]
    for col in exog:
        z_lev_cols.append(df_lev[col].to_numpy().astype(np.float64))
    n_iv_lev = len(z_lev_cols)

    n_iv = n_iv_diff + n_iv_lev

    # Build block-diagonal Z
    Z = np.zeros((n_total, n_iv))
    Z_diff_arr = np.column_stack(z_diff_cols)
    Z_lev_arr = np.column_stack(z_lev_cols)
    Z[:n_diff, :n_iv_diff] = Z_diff_arr
    Z[n_diff:, n_iv_diff:] = Z_lev_arr

    if n_iv < k:
        raise ValueError(
            f"Under-identified: {n_iv} instruments < {k} regressors. "
            "Increase maxlags or add exogenous variables."
        )

    # GMM estimation (shared solver)
    beta, resid, V, A_inv, W = _gmm_solve(X, y, Z, twostep=twostep)

    # Sargan/Hansen J test
    j_stat, j_pvalue = _j_test(Z, resid, W, n_iv, k)

    # AR tests on differenced residuals only
    entity_arr = df_diff[entity].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    resid_diff = resid[:n_diff]
    Z_diff_block = Z[:n_diff, :n_iv_diff]
    ZtX_diff = Z_diff_block.T @ X_diff
    W_diff = np.linalg.inv(Z_diff_block.T @ Z_diff_block)
    A_diff_inv = np.linalg.inv(ZtX_diff.T @ W_diff @ ZtX_diff)
    ar_args = (resid_diff, entity_arr, Z_diff_block, W_diff, ZtX_diff, A_diff_inv)
    ar1_stat, ar1_p = _ar_test(*ar_args, order=1)
    ar2_stat, ar2_p = _ar_test(*ar_args, order=2)

    # R²
    ss_res = resid @ resid
    y_dm = y - y.mean()
    ss_tot = y_dm @ y_dm
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n_total,
        k=k,
        df_r=n_total - k,
        r_squared=r2,
        r_squared_adj=r2,
        model_type="System GMM",
        vcov_type="twostep" if twostep else "onestep",
        j_stat=j_stat,
        j_pvalue=j_pvalue,
    )
    result._ar1 = (ar1_stat, ar1_p)
    result._ar2 = (ar2_stat, ar2_p)
    result._n_instruments = n_iv
    result.ssc = ssc
    return result


def _ar_test(
    resid: NDArray,
    entity_codes: NDArray,
    Z: NDArray,
    W: NDArray,
    ZtX: NDArray,
    A_inv: NDArray,
    order: int = 1,
) -> tuple[float, float]:
    """Arellano-Bond test for serial correlation of order m in Δe.

    Under H0 of no serial correlation of order m in levels e_it,
    the first-differenced residuals Δe_it should have no correlation
    of order m+1 (but do have order 1 correlation by construction).

    Note: Assumes observations within each entity are in time-sorted order
    in the residual array. The caller must ensure df was sorted by [entity, time]
    before computing residuals.

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
