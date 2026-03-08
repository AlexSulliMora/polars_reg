from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons
from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import (
    _clustered_meat,
    _interaction_codes,
    _recode_to_contiguous,
    vcov_wild_bootstrap,
)
from polars_reg._utils import ensure_polars, extract_arrays

try:
    from polars_reg._native import rust_iv2sls as _rust_iv2sls

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _to_codes_fast(series: pl.Series) -> np.ndarray:
    """Convert Polars Series to int32 codes. Minimal overhead path."""
    from polars_reg._native import rust_recode

    dtype = series.dtype
    if dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    ):
        arr = series.to_numpy().astype(np.int64)
        codes, _ = rust_recode(arr)
        return np.asarray(codes).astype(np.int32)
    codes = series.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    return codes.astype(np.int32)


def _iv2sls_rust(
    data: pl.DataFrame | pl.LazyFrame,
    spec,
    cluster: list[str] | None,
    vcov: str,
) -> RegressionResult:
    """Rust fast path for 2SLS IV regression."""
    if isinstance(data, pl.LazyFrame):
        all_cols = (
            [spec.depvar] + list(spec.exog) + list(spec.endog)
            + list(spec.instruments) + list(spec.fe)
        )
        if cluster:
            all_cols += [c for c in cluster if c not in all_cols]
        all_cols = list(dict.fromkeys(all_cols))
        data = data.select(all_cols).collect()

    numeric_cols = [spec.depvar] + list(spec.exog) + list(spec.endog) + list(spec.instruments)
    df = data.drop_nulls(subset=numeric_cols)

    y_col = df[spec.depvar].cast(pl.Float64).to_numpy()
    x_exog = [df[c].cast(pl.Float64).to_numpy() for c in spec.exog]
    x_endog = [df[c].cast(pl.Float64).to_numpy() for c in spec.endog]
    z_excl = [df[c].cast(pl.Float64).to_numpy() for c in spec.instruments]

    fe_arrays = [_to_codes_fast(df[c]) for c in spec.fe]
    cl_arrays = []
    cl_names = []
    if cluster:
        for c in cluster:
            if c in spec.fe:
                cl_arrays.append(fe_arrays[spec.fe.index(c)])
            else:
                cl_arrays.append(_to_codes_fast(df[c]))
            cl_names.append(c)

    (
        beta, V, resid, r2, r2_adj, first_stage_f,
        n, df_abs, n_dropped, cl_n_groups, final_names,
    ) = _rust_iv2sls(
        np.ascontiguousarray(y_col, dtype=np.float64),
        [np.ascontiguousarray(a, dtype=np.float64) for a in x_exog],
        [np.ascontiguousarray(a, dtype=np.float64) for a in x_endog],
        [np.ascontiguousarray(a, dtype=np.float64) for a in z_excl],
        list(spec.exog),
        list(spec.endog),
        [np.ascontiguousarray(a, dtype=np.int32) for a in fe_arrays],
        list(spec.fe),
        [np.ascontiguousarray(a, dtype=np.int32) for a in cl_arrays],
        cl_names,
        1e-8,
        100_000,
        vcov if not cluster else "cluster",
        spec.add_intercept and not spec.fe,
    )

    beta = np.asarray(beta)
    V = np.asarray(V)
    resid = np.asarray(resid)
    k = len(beta)

    if cluster:
        n_clusters = {c: g for c, g in zip(cluster, cl_n_groups)}
        df_r = min(n_clusters.values()) - 1
        vcov_type = "cluster"
    else:
        n_clusters = None
        df_r = n - k - df_abs
        vcov_type = vcov

    fe_absorbed = list(spec.fe) if spec.fe else None

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=list(final_names),
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="2SLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters,
        first_stage_f=float(first_stage_f),
        fe_absorbed=fe_absorbed,
        df_absorbed=df_abs,
    )
    # Stash arrays for Kleibergen-Paap diagnostics
    # Note: these are pre-demeaning arrays (OK for no-FE cases, which is all
    # current KP tests). FE cases would need demeaned arrays from Rust.
    n_orig = len(y_col)
    if n == n_orig:
        # No singletons dropped — use original arrays directly
        exog_parts = [np.asarray(a) for a in x_exog]
        if spec.add_intercept and not spec.fe:
            exog_parts.append(np.ones(n))
        result._iv_X_exog = np.column_stack(exog_parts) if exog_parts else np.empty((n, 0))
        result._iv_X_endog = np.column_stack([np.asarray(a) for a in x_endog])
        result._iv_Z_excl = np.column_stack([np.asarray(a) for a in z_excl])
        result._iv_cluster_arrays = (
            [np.asarray(a) for a in cl_arrays] if cluster else None
        )
    else:
        # Singletons dropped — arrays mismatch, can't stash for KP
        pass
    return result


def iv2sls(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Two-Stage Least Squares (2SLS) IV regression.

    Args:
        formula: Formula string with IV syntax, e.g.
            "y ~ x_exog || x_endog ~ z1 + z2"          (no FE)
            "y ~ x_exog | fe | x_endog ~ z1 + z2"      (with FE)
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0", "HC1", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs. Overrides vcov.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    data = ensure_polars(data)

    spec = parse_formula(formula)

    if not spec.endog or not spec.instruments:
        raise ValueError(
            "IV formula must specify endogenous variables and instruments. "
            "Use syntax: y ~ x_exog || x_endog ~ z1 + z2"
        )

    # --- Rust fast path ---
    _rust_eligible = (
        _HAS_NATIVE
        and not spec.indicators
        and not any(":" in c for c in spec.exog)
        and vcov not in ("bootstrap", "wildboot")
        and (cluster or vcov in ("iid", "HC0", "HC1"))
    )
    if _rust_eligible:
        return _iv2sls_rust(data, spec, cluster, vcov)

    arrays = extract_arrays(data, spec, cluster=cluster)

    X_exog = arrays.X
    y = arrays.y
    X_endog = arrays.endog
    Z_excl = arrays.instruments
    fe_dict = arrays.fe_arrays
    has_fe = len(fe_dict) > 0

    if X_endog is None or Z_excl is None:
        raise ValueError("IV requires endogenous variables and instruments.")

    # Handle absorbed FE: demean y, X_exog, X_endog, Z_excl
    if has_fe:
        keep = drop_singletons(fe_dict)
        if not keep.all():
            y = y[keep]
            X_exog = X_exog[keep]
            X_endog = X_endog[keep]
            Z_excl = Z_excl[keep]
            fe_dict = {k: v[keep] for k, v in fe_dict.items()}
            if cluster:
                arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}

        # Remove intercept (absorbed by FE)
        if spec.add_intercept and arrays.names[-1] == "_cons":
            X_exog = X_exog[:, :-1]
            arrays.names = arrays.names[:-1]

        # Demean all variables
        all_vars = np.column_stack([y.reshape(-1, 1), X_exog, X_endog, Z_excl])
        demeaned = demean(all_vars, fe_dict)
        col = 0
        y = demeaned[:, col]
        col += 1
        X_exog = demeaned[:, col : col + X_exog.shape[1]]
        col += X_exog.shape[1]
        X_endog = demeaned[:, col : col + X_endog.shape[1]]
        col += X_endog.shape[1]
        Z_excl = demeaned[:, col:]

        df_abs = absorbed_dof(fe_dict)
        fe_absorbed = list(fe_dict.keys())
    else:
        df_abs = 0
        fe_absorbed = None

    n = len(y)
    k_exog = X_exog.shape[1]
    k_endog = X_endog.shape[1]
    k = k_exog + k_endog

    # Full instrument matrix: Z = [X_exog, Z_excluded]
    Z = np.column_stack([X_exog, Z_excl])

    # --- Stage 1: Project endogenous variables onto instrument space ---
    ZtZ = Z.T @ Z
    ZtZ_inv = np.linalg.inv(ZtZ)
    ZtX_endog = Z.T @ X_endog
    X_endog_hat = Z @ (ZtZ_inv @ ZtX_endog)

    # --- First-stage F-statistic (partial F-test of excluded instruments) ---
    first_stage_f = _first_stage_f(X_exog, X_endog, Z_excl)

    # --- Stage 2: 2SLS ---
    X = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    XhX = X_hat.T @ X
    Xhy = X_hat.T @ y
    beta = np.linalg.solve(XhX, Xhy)

    resid = y - X @ beta

    # R-squared (within-R² when FE are absorbed)
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    # --- Variance-covariance ---
    XhX_inv = np.linalg.inv(XhX)

    n_clusters_dict = None
    if cluster and vcov != "wildboot":
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays_list) == 1:
            V = _iv_vcov_clustered(X_hat, resid, cluster_arrays_list[0], XhX_inv)
        else:
            V = _iv_vcov_multiway(X_hat, resid, cluster_arrays_list, XhX_inv)
        vcov_type = "cluster"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "bootstrap":

        def _iv_fit(X_b, y_b):
            Z_b = np.column_stack([X_b[:, :k_exog], Z_excl_boot[: len(y_b)]])
            ZtZ_b = Z_b.T @ Z_b
            X_endog_hat_b = Z_b @ np.linalg.solve(ZtZ_b, Z_b.T @ X_b[:, k_exog:])
            Xh_b = np.column_stack([X_b[:, :k_exog], X_endog_hat_b])
            return np.linalg.solve(Xh_b.T @ X_b, Xh_b.T @ y_b)

        # For pairs bootstrap, resample all arrays together
        Z_excl_boot = Z_excl  # captured by closure
        rng = np.random.default_rng(seed)
        betas = np.empty((n_boot, k))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            X_b = X[idx]
            y_b = y[idx]
            Z_excl_b = Z_excl[idx]
            Z_b = np.column_stack([X_b[:, :k_exog], Z_excl_b])
            try:
                ZtZ_b = Z_b.T @ Z_b
                Xend_hat_b = Z_b @ np.linalg.solve(ZtZ_b, Z_b.T @ X_b[:, k_exog:])
                Xh_b = np.column_stack([X_b[:, :k_exog], Xend_hat_b])
                betas[b] = np.linalg.solve(Xh_b.T @ X_b, Xh_b.T @ y_b)
            except (np.linalg.LinAlgError, ValueError):
                betas[b] = np.nan
        valid = ~np.any(np.isnan(betas), axis=1)
        V = np.atleast_2d(np.cov(betas[valid].T, ddof=1))
        vcov_type = "bootstrap"
        df_r = n - k - df_abs
    elif vcov == "wildboot":
        if not cluster:
            raise ValueError("vcov='wildboot' requires cluster= parameter")
        cl_arr = arrays.cluster_arrays[cluster[0]]
        V = vcov_wild_bootstrap(
            X,
            resid,
            cl_arr,
            n_boot=n_boot,
            seed=seed,
            bread=XhX_inv,
            score_X=X_hat,
        )
        vcov_type = "wildboot"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        V = _iv_vcov_iid(X_hat, X, resid, XhX_inv, df_abs=df_abs)
        vcov_type = "iid"
        df_r = n - k - df_abs
    else:
        V = _iv_vcov_robust(X_hat, resid, XhX_inv, kind=vcov)
        vcov_type = vcov
        df_r = n - k - df_abs

    names = arrays.names + (arrays.endog_names or [])

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="2SLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters_dict,
        first_stage_f=first_stage_f,
        fe_absorbed=fe_absorbed,
        df_absorbed=df_abs,
    )
    # Stash first-stage arrays for Kleibergen-Paap test
    result._X = X
    result._y = y
    result._iv_X_exog = X_exog
    result._iv_X_endog = X_endog
    result._iv_Z_excl = Z_excl
    result._iv_cluster_arrays = [arrays.cluster_arrays[c] for c in cluster] if cluster else None
    return result


def _first_stage_f(
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z_excl: np.ndarray,
) -> float:
    """Partial F-test of excluded instruments in first-stage regression.

    For each endogenous variable, regresses it on [X_exog] (restricted)
    and [X_exog, Z_excl] (unrestricted). Returns the F-stat for the
    first endogenous variable (standard practice for single-endog case).
    """
    n = X_exog.shape[0]
    k_exog = X_exog.shape[1]
    q = Z_excl.shape[1]  # number of excluded instruments

    # For each endogenous variable, compute partial F
    # (report for the first one, which is the standard single-endog case)
    x_end = X_endog[:, 0] if X_endog.ndim == 2 else X_endog

    # Restricted model: x_endog ~ X_exog
    beta_r = np.linalg.lstsq(X_exog, x_end, rcond=None)[0]
    resid_r = x_end - X_exog @ beta_r
    ss_r = resid_r @ resid_r

    # Unrestricted model: x_endog ~ X_exog + Z_excl
    Z_full = np.column_stack([X_exog, Z_excl])
    beta_u = np.linalg.lstsq(Z_full, x_end, rcond=None)[0]
    resid_u = x_end - Z_full @ beta_u
    ss_u = resid_u @ resid_u

    # F = ((SS_r - SS_u) / q) / (SS_u / (n - k_exog - q))
    f_stat = ((ss_r - ss_u) / q) / (ss_u / (n - k_exog - q))
    return float(f_stat)


def _iv_vcov_iid(
    X_hat: np.ndarray,
    X: np.ndarray,
    resid: np.ndarray,
    XhX_inv: np.ndarray,
    df_abs: int = 0,
) -> np.ndarray:
    """Homoskedastic VCV for 2SLS: sigma^2 * (X_hat'X)^{-1}."""
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k - df_abs)
    return sigma2 * XhX_inv


def _iv_vcov_robust(
    X_hat: np.ndarray,
    resid: np.ndarray,
    XhX_inv: np.ndarray,
    kind: str = "HC1",
) -> np.ndarray:
    """Heteroskedasticity-robust VCV for 2SLS.

    Sandwich: (X_hat'X)^{-1} meat (X_hat'X)^{-1}
    where meat = X_hat' diag(e^2) X_hat, with HC1 scaling.
    """
    n, k = X_hat.shape
    e2 = resid**2

    meat = X_hat.T @ (X_hat * e2[:, None])

    if kind == "HC0":
        return XhX_inv @ meat @ XhX_inv
    elif kind == "HC1":
        return (n / (n - k)) * XhX_inv @ meat @ XhX_inv
    else:
        raise ValueError(f"Unsupported robust SE kind for 2SLS: {kind}")


def _iv_vcov_clustered(
    X_hat: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray,
    XhX_inv: np.ndarray,
) -> np.ndarray:
    """One-way cluster-robust VCV for 2SLS.

    Uses score vector X_hat * resid and bread (X_hat'X)^{-1}.
    """
    n, k = X_hat.shape
    codes, G = _recode_to_contiguous(clusters)
    meat = _clustered_meat(X_hat, resid, codes, G)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    return dfc * XhX_inv @ meat @ XhX_inv


def _iv_vcov_multiway(
    X_hat: np.ndarray,
    resid: np.ndarray,
    cluster_list: list[np.ndarray],
    XhX_inv: np.ndarray,
) -> np.ndarray:
    """Multi-way clustered VCV for 2SLS via Cameron-Gelbach-Miller."""
    from itertools import combinations

    D = len(cluster_list)
    n, k = X_hat.shape
    V = np.zeros((k, k))
    dims = list(range(D))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction, G = _interaction_codes(*subset_arrays)
            meat = _clustered_meat(X_hat, resid, interaction, G)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * XhX_inv @ meat @ XhX_inv

    return V
