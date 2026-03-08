from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons
from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import (
    vcov_clustered,
    vcov_driscoll_kraay,
    vcov_hac,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
    vcov_robust,
    vcov_wild_bootstrap,
)
from polars_reg._utils import ensure_polars, extract_arrays

try:
    from polars_reg._native import (
        rust_ols_core as _rust_ols_core,
    )
    from polars_reg._native import (
        rust_ols_from_arrays as _rust_ols_from_arrays,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _is_nested(fe_codes: np.ndarray, cluster_codes: np.ndarray) -> bool:
    """Check if FE groups are nested within cluster groups.

    An FE is nested in a cluster if every FE group maps to exactly one cluster.
    Uses vectorized approach: for each FE group, check min==max of cluster codes.
    """
    n_fe = int(fe_codes.max()) + 1
    cl_min = np.full(n_fe, np.iinfo(np.int64).max, dtype=np.int64)
    cl_max = np.full(n_fe, np.iinfo(np.int64).min, dtype=np.int64)
    cl64 = cluster_codes.astype(np.int64)
    np.minimum.at(cl_min, fe_codes, cl64)
    np.maximum.at(cl_max, fe_codes, cl64)
    return bool(np.all(cl_min == cl_max))


def _non_nested_fe_dof(
    fe_dict: dict[str, np.ndarray],
    cluster_arrays: dict[str, np.ndarray],
    cluster: list[str],
) -> int:
    """Compute absorbed DoF from FE dimensions not nested in any cluster.

    reghdfe excludes nested FE from the dfc adjustment because the cluster
    correction already accounts for those degrees of freedom.
    """
    non_nested_dof = 0
    for fe_name, fe_codes in fe_dict.items():
        nested = False
        for cl_name in cluster:
            if cl_name in cluster_arrays:
                if _is_nested(fe_codes, cluster_arrays[cl_name]):
                    nested = True
                    break
        if not nested:
            n_groups = int(fe_codes.max()) + 1
            non_nested_dof += n_groups - 1  # subtract 1 for identification
    return non_nested_dof


def _to_codes_fast(series: pl.Series) -> np.ndarray:
    """Convert Polars Series to int32 codes via Rust recode. Minimal overhead path."""
    from polars_reg._native import rust_recode

    dtype = series.dtype
    if dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    ):
        arr = series.to_numpy().astype(np.int64)
        codes, _ = rust_recode(arr)
        return np.asarray(codes)
    # String/other types: use Polars categorical encoding
    codes = series.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    return codes.astype(np.int32)


def _ols_direct_rust(
    data: pl.DataFrame | pl.LazyFrame,
    spec,
    cluster: list[str] | None,
    vcov: str,
) -> RegressionResult:
    """Ultra-fast path: extract columns and run OLS entirely in Rust.

    Skips extract_arrays, column_stack, and astype overhead.
    """
    if isinstance(data, pl.LazyFrame):
        all_cols = [spec.depvar] + list(spec.exog) + list(spec.fe)
        if cluster:
            all_cols += [c for c in cluster if c not in all_cols]
        all_cols = list(dict.fromkeys(all_cols))
        data = data.select(all_cols).collect()

    # Drop nulls on numeric columns only
    numeric_cols = [spec.depvar] + list(spec.exog)
    df = data.drop_nulls(subset=numeric_cols)

    # Extract y as f64
    y_col = df[spec.depvar].cast(pl.Float64).to_numpy()

    # Extract X columns as f64 arrays (no column_stack!)
    x_arrays = [df[c].cast(pl.Float64).to_numpy() for c in spec.exog]
    x_names = list(spec.exog)

    # Extract FE as int32 codes
    fe_arrays = [_to_codes_fast(df[c]).astype(np.int32) for c in spec.fe]
    fe_names = list(spec.fe)

    # Extract cluster as int32 codes
    cl_arrays = []
    cl_names = []
    if cluster:
        for c in cluster:
            # Reuse FE codes if same column
            if c in spec.fe:
                idx = spec.fe.index(c)
                cl_arrays.append(fe_arrays[idx])
            else:
                cl_arrays.append(_to_codes_fast(df[c]).astype(np.int32))
            cl_names.append(c)

    (
        beta,
        V,
        resid,
        r2,
        r2_adj,
        n,
        df_abs,
        n_dropped,
        cl_n_groups,
        _x_names,
        _fe_names,
        _cl_names,
    ) = _rust_ols_from_arrays(
        np.ascontiguousarray(y_col, dtype=np.float64),
        [np.ascontiguousarray(a, dtype=np.float64) for a in x_arrays],
        x_names,
        [np.ascontiguousarray(a, dtype=np.int32) for a in fe_arrays],
        fe_names,
        [np.ascontiguousarray(a, dtype=np.int32) for a in cl_arrays],
        cl_names,
        1e-8,
        100_000,
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
        sigma2 = (resid @ resid) / (n - k - df_abs)
        XtX_inv = V / ((resid @ resid) / (n - k)) if (n - k) > 0 else V
        V = sigma2 * XtX_inv
        df_r = n - k - df_abs
        vcov_type = "iid"

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=x_names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="OLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters,
        fe_absorbed=fe_names,
        df_absorbed=df_abs,
    )


def _ols_rust_path(
    arrays,
    spec,
    fe_dict: dict[str, np.ndarray],
    cluster: list[str] | None,
    vcov: str,
) -> RegressionResult:
    """Fast Rust path for OLS with FE + optional clustering."""
    X, y = arrays.X, arrays.y

    # Remove intercept (absorbed by FE)
    if spec.add_intercept and arrays.names[-1] == "_cons":
        X = X[:, :-1]
        arrays.names = arrays.names[:-1]

    fe_codes_list = [np.ascontiguousarray(v, dtype=np.int32) for v in fe_dict.values()]
    n_groups_list = [int(c.max()) + 1 for c in fe_codes_list]

    cl_codes_list = []
    if cluster:
        cl_codes_list = [
            np.ascontiguousarray(arrays.cluster_arrays[c], dtype=np.int32) for c in cluster
        ]

    beta, V, resid, r2, r2_adj, n, df_abs, n_dropped, cl_n_groups = _rust_ols_core(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(X, dtype=np.float64),
        fe_codes_list,
        n_groups_list,
        cl_codes_list,
        1e-8,  # tol
        100_000,  # max_iter
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
        # Rust computed iid VCV with sigma^2/(n-k), adjust for df_abs
        sigma2 = (resid @ resid) / (n - k - df_abs)
        XtX_inv = V / ((resid @ resid) / (n - k)) if (n - k) > 0 else V
        V = sigma2 * XtX_inv
        df_r = n - k - df_abs
        vcov_type = "iid"

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="OLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters,
        fe_absorbed=list(fe_dict.keys()),
        df_absorbed=df_abs,
    )
    result._X = np.asarray(resid)  # placeholder (demeaned X not easily recoverable)
    result._y = np.asarray(resid)  # placeholder
    return result


def ols(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    time: str | None = None,
    bandwidth: int | None = None,
    weights: str | None = None,
    fweights: str | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Ordinary Least Squares regression (or Weighted Least Squares with weights).

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2" or "y ~ x1 + x2 | fe1 + fe2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0"-"HC3", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs. Overrides vcov (except wildboot).
        time: Column name for time ordering (required for NW/DK).
        bandwidth: Number of lags for HAC/DK. Default: Newey-West rule of thumb.
        weights: Column name for analytic weights (WLS). Minimizes sum w_i*(y_i - x_i'b)^2.
        fweights: Column name for frequency weights. Each obs counts f_i times.
        n_boot: Bootstrap replications (default 999). For vcov="bootstrap"/"wildboot".
        seed: Random seed for bootstrap reproducibility.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    if weights and fweights:
        raise ValueError("Cannot specify both weights and fweights")
    data = ensure_polars(data)

    spec = parse_formula(formula)

    # --- Ultra-fast Rust path: skip extract_arrays entirely ---
    # Eligible when: FE present, no weights, no interactions/indicators,
    # simple vcov (iid or cluster), no endog/IV, and native available
    use_direct = (
        _HAS_NATIVE
        and spec.fe
        and not weights
        and not fweights
        and not spec.endog
        and not spec.indicators
        and not any(":" in c for c in spec.exog)
        and vcov not in ("bootstrap", "wildboot", "NW", "DK", "HC2", "HC3")
        and (cluster or vcov == "iid")
    )
    if use_direct:
        return _ols_direct_rust(data, spec, cluster, vcov)

    weight_col = weights or fweights
    arrays = extract_arrays(data, spec, cluster=cluster, time=time, weights=weight_col)

    X, y = arrays.X, arrays.y
    w = arrays.weights
    fe_dict = arrays.fe_arrays
    has_fe = len(fe_dict) > 0

    # --- Rust fast path via extract_arrays (for weighted/complex cases with FE) ---
    use_rust = (
        _HAS_NATIVE
        and has_fe
        and w is None
        and vcov not in ("bootstrap", "wildboot", "NW", "DK", "HC2", "HC3")
        and (cluster or vcov == "iid")
    )
    if use_rust:
        return _ols_rust_path(
            arrays,
            spec,
            fe_dict,
            cluster,
            vcov,
        )

    # Handle frequency weights: expand effective sample size
    fw = None
    if fweights and w is not None:
        fw = w.copy()
        if np.any(fw < 1) or not np.allclose(fw, np.round(fw)):
            raise ValueError("Frequency weights must be positive integers")
        fw = np.round(fw).astype(np.int64)
        # For fweights, use sqrt(f) as regression weights (equivalent to replicating obs)
        w = fw.astype(np.float64)
        w = w * len(w) / w.sum()  # normalize like aweights

    # Normalize analytic weights to sum to N (Stata aweight convention)
    if weights and w is not None:
        if np.any(w <= 0):
            raise ValueError("Weights must be strictly positive")
        w = w * len(w) / w.sum()

    if has_fe:
        # Drop singletons
        keep = drop_singletons(fe_dict)
        if not keep.all():
            y = y[keep]
            X = X[keep]
            fe_dict = {k: v[keep] for k, v in fe_dict.items()}
            if w is not None:
                w = w[keep]
            if cluster:
                arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}
            if arrays.time_array is not None:
                arrays.time_array = arrays.time_array[keep]

        # Remove intercept (absorbed by FE)
        if spec.add_intercept and arrays.names[-1] == "_cons":
            X = X[:, :-1]
            arrays.names = arrays.names[:-1]

        # Demean y and X (weighted demeaning if WLS)
        all_vars = np.column_stack([y.reshape(-1, 1), X])
        demeaned = demean(all_vars, fe_dict, weights=w)
        y = demeaned[:, 0]
        X = demeaned[:, 1:]

        df_abs = absorbed_dof(fe_dict)
        fe_absorbed = list(fe_dict.keys())
    else:
        df_abs = 0
        fe_absorbed = None

    n, k = X.shape

    # For fweights, effective N = sum(f), adjusting DoF accordingly
    n_eff = int(fw.sum()) if fw is not None else n

    # For WLS: pre-multiply by sqrt(w); OLS on transformed data = WLS
    if w is not None:
        sqw = np.sqrt(w)
        Xw = X * sqw[:, None]
        yw = y * sqw
    else:
        Xw, yw = X, y

    # Solve: beta = (Xw'Xw)^{-1} Xw'yw
    beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    resid_w = yw - Xw @ beta  # weighted residuals (for SE computation)
    resid = y - X @ beta  # unweighted residuals (for output)

    # R-squared
    if w is not None:
        ss_res = np.sum(w * resid**2)
        y_wmean = np.sum(w * y) / np.sum(w)
        ss_tot = np.sum(w * (y - y_wmean) ** 2)
    else:
        ss_res = resid @ resid
        y_demean = y - y.mean()
        ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (n_eff - 1) / (n_eff - k - df_abs)

    # Variance-covariance (uses weighted X and residuals for sandwich)
    n_clusters = None
    if cluster and vcov != "wildboot":
        cluster_arrays = [arrays.cluster_arrays[c] for c in cluster]
        # Compute non-nested FE DoF for reghdfe-style dfc adjustment
        df_a_nn = _non_nested_fe_dof(fe_dict, arrays.cluster_arrays, cluster) if has_fe else -1
        if len(cluster_arrays) == 1:
            V = vcov_clustered(Xw, resid_w, cluster_arrays[0], df_a_non_nested=df_a_nn)
        else:
            V = vcov_multiway_clustered(Xw, resid_w, cluster_arrays, df_a_non_nested=df_a_nn)
        vcov_type = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov == "bootstrap":
        V = vcov_pairs_bootstrap(Xw, yw, n_boot=n_boot, seed=seed)
        vcov_type = "bootstrap"
        df_r = n_eff - k - df_abs
    elif vcov == "wildboot":
        if not cluster:
            raise ValueError("vcov='wildboot' requires cluster= parameter")
        cl_arr = arrays.cluster_arrays[cluster[0]]
        V = vcov_wild_bootstrap(Xw, resid_w, cl_arr, n_boot=n_boot, seed=seed)
        vcov_type = "wildboot"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        if vcov == "NW":
            V = vcov_hac(Xw, resid_w, arrays.time_array, bandwidth=bandwidth)
        else:
            V = vcov_driscoll_kraay(Xw, resid_w, arrays.time_array, bandwidth=bandwidth)
        vcov_type = vcov
        df_r = n_eff - k - df_abs
    elif vcov == "iid":
        V = vcov_iid(Xw, resid_w, df_abs=df_abs)
        vcov_type = "iid"
        df_r = n_eff - k - df_abs
    else:
        V = vcov_robust(Xw, resid_w, kind=vcov)
        vcov_type = vcov
        df_r = n_eff - k - df_abs

    if fweights:
        model_type = "OLS (fweight)"
    elif w is not None:
        model_type = "WLS"
    else:
        model_type = "OLS"
    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n_eff,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type=model_type,
        vcov_type=vcov_type,
        n_clusters=n_clusters,
        fe_absorbed=fe_absorbed,
        df_absorbed=df_abs,
    )
    result._X = X
    result._y = y
    return result
