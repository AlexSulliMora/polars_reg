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
    vcov_robust,
)
from polars_reg._utils import ensure_polars, extract_arrays


def _is_nested(fe_codes: np.ndarray, cluster_codes: np.ndarray) -> bool:
    """Check if FE groups are nested within cluster groups.

    An FE is nested in a cluster if every FE group maps to exactly one cluster.
    """
    # For each FE group, check if all observations map to the same cluster
    unique_fe = np.unique(fe_codes)
    for g in unique_fe:
        mask = fe_codes == g
        if len(np.unique(cluster_codes[mask])) > 1:
            return False
    return True


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


def ols(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    time: str | None = None,
    bandwidth: int | None = None,
) -> RegressionResult:
    """Ordinary Least Squares regression.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2" or "y ~ x1 + x2 | fe1 + fe2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0", "HC1", "HC2", "HC3", "NW" (Newey-West), or "DK" (Driscoll-Kraay)
        cluster: Column name(s) for clustered SEs. Overrides vcov.
        time: Column name for time ordering (required for NW/DK).
        bandwidth: Number of lags for HAC/DK. Default: Newey-West rule of thumb.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    data = ensure_polars(data)

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster, time=time)

    X, y = arrays.X, arrays.y
    fe_dict = arrays.fe_arrays
    has_fe = len(fe_dict) > 0

    if has_fe:
        # Drop singletons
        keep = drop_singletons(fe_dict)
        if not keep.all():
            y = y[keep]
            X = X[keep]
            fe_dict = {k: v[keep] for k, v in fe_dict.items()}
            if cluster:
                arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}
            if arrays.time_array is not None:
                arrays.time_array = arrays.time_array[keep]

        # Remove intercept (absorbed by FE)
        if spec.add_intercept and arrays.names[-1] == "_cons":
            X = X[:, :-1]
            arrays.names = arrays.names[:-1]

        # Demean y and X
        all_vars = np.column_stack([y.reshape(-1, 1), X])
        demeaned = demean(all_vars, fe_dict)
        y = demeaned[:, 0]
        X = demeaned[:, 1:]

        df_abs = absorbed_dof(fe_dict)
        fe_absorbed = list(fe_dict.keys())
    else:
        df_abs = 0
        fe_absorbed = None

    n, k = X.shape

    # Solve OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta

    # R-squared (within-R² when FE are absorbed)
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    # Variance-covariance
    if cluster:
        cluster_arrays = [arrays.cluster_arrays[c] for c in cluster]
        # Compute non-nested FE DoF for reghdfe-style dfc adjustment
        df_a_nn = _non_nested_fe_dof(fe_dict, arrays.cluster_arrays, cluster) if has_fe else -1
        if len(cluster_arrays) == 1:
            V = vcov_clustered(X, resid, cluster_arrays[0], df_a_non_nested=df_a_nn)
        else:
            V = vcov_multiway_clustered(X, resid, cluster_arrays, df_a_non_nested=df_a_nn)
        vcov_type = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        if vcov == "NW":
            V = vcov_hac(X, resid, arrays.time_array, bandwidth=bandwidth)
        else:
            V = vcov_driscoll_kraay(X, resid, arrays.time_array, bandwidth=bandwidth)
        vcov_type = vcov
        n_clusters = None
        df_r = n - k - df_abs
    elif vcov == "iid":
        V = vcov_iid(X, resid, df_abs=df_abs)
        vcov_type = "iid"
        n_clusters = None
        df_r = n - k - df_abs
    else:
        V = vcov_robust(X, resid, kind=vcov)
        vcov_type = vcov
        n_clusters = None
        df_r = n - k - df_abs

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
        fe_absorbed=fe_absorbed,
        df_absorbed=df_abs,
    )
    result._X = X
    result._y = y
    return result
