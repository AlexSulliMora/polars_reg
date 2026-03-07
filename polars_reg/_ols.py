from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_iid, vcov_multiway_clustered, vcov_robust
from polars_reg._utils import extract_arrays


def ols(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Ordinary Least Squares regression.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0", "HC1", "HC2", or "HC3"
        cluster: Column name(s) for clustered SEs. Overrides vcov.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    X, y = arrays.X, arrays.y
    n, k = X.shape

    # Solve OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta

    # R-squared
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Variance-covariance
    if cluster:
        cluster_arrays = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays) == 1:
            V = vcov_clustered(X, resid, cluster_arrays[0])
        else:
            V = vcov_multiway_clustered(X, resid, cluster_arrays)
        vcov_type = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov == "iid":
        V = vcov_iid(X, resid)
        vcov_type = "iid"
        n_clusters = None
        df_r = n - k
    else:
        V = vcov_robust(X, resid, kind=vcov)
        vcov_type = vcov
        n_clusters = None
        df_r = n - k

    return RegressionResult(
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
    )
