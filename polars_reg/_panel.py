from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons
from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_multiway_clustered
from polars_reg._utils import extract_arrays


def panel_fe(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Panel fixed effects (within) estimator.

    Demeans by entity (and optionally time), then OLS on demeaned data.
    Default clusters SEs by entity.
    """
    if cluster is None:
        cluster = [entity]
    elif isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    spec.fe = [entity] + ([time] if time else [])
    spec.add_intercept = False

    arrays = extract_arrays(data, spec, cluster=cluster)
    y, X = arrays.y, arrays.X
    fe_dict = arrays.fe_arrays

    # Remove intercept column if present
    if arrays.names and arrays.names[-1] == "_cons":
        X = X[:, :-1]
        arrays.names = arrays.names[:-1]

    keep = drop_singletons(fe_dict)
    if not keep.all():
        y, X = y[keep], X[keep]
        fe_dict = {k: v[keep] for k, v in fe_dict.items()}
        arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}

    all_vars = np.column_stack([y.reshape(-1, 1), X])
    demeaned = demean(all_vars, fe_dict)
    y_dm, X_dm = demeaned[:, 0], demeaned[:, 1:]

    n, k = X_dm.shape
    df_abs = absorbed_dof(fe_dict)

    beta = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ y_dm)
    resid = y_dm - X_dm @ beta

    ss_res = resid @ resid
    ss_tot = (y_dm - y_dm.mean()) @ (y_dm - y_dm.mean())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
    if len(cluster_arrays_list) == 1:
        V = vcov_clustered(X_dm, resid, cluster_arrays_list[0])
    else:
        V = vcov_multiway_clustered(X_dm, resid, cluster_arrays_list)
    n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=min(n_clusters_dict.values()) - 1,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="Panel FE",
        vcov_type="cluster",
        n_clusters=n_clusters_dict,
        fe_absorbed=list(fe_dict.keys()),
        df_absorbed=df_abs,
    )
