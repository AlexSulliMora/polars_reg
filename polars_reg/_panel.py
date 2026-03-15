from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons, reindex_fe_codes
from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import (
    _dk_meat,
    _hac_meat,
    vcov_clustered,
    vcov_driscoll_kraay,
    vcov_hac,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
    vcov_robust,
    vcov_wild_bootstrap,
)
from polars_reg._utils import ensure_polars, extract_arrays, sanitize_inf, validate_vcov


def panel_fe(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Panel fixed effects (within) estimator.

    Demeans by entity (and optionally time), then OLS on demeaned data.
    Default clusters SEs by entity.

    Args:
        vcov: "iid", "NW", "DK", "bootstrap", or "wildboot".
        bandwidth: Number of lags for NW/DK. Default: Newey-West rule of thumb.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    elif isinstance(cluster, list) and len(cluster) == 0:
        # Explicit empty list → iid SEs (useful for Hausman test)
        pass
    elif cluster is None:
        cluster = [entity]
    _fe_vcov = {"iid", "NW", "DK", "bootstrap", "wildboot"}
    validate_vcov(vcov, _fe_vcov, "Panel FE")
    data = ensure_polars(data)

    spec = parse_formula(formula)
    spec.fe = [entity] + ([time] if time else [])
    spec.add_intercept = False

    arrays = extract_arrays(data, spec, cluster=cluster if cluster else None, time=time)
    y, X = arrays.y, arrays.X
    fe_dict = arrays.fe_arrays

    # Remove intercept column if present
    if arrays.names and arrays.names[-1] == "_cons":
        X = X[:, :-1]
        arrays.names = arrays.names[:-1]

    keep = drop_singletons(fe_dict)
    if not keep.all():
        if keep.sum() == 0:
            raise ValueError("All observations dropped as singletons")
        y, X = y[keep], X[keep]
        fe_dict = {k: v[keep] for k, v in fe_dict.items()}
        fe_dict = reindex_fe_codes(fe_dict)
        arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}
        if arrays.time_array is not None:
            arrays.time_array = arrays.time_array[keep]

    all_vars = np.column_stack([y.reshape(-1, 1), X])
    demeaned = demean(all_vars, fe_dict)
    y_dm, X_dm = demeaned[:, 0], demeaned[:, 1:]

    n, k = X_dm.shape
    df_abs = absorbed_dof(fe_dict)

    try:
        beta = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ y_dm)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Design matrix is singular after within-transformation (perfect collinearity)."
        )
    resid = y_dm - X_dm @ beta

    ss_res = resid @ resid
    ss_tot = (y_dm - y_dm.mean()) @ (y_dm - y_dm.mean())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    n_clusters_dict = None
    if vcov == "bootstrap":
        V = vcov_pairs_bootstrap(X_dm, y_dm, n_boot=n_boot, seed=seed)
        vcov_type_str = "bootstrap"
        df_r = n - k - df_abs
    elif vcov == "wildboot":
        if not cluster:
            raise ValueError("vcov='wildboot' requires cluster= parameter")
        cl_arr = arrays.cluster_arrays[cluster[0]]
        V = vcov_wild_bootstrap(X_dm, resid, cl_arr, n_boot=n_boot, seed=seed)
        vcov_type_str = "wildboot"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        if vcov == "NW":
            V = vcov_hac(X_dm, resid, arrays.time_array, bandwidth=bandwidth)
        else:
            V = vcov_driscoll_kraay(X_dm, resid, arrays.time_array, bandwidth=bandwidth)
        vcov_type_str = vcov
        df_r = n - k - df_abs
    elif cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays_list) == 1:
            V = vcov_clustered(X_dm, resid, cluster_arrays_list[0])
        else:
            V = vcov_multiway_clustered(X_dm, resid, cluster_arrays_list)
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
        vcov_type_str = "cluster"
    else:
        V = vcov_iid(X_dm, resid, df_abs=df_abs)
        df_r = n - k - df_abs
        vcov_type_str = "iid"

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
        model_type="Panel FE",
        vcov_type=vcov_type_str,
        n_clusters=n_clusters_dict,
        fe_absorbed=list(fe_dict.keys()),
        df_absorbed=df_abs,
    )


def _group_means(arr: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Compute group means for 1D or 2D array."""
    if arr.ndim == 1:
        sums = np.bincount(codes, weights=arr, minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups)
        return sums / counts
    means = np.zeros((n_groups, arr.shape[1]))
    counts = np.bincount(codes, minlength=n_groups)
    for j in range(arr.shape[1]):
        means[:, j] = np.bincount(codes, weights=arr[:, j], minlength=n_groups) / counts
    return means


def panel_re(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Panel random effects (GLS) estimator.

    Uses Swamy-Arora method to estimate variance components, then
    performs quasi-demeaning with theta = 1 - sqrt(sigma_e^2 / (T*sigma_u^2 + sigma_e^2)).

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity (panel) identifier
        time: Column name for time identifier (required for NW/DK)
        vcov: "iid", "HC1", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs
        bandwidth: Number of lags for NW/DK. Default: Newey-West rule of thumb.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    data = ensure_polars(data)
    if isinstance(cluster, str):
        cluster = [cluster]
    _re_vcov = {"iid", "HC1", "NW", "DK", "bootstrap", "wildboot"}
    validate_vcov(vcov, _re_vcov, "Panel RE")
    spec = parse_formula(formula)

    # Drop rows with nulls in relevant columns to stay aligned with extract_arrays
    relevant_cols = [spec.depvar] + spec.exog + [entity]
    if time is not None:
        relevant_cols.append(time)
    if cluster:
        relevant_cols.extend(c for c in cluster if c not in relevant_cols)
    relevant_cols = list(dict.fromkeys(relevant_cols))
    data = data.drop_nulls(subset=relevant_cols)

    # Don't put entity in spec.fe — RE keeps the intercept and handles
    # entity codes internally for variance component estimation.
    arrays = extract_arrays(data, spec, cluster=cluster if cluster else None, time=time)
    y, X = arrays.y, arrays.X

    # Extract entity codes from the data directly
    if isinstance(data, pl.LazyFrame):
        cols = spec.exog + [spec.depvar, entity]
        if time is not None:
            cols.append(time)
        if cluster:
            cols.extend(c for c in cluster if c not in cols)
        cols = list(dict.fromkeys(cols))  # deduplicate
        data = data.select(cols).collect()
    entity_codes = (
        data[entity].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy().astype(np.int32)
    )
    n_entities = int(entity_codes.max()) + 1
    n = len(y)
    k = X.shape[1]

    # For within regression, use only non-intercept columns
    has_cons = arrays.names and arrays.names[-1] == "_cons"
    X_nocons = X[:, :-1] if has_cons else X
    k_nocons = X_nocons.shape[1]

    # Step 1: Estimate sigma_e^2 from within (FE) regression
    entity_means_y = _group_means(y, entity_codes, n_entities)
    entity_means_X = _group_means(X_nocons, entity_codes, n_entities)
    y_within = y - entity_means_y[entity_codes]
    X_within = X_nocons - entity_means_X[entity_codes]

    try:
        beta_within = np.linalg.solve(X_within.T @ X_within, X_within.T @ y_within)
    except np.linalg.LinAlgError:
        raise ValueError("Within-transformation design matrix is singular (perfect collinearity).")
    resid_within = y_within - X_within @ beta_within
    sigma_e2 = (resid_within @ resid_within) / (n - n_entities - k_nocons)

    # Step 2: Estimate sigma_u^2 from between regression
    T_i = np.bincount(entity_codes, minlength=n_entities).astype(float)
    T_bar = n / n_entities
    # Between regression uses group means including intercept
    X_bar = entity_means_X
    if has_cons:
        X_bar = np.column_stack([X_bar, np.ones(n_entities)])
    beta_between = np.linalg.lstsq(X_bar, entity_means_y, rcond=None)[0]
    resid_between = entity_means_y - X_bar @ beta_between
    sigma_b2 = (resid_between @ resid_between) / (n_entities - k)
    sigma_u2 = max(0.0, sigma_b2 - sigma_e2 / T_bar)

    # Step 3: Compute theta per entity (quasi-demeaning parameter)
    denom = T_i * sigma_u2 + sigma_e2
    theta = np.where(denom > 0, 1.0 - np.sqrt(np.maximum(sigma_e2, 0.0) / denom), 0.0)

    # Step 4: Quasi-demean (full X including intercept)
    entity_means_X_full = _group_means(X, entity_codes, n_entities)
    y_re = y - theta[entity_codes] * entity_means_y[entity_codes]
    X_re = X - theta[entity_codes, None] * entity_means_X_full[entity_codes]

    # Swamy & Arora (1972) GLS on quasi-demeaned data
    try:
        beta = np.linalg.solve(X_re.T @ X_re, X_re.T @ y_re)
    except np.linalg.LinAlgError:
        raise ValueError("GLS design matrix is singular after quasi-demeaning.")

    ss_res = (y - X @ beta) @ (y - X @ beta)
    y_dm = y - y.mean()
    ss_tot = y_dm @ y_dm
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Original residuals (for R² and output)
    resid = y - X @ beta
    # Quasi-demeaned residuals (for sandwich VCV and iid VCV)
    resid_re = y_re - X_re @ beta

    n_clusters_dict = None
    if vcov == "bootstrap":
        V = vcov_pairs_bootstrap(X_re, y_re, n_boot=n_boot, seed=seed)
        vcov_type_str = "bootstrap"
        df_r = n - k
    elif vcov == "wildboot":
        if not cluster:
            cluster = [entity]
        cl_codes = (
            entity_codes
            if cluster == [entity]
            else (
                data[cluster[0]]
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
                .astype(np.int32)
            )
        )
        V = vcov_wild_bootstrap(X_re, resid_re, cl_codes, n_boot=n_boot, seed=seed)
        vcov_type_str = "wildboot"
        n_clusters_dict = {cluster[0]: len(np.unique(cl_codes))}
        df_r = n_clusters_dict[cluster[0]] - 1
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        try:
            XtX_inv = np.linalg.inv(X_re.T @ X_re)
        except np.linalg.LinAlgError:
            raise ValueError("GLS design matrix is singular for HAC VCV.")
        score = X_re * resid_re[:, None]
        if vcov == "NW":
            S = _hac_meat(score, arrays.time_array, bandwidth)
            dfc = n / (n - k)
        else:
            S = _dk_meat(score, arrays.time_array, bandwidth)
            T_unique = len(np.unique(arrays.time_array))
            dfc = T_unique / (T_unique - 1)
        V = dfc * XtX_inv @ S @ XtX_inv
        vcov_type_str = vcov
        df_r = n - k
    elif cluster or vcov in ("HC0", "HC1"):
        if not cluster:
            cluster = [entity]
        # Need cluster codes
        cluster_arrays_list = []
        for c in cluster:
            if c == entity:
                cluster_arrays_list.append(entity_codes)
            elif arrays.cluster_arrays and c in arrays.cluster_arrays:
                cluster_arrays_list.append(arrays.cluster_arrays[c])
            else:
                cl = (
                    data[c]
                    .cast(pl.Utf8)
                    .cast(pl.Categorical)
                    .to_physical()
                    .to_numpy()
                    .astype(np.int32)
                )
                cluster_arrays_list.append(cl)

        if vcov in ("HC0", "HC1"):
            V = vcov_robust(X_re, resid_re, kind=vcov)
        elif len(cluster_arrays_list) == 1:
            V = vcov_clustered(X_re, resid_re, cluster_arrays_list[0])
        else:
            V = vcov_multiway_clustered(X_re, resid_re, cluster_arrays_list)
        if vcov in ("HC0", "HC1"):
            n_clusters_dict = None  # HC0/HC1 is robust, not clustered
            vcov_type_str = vcov
            df_r = n - k
        else:
            n_clusters_dict = {c: len(np.unique(a)) for c, a in zip(cluster, cluster_arrays_list)}
            vcov_type_str = "cluster"
            df_r = min(n_clusters_dict.values()) - 1
    else:
        V = vcov_iid(X_re, resid_re)
        vcov_type_str = "iid"
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
        model_type="Panel RE",
        vcov_type=vcov_type_str,
        n_clusters=n_clusters_dict,
    )


def panel_fd(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Panel first-difference estimator.

    Takes first differences within entity groups (sorted by time),
    then runs OLS on the differenced data.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity identifier
        time: Column name for time identifier
        vcov: "iid", "HC1", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    if isinstance(cluster, str):
        cluster = [cluster]
    if cluster is None:
        cluster = [entity]
    _fd_vcov = {"iid", "HC1", "bootstrap", "wildboot"}
    validate_vcov(vcov, _fd_vcov, "Panel FD")
    data = ensure_polars(data)

    spec = parse_formula(formula)
    # We need entity and time columns but not as FE
    all_needed = [spec.depvar] + spec.exog + [entity, time]
    if cluster:
        all_needed += [c for c in cluster if c not in all_needed]
    all_needed = list(dict.fromkeys(all_needed))

    if isinstance(data, pl.LazyFrame):
        data = data.select(all_needed).collect()

    # Sanitize inf/NaN before any computation
    data = sanitize_inf(data, all_needed)

    # Sort by entity then time, compute first differences
    df_sorted = data.select(all_needed).sort([entity, time])

    # Compute differences within entities using Polars
    diff_exprs = []
    numeric_cols = [spec.depvar] + spec.exog
    for col in numeric_cols:
        diff_exprs.append((pl.col(col) - pl.col(col).shift(1)).over(entity).alias(f"d_{col}"))

    df_diff = df_sorted.with_columns(diff_exprs)

    # Mark first observation per entity (diff is null)
    df_diff = df_diff.with_columns((pl.col(f"d_{spec.depvar}").is_not_null()).alias("_has_diff"))
    df_diff = df_diff.filter(pl.col("_has_diff"))

    # Extract differenced arrays
    n = len(df_diff)
    y_d = df_diff[f"d_{spec.depvar}"].to_numpy().astype(np.float64)

    x_cols = []
    names = []
    for col in spec.exog:
        x_cols.append(df_diff[f"d_{col}"].to_numpy().astype(np.float64))
        names.append(col)

    if spec.add_intercept:
        x_cols.append(np.ones(n, dtype=np.float64))
        names.append("_cons")

    X_d = np.column_stack(x_cols) if x_cols else np.empty((n, 0))
    k = X_d.shape[1]

    beta = np.linalg.solve(X_d.T @ X_d, X_d.T @ y_d)
    resid = y_d - X_d @ beta

    ss_res = resid @ resid
    y_dm = y_d - y_d.mean()
    ss_tot = y_dm @ y_dm
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Cluster SEs by entity
    cluster_arrays: dict[str, np.ndarray] = {}
    for c in cluster:
        raw = df_diff[c].cast(pl.Utf8).cast(pl.Categorical).to_physical()
        codes = raw.to_numpy().astype(np.int32)
        cluster_arrays[c] = codes

    n_clusters_dict = {c: len(np.unique(cluster_arrays[c])) for c in cluster}

    if vcov == "bootstrap":
        V = vcov_pairs_bootstrap(X_d, y_d, n_boot=n_boot, seed=seed)
        vcov_type_str = "bootstrap"
        df_r = n - k
    elif vcov == "wildboot":
        cl_arr = cluster_arrays[cluster[0]]
        V = vcov_wild_bootstrap(X_d, resid, cl_arr, n_boot=n_boot, seed=seed)
        vcov_type_str = "wildboot"
        df_r = min(n_clusters_dict.values()) - 1
    elif cluster:
        cluster_list = [cluster_arrays[c] for c in cluster]
        if len(cluster_list) == 1:
            V = vcov_clustered(X_d, resid, cluster_list[0])
        else:
            V = vcov_multiway_clustered(X_d, resid, cluster_list)
        vcov_type_str = "cluster"
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        V = vcov_iid(X_d, resid)
        vcov_type_str = "iid"
        df_r = n - k
    elif vcov in ("HC1", "HC0", "HC2", "HC3"):
        V = vcov_robust(X_d, resid, kind=vcov)
        vcov_type_str = vcov
        df_r = n - k
    else:
        V = vcov_iid(X_d, resid)
        vcov_type_str = "iid"
        df_r = n - k

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="Panel FD",
        vcov_type=vcov_type_str,
        n_clusters=n_clusters_dict,
    )
