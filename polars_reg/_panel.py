from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._ols import ols
from polars_reg._results import RegressionResult
from polars_reg._se import compute_vcov
from polars_reg._ssc import SSC, _default_ssc
from polars_reg._utils import ensure_polars, extract_arrays, sanitize_inf, validate_vcov


def _inject_fe(formula: str, entity: str, time: str | None = None) -> str:
    """Inject entity (and optionally time) as FE into a formula string.

    'y ~ x1 + x2' + entity='firm' -> 'y ~ x1 + x2 | firm'
    'y ~ x1 + x2' + entity='firm', time='year' -> 'y ~ x1 + x2 | firm + year'

    Raises ValueError if formula already contains '|' (FE already specified).
    """
    if "|" in formula:
        raise ValueError(
            "Formula already contains fixed effects ('|'). "
            "Use ols() directly instead of panel_fe() when specifying FE in the formula."
        )
    fe_part = entity if time is None else f"{entity} + {time}"
    return f"{formula} | {fe_part}"


def panel_fe(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
    weights: str | None = None,
    fweights: str | None = None,
    time_col: str | None = None,
) -> RegressionResult:
    """Panel fixed effects (within) estimator.

    Sugar for ols() with absorbed entity (and optionally time) fixed effects.
    Default clusters SEs by entity. See Wooldridge (2010), ch. 10.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity (panel) identifier
        time: Column name for time identifier (optional, adds time FE)
        vcov: "iid", "HC0"-"HC3", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs. Default: [entity].
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
        bandwidth: Number of lags for NW/DK. Default: Newey-West rule of thumb.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
        weights: Column name for analytic weights (WLS).
        fweights: Column name for frequency weights.
        time_col: Column name for time ordering (for NW/DK). Defaults to time if provided.
    """
    # Handle cluster defaults.
    # In ols(), cluster overrides vcov (except wildboot), so we only default
    # cluster=[entity] when the vcov type won't conflict with clustering.
    # NW/DK/bootstrap need cluster=None so ols() uses the requested vcov.
    _no_default_cluster = {"NW", "DK", "bootstrap"}
    if isinstance(cluster, str):
        cluster = [cluster]
    elif isinstance(cluster, list) and len(cluster) == 0:
        cluster = None  # Explicit empty list -> iid SEs (Hausman test pattern)
    elif cluster is None:
        if vcov not in _no_default_cluster:
            cluster = [entity]
        # else: leave cluster=None so ols() uses the vcov-specific path

    # Inject FE into formula
    fe_formula = _inject_fe(formula, entity, time)

    # For HAC/DK, time_col defaults to time if provided
    if time_col is None and time is not None and vcov in ("NW", "DK"):
        time_col = time

    result = ols(
        fe_formula,
        data,
        vcov=vcov,
        cluster=cluster,
        ssc=ssc,
        time=time_col,
        bandwidth=bandwidth,
        weights=weights,
        fweights=fweights,
        n_boot=n_boot,
        seed=seed,
    )
    result.model_type = "Panel FE"
    return result


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
    ssc: SSC | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Panel random effects (GLS) estimator.

    Uses Swamy & Arora (1972) method to estimate variance components, then
    performs quasi-demeaning with theta = 1 - sqrt(sigma_e^2 / (T*sigma_u^2 + sigma_e^2)).

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity (panel) identifier
        time: Column name for time identifier (required for NW/DK)
        vcov: "iid", "HC1", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
        bandwidth: Number of lags for NW/DK. Default: Newey-West rule of thumb.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    if ssc is None:
        ssc = _default_ssc()
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

    # Don't put entity in spec.fe -- RE keeps the intercept and handles
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

    resid = y - X @ beta
    resid_re = y_re - X_re @ beta

    # Build cluster arrays (needed for cluster/wildboot VCV)
    n_clusters_dict = None
    cl_list = None
    if cluster or vcov in ("wildboot", "HC0", "HC1"):
        if not cluster:
            cluster = [entity]
        cl_list = []
        for c in cluster:
            if c == entity:
                cl_list.append(entity_codes)
            elif arrays.cluster_arrays and c in arrays.cluster_arrays:
                cl_list.append(arrays.cluster_arrays[c])
            else:
                cl = (
                    data[c]
                    .cast(pl.Utf8)
                    .cast(pl.Categorical)
                    .to_physical()
                    .to_numpy()
                    .astype(np.int32)
                )
                cl_list.append(cl)

    if cluster and vcov not in ("HC0", "HC1", "wildboot", "bootstrap", "NW", "DK"):
        V = compute_vcov(X_re, resid_re, vcov, ssc, cluster_arrays=cl_list)
        n_clusters_dict = {c: len(np.unique(a)) for c, a in zip(cluster, cl_list)}
        vcov_type_str = "cluster"
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "wildboot":
        V = compute_vcov(
            X_re, resid_re, vcov, ssc, cluster_arrays=cl_list, n_boot=n_boot, seed=seed
        )
        vcov_type_str = "wildboot"
        n_clusters_dict = {cluster[0]: len(np.unique(cl_list[0]))}
        df_r = n_clusters_dict[cluster[0]] - 1
    elif vcov in ("HC0", "HC1"):
        V = compute_vcov(X_re, resid_re, vcov, ssc)
        n_clusters_dict = None
        vcov_type_str = vcov
        df_r = n - k
    else:
        V = compute_vcov(
            X_re,
            resid_re,
            vcov,
            ssc,
            time_array=arrays.time_array,
            bandwidth=bandwidth,
            n_boot=n_boot,
            seed=seed,
            y=y_re,
        )
        vcov_type_str = vcov
        df_r = n - k

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
        model_type="Panel RE",
        vcov_type=vcov_type_str,
        n_clusters=n_clusters_dict,
    )
    result.ssc = ssc
    return result


def _first_difference(
    data: pl.DataFrame,
    entity: str,
    time: str,
    formula: str,
    cluster: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """First-difference data within entity groups.

    Sorts by [entity, time], computes within-entity diffs for all numeric
    columns in the formula, drops first obs per entity.

    Returns (diffed_data, diff_col_names) where diffed_data has columns
    named with original names (containing differenced values), plus any
    cluster and entity columns preserved.
    """
    spec = parse_formula(formula)
    all_needed = [spec.depvar] + spec.exog + [entity, time]
    if cluster:
        all_needed += [c for c in cluster if c not in all_needed]
    all_needed = list(dict.fromkeys(all_needed))

    data = sanitize_inf(data, all_needed)
    df_sorted = data.select(all_needed).sort([entity, time])

    diff_exprs = []
    numeric_cols = [spec.depvar] + spec.exog
    for col in numeric_cols:
        diff_exprs.append((pl.col(col) - pl.col(col).shift(1)).over(entity).alias(f"d_{col}"))

    df_diff = df_sorted.with_columns(diff_exprs)
    df_diff = df_diff.with_columns((pl.col(f"d_{spec.depvar}").is_not_null()).alias("_has_diff"))
    df_diff = df_diff.filter(pl.col("_has_diff"))

    rename_exprs = []
    diff_col_names = []
    for col in numeric_cols:
        rename_exprs.append(pl.col(f"d_{col}").alias(col))
        diff_col_names.append(f"d_{col}")

    keep_cols = [entity, time]
    if cluster:
        keep_cols += [c for c in cluster if c not in keep_cols]
    keep_cols = list(dict.fromkeys(keep_cols))

    diffed_data = df_diff.select(rename_exprs + [pl.col(c) for c in keep_cols])
    return diffed_data, diff_col_names


def panel_fd(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """First-difference estimator.

    Differences data within entity, then runs OLS on differenced data.
    See Wooldridge (2010), ch. 13.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity identifier
        time: Column name for time identifier
        vcov: "iid", "HC0"-"HC3", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs. Default: [entity].
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
    _no_default_cluster = {"NW", "DK", "bootstrap"}
    if isinstance(cluster, str):
        cluster = [cluster]
    elif isinstance(cluster, list) and len(cluster) == 0:
        cluster = None
    elif cluster is None:
        if vcov not in _no_default_cluster:
            cluster = [entity]

    data = ensure_polars(data)
    if isinstance(data, pl.LazyFrame):
        spec = parse_formula(formula)
        all_needed = [spec.depvar] + spec.exog + [entity, time]
        if cluster:
            all_needed += [c for c in cluster if c not in all_needed]
        all_needed = list(dict.fromkeys(all_needed))
        data = data.select(all_needed).collect()

    fd_cluster = cluster if cluster else [entity]
    diffed_data, _diff_cols = _first_difference(data, entity, time, formula, fd_cluster)

    result = ols(
        formula,
        diffed_data,
        vcov=vcov,
        cluster=cluster,
        ssc=ssc,
        n_boot=n_boot,
        seed=seed,
    )
    result.model_type = "Panel FD"
    return result
