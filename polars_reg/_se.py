from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from polars_reg._native import rust_clustered_meat as _rust_clustered_meat
from polars_reg._native import rust_dk_meat as _rust_dk_meat
from polars_reg._native import rust_hac_meat as _rust_hac_meat
from polars_reg._native import rust_recode as _rust_recode
from polars_reg._ssc import SSC, _compute_k_eff, _default_ssc

# Webb 6-point distribution for wild bootstrap
_WEBB6 = np.array([-np.sqrt(3 / 2), -1.0, -np.sqrt(1 / 2), np.sqrt(1 / 2), 1.0, np.sqrt(3 / 2)])


def vcov_iid(X: NDArray, resid: NDArray, ssc: SSC | None = None, df_abs: int = 0) -> NDArray:
    """Homoskedastic VCV: sigma^2 * (X'X)^{-1}.

    Args:
        ssc: Small-sample correction configuration.
        df_abs: Additional absorbed degrees of freedom (e.g., from fixed effects).
    """
    if ssc is None:
        ssc = _default_ssc()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    k_eff = _compute_k_eff(k, ssc.k_fixef, df_abs, 0)
    sigma2 = resid @ resid / (n - k_eff) if ssc.k_adj else resid @ resid / n
    return sigma2 * XtX_inv


def vcov_robust(
    X: NDArray, resid: NDArray, kind: str = "HC1", ssc: SSC | None = None, df_abs: int = 0
) -> NDArray:
    """Heteroskedasticity-robust VCV (HC0, HC1, HC2, HC3).

    All use sandwich form: (X'X)^{-1} X' diag(w) X (X'X)^{-1}
    HC0: w_i = e_i^2
    HC1: w_i = e_i^2 * n/(n-k_eff)  (k_eff includes FE per ssc.k_fixef)
    HC2: w_i = e_i^2 / (1 - h_ii)
    HC3: w_i = e_i^2 / (1 - h_ii)^2
    where h_ii = x_i' (X'X)^{-1} x_i (hat matrix diagonal)

    Args:
        ssc: Small-sample correction configuration.
        df_abs: Additional absorbed degrees of freedom (e.g., from fixed effects).
    """
    if ssc is None:
        ssc = _default_ssc()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    if kind == "HC0":
        meat = X.T @ (X * (resid**2)[:, None])
        return XtX_inv @ meat @ XtX_inv
    elif kind == "HC1":
        meat = X.T @ (X * (resid**2)[:, None])
        k_eff = _compute_k_eff(k, ssc.k_fixef, df_abs, 0)
        scale = (n / (n - k_eff)) if ssc.k_adj else 1.0
        return scale * XtX_inv @ meat @ XtX_inv
    elif kind in ("HC2", "HC3"):
        hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        if kind == "HC2":
            weights = resid**2 / (1.0 - hat)
        else:
            weights = resid**2 / (1.0 - hat) ** 2
        meat = X.T @ (X * weights[:, None])
        return XtX_inv @ meat @ XtX_inv
    else:
        raise ValueError(f"Unknown robust SE kind: {kind}")


def _recode_to_contiguous(arr: NDArray) -> tuple[NDArray, int]:
    """Remap arbitrary integer codes to contiguous 0..G-1."""
    if len(arr) == 0:
        return arr.astype(np.int32), 0
    codes, n_groups = _rust_recode(arr.astype(np.int64))
    return codes, n_groups


def _clustered_meat(X: NDArray, resid: NDArray, codes: NDArray, n_groups: int) -> NDArray:
    """Compute the clustered sandwich meat: sum_g (s_g s_g').

    Uses Rust native extension for O(n*k) aggregation.
    Expects pre-computed contiguous codes and group count.
    """
    return np.asarray(
        _rust_clustered_meat(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(resid, dtype=np.float64),
            np.ascontiguousarray(codes, dtype=np.int32),
            n_groups,
        )
    )


def vcov_clustered(
    X: NDArray,
    resid: NDArray,
    clusters: NDArray,
    ssc: SSC | None = None,
    df_a_non_nested: int = 0,
) -> NDArray:
    """One-way cluster-robust VCV (CRV1).

    Cameron, Gelbach & Miller (2011), "Robust Inference with Multiway
    Clustering", JBES 29(2). See eq. 2 for the one-way case.

    V = dfc * (X'X)^{-1} * meat * (X'X)^{-1}

    dfc = G_adj_factor * k_adj_factor
    where G_adj_factor = G/(G-1) if ssc.G_adj else 1
    and   k_adj_factor = (N-1)/(N-k_eff) if ssc.k_adj else 1

    Args:
        ssc: Small-sample correction configuration.
        df_a_non_nested: Non-nested FE degrees of freedom for k_fixef computation.
    """
    if ssc is None:
        ssc = _default_ssc()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    codes, G = _recode_to_contiguous(clusters)
    if G < 2:
        raise ValueError("Clustered SEs require at least 2 cluster groups")
    meat = _clustered_meat(X, resid, codes, G)
    k_eff = _compute_k_eff(k, ssc.k_fixef, 0, df_a_non_nested)
    k_adj_factor = (n - 1) / (n - k_eff) if ssc.k_adj else 1.0
    G_adj_factor = G / (G - 1) if ssc.G_adj else 1.0
    dfc = k_adj_factor * G_adj_factor
    return dfc * XtX_inv @ meat @ XtX_inv


def _hac_meat(
    score: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Compute Newey-West HAC meat matrix from score vectors.

    Newey & West (1987), "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
    Econometrica 55(3). Uses Bartlett kernel: w(j) = 1 - j/(L+1).

    Args:
        score: n x k score matrix (typically X * resid[:, None]).
        time_ids: Time period identifiers. Scores are sorted by these.
        bandwidth: Number of lags. Default: floor(4*(n/100)^(2/9)).

    Returns:
        k x k meat matrix.
    """
    return np.asarray(
        _rust_hac_meat(
            np.ascontiguousarray(score, dtype=np.float64),
            np.ascontiguousarray(time_ids, dtype=np.float64),
            bandwidth if bandwidth is not None else -1,
        )
    )


def _dk_meat(
    score: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Compute Driscoll-Kraay meat matrix from score vectors.

    Driscoll & Kraay (1998), "Consistent Covariance Matrix Estimation with
    Spatially Dependent Panel Data", Review of Economics and Statistics 80(4).
    Aggregates scores by time period, then applies Newey-West kernel.

    Args:
        score: n x k score matrix (typically X * resid[:, None]).
        time_ids: Time period identifiers.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).

    Returns:
        k x k meat matrix.
    """
    return np.asarray(
        _rust_dk_meat(
            np.ascontiguousarray(score, dtype=np.float64),
            np.ascontiguousarray(time_ids, dtype=np.float64),
            bandwidth if bandwidth is not None else -1,
        )
    )


def vcov_hac(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
    ssc: SSC | None = None,
    df_abs: int = 0,
) -> NDArray:
    """Newey-West HAC VCV (heteroskedasticity and autocorrelation consistent).

    Uses Bartlett kernel: w(j) = 1 - j/(L+1).

    Args:
        time_ids: Time period identifiers. Observations are sorted by these.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).
        ssc: Small-sample correction configuration.
        df_abs: Total absorbed FE degrees of freedom.
    """
    if ssc is None:
        ssc = _default_ssc()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]
    S = _hac_meat(score, time_ids, bandwidth)
    k_eff = _compute_k_eff(k, ssc.k_fixef, df_abs, 0)
    dfc = n / (n - k_eff) if ssc.k_adj else 1.0
    return dfc * XtX_inv @ S @ XtX_inv


def vcov_driscoll_kraay(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
    ssc: SSC | None = None,
    df_abs: int = 0,
) -> NDArray:
    """Driscoll-Kraay VCV for panel data.

    Robust to cross-sectional dependence, heteroskedasticity, and autocorrelation.
    Aggregates score vectors by time period, then applies Newey-West.

    The T/(T-1) correction is intrinsic to the DK estimator and NOT controlled
    by ssc.G_adj. Only ssc.k_adj affects an additional N/(N-k) factor if present.

    Args:
        time_ids: Time period identifiers for each observation.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).
        ssc: Small-sample correction configuration.
        df_abs: Total absorbed FE degrees of freedom.
    """
    if ssc is None:
        ssc = _default_ssc()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]
    S = _dk_meat(score, time_ids, bandwidth)
    T = len(np.unique(time_ids))
    if T < 2:
        raise ValueError("Driscoll-Kraay SEs require at least 2 time periods")
    # T/(T-1) is intrinsic to DK, always applied
    dfc = T / (T - 1)
    return dfc * XtX_inv @ S @ XtX_inv


def _interaction_codes(*arrays: NDArray) -> tuple[NDArray, int]:
    """Create unique integer codes for the interaction of multiple cluster arrays.

    Returns (codes, n_groups) tuple with contiguous 0..G-1 codes.
    """
    if len(arrays) == 1:
        return _recode_to_contiguous(arrays[0])
    # Fast interaction coding: combine arrays into a single code via
    # multiplied offsets (avoids expensive structured array + np.unique).
    combined = arrays[0].astype(np.int64)
    for arr in arrays[1:]:
        combined = combined * (int(arr.max()) + 1) + arr.astype(np.int64)
    return _recode_to_contiguous(combined)


def vcov_multiway_clustered(
    X: NDArray,
    resid: NDArray,
    cluster_list: list[NDArray],
    ssc: SSC | None = None,
    df_a_non_nested: int = 0,
) -> NDArray:
    """Multi-way clustered VCV via Cameron, Gelbach & Miller (2011) inclusion-exclusion.

    V = sum over non-empty subsets S of (-1)^(|S|+1) * V_S
    where V_S is one-way clustered VCV using intersection of dimensions in S.

    For D=2: V = V_A + V_B - V_{A*B}
    For D=3: V = V_A + V_B + V_C - V_AB - V_AC - V_BC + V_ABC

    Args:
        ssc: Small-sample correction configuration.
            G_df="min": single dfc using min(G) for all terms (reghdfe-style).
            G_df="conventional": per-term G_i/(G_i-1) for each summand.
        df_a_non_nested: Non-nested FE degrees of freedom for k_fixef computation.
    """
    if ssc is None:
        ssc = _default_ssc()
    D = len(cluster_list)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    V = np.zeros((k, k))
    dims = list(range(D))

    # Precompute contiguous codes for each individual cluster dimension
    precomputed: list[tuple[NDArray, int]] = []
    for cl in cluster_list:
        precomputed.append(_recode_to_contiguous(cl))

    k_eff = _compute_k_eff(k, ssc.k_fixef, 0, df_a_non_nested)
    k_adj_factor = (n - 1) / (n - k_eff) if ssc.k_adj else 1.0

    if ssc.G_df == "min":
        # reghdfe-style: single G factor using min G across individual cluster dims
        G_min = min(g for _, g in precomputed)
        G_adj_global = G_min / (G_min - 1) if ssc.G_adj else 1.0

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            if len(subset) == 1:
                codes, G = precomputed[subset[0]]
            else:
                subset_arrays = [cluster_list[d] for d in subset]
                codes, G = _interaction_codes(*subset_arrays)
            meat = _clustered_meat(X, resid, codes, G)
            if ssc.G_df == "min":
                G_adj_factor = G_adj_global
            else:
                # conventional: per-term G/(G-1)
                G_adj_factor = G / (G - 1) if ssc.G_adj else 1.0
            dfc = k_adj_factor * G_adj_factor
            V += sign * dfc * XtX_inv @ meat @ XtX_inv

    return V


def vcov_pairs_bootstrap(
    X: NDArray,
    y: NDArray,
    n_boot: int = 999,
    seed: int | None = None,
    fit_fn: object = None,
    ssc: SSC | None = None,
) -> NDArray:
    """Pairs bootstrap VCV.

    Resamples (y_i, x_i) pairs with replacement and re-estimates.
    Returns the sample covariance of bootstrap coefficient estimates.

    Args:
        fit_fn: Optional callable (X, y) -> beta. Defaults to OLS.
        ssc: Accepted for API consistency but not used (bootstrap is
            resampling-based inference; SSC does not apply).
    """
    rng = np.random.default_rng(seed)
    n, k = X.shape
    betas = np.empty((n_boot, k))

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b = X[idx]
        y_b = y[idx]
        try:
            if fit_fn is not None:
                betas[b] = fit_fn(X_b, y_b)
            else:
                betas[b] = np.linalg.solve(X_b.T @ X_b, X_b.T @ y_b)
        except (np.linalg.LinAlgError, ValueError):
            betas[b] = np.nan

    valid = ~np.any(np.isnan(betas), axis=1)
    n_valid = valid.sum()
    if n_valid < 2:
        raise ValueError(
            f"Too few valid bootstrap replicates ({n_valid}/{n_boot}). "
            "Data may be nearly collinear."
        )
    V = np.cov(betas[valid].T, ddof=1)
    return np.atleast_2d(V)


def vcov_wild_bootstrap(
    X: NDArray,
    resid: NDArray,
    clusters: NDArray,
    n_boot: int = 999,
    seed: int | None = None,
    bread: NDArray | None = None,
    score_X: NDArray | None = None,
    ssc: SSC | None = None,
) -> NDArray:
    """Wild cluster bootstrap VCV using Webb 6-point distribution.

    Draws one random multiplier per cluster from the Webb 6-point set,
    creates perturbed residuals, re-estimates beta, and returns the
    sample covariance of bootstrap coefficient estimates.

    For OLS: bread = (X'X)^{-1}, score_X = X (default).
    For IV:  bread = (X_hat'X)^{-1}, score_X = X_hat.

    Args:
        ssc: Accepted for API consistency but not used (bootstrap is
            resampling-based inference; SSC does not apply).
    """
    rng = np.random.default_rng(seed)
    _, codes = np.unique(clusters, return_inverse=True)
    G = codes.max() + 1
    n, k = X.shape

    if bread is None:
        bread = np.linalg.inv(X.T @ X)
    if score_X is None:
        score_X = X

    # beta_star - beta = bread @ score_X' @ (resid * eta)
    betas_centered = np.empty((n_boot, k))
    for b in range(n_boot):
        multipliers = rng.choice(_WEBB6, size=G)
        eta = multipliers[codes]
        betas_centered[b] = bread @ (score_X.T @ (resid * eta))

    V = np.cov(betas_centered.T, ddof=1)
    return np.atleast_2d(V)


def _mle_multiway_clustered(
    X: NDArray,
    score_resid: NDArray,
    cluster_list: list[NDArray],
    H_inv: NDArray,
    n: int,
    k: int,
    ssc: SSC | None = None,
) -> NDArray:
    """Multi-way clustered VCV for MLE models (CGM inclusion-exclusion)."""
    if ssc is None:
        ssc = _default_ssc()
    D = len(cluster_list)
    V = np.zeros((k, k))
    dims = list(range(D))

    k_adj_factor = (n - 1) / (n - k) if ssc.k_adj else 1.0

    # Precompute codes for min-G computation
    precomputed: list[tuple[NDArray, int]] = []
    for cl in cluster_list:
        precomputed.append(_recode_to_contiguous(cl))

    if ssc.G_df == "min":
        G_min = min(g for _, g in precomputed)
        G_adj_global = G_min / (G_min - 1) if ssc.G_adj else 1.0

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction, G = _interaction_codes(*subset_arrays)
            meat = _clustered_meat(X, score_resid, interaction, G)
            if ssc.G_df == "min":
                G_adj_factor = G_adj_global
            else:
                G_adj_factor = G / (G - 1) if ssc.G_adj else 1.0
            dfc = k_adj_factor * G_adj_factor
            V += sign * dfc * H_inv @ meat @ H_inv

    return V


def compute_vcov(
    X: NDArray,
    resid: NDArray,
    vcov: str,
    ssc: SSC,
    *,
    cluster_arrays: list[NDArray] | None = None,
    time_array: NDArray | None = None,
    bandwidth: int | None = None,
    df_abs: int = 0,
    df_a_non_nested: int = 0,
    n_boot: int = 999,
    seed: int | None = None,
    y: NDArray | None = None,
    bread: NDArray | None = None,
    score_X: NDArray | None = None,
) -> NDArray:
    """Unified VCV dispatch.

    Replaces copy-pasted if/elif chains in estimators. Delegates to the
    individual ``vcov_*`` functions for OLS-family models, or computes
    bread-meat-bread sandwiches directly when a custom bread matrix is
    provided (for MLE / IV models).

    For OLS-family: bread and score_X default to (X'X)^{-1} and X.
    For IV: pass bread=(X_hat'X)^{-1} and score_X=X_hat.
    For MLE: pass bread=H_inv (and compute meat from per-obs scores).

    Args:
        X: Design matrix (n x k).
        resid: Residual vector (n,). For MLE models, this is the score
            residual (e.g. y - mu for PPML, lambda for probit).
        vcov: VCV type string ("iid", "HC0"-"HC3", "NW", "DK",
            "bootstrap", "wildboot").
        ssc: Small-sample correction configuration.
        cluster_arrays: List of cluster code arrays. If provided and vcov
            is not "wildboot", clustered SEs are computed.
        time_array: Time identifiers (required for NW/DK).
        bandwidth: Number of lags for HAC/DK.
        df_abs: Total absorbed FE degrees of freedom.
        df_a_non_nested: Non-nested FE degrees of freedom.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap.
        y: Dependent variable (required for pairs bootstrap).
        bread: Override the bread matrix. When None, (X'X)^{-1} is used.
        score_X: Override X used in score/meat computation. When None, X
            itself is used.
    """
    n, k = X.shape

    if bread is not None:
        # Custom bread (MLE or IV) — compute sandwich directly
        effective_X = score_X if score_X is not None else X

        if cluster_arrays and vcov != "wildboot":
            if len(cluster_arrays) == 1:
                codes, G = _recode_to_contiguous(cluster_arrays[0])
                if G < 2:
                    raise ValueError("Clustered SEs require at least 2 cluster groups")
                meat = _clustered_meat(effective_X, resid, codes, G)
                k_eff = _compute_k_eff(k, ssc.k_fixef, 0, 0)
                k_adj_factor = (n - 1) / (n - k_eff) if ssc.k_adj else 1.0
                G_adj_factor = G / (G - 1) if ssc.G_adj else 1.0
                dfc = k_adj_factor * G_adj_factor
                return dfc * bread @ meat @ bread
            else:
                return _mle_multiway_clustered(
                    effective_X,
                    resid,
                    cluster_arrays,
                    bread,
                    n,
                    k,
                    ssc=ssc,
                )
        elif vcov == "wildboot":
            if not cluster_arrays:
                raise ValueError("vcov='wildboot' requires cluster_arrays")
            return vcov_wild_bootstrap(
                X,
                resid,
                cluster_arrays[0],
                n_boot=n_boot,
                seed=seed,
                bread=bread,
                score_X=score_X,
                ssc=ssc,
            )
        elif vcov in ("HC0", "HC1"):
            meat = effective_X.T @ (effective_X * (resid**2)[:, None])
            k_eff = _compute_k_eff(k, ssc.k_fixef, 0, 0)
            dfc = (n / (n - k_eff)) if (ssc.k_adj and vcov == "HC1") else 1.0
            return dfc * bread @ meat @ bread
        elif vcov == "NW":
            if time_array is None:
                raise ValueError("vcov='NW' requires time_array")
            score = effective_X * resid[:, None]
            S = _hac_meat(score, time_array, bandwidth)
            k_eff = _compute_k_eff(k, ssc.k_fixef, df_abs, 0)
            dfc = n / (n - k_eff) if ssc.k_adj else 1.0
            return dfc * bread @ S @ bread
        elif vcov == "DK":
            if time_array is None:
                raise ValueError("vcov='DK' requires time_array")
            score = effective_X * resid[:, None]
            S = _dk_meat(score, time_array, bandwidth)
            T = len(np.unique(time_array))
            if T < 2:
                raise ValueError("Driscoll-Kraay SEs require at least 2 time periods")
            dfc = T / (T - 1)
            return dfc * bread @ S @ bread
        elif vcov == "bootstrap":
            if y is None:
                raise ValueError("Pairs bootstrap requires y")
            return vcov_pairs_bootstrap(X, y, n_boot=n_boot, seed=seed, ssc=ssc)
        elif vcov == "iid":
            # MLE iid: bread is already the VCV (inverse information matrix)
            return bread
        else:
            raise ValueError(f"Unknown vcov type: {vcov}")
    else:
        # Standard OLS bread — delegate to existing functions
        if cluster_arrays and vcov != "wildboot":
            if len(cluster_arrays) == 1:
                return vcov_clustered(
                    X,
                    resid,
                    cluster_arrays[0],
                    ssc=ssc,
                    df_a_non_nested=df_a_non_nested,
                )
            else:
                return vcov_multiway_clustered(
                    X,
                    resid,
                    cluster_arrays,
                    ssc=ssc,
                    df_a_non_nested=df_a_non_nested,
                )
        elif vcov == "bootstrap":
            if y is None:
                raise ValueError("Pairs bootstrap requires y")
            return vcov_pairs_bootstrap(X, y, n_boot=n_boot, seed=seed, ssc=ssc)
        elif vcov == "wildboot":
            if not cluster_arrays:
                raise ValueError("vcov='wildboot' requires cluster_arrays")
            return vcov_wild_bootstrap(
                X,
                resid,
                cluster_arrays[0],
                n_boot=n_boot,
                seed=seed,
                ssc=ssc,
            )
        elif vcov in ("NW", "DK"):
            if time_array is None:
                raise ValueError(f"vcov='{vcov}' requires time_array")
            if vcov == "NW":
                return vcov_hac(
                    X,
                    resid,
                    time_array,
                    bandwidth=bandwidth,
                    ssc=ssc,
                    df_abs=df_abs,
                )
            else:
                return vcov_driscoll_kraay(
                    X,
                    resid,
                    time_array,
                    bandwidth=bandwidth,
                    ssc=ssc,
                    df_abs=df_abs,
                )
        elif vcov == "iid":
            return vcov_iid(X, resid, ssc=ssc, df_abs=df_abs)
        elif vcov in ("HC0", "HC1", "HC2", "HC3"):
            return vcov_robust(X, resid, kind=vcov, ssc=ssc, df_abs=df_abs)
        else:
            raise ValueError(f"Unknown vcov type: {vcov}")
