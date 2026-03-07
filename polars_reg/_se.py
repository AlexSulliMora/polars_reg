from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

# Webb 6-point distribution for wild bootstrap
_WEBB6 = np.array([-np.sqrt(3 / 2), -1.0, -np.sqrt(1 / 2), np.sqrt(1 / 2), 1.0, np.sqrt(3 / 2)])


def vcov_iid(X: NDArray, resid: NDArray, df_abs: int = 0) -> NDArray:
    """Homoskedastic VCV: sigma^2 * (X'X)^{-1}.

    Args:
        df_abs: Additional absorbed degrees of freedom (e.g., from fixed effects).
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    sigma2 = resid @ resid / (n - k - df_abs)
    return sigma2 * XtX_inv


def vcov_robust(X: NDArray, resid: NDArray, kind: str = "HC1") -> NDArray:
    """Heteroskedasticity-robust VCV (HC0, HC1, HC2, HC3).

    All use sandwich form: (X'X)^{-1} X' diag(w) X (X'X)^{-1}
    HC0: w_i = e_i^2
    HC1: w_i = e_i^2 * n/(n-k)
    HC2: w_i = e_i^2 / (1 - h_ii)
    HC3: w_i = e_i^2 / (1 - h_ii)^2
    where h_ii = x_i' (X'X)^{-1} x_i (hat matrix diagonal)
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    if kind == "HC0":
        meat = X.T @ (X * (resid**2)[:, None])
        return XtX_inv @ meat @ XtX_inv
    elif kind == "HC1":
        meat = X.T @ (X * (resid**2)[:, None])
        return (n / (n - k)) * XtX_inv @ meat @ XtX_inv
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


def _clustered_meat(X: NDArray, resid: NDArray, clusters: NDArray) -> NDArray:
    """Compute the clustered sandwich meat: sum_g (s_g s_g').

    Uses np.bincount to aggregate scores by cluster in O(n*k) instead of
    looping over groups in Python.
    """
    k = X.shape[1]
    score = X * resid[:, None]
    # Remap clusters to contiguous 0..G-1 codes
    _, codes = np.unique(clusters, return_inverse=True)
    G = codes.max() + 1
    # Aggregate scores by cluster: S[g, j] = sum of score[i, j] for i in cluster g
    S = np.empty((G, k))
    for j in range(k):
        S[:, j] = np.bincount(codes, weights=score[:, j], minlength=G)
    # meat = S' @ S = sum_g outer(s_g, s_g)
    return S.T @ S


def vcov_clustered(
    X: NDArray,
    resid: NDArray,
    clusters: NDArray,
    df_correction: bool = True,
    df_a_non_nested: int = -1,
) -> NDArray:
    """One-way cluster-robust VCV (CRV1).

    V = dfc * (X'X)^{-1} * meat * (X'X)^{-1}

    Without absorbed FE: dfc = G/(G-1) * (n-1)/(n-k)      (matches Stata reg)
    With absorbed FE:    dfc = G/(G-1) * (n-d)/(n-d-k)     (matches Stata reghdfe)
        where d = DoF from FE dimensions NOT nested in the cluster.

    Args:
        df_a_non_nested: DoF absorbed by FE not nested in the cluster variable.
            When > -1, indicates FE are absorbed and reghdfe-style dfc is used:
            dfc = G/(G-1) * (n-d)/(n-d-k) where d = df_a_non_nested.
            When -1 (default), standard Stata reg dfc: G/(G-1) * (n-1)/(n-k).
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = _clustered_meat(X, resid, clusters)
    G = len(np.unique(clusters))
    if df_correction:
        if df_a_non_nested >= 0:
            # reghdfe-style: G/(G-1) * N/(N-d-k) where d = non-nested FE DoF
            dfc = (G / (G - 1)) * (n / (n - df_a_non_nested - k))
        else:
            # Standard Stata reg: (n-1)/(n-k)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    else:
        dfc = 1.0
    return dfc * XtX_inv @ meat @ XtX_inv


def vcov_hac(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Newey-West HAC VCV (heteroskedasticity and autocorrelation consistent).

    Uses Bartlett kernel: w(j) = 1 - j/(L+1).

    Args:
        time_ids: Time period identifiers. Observations are sorted by these.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]

    # Sort by time to ensure proper lag structure
    order = np.argsort(time_ids)
    score = score[order]

    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    # Meat: Γ_0 + Σ_{j=1}^{L} w(j) * (Γ_j + Γ_j')
    S = score.T @ score
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1)
        Gamma_j = score[j:].T @ score[:-j]
        S += w * (Gamma_j + Gamma_j.T)

    dfc = n / (n - k)
    return dfc * XtX_inv @ S @ XtX_inv


def vcov_driscoll_kraay(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Driscoll-Kraay VCV for panel data.

    Robust to cross-sectional dependence, heteroskedasticity, and autocorrelation.
    Aggregates score vectors by time period, then applies Newey-West.

    Args:
        time_ids: Time period identifiers for each observation.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]

    # Aggregate scores by time period
    unique_times, time_idx = np.unique(time_ids, return_inverse=True)
    T = len(unique_times)

    h = np.zeros((T, k))
    for j in range(k):
        h[:, j] = np.bincount(time_idx, weights=score[:, j], minlength=T)

    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))

    # Newey-West on cross-sectional sums
    S = h.T @ h
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1)
        Gamma_j = h[j:].T @ h[:-j]
        S += w * (Gamma_j + Gamma_j.T)

    # Small-sample correction: T/(T-1) following Hoechle (2007)
    dfc = T / (T - 1)
    return dfc * XtX_inv @ S @ XtX_inv


def _interaction_codes(*arrays: NDArray) -> NDArray:
    """Create unique integer codes for the interaction of multiple cluster arrays."""
    if len(arrays) == 1:
        return arrays[0]
    n = len(arrays[0])
    dtype = [(f"f{i}", arr.dtype) for i, arr in enumerate(arrays)]
    structured = np.empty(n, dtype=dtype)
    for i, arr in enumerate(arrays):
        structured[f"f{i}"] = arr
    _, codes = np.unique(structured, return_inverse=True)
    return codes


def vcov_multiway_clustered(
    X: NDArray,
    resid: NDArray,
    cluster_list: list[NDArray],
    df_a_non_nested: int = -1,
) -> NDArray:
    """Multi-way clustered VCV via Cameron-Gelbach-Miller inclusion-exclusion.

    V = sum over non-empty subsets S of (-1)^(|S|+1) * V_S
    where V_S is one-way clustered VCV using intersection of dimensions in S.

    For D=2: V = V_A + V_B - V_{A*B}
    For D=3: V = V_A + V_B + V_C - V_AB - V_AC - V_BC + V_ABC

    Uses a single dfc (based on min cluster count) for all CGM terms,
    matching reghdfe behavior.

    Args:
        df_a_non_nested: DoF absorbed by FE not nested in any cluster dimension.
            When >= 0, uses reghdfe-style dfc: G_min/(G_min-1) * N/(N-d-k).
            When -1 (default), uses per-term dfc: G/(G-1) * (N-1)/(N-k).
    """
    D = len(cluster_list)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    V = np.zeros((k, k))
    dims = list(range(D))

    if df_a_non_nested >= 0:
        # reghdfe-style: single dfc using min G across individual cluster dims
        G_min = min(len(np.unique(cl)) for cl in cluster_list)
        dfc = (G_min / (G_min - 1)) * (n / (n - df_a_non_nested - k))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction = _interaction_codes(*subset_arrays)
            meat = _clustered_meat(X, resid, interaction)
            if df_a_non_nested < 0:
                # Standard: per-term G/(G-1) * (N-1)/(N-k)
                G = len(np.unique(interaction))
                dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * XtX_inv @ meat @ XtX_inv

    return V


def vcov_pairs_bootstrap(
    X: NDArray,
    y: NDArray,
    n_boot: int = 999,
    seed: int | None = None,
    fit_fn: object = None,
) -> NDArray:
    """Pairs bootstrap VCV.

    Resamples (y_i, x_i) pairs with replacement and re-estimates.
    Returns the sample covariance of bootstrap coefficient estimates.

    Args:
        fit_fn: Optional callable (X, y) -> beta. Defaults to OLS.
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
) -> NDArray:
    """Wild cluster bootstrap VCV using Webb 6-point distribution.

    Draws one random multiplier per cluster from the Webb 6-point set,
    creates perturbed residuals, re-estimates beta, and returns the
    sample covariance of bootstrap coefficient estimates.

    For OLS: bread = (X'X)^{-1}, score_X = X (default).
    For IV:  bread = (X_hat'X)^{-1}, score_X = X_hat.
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
