from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from numpy.typing import NDArray

try:
    from polars_reg._native import rust_demean as _rust_demean

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _group_counts(codes: NDArray, n_groups: int, w: NDArray | None = None) -> NDArray:
    """Precompute group counts/weight sums (cached outside the hot loop)."""
    if w is None:
        return np.bincount(codes, minlength=n_groups).astype(np.float64)
    return np.bincount(codes, weights=w, minlength=n_groups)


def _subtract_group_means(
    x: NDArray, codes: NDArray, n_groups: int, denom: NDArray, w: NDArray | None = None
) -> NDArray:
    """Subtract group means from x in-place. x can be 1D or 2D.

    Uses pre-computed denominators to avoid recomputing group counts each call.
    """
    if x.ndim == 1:
        if w is None:
            sums = np.bincount(codes, weights=x, minlength=n_groups)
        else:
            sums = np.bincount(codes, weights=w * x, minlength=n_groups)
        x -= (sums / denom)[codes]
        return x
    k = x.shape[1]
    for j in range(k):
        col = x[:, j]
        if w is None:
            sums = np.bincount(codes, weights=col, minlength=n_groups)
        else:
            sums = np.bincount(codes, weights=w * col, minlength=n_groups)
        col -= (sums / denom)[codes]
    return x


def _group_means(x: NDArray, codes: NDArray, n_groups: int, w: NDArray | None = None) -> NDArray:
    """Compute (optionally weighted) group means. x can be 1D or 2D."""
    denom = _group_counts(codes, n_groups, w)
    if x.ndim == 1:
        if w is None:
            sums = np.bincount(codes, weights=x, minlength=n_groups)
        else:
            sums = np.bincount(codes, weights=w * x, minlength=n_groups)
        return sums / denom
    k = x.shape[1]
    means = np.empty((n_groups, k))
    for j in range(k):
        if w is None:
            sums = np.bincount(codes, weights=x[:, j], minlength=n_groups)
        else:
            sums = np.bincount(codes, weights=w * x[:, j], minlength=n_groups)
        means[:, j] = sums / denom
    return means


def demean(
    X: NDArray,
    fe_dict: dict[str, NDArray],
    tol: float = 1e-8,
    max_iter: int = 100_000,
    weights: NDArray | None = None,
) -> NDArray:
    """Demean columns of X by absorbing multiple fixed effects.

    Uses Symmetric Kaczmarz + CG acceleration (Correia 2016).
    Single FE: exact in one pass. Multiple FE: iterative.

    Args:
        weights: Optional analytic weights for weighted group means.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    X = X.copy().astype(np.float64)
    fe_list = list(fe_dict.values())
    n_groups_list = [int(codes.max()) + 1 for codes in fe_list]

    # Use Rust native path for unweighted demeaning
    if _HAS_NATIVE and weights is None:
        fe_codes_list = [np.ascontiguousarray(c, dtype=np.int32) for c in fe_list]
        result = np.asarray(
            _rust_demean(
                np.ascontiguousarray(X),
                fe_codes_list,
                n_groups_list,
                tol,
                max_iter,
            )
        )
        return result.squeeze(axis=1) if squeeze else result

    if len(fe_list) == 1:
        codes = fe_list[0]
        n_g = n_groups_list[0]
        denom = _group_counts(codes, n_g, w=weights)
        _subtract_group_means(X, codes, n_g, denom, w=weights)
        return X.squeeze(axis=1) if squeeze else X

    result = _demean_cg(X, fe_list, n_groups_list, tol, max_iter, weights=weights)
    return result.squeeze(axis=1) if squeeze else result


def _symmetric_kaczmarz(
    X: NDArray,
    fe_list: list[NDArray],
    n_groups_list: list[int],
    denoms: list[NDArray],
    w: NDArray | None = None,
) -> NDArray:
    """One sweep of symmetric Kaczmarz: forward then backward.

    Uses pre-computed group counts (denoms) to avoid redundant bincount calls.
    """
    # Forward
    for codes, n_g, denom in zip(fe_list, n_groups_list, denoms):
        _subtract_group_means(X, codes, n_g, denom, w=w)
    # Backward (skip last since forward already did it)
    for codes, n_g, denom in zip(
        reversed(fe_list[:-1]), reversed(n_groups_list[:-1]), reversed(denoms[:-1])
    ):
        _subtract_group_means(X, codes, n_g, denom, w=w)
    return X


def _demean_cg(
    X: NDArray,
    fe_list: list[NDArray],
    n_groups_list: list[int],
    tol: float,
    max_iter: int,
    weights: NDArray | None = None,
) -> NDArray:
    """CG-accelerated demeaning with symmetric Kaczmarz transform."""
    # Precompute group counts once (avoids redundant bincount in every iteration)
    denoms = [_group_counts(codes, n_g, w=weights) for codes, n_g in zip(fe_list, n_groups_list)]

    # Pre-allocate work arrays to avoid copies in the hot loop
    x = X.copy()
    tmp = np.empty_like(x)

    # Initial residual: r = T(x) - x
    np.copyto(tmp, x)
    _symmetric_kaczmarz(tmp, fe_list, n_groups_list, denoms, w=weights)
    r = tmp - x
    u = r.copy()
    ssr = np.sum(r * r)

    for _ in range(max_iter):
        x_norm = np.sum(x * x)
        if ssr < tol**2 * max(x_norm, 1e-16):
            break

        # v = u - T(u) = A*u where A = I - T
        np.copyto(tmp, u)
        _symmetric_kaczmarz(tmp, fe_list, n_groups_list, denoms, w=weights)
        v = u - tmp
        uv = np.sum(u * v)
        if abs(uv) < 1e-30:
            break
        alpha = ssr / uv
        x += alpha * u
        r -= alpha * v
        ssr_new = np.sum(r * r)
        beta = ssr_new / ssr
        u = r + beta * u
        ssr = ssr_new
    else:
        warnings.warn(f"Demeaning did not converge after {max_iter} iterations")

    return x


def drop_singletons(fe_dict: dict[str, NDArray]) -> NDArray:
    """Return boolean mask of observations to keep (iteratively drop singletons)."""
    n = len(next(iter(fe_dict.values())))
    keep = np.ones(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for codes in fe_dict.values():
            active_codes = codes[keep]
            counts = np.bincount(active_codes)
            # Vectorized: mark all obs in singleton groups for removal
            is_singleton = counts[active_codes] == 1
            if is_singleton.any():
                idx = np.where(keep)[0]
                keep[idx[is_singleton]] = False
                changed = True
    return keep


def absorbed_dof(fe_dict: dict[str, NDArray]) -> int:
    """Count degrees of freedom absorbed by fixed effects.

    Single FE: number of groups.
    Two+ FE: sum of groups minus connected components (pairwise method).
    """
    fe_list = list(fe_dict.values())
    n_groups = [int(codes.max()) + 1 for codes in fe_list]
    total_dof = n_groups[0]

    for i in range(1, len(fe_list)):
        total_dof += n_groups[i]
        max_components = 0
        for j in range(i):
            c = _connected_components(fe_list[j], n_groups[j], fe_list[i], n_groups[i])
            max_components = max(max_components, c)
        total_dof -= max_components

    return total_dof


def _connected_components(codes_a: NDArray, n_a: int, codes_b: NDArray, n_b: int) -> int:
    """Count connected components in bipartite graph of two FE dimensions.

    Builds a sparse adjacency using only unique edges (deduplicated) to reduce
    the size of the sparse matrix construction + sort_indices bottleneck.
    """
    # Deduplicate edges: only keep unique (a, b) pairs
    edge_codes = codes_a.astype(np.int64) * n_b + codes_b.astype(np.int64)
    unique_edges = np.unique(edge_codes)
    ua = (unique_edges // n_b).astype(np.int32)
    ub = (unique_edges % n_b).astype(np.int32)

    total = n_a + n_b
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(ua)), (ua, n_a + ub)),
        shape=(total, total),
    )
    graph = graph + graph.T
    n_components = scipy.sparse.csgraph.connected_components(graph, directed=False)[0]
    return n_components
