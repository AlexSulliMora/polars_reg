from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from numpy.typing import NDArray


def _group_means(x: NDArray, codes: NDArray, n_groups: int) -> NDArray:
    """Compute group means. x can be 1D or 2D."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    if x.ndim == 1:
        sums = np.bincount(codes, weights=x, minlength=n_groups)
        return sums / counts
    else:
        k = x.shape[1]
        means = np.empty((n_groups, k))
        for j in range(k):
            sums = np.bincount(codes, weights=x[:, j], minlength=n_groups)
            means[:, j] = sums / counts
        return means


def demean(
    X: NDArray,
    fe_dict: dict[str, NDArray],
    tol: float = 1e-8,
    max_iter: int = 100_000,
) -> NDArray:
    """Demean columns of X by absorbing multiple fixed effects.

    Uses Symmetric Kaczmarz + CG acceleration (Correia 2016).
    Single FE: exact in one pass. Multiple FE: iterative.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    X = X.copy().astype(np.float64)
    fe_list = list(fe_dict.values())
    n_groups_list = [int(codes.max()) + 1 for codes in fe_list]

    if len(fe_list) == 1:
        codes = fe_list[0]
        n_g = n_groups_list[0]
        means = _group_means(X, codes, n_g)
        X -= means[codes]
        return X.squeeze(axis=1) if squeeze else X

    result = _demean_cg(X, fe_list, n_groups_list, tol, max_iter)
    return result.squeeze(axis=1) if squeeze else result


def _symmetric_kaczmarz(
    X: NDArray, fe_list: list[NDArray], n_groups_list: list[int]
) -> NDArray:
    """One sweep of symmetric Kaczmarz: forward then backward."""
    # Forward
    for codes, n_g in zip(fe_list, n_groups_list):
        means = _group_means(X, codes, n_g)
        X = X - means[codes]
    # Backward (skip last since forward already did it)
    for codes, n_g in zip(reversed(fe_list[:-1]), reversed(n_groups_list[:-1])):
        means = _group_means(X, codes, n_g)
        X = X - means[codes]
    return X


def _demean_cg(
    X: NDArray,
    fe_list: list[NDArray],
    n_groups_list: list[int],
    tol: float,
    max_iter: int,
) -> NDArray:
    """CG-accelerated demeaning with symmetric Kaczmarz transform."""
    x = X.copy()

    # Initial residual: r = T(x) - x
    r = _symmetric_kaczmarz(x.copy(), fe_list, n_groups_list) - x
    u = r.copy()
    ssr = np.sum(r**2)

    for _ in range(max_iter):
        x_norm = np.sum(x**2)
        if ssr < tol**2 * max(x_norm, 1e-16):
            break

        # v = u - T(u) = A*u where A = I - T
        Tu = _symmetric_kaczmarz(u.copy(), fe_list, n_groups_list)
        v = u - Tu
        uv = np.sum(u * v)
        if abs(uv) < 1e-30:
            break
        alpha = ssr / uv
        x = x + alpha * u
        r = r - alpha * v
        ssr_new = np.sum(r**2)
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
            singleton_groups = set(np.where(counts == 1)[0])
            if singleton_groups:
                for i in range(n):
                    if keep[i] and codes[i] in singleton_groups:
                        keep[i] = False
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


def _connected_components(
    codes_a: NDArray, n_a: int, codes_b: NDArray, n_b: int
) -> int:
    """Count connected components in bipartite graph of two FE dimensions."""
    total = n_a + n_b
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(codes_a)), (codes_a, n_a + codes_b)),
        shape=(total, total),
    )
    graph = graph + graph.T
    n_components = scipy.sparse.csgraph.connected_components(graph, directed=False)[0]
    return n_components
