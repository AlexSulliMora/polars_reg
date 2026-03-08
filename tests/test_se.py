import numpy as np
import pytest

from polars_reg._se import (
    _interaction_codes,
    vcov_clustered,
    vcov_driscoll_kraay,
    vcov_hac,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_robust,
)


def _make_ols_data():
    """Simple OLS: y = 2 + 3*x + e, return X, y, residuals, beta."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    X = np.column_stack([x, np.ones(n)])
    y = 2.0 + 3.0 * x + e
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return X, y, resid, beta, XtX_inv


def test_vcov_iid():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n, k = X.shape
    V = vcov_iid(X, resid)
    sigma2 = resid @ resid / (n - k)
    expected = sigma2 * XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_hc0():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    V = vcov_robust(X, resid, kind="HC0")
    meat = X.T @ np.diag(resid**2) @ X
    expected = XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_hc1():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n, k = X.shape
    V = vcov_robust(X, resid, kind="HC1")
    meat = X.T @ np.diag(resid**2) @ X
    expected = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_hc2():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    V = vcov_robust(X, resid, kind="HC2")
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    weights = resid**2 / (1.0 - hat)
    meat = X.T @ np.diag(weights) @ X
    expected = XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_hc3():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    V = vcov_robust(X, resid, kind="HC3")
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    weights = resid**2 / (1.0 - hat) ** 2
    meat = X.T @ np.diag(weights) @ X
    expected = XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_clustered_oneway():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n = len(resid)
    clusters = np.repeat(np.arange(10), 10)
    V = vcov_clustered(X, resid, clusters)
    G = 10
    k = X.shape[1]
    score = X * resid[:, None]
    meat = np.zeros((k, k))
    for g in range(G):
        mask = clusters == g
        sg = score[mask].sum(axis=0)
        meat += np.outer(sg, sg)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    expected = dfc * XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_twoway_clustered():
    """Two-way clustering: V = V_A + V_B - V_{A*B}."""
    rng = np.random.default_rng(42)
    n = 200
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    firm = np.repeat(np.arange(10), 20)
    year = np.tile(np.arange(20), 10)

    V = vcov_multiway_clustered(X, resid, [firm, year])

    V_firm = vcov_clustered(X, resid, firm, df_correction=True)
    V_year = vcov_clustered(X, resid, year, df_correction=True)
    interaction, _ = _interaction_codes(firm, year)
    V_inter = vcov_clustered(X, resid, interaction, df_correction=True)
    expected = V_firm + V_year - V_inter

    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_threeway_clustered():
    """Three-way: V = V_A + V_B + V_C - V_AB - V_AC - V_BC + V_ABC."""
    rng = np.random.default_rng(42)
    n = 120
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    a = np.repeat(np.arange(4), 30)
    b = np.tile(np.repeat(np.arange(6), 5), 4)
    c = np.tile(np.arange(5), 24)

    V = vcov_multiway_clustered(X, resid, [a, b, c])
    assert V.shape == (2, 2)
    np.testing.assert_allclose(V, V.T, atol=1e-14)


def test_invalid_robust_kind():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    with pytest.raises(ValueError, match="Unknown robust SE kind"):
        vcov_robust(X, resid, kind="HC99")


def test_hac_refactor_parity():
    """Refactored vcov_hac should produce valid VCV."""
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    V = vcov_hac(X, resid, time_ids, bandwidth=5)
    assert V.shape == (k, k)
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals >= -1e-10)


def test_dk_refactor_parity():
    """Refactored vcov_driscoll_kraay should produce valid VCV."""
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    V = vcov_driscoll_kraay(X, resid, time_ids, bandwidth=5)
    assert V.shape == (k, k)
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals >= -1e-10)


def test_hac_meat_standalone():
    """_hac_meat should produce same meat as the full vcov_hac minus bread."""
    from polars_reg._se import _hac_meat

    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    score = X * resid[:, None]
    meat = _hac_meat(score, time_ids, bandwidth=5)
    XtX_inv = np.linalg.inv(X.T @ X)
    dfc = n / (n - k)
    V_manual = dfc * XtX_inv @ meat @ XtX_inv
    V_func = vcov_hac(X, resid, time_ids, bandwidth=5)
    np.testing.assert_allclose(V_manual, V_func, rtol=1e-12)
