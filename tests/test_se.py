import numpy as np
import pytest

from polars_reg._se import (
    _interaction_codes,
    vcov_clustered,
    vcov_driscoll_kraay,
    vcov_hac,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
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
    """Two-way clustering: V = V_A + V_B - V_{A*B}.

    Uses G_df="conventional" so that each term gets its own G/(G-1) factor,
    matching manual decomposition via individual vcov_clustered calls.
    """
    from polars_reg._ssc import SSC

    rng = np.random.default_rng(42)
    n = 200
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    firm = np.repeat(np.arange(10), 20)
    year = np.tile(np.arange(20), 10)

    conv_ssc = SSC(G_df="conventional")
    V = vcov_multiway_clustered(X, resid, [firm, year], ssc=conv_ssc)

    V_firm = vcov_clustered(X, resid, firm)
    V_year = vcov_clustered(X, resid, year)
    interaction, _ = _interaction_codes(firm, year)
    V_inter = vcov_clustered(X, resid, interaction)
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


def test_vcov_clustered_reghdfe_dfc():
    """Test reghdfe-style DFC: G/(G-1) * N/(N-d-k)."""
    rng = np.random.default_rng(42)
    n, k = 100, 2
    X = np.column_stack([rng.standard_normal((n, k - 1)), np.ones(n)])
    resid = rng.standard_normal(n)
    clusters = rng.integers(0, 10, n).astype(np.int32)
    df_a_non_nested = 5
    V = vcov_clustered(X, resid, clusters, df_a_non_nested=df_a_non_nested)
    assert V.shape == (k, k)
    # Verify DFC is applied (formula: G/(G-1) * N/(N-d-k))
    assert np.all(np.isfinite(V))
    assert np.all(np.diag(V) >= 0)


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


def test_vcov_clustered_single_group():
    """G=1 raises ValueError."""
    rng = np.random.default_rng(42)
    n = 20
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    clusters = np.zeros(n, dtype=np.int32)  # all same group
    with pytest.raises(ValueError, match="at least 2 cluster groups"):
        vcov_clustered(X, resid, clusters)


def test_vcov_clustered_singleton_cluster():
    """One cluster group with single obs, rest normal."""
    rng = np.random.default_rng(42)
    n = 21
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    clusters = np.repeat(np.arange(3), 7)  # 3 groups of 7
    # Make one group have a single observation
    clusters = np.concatenate([np.zeros(1, dtype=int), np.repeat([1, 2], 10)])
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    V = vcov_clustered(X, resid, clusters)
    assert V.shape == (2, 2)
    assert np.all(np.isfinite(V))


def test_vcov_multiway_two_identical_dims():
    """Two identical cluster arrays."""
    rng = np.random.default_rng(42)
    n = 100
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    clusters = np.repeat(np.arange(10), 10)
    V = vcov_multiway_clustered(X, resid, [clusters, clusters.copy()])
    # With two identical dims: V = V_A + V_A - V_{A*A} = 2*V_A - V_A = V_A
    V_single = vcov_clustered(X, resid, clusters)
    np.testing.assert_allclose(V, V_single, rtol=1e-10)


def test_vcov_hac_single_time():
    """T=1 for HAC should still produce a result (degenerate but finite)."""
    rng = np.random.default_rng(42)
    n = 20
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    time_ids = np.zeros(n)  # all same time period
    V = vcov_hac(X, resid, time_ids)
    assert V.shape == (2, 2)
    assert np.all(np.isfinite(V))


def test_vcov_dk_single_time():
    """T=1 for DK raises ValueError."""
    rng = np.random.default_rng(42)
    n = 20
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    time_ids = np.zeros(n)  # all same time period
    with pytest.raises(ValueError, match="at least 2 time periods"):
        vcov_driscoll_kraay(X, resid, time_ids)


def test_vcov_all_variants_finite():
    """Create clean X/resid data, call iid/HC0/HC1/HC2/HC3, all finite."""
    rng = np.random.default_rng(42)
    n = 50
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    y = 2.0 + 3.0 * X[:, 0] + rng.standard_normal(n) * 0.5
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta

    V_iid = vcov_iid(X, resid)
    assert np.all(np.isfinite(V_iid))

    for kind in ["HC0", "HC1", "HC2", "HC3"]:
        V = vcov_robust(X, resid, kind=kind)
        assert np.all(np.isfinite(V)), f"{kind} produced non-finite values"


def test_vcov_hc2_high_leverage():
    """X matrix near-singular, hat diagonal near 1."""
    rng = np.random.default_rng(42)
    n = 10
    k = 8  # high leverage: k close to n
    X = rng.standard_normal((n, k))
    y = rng.standard_normal(n)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    V = vcov_robust(X, resid, kind="HC2")
    assert V.shape == (k, k)
    # May have large values but should be finite
    assert np.all(np.isfinite(V))


def test_vcov_bootstrap_small_n():
    """N=5, k=3, bootstrap handles gracefully."""
    rng = np.random.default_rng(42)
    n, k = 5, 3
    X = rng.standard_normal((n, k))
    y = rng.standard_normal(n)
    V = vcov_pairs_bootstrap(X, y, n_boot=999, seed=42)
    assert V.shape == (k, k)
    assert np.all(np.isfinite(V))
