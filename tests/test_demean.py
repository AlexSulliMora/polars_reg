import numpy as np

from polars_reg._demean import absorbed_dof, demean, drop_singletons


def test_single_fe_demean():
    """Demeaning by one FE should subtract group means."""
    rng = np.random.default_rng(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x = rng.standard_normal(n)

    result = demean(x.reshape(-1, 1), {"g": groups})
    for g in range(10):
        mask = groups == g
        np.testing.assert_allclose(result[mask, 0].mean(), 0.0, atol=1e-12)


def test_demean_preserves_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    groups = np.repeat(np.arange(10), 10)
    result = demean(X, {"g": groups})
    assert result.shape == (100, 3)


def test_demean_1d_input():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    groups = np.repeat(np.arange(10), 10)
    result = demean(x, {"g": groups})
    assert result.ndim == 1
    assert result.shape == (100,)


def test_twoway_fe_demean():
    """Two-way demeaning should match brute-force LSDV projection."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 10, 5
    n = n_firms * n_years
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    x = rng.standard_normal(n)

    # Brute force: regress x on firm + year dummies, take residuals
    D_firm = np.eye(n_firms)[firm]
    D_year = np.eye(n_years)[year]
    D = np.column_stack([D_firm, D_year])
    proj = D @ np.linalg.lstsq(D, x, rcond=None)[0]
    expected = x - proj

    result = demean(x.reshape(-1, 1), {"firm": firm, "year": year})
    np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)


def test_threeway_fe_demean():
    """Three-way demeaning should match brute-force."""
    rng = np.random.default_rng(42)
    n = 60
    a = np.repeat(np.arange(3), 20)
    b = np.tile(np.repeat(np.arange(4), 5), 3)
    c = np.tile(np.arange(5), 12)
    x = rng.standard_normal(n)

    D_a = np.eye(3)[a]
    D_b = np.eye(4)[b]
    D_c = np.eye(5)[c]
    D = np.column_stack([D_a, D_b, D_c])
    proj = D @ np.linalg.lstsq(D, x, rcond=None)[0]
    expected = x - proj

    result = demean(x.reshape(-1, 1), {"a": a, "b": b, "c": c})
    np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)


def test_drop_singletons():
    a = np.array([99, 0, 0, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 0, 1, 0, 1, 2])
    mask = drop_singletons({"a": a, "b": b})
    assert not mask[0]  # singleton in group a=99
    assert not mask[7]  # cascading singleton: b=2 only had obs 0 and 7
    assert mask[1:7].all()  # remaining observations survive


def test_drop_singletons_no_singletons():
    a = np.array([0, 0, 1, 1, 2, 2])
    mask = drop_singletons({"a": a})
    assert mask.all()


def test_absorbed_dof_single_fe():
    codes = np.array([0, 0, 1, 1, 2, 2])
    dof = absorbed_dof({"g": codes})
    assert dof == 3


def test_absorbed_dof_twoway():
    """Two-way FE: dof = g1 + g2 - connected_components."""
    firm = np.array([0, 0, 1, 1])
    year = np.array([0, 1, 0, 1])
    # Fully connected: 1 component
    dof = absorbed_dof({"firm": firm, "year": year})
    assert dof == 2 + 2 - 1  # = 3


def test_absorbed_dof_disconnected():
    """Two separate groups that don't share any year => 2 components."""
    firm = np.array([0, 0, 1, 1])
    year = np.array([0, 1, 2, 3])
    # No shared years: 2 components
    dof = absorbed_dof({"firm": firm, "year": year})
    assert dof == 2 + 4 - 2  # = 4
