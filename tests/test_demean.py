import numpy as np

from polars_reg._demean import absorbed_dof, demean, drop_singletons, reindex_fe_codes


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


def test_reindex_fe_codes():
    """reindex_fe_codes maps non-contiguous codes to 0..N-1."""
    fe_dict = {"a": np.array([0, 3, 7, 3, 7]), "b": np.array([0, 2, 5, 2, 5])}
    result = reindex_fe_codes(fe_dict)
    for codes in result.values():
        unique = np.unique(codes)
        assert np.array_equal(unique, np.arange(len(unique)))


def test_demean_after_singleton_drop():
    """Demeaning must produce finite results after singleton removal leaves non-contiguous codes."""
    rng = np.random.default_rng(42)
    n = 200
    fe_a = rng.integers(0, 10, size=n)
    fe_b = rng.integers(0, 8, size=n)
    # Force singletons
    fe_a[0] = 99
    fe_b[1] = 88

    fe_dict = {"a": fe_a, "b": fe_b}
    keep = drop_singletons(fe_dict)
    fe_filtered = {k: v[keep] for k, v in fe_dict.items()}
    fe_reindexed = reindex_fe_codes(fe_filtered)

    X = rng.standard_normal((keep.sum(), 3))
    result = demean(X, fe_reindexed)
    assert np.all(np.isfinite(result))


def test_demean_empty_array():
    """demean with empty arrays returns empty."""
    result = demean(np.empty((0, 3)), {"g": np.array([], dtype=np.int64)})
    assert result.shape == (0, 3)


def test_demean_single_observation():
    """demean with 1 obs should return zeros."""
    X = np.array([[5.0, 3.0, 1.0]])
    result = demean(X, {"g": np.array([0])})
    np.testing.assert_allclose(result, np.zeros((1, 3)), atol=1e-14)


def test_demean_constant_column():
    """Column with zero variance demeaned correctly (all zeros)."""
    rng = np.random.default_rng(42)
    n = 50
    groups = np.repeat(np.arange(5), 10)
    X = np.column_stack([rng.standard_normal(n), np.ones(n) * 7.0])
    result = demean(X, {"g": groups})
    np.testing.assert_allclose(result[:, 1], np.zeros(n), atol=1e-12)


def test_demean_extreme_values():
    """Values near 1e15 produce finite results."""
    rng = np.random.default_rng(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    X = rng.standard_normal((n, 2)) * 1e15
    result = demean(X, {"g": groups})
    assert np.all(np.isfinite(result))


def test_demean_many_fe_levels():
    """500 FE levels with N=1000."""
    rng = np.random.default_rng(42)
    n = 1000
    groups = rng.integers(0, 500, size=n)
    X = rng.standard_normal((n, 2))
    result = demean(X, {"g": groups})
    assert result.shape == (n, 2)
    assert np.all(np.isfinite(result))


def test_demean_weighted_singletons():
    """Weighted demeaning with singletons dropped and reindexed."""
    rng = np.random.default_rng(42)
    n = 100
    fe_a = rng.integers(0, 10, size=n)
    fe_a[0] = 99  # create singleton
    fe_dict = {"a": fe_a}
    keep = drop_singletons(fe_dict)
    fe_filtered = {k: v[keep] for k, v in fe_dict.items()}
    fe_reindexed = reindex_fe_codes(fe_filtered)
    n_kept = keep.sum()
    X = rng.standard_normal((n_kept, 2))
    weights = rng.uniform(0.5, 2.0, size=n_kept)
    result = demean(X, fe_reindexed, weights=weights)
    assert result.shape == (n_kept, 2)
    assert np.all(np.isfinite(result))


def test_absorbed_dof_three_way():
    """3 FE dimensions, verify against brute-force LSDV DoF."""
    a = np.repeat(np.arange(3), 20)
    b = np.tile(np.repeat(np.arange(4), 5), 3)
    c = np.tile(np.arange(5), 12)

    dof = absorbed_dof({"a": a, "b": b, "c": c})

    # Brute-force: build full LSDV dummy matrix, compute rank
    D_a = np.eye(3)[a]
    D_b = np.eye(4)[b]
    D_c = np.eye(5)[c]
    D = np.column_stack([D_a, D_b, D_c])
    lsdv_rank = np.linalg.matrix_rank(D)
    assert dof == lsdv_rank


def test_absorbed_dof_empty():
    """absorbed_dof({}) returns 0."""
    assert absorbed_dof({}) == 0


def test_drop_singletons_cascading():
    """Chain reaction: dropping one singleton creates another."""
    # Group a: [0, 0, 1, 1, 2]  -- a=2 is singleton
    # Group b: [0, 1, 0, 1, 1]  -- after a=2 dropped, b still has {0,1}
    # But design a different case where cascade happens:
    # a: [0, 1, 2, 2, 3]
    # b: [0, 0, 1, 1, 0]
    # Initially a=0 is singleton -> drop obs 0
    # After drop: a=[1,2,2,3], b=[0,1,1,0]
    # Now a=1 is singleton -> drop obs 1
    # After drop: a=[2,2,3], b=[1,1,0]
    # Now a=3 is singleton -> drop obs 4
    # After drop: a=[2,2], b=[1,1]
    # Now b=1 still has 2 obs -> done
    a = np.array([0, 1, 2, 2, 3])
    b = np.array([0, 0, 1, 1, 0])
    mask = drop_singletons({"a": a, "b": b})
    # Only obs 2 and 3 survive (a=2, b=1)
    assert mask.sum() == 2
    assert mask[2] and mask[3]


def test_drop_singletons_all_removed():
    """All obs are singletons -> all False mask."""
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 1, 2, 3])
    mask = drop_singletons({"a": a, "b": b})
    assert not mask.any()


def test_reindex_fe_codes_already_contiguous():
    """[0,1,2,1,0] stays [0,1,2,1,0]."""
    codes = np.array([0, 1, 2, 1, 0])
    result = reindex_fe_codes({"g": codes})
    np.testing.assert_array_equal(result["g"], codes)


def test_reindex_fe_codes_empty():
    """Empty dict -> empty dict."""
    result = reindex_fe_codes({})
    assert result == {}
