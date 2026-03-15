"""Tests for GRS (Gibbons-Ross-Shanken 1989) F-test."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def grs_panel_nonzero():
    """Balanced panel with nonzero alphas (GRS should reject)."""
    rng = np.random.default_rng(42)
    N, T, K = 5, 120, 3

    # Factor moments
    mu_f = np.array([0.005, 0.003, 0.002])
    Omega_f = np.array(
        [
            [0.004, 0.001, 0.0005],
            [0.001, 0.003, 0.0003],
            [0.0005, 0.0003, 0.002],
        ]
    )
    L_f = np.linalg.cholesky(Omega_f)

    # Nonzero alphas
    true_alphas = np.array([0.003, -0.004, 0.005, -0.002, 0.006])
    betas = rng.standard_normal((N, K)) * 0.5 + 1.0

    # Residual covariance
    Sigma_e = np.eye(N) * 0.001
    L_e = np.linalg.cholesky(Sigma_e)

    rows = []
    for t in range(T):
        f_t = mu_f + L_f @ rng.standard_normal(K)
        e_t = L_e @ rng.standard_normal(N)
        for i in range(N):
            r = true_alphas[i] + betas[i] @ f_t + e_t[i]
            rows.append(
                {
                    "date": t,
                    "portfolio": f"port_{i:02d}",
                    "ret": r,
                    "f1": f_t[0],
                    "f2": f_t[1],
                    "f3": f_t[2],
                }
            )

    return pl.DataFrame(rows), true_alphas


@pytest.fixture
def grs_panel_zero():
    """Balanced panel with zero alphas (GRS should NOT reject)."""
    rng = np.random.default_rng(99)
    N, T, K = 5, 120, 3

    mu_f = np.array([0.005, 0.003, 0.002])
    Omega_f = np.array(
        [
            [0.004, 0.001, 0.0005],
            [0.001, 0.003, 0.0003],
            [0.0005, 0.0003, 0.002],
        ]
    )
    L_f = np.linalg.cholesky(Omega_f)

    betas = rng.standard_normal((N, K)) * 0.5 + 1.0
    Sigma_e = np.eye(N) * 0.001
    L_e = np.linalg.cholesky(Sigma_e)

    rows = []
    for t in range(T):
        f_t = mu_f + L_f @ rng.standard_normal(K)
        e_t = L_e @ rng.standard_normal(N)
        for i in range(N):
            r = betas[i] @ f_t + e_t[i]  # alpha = 0
            rows.append(
                {
                    "date": t,
                    "portfolio": f"port_{i:02d}",
                    "ret": r,
                    "f1": f_t[0],
                    "f2": f_t[1],
                    "f3": f_t[2],
                }
            )

    return pl.DataFrame(rows)


# ── Core tests ─────────────────────────────────────────────────────


def test_grs_rejects_nonzero_alphas(grs_panel_nonzero):
    """GRS should reject when alphas are nonzero."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    assert result.pvalue < 0.05
    assert result.statistic > 0


def test_grs_accepts_zero_alphas(grs_panel_zero):
    """GRS should fail to reject when alphas are zero."""
    df = grs_panel_zero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    assert result.pvalue > 0.05


def test_grs_from_group_matches_raw(grs_panel_nonzero):
    """Both API paths should produce identical results."""
    df, _ = grs_panel_nonzero
    formula = "ret ~ f1 + f2 + f3"

    # Raw data path
    raw = pr.grs_test(formula, data=df, assets="portfolio", time="date")

    # Group path
    grp = pr.groupby_reg(pr.ols, formula, data=df, group_by="portfolio")
    from_grp = pr.grs_test_from_group(grp, formula, data=df, assets="portfolio", time="date")

    np.testing.assert_allclose(raw.statistic, from_grp.statistic, rtol=1e-10)
    np.testing.assert_allclose(raw.pvalue, from_grp.pvalue, rtol=1e-10)
    np.testing.assert_allclose(raw.alphas, from_grp.alphas, rtol=1e-10)
    np.testing.assert_allclose(raw.wald_statistic, from_grp.wald_statistic, rtol=1e-10)


def test_grs_wald_statistic(grs_panel_nonzero):
    """Wald chi2 should equal N * GRS F."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    np.testing.assert_allclose(
        result.wald_statistic, result.n_assets * result.statistic, rtol=1e-10
    )


def test_grs_sharpe_ratios(grs_panel_nonzero):
    """Sharpe decomposition: tangency = alpha'Sigma_inv alpha + factors."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    # Verify decomposition identity
    Sigma_inv = np.linalg.inv(result.sigma)
    alpha_quad = float(result.alphas @ Sigma_inv @ result.alphas)
    np.testing.assert_allclose(
        result.sharpe_sq_tangency,
        alpha_quad + result.sharpe_sq_factors,
        rtol=1e-10,
    )


# ── Edge cases ─────────────────────────────────────────────────────


def test_grs_unbalanced_panel_error():
    """Should raise ValueError on unbalanced panel."""
    df = pl.DataFrame(
        {
            "date": [1, 2, 3, 1, 2],
            "asset": ["A", "A", "A", "B", "B"],
            "ret": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f1": [0.01, 0.02, 0.03, 0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="[Uu]nbalanced"):
        pr.grs_test("ret ~ f1", data=df, assets="asset", time="date")


def test_grs_t_leq_n_plus_k_error():
    """Should raise when T <= N + K."""
    rng = np.random.default_rng(42)
    N, T, _K = 10, 10, 3  # T = N + K - 3, so T <= N+K
    rows = []
    for t in range(T):
        for i in range(N):
            rows.append(
                {
                    "date": t,
                    "asset": f"a{i}",
                    "ret": rng.standard_normal(),
                    "f1": rng.standard_normal(),
                    "f2": rng.standard_normal(),
                    "f3": rng.standard_normal(),
                }
            )
    df = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="T > N \\+ K"):
        pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="asset", time="date")


def test_grs_no_intercept_error():
    """Should raise when formula has -1 (no intercept)."""
    df = pl.DataFrame(
        {
            "date": [1, 2, 1, 2],
            "asset": ["A", "A", "B", "B"],
            "ret": [0.1, 0.2, 0.3, 0.4],
            "f1": [0.01, 0.02, 0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="intercept"):
        pr.grs_test("ret ~ f1 - 1", data=df, assets="asset", time="date")


def test_grs_single_factor(grs_panel_nonzero):
    """K=1 (CAPM-style) should work correctly."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1", data=df, assets="portfolio", time="date")
    assert result.n_factors == 1
    assert result.statistic > 0
    assert 0 <= result.pvalue <= 1
    assert result.df == (5, 120 - 5 - 1)


def test_grs_summary_format(grs_panel_nonzero):
    """summary() should return a non-empty formatted string."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    s = result.summary()
    assert len(s) > 0
    assert "GRS" in s
    assert "F-statistic" in s
    assert "Wald" in s
    assert "Sharpe" in s


def test_grs_alpha_table_columns(grs_panel_nonzero):
    """alpha_table() should have correct schema."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    tbl = result.alpha_table()
    assert isinstance(tbl, pl.DataFrame)
    assert tbl.columns == ["asset", "alpha", "se", "t", "p"]
    assert len(tbl) == 5


def test_grs_lazyframe_input(grs_panel_nonzero):
    """LazyFrame should be accepted and produce correct results."""
    df, _ = grs_panel_nonzero
    lazy = df.lazy()
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=lazy, assets="portfolio", time="date")
    assert result.statistic > 0
    assert result.n_assets == 5


def test_grs_rejects_pandas(grs_panel_nonzero):
    """pandas DataFrame should be rejected with helpful message."""
    pytest.importorskip("pandas")
    df, _ = grs_panel_nonzero
    pdf = df.to_pandas()
    with pytest.raises(TypeError, match="pl.from_pandas"):
        pr.grs_test("ret ~ f1 + f2 + f3", data=pdf, assets="portfolio", time="date")


def test_grs_failed_groups_error(grs_panel_nonzero):
    """Should raise if GroupRegressionResult has failures."""
    from polars_reg._groupby import GroupRegressionResult

    grp = GroupRegressionResult()
    grp.failed["bad_group"] = "some error"
    with pytest.raises(ValueError, match="failed"):
        pr.grs_test_from_group(grp, "ret ~ f1", data=pl.DataFrame(), assets="a", time="t")


def test_grs_intercept_only_model():
    """K=0 (no factors, intercept only) should work — tests mean returns jointly zero."""
    rng = np.random.default_rng(42)
    N, T = 3, 50
    rows = []
    # Generate assets with nonzero means
    means = [0.01, -0.02, 0.03]
    for t in range(T):
        for i in range(N):
            rows.append(
                {
                    "date": t,
                    "asset": f"a{i}",
                    "ret": means[i] + rng.standard_normal() * 0.05,
                }
            )
    df = pl.DataFrame(rows)
    result = pr.grs_test("ret ~ 1", data=df, assets="asset", time="date")
    assert result.n_factors == 0
    assert result.statistic > 0
    assert result.df == (3, 50 - 3 - 0)


# ── Analytical validation ──────────────────────────────────────────


def test_grs_analytical_small_example():
    """Verify GRS against hand-computed values for a tiny example.

    N=2, T=50, K=1. We compute the statistic manually and compare.
    """
    rng = np.random.default_rng(123)
    N, T, K = 2, 50, 1

    # Generate factor
    f = rng.standard_normal(T) * 0.02 + 0.005

    # Generate returns with known betas and alphas
    betas = np.array([1.2, 0.8])
    true_alpha = np.array([0.005, -0.003])
    eps = rng.standard_normal((T, N)) * 0.03

    rows = []
    for t in range(T):
        for i in range(N):
            r = true_alpha[i] + betas[i] * f[t] + eps[t, i]
            rows.append({"date": t, "asset": f"a{i}", "ret": r, "f1": f[t]})

    df = pl.DataFrame(rows)
    result = pr.grs_test("ret ~ f1", data=df, assets="asset", time="date")

    # Manually compute GRS
    # Run OLS per asset
    ones = np.ones(T)
    X = np.column_stack([f, ones])
    residuals = np.zeros((T, N))
    alphas_manual = np.zeros(N)
    for i in range(N):
        y_i = np.array([true_alpha[i] + betas[i] * f[t] + eps[t, i] for t in range(T)])
        b = np.linalg.lstsq(X, y_i, rcond=None)[0]
        alphas_manual[i] = b[1]  # intercept
        residuals[:, i] = y_i - X @ b

    Sigma_hat = (1.0 / (T - K - 1)) * residuals.T @ residuals
    Sigma_inv = np.linalg.inv(Sigma_hat)

    mu_f = f.mean()
    Omega_tilde = (1.0 / T) * np.sum((f - mu_f) ** 2)
    mu_Omega_inv_mu = mu_f**2 / Omega_tilde

    a_S_a = alphas_manual @ Sigma_inv @ alphas_manual
    grs_manual = (T / N) * ((T - N - K) / (T - K - 1)) * a_S_a / (1.0 + mu_Omega_inv_mu)

    np.testing.assert_allclose(result.statistic, grs_manual, rtol=1e-10)
    np.testing.assert_allclose(result.alphas, alphas_manual, rtol=1e-10)


def test_grs_dimensions(grs_panel_nonzero):
    """Check dimensions of stored matrices."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    assert result.sigma.shape == (5, 5)
    assert result.factor_means.shape == (3,)
    assert result.factor_cov.shape == (3, 3)
    assert result.alphas.shape == (5,)
    assert result.alpha_se.shape == (5,)
    assert len(result.alpha_names) == 5


def test_grs_repr(grs_panel_nonzero):
    """repr() should return the summary string."""
    df, _ = grs_panel_nonzero
    result = pr.grs_test("ret ~ f1 + f2 + f3", data=df, assets="portfolio", time="date")
    assert repr(result) == result.summary()
