"""Tests for Arellano-Bond dynamic panel GMM estimator."""

import numpy as np
import polars as pl
import pytest

from polars_reg import panel_ab, panel_sys_gmm


@pytest.fixture
def ab_data():
    """Dynamic panel dataset with persistence."""
    rng = np.random.default_rng(42)
    n_entities, n_times = 100, 10
    rho = 0.5  # true autoregressive coefficient
    beta_x = 1.0

    y = np.zeros((n_entities, n_times))
    x = rng.standard_normal((n_entities, n_times))
    alpha = rng.standard_normal(n_entities) * 0.5  # entity FE
    e = rng.standard_normal((n_entities, n_times)) * 0.3

    # Generate AR(1) process with entity FE
    y[:, 0] = alpha + e[:, 0]
    for t in range(1, n_times):
        y[:, t] = rho * y[:, t - 1] + beta_x * x[:, t] + alpha + e[:, t]

    # Flatten to long format
    entity_id = np.repeat(np.arange(n_entities), n_times)
    time_id = np.tile(np.arange(n_times), n_entities)
    y_flat = y.ravel()
    x_flat = x.ravel()

    return pl.DataFrame({"y": y_flat, "x": x_flat, "entity": entity_id, "time": time_id})


def test_ab_basic(ab_data):
    """AB estimator runs and produces valid output."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    assert r.model_type == "Arellano-Bond"
    assert "L.y" in r.names
    assert "x" in r.names
    assert r.n_obs > 0


def test_ab_coefficient_signs(ab_data):
    """AR coefficient should be positive, x coefficient positive."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    idx_ly = r.names.index("L.y")
    idx_x = r.names.index("x")
    # AR coefficient should be positive (true rho=0.5)
    assert r.coefficients[idx_ly] > 0
    # x coefficient should be positive (true beta=1.0)
    assert r.coefficients[idx_x] > 0


def test_ab_coefficient_close(ab_data):
    """Coefficients should be reasonably close to true values."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    idx_ly = r.names.index("L.y")
    idx_x = r.names.index("x")
    # With n=100, T=10: should be within 0.3 of true values
    assert abs(r.coefficients[idx_ly] - 0.5) < 0.3
    assert abs(r.coefficients[idx_x] - 1.0) < 0.3


def test_ab_se_positive(ab_data):
    """SEs should be positive."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    assert np.all(r.se > 0)


def test_ab_sargan_test(ab_data):
    """Sargan/Hansen J test should be present."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    assert r.j_stat is not None
    assert r.j_pvalue is not None
    assert r.j_stat >= 0
    assert 0 <= r.j_pvalue <= 1


def test_ab_ar_tests(ab_data):
    """AR(1) and AR(2) test results should be available."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    ar1_stat, ar1_p = r._ar1
    ar2_stat, ar2_p = r._ar2
    assert not np.isnan(ar1_stat)
    assert not np.isnan(ar2_stat)
    assert 0 <= ar1_p <= 1
    assert 0 <= ar2_p <= 1


def test_ab_twostep(ab_data):
    """Two-step GMM should produce different (typically better) SEs."""
    r1 = panel_ab("y ~ x", data=ab_data, entity="entity", time="time", twostep=False)
    r2 = panel_ab("y ~ x", data=ab_data, entity="entity", time="time", twostep=True)
    # Both should run
    assert r1.model_type == "Arellano-Bond"
    assert r2.model_type == "Arellano-Bond"
    assert r2.vcov_type == "twostep"
    # Coefficients may differ slightly
    assert r1.n_obs == r2.n_obs


def test_ab_maxlags(ab_data):
    """Limiting maxlags reduces instruments."""
    r_all = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    r_lim = panel_ab("y ~ x", data=ab_data, entity="entity", time="time", maxlags=3)
    assert r_lim._n_instruments <= r_all._n_instruments


def test_ab_summary(ab_data):
    """Summary displays correctly."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    s = r.summary()
    assert "Arellano-Bond" in s
    assert "L.y" in s
    assert "x" in s


def test_ab_coef_table(ab_data):
    """Coef table is a valid DataFrame."""
    r = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    ct = r.coef_table()
    assert ct.shape[0] == 2  # L.y and x
    assert "coef" in ct.columns
    assert "se" in ct.columns


def test_ab_no_exog():
    """AB with no exogenous regressors (pure AR)."""
    rng = np.random.default_rng(42)
    n_e, n_t = 200, 12
    rho = 0.5
    y = np.zeros((n_e, n_t))
    alpha = rng.standard_normal(n_e) * 0.3
    e = rng.standard_normal((n_e, n_t)) * 0.2
    y[:, 0] = alpha + e[:, 0]
    for t in range(1, n_t):
        y[:, t] = rho * y[:, t - 1] + alpha + e[:, t]
    entity = np.repeat(np.arange(n_e), n_t)
    time_id = np.tile(np.arange(n_t), n_e)
    df = pl.DataFrame({"y": y.ravel(), "entity": entity, "time": time_id})
    # Formula with no exogenous vars — only the AR term
    r = panel_ab("y ~ 0", data=df, entity="entity", time="time")
    assert "L.y" in r.names
    assert r.n_obs > 0
    assert np.all(r.se > 0)


def test_ab_multiple_exog():
    """AB with multiple exogenous regressors."""
    rng = np.random.default_rng(42)
    n_e, n_t = 80, 8
    y = np.zeros((n_e, n_t))
    x1 = rng.standard_normal((n_e, n_t))
    x2 = rng.standard_normal((n_e, n_t))
    alpha = rng.standard_normal(n_e) * 0.3
    e = rng.standard_normal((n_e, n_t)) * 0.2
    y[:, 0] = alpha + e[:, 0]
    for t in range(1, n_t):
        y[:, t] = 0.4 * y[:, t - 1] + 1.0 * x1[:, t] - 0.5 * x2[:, t] + alpha + e[:, t]
    entity = np.repeat(np.arange(n_e), n_t)
    time_id = np.tile(np.arange(n_t), n_e)
    df = pl.DataFrame(
        {
            "y": y.ravel(),
            "x1": x1.ravel(),
            "x2": x2.ravel(),
            "entity": entity,
            "time": time_id,
        }
    )
    r = panel_ab("y ~ x1 + x2", data=df, entity="entity", time="time")
    assert len(r.names) == 3  # L.y, x1, x2
    assert np.all(r.se > 0)


# ── System GMM (Blundell-Bond) ──────────────────────────────────


def test_sys_gmm_basic(ab_data):
    """System GMM runs and produces valid output."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    assert r.model_type == "System GMM"
    assert "L.y" in r.names
    assert "x" in r.names
    assert r.n_obs > 0


def test_sys_gmm_coefficient_signs(ab_data):
    """AR coefficient positive, x coefficient positive."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    idx_ly = r.names.index("L.y")
    idx_x = r.names.index("x")
    assert r.coefficients[idx_ly] > 0
    assert r.coefficients[idx_x] > 0


def test_sys_gmm_more_instruments(ab_data):
    """System GMM should have more instruments than AB (additional level eq)."""
    r_ab = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    r_sys = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    assert r_sys._n_instruments > r_ab._n_instruments


def test_sys_gmm_more_obs(ab_data):
    """System GMM should have more observations (stacked diff + level)."""
    r_ab = panel_ab("y ~ x", data=ab_data, entity="entity", time="time")
    r_sys = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    assert r_sys.n_obs > r_ab.n_obs


def test_sys_gmm_se_positive(ab_data):
    """SEs should be positive."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    assert np.all(r.se > 0)


def test_sys_gmm_sargan(ab_data):
    """Sargan/Hansen J test present."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    assert r.j_stat is not None
    assert r.j_pvalue is not None
    assert r.j_stat >= 0


def test_sys_gmm_ar_tests(ab_data):
    """AR tests available."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    ar1_stat, ar1_p = r._ar1
    ar2_stat, ar2_p = r._ar2
    assert not np.isnan(ar1_stat)
    assert 0 <= ar1_p <= 1
    assert not np.isnan(ar2_stat)
    assert 0 <= ar2_p <= 1


def test_sys_gmm_twostep(ab_data):
    """Two-step system GMM runs."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time", twostep=True)
    assert r.vcov_type == "twostep"
    assert r.model_type == "System GMM"
    assert np.all(r.se > 0)


def test_sys_gmm_summary(ab_data):
    """Summary displays correctly."""
    r = panel_sys_gmm("y ~ x", data=ab_data, entity="entity", time="time")
    s = r.summary()
    assert "System GMM" in s
    assert "L.y" in s


def test_sys_gmm_no_exog():
    """System GMM with pure AR model."""
    rng = np.random.default_rng(42)
    n_e, n_t = 200, 12
    rho = 0.5
    y = np.zeros((n_e, n_t))
    alpha = rng.standard_normal(n_e) * 0.3
    e = rng.standard_normal((n_e, n_t)) * 0.2
    y[:, 0] = alpha + e[:, 0]
    for t in range(1, n_t):
        y[:, t] = rho * y[:, t - 1] + alpha + e[:, t]
    entity = np.repeat(np.arange(n_e), n_t)
    time_id = np.tile(np.arange(n_t), n_e)
    df = pl.DataFrame({"y": y.ravel(), "entity": entity, "time": time_id})
    r = panel_sys_gmm("y ~ 0", data=df, entity="entity", time="time")
    assert "L.y" in r.names
    assert r.n_obs > 0
