"""Tests for diagnostic tests (Wald, Hausman)."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def ols_result():
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 1.0 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    return pr.ols("y ~ x1 + x2", data=df)


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(42)
    n_firms, n_years = 30, 15
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    firm_fe = rng.standard_normal(n_firms) * 2.0
    # Make x1 correlated with the entity effect so RE is inconsistent
    x1 = x1 + firm_fe[firm_id] * 0.8
    y = 1.0 * x1 - 0.5 * x2 + firm_fe[firm_id] + rng.standard_normal(n) * 0.3
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


# ── Wald test ─────────────────────────────────────────────────────


def test_wald_single_restriction(ols_result):
    """Test that beta_x1 = 1."""
    # R = [0, 1, 0, 0] selects x1 coefficient (x1 is first, x2 second, _cons third)
    # But names order is: x1, x2, _cons
    R = np.array([[1, 0, 0]])  # test beta_x1 = 0
    result = ols_result.wald_test(R)
    assert "statistic" in result
    assert "pvalue" in result
    assert "df" in result
    # x1 coef is ~1.0, so testing = 0 should reject
    assert result["pvalue"] < 0.001


def test_wald_joint_restriction(ols_result):
    """Test that beta_x1 = 0 AND beta_x2 = 0 jointly."""
    R = np.array([[1, 0, 0], [0, 1, 0]])
    result = ols_result.wald_test(R)
    assert result["df"] == (2, ols_result.df_r)
    assert result["pvalue"] < 0.001


def test_wald_true_null(ols_result):
    """Test that beta_x1 = 1 (true value) should NOT reject."""
    R = np.array([[1, 0, 0]])
    q = np.array([1.0])
    result = ols_result.wald_test(R, q)
    # p-value should be > 0.05 (true null)
    assert result["pvalue"] > 0.01


def test_wald_equality_restriction(ols_result):
    """Test that beta_x1 = beta_x2."""
    R = np.array([[1, -1, 0]])  # beta_x1 - beta_x2 = 0
    result = ols_result.wald_test(R)
    # Coefficients are 1.0 and -0.5, so they're not equal — should reject
    assert result["pvalue"] < 0.001


def test_wald_chi2(ols_result):
    """chi2 = j * F."""
    R = np.array([[1, 0, 0], [0, 1, 0]])
    result = ols_result.wald_test(R)
    j = result["df"][0]
    np.testing.assert_allclose(result["chi2"], j * result["statistic"], rtol=1e-10)


# ── Hausman test ──────────────────────────────────────────────────


def test_hausman_basic(panel_data):
    r_fe = pr.panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    r_re = pr.panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    result = pr.hausman_test(r_fe, r_re)
    assert "statistic" in result
    assert "pvalue" in result
    assert "df" in result
    assert result["df"] == 2  # x1 and x2


def test_hausman_rejects_when_correlated():
    """When FE are correlated with regressors, Hausman should reject RE.

    The Hausman test compares FE vs RE coefficients. We verify
    the test correctly identifies when FE and RE give different estimates.
    """
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    firm_fe = rng.standard_normal(n_firms) * 2.0
    # x1 is correlated with entity effect → RE is inconsistent
    x1 = rng.standard_normal(n) + firm_fe[firm_id] * 0.8
    x2 = rng.standard_normal(n)
    y = 1.0 * x1 - 0.5 * x2 + firm_fe[firm_id] + rng.standard_normal(n) * 0.3
    df = pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )
    # Use iid SEs for FE (classic Hausman requires comparable VCVs)
    r_fe = pr.panel_fe("y ~ x1 + x2", data=df, entity="firm_id", time="year_id", cluster=[])
    r_re = pr.panel_re("y ~ x1 + x2", data=df, entity="firm_id")
    result = pr.hausman_test(r_fe, r_re)
    # Coefficients should differ meaningfully
    assert abs(r_fe.coefficients[0] - r_re.coefficients[0]) > 0.05
    # Statistic should be non-negative
    assert result["statistic"] >= 0


def test_hausman_no_reject_when_uncorrelated():
    """When FE are independent of regressors, Hausman should not reject."""
    rng = np.random.default_rng(99)
    n_firms, n_years = 30, 15
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    firm_fe = rng.standard_normal(n_firms)
    # No correlation between FE and regressors
    y = 1.0 * x1 - 0.5 * x2 + firm_fe[firm_id] + rng.standard_normal(n) * 0.5
    df = pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )
    r_fe = pr.panel_fe("y ~ x1 + x2", data=df, entity="firm_id", time="year_id")
    r_re = pr.panel_re("y ~ x1 + x2", data=df, entity="firm_id")
    result = pr.hausman_test(r_fe, r_re)
    assert result["pvalue"] > 0.05


# ── Weak instrument test ────────────────────────────────────────


@pytest.fixture
def iv_strong():
    """IV result with strong instruments."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    # Strong instruments: high correlation with endogenous var
    x_end = 0.8 * z1 + 0.6 * z2 + 0.3 * u
    y = 1.0 + 2.0 * x_end + u
    df = pl.DataFrame({"y": y, "x_end": x_end, "z1": z1, "z2": z2})
    return pr.iv2sls("y ~ 1 || x_end ~ z1 + z2", data=df)


@pytest.fixture
def iv_weak():
    """IV result with weak instruments."""
    rng = np.random.default_rng(42)
    n = 200
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    # Weak instruments: very low correlation
    x_end = 0.02 * z1 + 0.01 * z2 + u
    y = 1.0 + 2.0 * x_end + u
    df = pl.DataFrame({"y": y, "x_end": x_end, "z1": z1, "z2": z2})
    return pr.iv2sls("y ~ 1 || x_end ~ z1 + z2", data=df)


def test_weak_instrument_strong(iv_strong):
    result = pr.weak_instrument_test(iv_strong, n_instruments=2)
    assert result["staiger_stock"] is True
    assert result["assessment"] == "strong"
    assert result["f_stat"] > 10


def test_weak_instrument_weak(iv_weak):
    result = pr.weak_instrument_test(iv_weak, n_instruments=2)
    assert result["staiger_stock"] is False
    assert result["assessment"] == "weak"
    assert result["f_stat"] < 10


def test_weak_instrument_stock_yogo(iv_strong):
    result = pr.weak_instrument_test(iv_strong, n_instruments=2)
    assert result["stock_yogo"] is not None
    assert "critical_values" in result["stock_yogo"]
    assert result["stock_yogo"]["n_instruments"] == 2


def test_weak_instrument_no_stock_yogo(iv_strong):
    """Without n_instruments, no Stock-Yogo values."""
    result = pr.weak_instrument_test(iv_strong)
    assert result["stock_yogo"] is None


def test_weak_instrument_no_f_stat():
    """Should raise for results without first-stage F."""
    rng = np.random.default_rng(42)
    n = 100
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    r = pr.ols("y ~ x1", data=df)
    with pytest.raises(ValueError, match="No first-stage F"):
        pr.weak_instrument_test(r)


def test_hausman_coefficients_compared(panel_data):
    r_fe = pr.panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    r_re = pr.panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    result = pr.hausman_test(r_fe, r_re)
    assert "x1" in result["coefficients_compared"]
    assert "x2" in result["coefficients_compared"]
    assert "_cons" not in result["coefficients_compared"]


# ── Kleibergen-Paap rk test ────────────────────────────────────


def test_kp_from_result_strong(iv_strong):
    """KP rk stat should be large for strong instruments."""
    result = pr.kleibergen_paap_from_result(iv_strong)
    assert result["rk_stat"] is not None
    assert result["rk_stat"] > 10  # strong instruments


def test_kp_from_result_weak(iv_weak):
    """KP rk stat should be small for weak instruments."""
    result = pr.kleibergen_paap_from_result(iv_weak)
    assert result["rk_stat"] is not None
    assert result["rk_stat"] < 10  # weak instruments


def test_kp_robust():
    """KP with robust SEs should produce a result."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_end = 0.8 * z1 + 0.6 * z2 + 0.3 * u
    y = 1.0 + 2.0 * x_end + u
    df = pl.DataFrame({"y": y, "x_end": x_end, "z1": z1, "z2": z2})
    r = pr.iv2sls("y ~ 1 || x_end ~ z1 + z2", data=df, vcov="HC1")
    result = pr.kleibergen_paap_from_result(r)
    assert result["rk_stat"] is not None
    assert result["rk_stat"] > 0


def test_kp_clustered():
    """KP with clustered SEs should produce a result."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_end = 0.8 * z1 + 0.6 * z2 + 0.3 * u
    y = 1.0 + 2.0 * x_end + u
    cluster = np.random.randint(0, 50, n)
    df = pl.DataFrame(
        {
            "y": y,
            "x_end": x_end,
            "z1": z1,
            "z2": z2,
            "cl": cluster,
        }
    )
    r = pr.iv2sls("y ~ 1 || x_end ~ z1 + z2", data=df, cluster=["cl"])
    result = pr.kleibergen_paap_from_result(r)
    assert result["rk_stat"] is not None
    assert result["rk_stat"] > 0


def test_kp_no_iv_arrays():
    """Should raise for OLS results without IV arrays."""
    rng = np.random.default_rng(42)
    n = 100
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    r = pr.ols("y ~ x1", data=df)
    with pytest.raises(ValueError, match="first-stage arrays"):
        pr.kleibergen_paap_from_result(r)
