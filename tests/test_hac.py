"""Tests for HAC (Newey-West) and Driscoll-Kraay standard errors."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def ts_data():
    """Time series data with AR(1) errors."""
    rng = np.random.default_rng(42)
    T = 200
    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)

    # AR(1) errors with rho = 0.7
    e = np.zeros(T)
    e[0] = rng.standard_normal()
    for t in range(1, T):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal()

    y = 1.0 + 2.0 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "time": np.arange(T)})


@pytest.fixture
def panel_data():
    """Panel data with cross-sectional dependence."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 30, 50
    n = n_firms * n_years

    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(2000, 2000 + n_years), n_firms)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    # Common time shock (cross-sectional dependence)
    time_shock = rng.standard_normal(n_years) * 2.0
    firm_fe = rng.standard_normal(n_firms)

    e = rng.standard_normal(n) * 0.5
    y = 1.0 * x1 - 0.5 * x2 + firm_fe[firm_id] + time_shock[year_id - 2000] + e
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


# ── Newey-West HAC tests ────────────────────────────────────────


def test_nw_basic(ts_data):
    """Newey-West SEs should be computable without error."""
    result = pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW", time="time")
    assert result.vcov_type == "NW"
    assert result.n_obs == 200
    assert len(result.se) == 3


def test_nw_larger_than_iid(ts_data):
    """With AR(1) errors, NW SEs should be larger than iid SEs."""
    r_iid = pr.ols("y ~ x1 + x2", data=ts_data)
    r_nw = pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW", time="time")
    # NW SEs should generally be larger with autocorrelation
    # Check for intercept (most affected by persistent errors)
    idx_cons = r_nw.names.index("_cons")
    assert r_nw.se[idx_cons] > r_iid.se[idx_cons]


def test_nw_coefficients_match_ols(ts_data):
    """NW should give same coefficients as OLS (only SEs differ)."""
    r_iid = pr.ols("y ~ x1 + x2", data=ts_data)
    r_nw = pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW", time="time")
    np.testing.assert_allclose(r_nw.coefficients, r_iid.coefficients, rtol=1e-10)


def test_nw_explicit_bandwidth(ts_data):
    """Explicit bandwidth should work."""
    r1 = pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW", time="time", bandwidth=3)
    r2 = pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW", time="time", bandwidth=10)
    # More lags = different (generally larger) SEs with autocorrelation
    assert not np.allclose(r1.se, r2.se)


def test_nw_requires_time(ts_data):
    """NW without time= should raise an error."""
    with pytest.raises(ValueError, match="requires time"):
        pr.ols("y ~ x1 + x2", data=ts_data, vcov="NW")


def test_nw_with_fe(panel_data):
    """NW should work with absorbed fixed effects."""
    result = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert result.fe_absorbed == ["firm_id"]


# ── Driscoll-Kraay tests ────────────────────────────────────────


def test_dk_basic(panel_data):
    """Driscoll-Kraay SEs should be computable without error."""
    result = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
    assert len(result.se) == 2  # x1, x2 (no intercept with FE)


def test_dk_coefficients_match_ols(panel_data):
    """DK should give same coefficients as clustered (only SEs differ)."""
    r_cl = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        cluster=["firm_id"],
    )
    r_dk = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="DK",
        time="year_id",
    )
    np.testing.assert_allclose(r_dk.coefficients, r_cl.coefficients, rtol=1e-10)


def test_dk_differs_from_cluster(panel_data):
    """DK SEs should differ from cluster SEs (different assumptions)."""
    r_cl = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        cluster=["firm_id"],
    )
    r_dk = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="DK",
        time="year_id",
    )
    # SEs should differ
    assert not np.allclose(r_cl.se, r_dk.se, rtol=0.01)


def test_dk_requires_time(panel_data):
    """DK without time= should raise an error."""
    with pytest.raises(ValueError, match="requires time"):
        pr.ols("y ~ x1 + x2 | firm_id", data=panel_data, vcov="DK")


def test_dk_explicit_bandwidth(panel_data):
    """Explicit bandwidth for DK."""
    r1 = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="DK",
        time="year_id",
        bandwidth=2,
    )
    r2 = pr.ols(
        "y ~ x1 + x2 | firm_id",
        data=panel_data,
        vcov="DK",
        time="year_id",
        bandwidth=10,
    )
    assert not np.allclose(r1.se, r2.se)


# ── Panel FE with HAC/DK ────────────────────────────────────────


def test_panel_fe_nw(panel_data):
    """panel_fe should support NW SEs."""
    result = pr.panel_fe(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        time="year_id",
        vcov="NW",
    )
    assert result.vcov_type == "NW"
    assert len(result.se) == 2


def test_panel_fe_dk(panel_data):
    """panel_fe should support DK SEs."""
    result = pr.panel_fe(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        time="year_id",
        vcov="DK",
    )
    assert result.vcov_type == "DK"
    assert len(result.se) == 2


def test_panel_fe_dk_coeffs_match(panel_data):
    """DK coefficients should match cluster coefficients."""
    r_cl = pr.panel_fe(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        time="year_id",
    )
    r_dk = pr.panel_fe(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        time="year_id",
        vcov="DK",
    )
    np.testing.assert_allclose(r_dk.coefficients, r_cl.coefficients, rtol=1e-10)


def test_panel_fe_nw_requires_time(panel_data):
    """panel_fe with NW but no time should fail."""
    with pytest.raises(ValueError, match="requires time"):
        pr.panel_fe(
            "y ~ x1 + x2",
            data=panel_data,
            entity="firm_id",
            vcov="NW",
        )
