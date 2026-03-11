import numpy as np
import polars as pl

from polars_reg._ols import ols
from polars_reg._panel import panel_fd, panel_fe, panel_re


def test_panel_fe_basic(panel_data):
    """Panel FE should recover coefficients."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert result.model_type == "Panel FE"
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.15)  # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.15)  # x2


def test_panel_fe_default_cluster(panel_data):
    """Should default to clustering by entity."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_panel_fe_matches_ols_fe(panel_data):
    """Panel FE should give same coefficients as OLS with absorbed FE."""
    # Note: only coefficients are compared. SEs differ because panel_fe and ols
    # use different degrees-of-freedom corrections (panel_fe uses reg-style,
    # ols with FE uses reghdfe-style dfc).
    panel_result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    ols_result = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data, cluster=["firm_id"])
    np.testing.assert_allclose(panel_result.coefficients, ols_result.coefficients, rtol=1e-6)


def test_panel_fe_entity_only(panel_data):
    """Should work with entity FE only (no time FE)."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.fe_absorbed == ["firm_id"]
    assert result.n_obs == 1000


def test_panel_fe_summary(panel_data):
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    s = result.summary()
    assert "Panel FE" in s
    assert "Absorbed FE" in s


# ── panel_re tests ──────────────────────────────────────────────────


def test_panel_re_basic(panel_data):
    """Panel RE should recover reasonable coefficients."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.model_type == "Panel RE"
    assert result.vcov_type == "iid"
    # RE coefficients should be close to FE coefficients for this DGP
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.15)  # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.15)  # x2


def test_panel_re_has_intercept(panel_data):
    """RE includes an intercept (unlike FE)."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert "_cons" in result.names


def test_panel_re_nobs(panel_data):
    """RE should use all observations."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.n_obs == 1000


def test_panel_re_vs_fe(panel_data):
    """RE and FE slope coefficients should be similar (no correlation in DGP)."""
    re_result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    fe_result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id")
    # Slope coefficients (exclude intercept from RE)
    re_slopes = re_result.coefficients[:2]
    fe_slopes = fe_result.coefficients[:2]
    np.testing.assert_allclose(re_slopes, fe_slopes, atol=0.1)


def test_panel_re_summary(panel_data):
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    s = result.summary()
    assert "Panel RE" in s


def test_panel_re_cluster(panel_data):
    """RE with clustered SEs by entity."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_panel_re_robust(panel_data):
    """RE with HC1 robust SEs."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", vcov="HC1")
    assert result.vcov_type == "HC1"
    # Robust SEs should differ from iid
    result_iid = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert not np.allclose(result.se, result_iid.se)


def test_panel_re_nw(panel_data):
    """RE with Newey-West SEs."""
    result = panel_re(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert all(se > 0 for se in result.se)


def test_panel_re_dk(panel_data):
    """RE with Driscoll-Kraay SEs."""
    result = panel_re(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"


def test_panel_re_wildboot(panel_data):
    """RE with wild cluster bootstrap SEs."""
    result = panel_re(
        "y ~ x1 + x2",
        data=panel_data,
        entity="firm_id",
        vcov="wildboot",
        cluster=["firm_id"],
        seed=42,
    )
    assert result.vcov_type == "wildboot"


def test_panel_re_nw_requires_time(panel_data):
    """NW vcov should raise without time parameter."""
    import pytest

    with pytest.raises(ValueError, match="requires time"):
        panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", vcov="NW")


# ── panel_fd tests ──────────────────────────────────────────────────


def test_panel_fd_basic(panel_data):
    """Panel FD should recover reasonable coefficients."""
    result = panel_fd("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert result.model_type == "Panel FD"
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.2)  # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.2)  # x2


def test_panel_fd_nobs(panel_data):
    """FD loses one observation per entity."""
    result = panel_fd("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    # 50 firms * 20 years = 1000, lose 1 per firm = 950
    assert result.n_obs == 950


def test_panel_fd_default_cluster(panel_data):
    """FD should default to clustering by entity."""
    result = panel_fd("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_panel_fd_has_intercept(panel_data):
    """FD includes an intercept (drift term)."""
    result = panel_fd("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert "_cons" in result.names


def test_panel_fd_summary(panel_data):
    result = panel_fd("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    s = result.summary()
    assert "Panel FD" in s


def test_panel_fd_manual_check():
    """Verify FD with a tiny hand-constructed panel."""
    # 2 entities, 3 periods each
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "t": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 3.0, 6.0, 2.0, 5.0, 9.0],
            "x": [1.0, 2.0, 3.0, 1.0, 3.0, 5.0],
        }
    )
    # dy = [2, 3, 3, 4], dx = [1, 1, 2, 2]
    # OLS of dy on dx with intercept:
    # dy = a + b*dx => b = cov(dx,dy)/var(dx)
    result = panel_fd("y ~ x", data=df, entity="id", time="t")
    assert result.n_obs == 4  # 6 obs - 2 entities = 4
    # With the differenced data: dy=[2,3,3,4], dx=[1,1,2,2]
    # mean(dx)=1.5, mean(dy)=3.0
    # cov = ((1-1.5)*(2-3) + (1-1.5)*(3-3) + (2-1.5)*(3-3) + (2-1.5)*(4-3))/3
    #      = (0.5 + 0 + 0 + 0.5)/3 = 1/3
    # var = ((-.5)^2 + (-.5)^2 + .5^2 + .5^2)/3 = 1/3
    # b = 1, a = 3 - 1*1.5 = 1.5
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=1e-10)  # x coef
    np.testing.assert_allclose(result.coefficients[1], 1.5, atol=1e-10)  # intercept


# ── Additional robustness tests ───────────────────────────────────


def test_panel_fe_single_entity():
    """Panel with 1 entity, many time periods (entity FE only)."""
    rng = np.random.default_rng(42)
    n = 50
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "entity": np.zeros(n, dtype=int),
        }
    )
    # Entity-only FE (no time FE) -- single entity means FE just demeans
    result = panel_fe("y ~ x1", data=df, entity="entity", cluster=[])
    assert result.n_obs == n
    assert result.model_type == "Panel FE"
    assert len(result.coefficients) == 1


def test_panel_fe_nan_in_entity():
    """NaN in entity column is handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 100
    entity = np.repeat(np.arange(10), 10).astype(float)
    entity[0] = np.nan  # inject NaN into entity column
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "entity": entity,
            "time": np.tile(np.arange(10), 10),
        }
    )
    result = panel_fe("y ~ x1", data=df, entity="entity", time="time")
    # One observation dropped due to NaN in entity
    assert result.n_obs < n


def test_panel_re_lazyframe():
    """LazyFrame input works for panel_re."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 20, 10
    n = n_firms * n_years
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "entity": np.repeat(np.arange(n_firms), n_years),
            "time": np.tile(np.arange(n_years), n_firms),
        }
    )
    lazy = df.lazy()
    result = panel_re("y ~ x1", data=lazy, entity="entity")
    assert result.model_type == "Panel RE"
    assert result.n_obs == n


def test_panel_fd_two_periods_only():
    """T=2 minimum viable first-difference."""
    rng = np.random.default_rng(42)
    n_entities = 30
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n_entities * 2),
            "x1": rng.standard_normal(n_entities * 2),
            "entity": np.repeat(np.arange(n_entities), 2),
            "time": np.tile([0, 1], n_entities),
        }
    )
    result = panel_fd("y ~ x1", data=df, entity="entity", time="time")
    assert result.model_type == "Panel FD"
    # With T=2, lose 1 obs per entity
    assert result.n_obs == n_entities


def test_panel_fe_iid_se_explicit(panel_data):
    """cluster=[] forces iid SEs in panel_fe."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", cluster=[])
    assert result.vcov_type == "iid"


def test_panel_unbalanced_extreme():
    """Unbalanced panel: some entities 2 obs, others 50."""
    rng = np.random.default_rng(42)
    entity_id = []
    time_id = []
    idx = 0
    # 10 entities with 2 obs each
    for e in range(10):
        for t in range(2):
            entity_id.append(e)
            time_id.append(t)
            idx += 1
    # 5 entities with 50 obs each
    for e in range(10, 15):
        for t in range(50):
            entity_id.append(e)
            time_id.append(t)
            idx += 1
    n = len(entity_id)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "entity": entity_id,
            "time": time_id,
        }
    )
    result = panel_fe("y ~ x1", data=df, entity="entity", time="time")
    assert result.model_type == "Panel FE"
    assert result.n_obs > 0


def test_panel_fd_iid_vcov():
    """panel_fd with vcov='iid' works correctly."""
    rng = np.random.default_rng(42)
    n_entities, n_times = 30, 10
    n = n_entities * n_times
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "entity": np.repeat(np.arange(n_entities), n_times),
            "time": np.tile(np.arange(n_times), n_entities),
        }
    )
    result = panel_fd("y ~ x1", data=df, entity="entity", time="time", vcov="iid", cluster=[])
    assert result.vcov_type == "iid"
    assert np.all(result.se > 0)
