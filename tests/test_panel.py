import numpy as np
from polars_reg._panel import panel_fe
from polars_reg._ols import ols


def test_panel_fe_basic(panel_data):
    """Panel FE should recover coefficients."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert result.model_type == "Panel FE"
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.15)   # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.15)  # x2


def test_panel_fe_default_cluster(panel_data):
    """Should default to clustering by entity."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_panel_fe_matches_ols_fe(panel_data):
    """Panel FE should give same coefficients as OLS with absorbed FE."""
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
