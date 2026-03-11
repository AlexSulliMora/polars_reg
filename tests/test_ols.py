import numpy as np
import polars as pl

from polars_reg._ols import ols


def test_ols_basic(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data)
    # DGP: y = 2.0 + 1.5*x1 - 0.5*x2 + N(0, 0.25)
    assert result.n_obs == 1000
    assert result.model_type == "OLS"
    np.testing.assert_allclose(result.coefficients[0], 1.5, atol=0.1)  # x1
    np.testing.assert_allclose(result.coefficients[1], -0.5, atol=0.1)  # x2
    np.testing.assert_allclose(result.coefficients[2], 2.0, atol=0.1)  # _cons
    assert result.r_squared > 0.8
    assert result.names == ["x1", "x2", "_cons"]


def test_ols_robust(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data, vcov="HC1")
    assert result.vcov_type == "HC1"
    assert len(result.se) == 3
    # Robust SEs should be close to iid SEs for homoskedastic data
    iid_result = ols("y ~ x1 + x2", data=simple_data)
    np.testing.assert_allclose(result.se, iid_result.se, rtol=0.2)


def test_ols_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50}
    assert result.df_r == 49


def test_ols_twoway_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id", "year_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50, "year_id": 20}
    assert result.df_r == 19  # min(50, 20) - 1


def test_ols_summary(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data)
    s = result.summary()
    assert "OLS Regression" in s
    assert "x1" in s
    assert "x2" in s
    assert "_cons" in s


def test_ols_no_intercept(simple_data):
    result = ols("y ~ x1 + x2 - 1", data=simple_data)
    assert "_cons" not in result.names
    assert len(result.coefficients) == 2


def test_ols_with_fe(panel_data):
    """OLS with absorbed FE should recover coefficients without FE bias."""
    result = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data)
    assert result.fe_absorbed == ["firm_id", "year_id"]
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.15)  # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.15)  # x2
    assert len(result.names) == 2  # no intercept when FE absorbed
    assert result.df_absorbed > 0


def test_ols_fe_clustered(panel_data):
    result = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data, cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50}
    assert result.fe_absorbed == ["firm_id", "year_id"]


def test_ols_fe_twoway_clustered(panel_data):
    result = ols(
        "y ~ x1 + x2 | firm_id + year_id",
        data=panel_data,
        cluster=["firm_id", "year_id"],
    )
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50, "year_id": 20}


# ── Interaction terms ────────────────────────────────────────────


def test_interaction_colon():
    """x1:x2 creates an elementwise product column."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * x1 * x2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    r = ols("y ~ x1 + x2 + x1:x2", data=df)
    assert "x1:x2" in r.names
    idx = r.names.index("x1:x2")
    np.testing.assert_allclose(r.coefficients[idx], 0.5, atol=0.15)


def test_interaction_star():
    """x1*x2 expands to x1 + x2 + x1:x2 and matches explicit form."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * x1 * x2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    r_star = ols("y ~ x1*x2", data=df)
    r_explicit = ols("y ~ x1 + x2 + x1:x2", data=df)
    np.testing.assert_allclose(r_star.coefficients, r_explicit.coefficients)
    np.testing.assert_allclose(r_star.se, r_explicit.se)


def test_interaction_with_fe():
    """Interaction terms work with absorbed fixed effects."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    fe = rng.integers(0, 20, n)
    y = 2.0 * x1 + 3.0 * x2 + 0.5 * x1 * x2 + fe * 0.1 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "fe": fe})
    r = ols("y ~ x1*x2 | fe", data=df)
    assert "x1:x2" in r.names
    assert r.n_obs == 500
    np.testing.assert_allclose(r.coefficients[r.names.index("x1:x2")], 0.5, atol=0.15)


def test_interaction_robust_se():
    """Interaction terms work with robust SEs."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 0.5 * x1 * x2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    r = ols("y ~ x1 + x2 + x1:x2", data=df, vcov="HC1")
    assert r.vcov_type == "HC1"
    assert np.all(r.se > 0)


def test_three_way_interaction():
    """x1*x2*x3 produces all subset interactions."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1.0 + x1 + x2 + x3 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    r = ols("y ~ x1*x2*x3", data=df)
    expected = ["x1", "x2", "x3", "x1:x2", "x1:x3", "x2:x3", "x1:x2:x3", "_cons"]
    assert r.names == expected


# ── Indicator variables ──────────────────────────────────────────


def test_indicator_basic():
    """i.group expands to K-1 dummy variables."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal(n)
    group = rng.choice([1, 2, 3, 4], n)
    y = 1.0 + 0.5 * x + (group == 2) + (group == 3) * 2 + (group == 4) * 3
    y = y + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x": x, "group": group})
    r = ols("y ~ x + i.group", data=df)
    # 3 dummies (group=2, 3, 4) + x + _cons = 5 coefficients
    assert len(r.coefficients) == 5
    assert "group=2" in r.names
    assert "group=3" in r.names
    assert "group=4" in r.names
    # Coefficients close to true values
    np.testing.assert_allclose(r.coefficients[r.names.index("group=2")], 1.0, atol=0.2)
    np.testing.assert_allclose(r.coefficients[r.names.index("group=4")], 3.0, atol=0.2)


def test_indicator_string_levels():
    """i. works with string categories."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal(n)
    group = rng.choice(["A", "B", "C"], n)
    y = 1.0 + x + (group == "B") * 2.0 + (group == "C") * 4.0 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x": x, "group": group})
    r = ols("y ~ x + i.group", data=df)
    # A is reference (sorted first)
    assert "group=B" in r.names
    assert "group=C" in r.names
    assert "group=A" not in r.names


def test_indicator_star_interaction():
    """i.group*x gives dummies + continuous + dummy:continuous."""
    rng = np.random.default_rng(42)
    n = 400
    x = rng.standard_normal(n)
    group = rng.choice([1, 2, 3], n)
    y = x + (group == 2) + (group == 3) * 2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x": x, "group": group})
    r = ols("y ~ i.group*x", data=df)
    # group=2, group=3, x, group=2:x, group=3:x, _cons = 6
    assert len(r.coefficients) == 6
    assert "group=2:x" in r.names
    assert "group=3:x" in r.names


def test_indicator_with_fe():
    """Indicator variables work with absorbed FE."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.standard_normal(n)
    group = rng.choice([1, 2, 3], n)
    fe = rng.integers(0, 10, n)
    y = x + (group == 2) + (group == 3) * 2 + fe * 0.1 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x": x, "group": group, "fe": fe})
    r = ols("y ~ x + i.group | fe", data=df)
    assert "group=2" in r.names
    assert "group=3" in r.names
    assert np.all(r.se > 0)
