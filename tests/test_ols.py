import warnings

import numpy as np
import polars as pl
import pytest

import polars_reg as pr
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


def test_ols_with_singletons_no_warnings():
    """OLS with FE should not produce RuntimeWarnings when singletons are dropped."""
    rng = np.random.default_rng(123)
    n = 500
    fe1 = rng.integers(0, 50, size=n)
    fe2 = rng.integers(0, 30, size=n)
    # Force singletons in first 3 observations
    fe1[:3] = [997, 998, 999]

    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "fe1": fe1,
            "fe2": fe2,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = pr.ols("y ~ x1 | fe1 + fe2", data=df)

    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.se))


# ── Robustness edge cases ──────────────────────────────────────


def test_ols_nan_in_x_dropped():
    """Null rows in x should be dropped, producing finite results with fewer N."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1})
    # Set first 10 x1 values to null via Polars
    mask = pl.Series("mask", [True] * 10 + [False] * (n - 10))
    df = df.with_columns(pl.when(mask).then(None).otherwise(pl.col("x1")).alias("x1"))
    assert df["x1"].null_count() == 10
    result = ols("y ~ x1", data=df)
    assert result.n_obs == n - 10
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.se))


def test_ols_lazyframe_input(simple_data):
    """LazyFrame input should produce same coefficients as DataFrame."""
    result_df = ols("y ~ x1 + x2", data=simple_data)
    result_lf = ols("y ~ x1 + x2", data=simple_data.lazy())
    np.testing.assert_allclose(result_lf.coefficients, result_df.coefficients, rtol=1e-10)


def test_ols_integer_columns():
    """OLS should handle integer-typed y and x columns without error."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.integers(0, 10, size=n)
    y = 2 * x + rng.integers(-2, 3, size=n)
    df = pl.DataFrame({"y": y, "x": x})
    result = ols("y ~ x", data=df)
    assert np.all(np.isfinite(result.coefficients))
    assert result.n_obs == n


def test_ols_all_vcov_variants_finite():
    """All vcov types (iid, HC0-HC3, cluster) should produce finite SEs."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    fe1 = rng.integers(0, 10, size=n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "fe1": fe1})
    for v in ["iid", "HC0", "HC1", "HC2", "HC3"]:
        result = ols("y ~ x1", data=df, vcov=v)
        assert np.all(np.isfinite(result.se)), f"Non-finite SEs for vcov={v}"
    result_cl = ols("y ~ x1", data=df, cluster=["fe1"])
    assert np.all(np.isfinite(result_cl.se)), "Non-finite SEs for cluster"


def test_ols_large_fe_ratio():
    """Large number of FE levels relative to N should not crash."""
    rng = np.random.default_rng(42)
    n = 300
    fe = np.concatenate([np.arange(200), rng.integers(0, 200, size=n - 200)])
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "fe": fe})
    result = ols("y ~ x1 | fe", data=df)
    assert np.all(np.isfinite(result.coefficients))


def test_ols_all_singletons_raises():
    """All observations in unique FE groups should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": x1,
            "x2": x2,
            "fe": np.arange(n),
        }
    )
    # Use interaction + HC3 vcov to force pure Python path
    with pytest.raises(ValueError, match="singletons"):
        ols("y ~ x1 + x1:x2 | fe", data=df, vcov="HC3")


def test_ols_collinear_raises():
    """Perfectly collinear regressors should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": x1,
            "x2": x1,  # identical copy
        }
    )
    with pytest.raises(ValueError, match="(?i)singular"):
        ols("y ~ x1 + x2 - 1", data=df)
