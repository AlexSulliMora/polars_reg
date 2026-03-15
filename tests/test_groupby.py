"""Tests for GroupBy regression."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def grouped_data():
    """Dataset with groups for per-group regression."""
    rng = np.random.default_rng(42)
    n_per_group = 100
    groups = ["A", "B", "C"]
    dfs = []
    for g in groups:
        x1 = rng.standard_normal(n_per_group)
        x2 = rng.standard_normal(n_per_group)
        e = rng.standard_normal(n_per_group) * 0.5
        y = 1.0 * x1 - 0.5 * x2 + e
        dfs.append(pl.DataFrame({"y": y, "x1": x1, "x2": x2, "group": [g] * n_per_group}))
    return pl.concat(dfs)


def test_groupby_basic(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    assert len(result) == 3
    assert "A" in result
    assert "B" in result
    assert "C" in result


def test_groupby_keys(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    keys = list(result.keys())
    assert len(keys) == 3


def test_groupby_individual_result(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    r = result["A"]
    assert isinstance(r, pr.RegressionResult)
    assert r.n_obs == 100
    assert len(r.coefficients) == 3  # x1, x2, _cons


def test_groupby_coef_table(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    table = result.coef_table()
    assert "group" in table.columns
    assert len(table) == 9  # 3 groups * 3 coefficients each


def test_groupby_summary(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    s = result.summary()
    assert "3 groups succeeded" in s
    assert "Group: A" in s
    assert "Group: B" in s


def test_groupby_iteration(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    count = 0
    for key in result:
        count += 1
        assert isinstance(result[key], pr.RegressionResult)
    assert count == 3


def test_groupby_values(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    vals = list(result.values())
    assert len(vals) == 3
    assert all(isinstance(v, pr.RegressionResult) for v in vals)


def test_groupby_min_obs(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group", min_obs=200)
    assert len(result) == 0
    assert len(result.failed) == 3


def test_groupby_with_robust(grouped_data):
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group", vcov="HC1")
    assert len(result) == 3
    for r in result.values():
        assert r.vcov_type == "HC1"


def test_groupby_singular_group():
    """Groups with singular X'X should fail gracefully."""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [1.0, 1.0, 1.0],  # no variation
            "x2": [0.0, 0.0, 0.0],
            "group": ["A", "A", "A"],
        }
    )
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", df, group_by="group")
    # Should fail gracefully: the singular group should fail, not succeed
    assert len(result) == 0
    assert len(result.failed) > 0


def test_groupby_regtable_integration(grouped_data):
    """GroupBy results should work with regtable."""
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", grouped_data, group_by="group")
    table = pr.regtable(*result.values(), labels=list(result.keys()))
    assert "x1" in table
    assert "x2" in table


def test_groupby_multikey():
    """GroupBy with multiple key columns."""
    rng = np.random.default_rng(42)
    n = 200
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "sector": np.repeat(["Tech", "Fin"], n // 2),
            "region": np.tile(["US", "EU"], n // 2),
        }
    )
    result = pr.groupby_reg(pr.ols, "y ~ x1", df, group_by=["sector", "region"])
    assert len(result) == 4  # 2 sectors * 2 regions


def test_groupby_rejects_pandas(grouped_data):
    """GroupBy should reject pandas DataFrames with helpful error."""
    pytest.importorskip("pandas")
    pd_df = grouped_data.to_pandas()
    with pytest.raises(TypeError, match="pl.from_pandas"):
        pr.groupby_reg(pr.ols, "y ~ x1 + x2", pd_df, group_by="group")


def test_groupby_iv(grouped_data):
    """GroupBy should work with IV estimators."""
    rng = np.random.default_rng(99)
    n = 300
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_end = 0.5 * z1 + 0.3 * z2 + 0.5 * u
    y = 1.0 + 2.0 * x_end + u
    df = pl.DataFrame(
        {
            "y": y,
            "x_end": x_end,
            "z1": z1,
            "z2": z2,
            "group": np.repeat(["A", "B", "C"], 100),
        }
    )
    result = pr.groupby_reg(pr.iv2sls, "y ~ 1 || x_end ~ z1 + z2", df, group_by="group")
    assert len(result) == 3


# ── Additional robustness tests ───────────────────────────────────


def test_groupby_group_fewer_obs_than_params():
    """Group with N < k handled gracefully (fails, not crashes)."""
    rng = np.random.default_rng(42)
    # Group A has only 1 obs but 3 params (x1, x2, _cons) -> singular
    df = pl.DataFrame(
        {
            "y": [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x1": rng.standard_normal(7).tolist(),
            "x2": rng.standard_normal(7).tolist(),
            "group": ["A", "B", "B", "B", "B", "B", "B"],
        }
    )
    result = pr.groupby_reg(pr.ols, "y ~ x1 + x2", df, group_by="group")
    # Group A should fail (N=1 < k=3), group B should succeed (N=6 >= k=3)
    assert "B" in result
    assert "A" in result.failed
