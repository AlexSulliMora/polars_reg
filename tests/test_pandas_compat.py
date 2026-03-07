"""Tests for pandas DataFrame compatibility."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 200
    return {
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "firm": rng.choice(["A", "B", "C", "D"], n),
        "z1": rng.standard_normal(n),
        "z2": rng.standard_normal(n),
    }


@pytest.fixture
def pl_df(sample_data):
    return pl.DataFrame(sample_data)


@pytest.fixture
def pd_df(sample_data):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(sample_data)


def test_ols_pandas_matches_polars(pl_df, pd_df):
    r_pl = pr.ols("y ~ x1 + x2", data=pl_df)
    r_pd = pr.ols("y ~ x1 + x2", data=pd_df)
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)
    np.testing.assert_allclose(r_pd.se, r_pl.se)


def test_ols_robust_pandas(pl_df, pd_df):
    r_pl = pr.ols("y ~ x1 + x2", data=pl_df, vcov="HC1")
    r_pd = pr.ols("y ~ x1 + x2", data=pd_df, vcov="HC1")
    np.testing.assert_allclose(r_pd.se, r_pl.se)


def test_ols_cluster_pandas(pl_df, pd_df):
    r_pl = pr.ols("y ~ x1 + x2", data=pl_df, cluster="firm")
    r_pd = pr.ols("y ~ x1 + x2", data=pd_df, cluster="firm")
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_ols_fe_pandas(pl_df, pd_df):
    r_pl = pr.ols("y ~ x1 + x2 | firm", data=pl_df, cluster="firm")
    r_pd = pr.ols("y ~ x1 + x2 | firm", data=pd_df, cluster="firm")
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_iv2sls_pandas(pl_df, pd_df):
    # Make x_endog from z1/z2 so IV is identified
    pl_iv = pl_df.with_columns((pl.col("z1") * 0.5 + pl.col("z2") * 0.3).alias("x_end"))
    pd_iv = pl_iv.to_pandas()
    r_pl = pr.iv2sls("y ~ x1 || x_end ~ z1 + z2", data=pl_iv)
    r_pd = pr.iv2sls("y ~ x1 || x_end ~ z1 + z2", data=pd_iv)
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_liml_pandas(pl_df, pd_df):
    pl_iv = pl_df.with_columns((pl.col("z1") * 0.5 + pl.col("z2") * 0.3).alias("x_end"))
    pd_iv = pl_iv.to_pandas()
    r_pl = pr.liml("y ~ x1 || x_end ~ z1 + z2", data=pl_iv)
    r_pd = pr.liml("y ~ x1 || x_end ~ z1 + z2", data=pd_iv)
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_gmm_pandas(pl_df, pd_df):
    pl_iv = pl_df.with_columns((pl.col("z1") * 0.5 + pl.col("z2") * 0.3).alias("x_end"))
    pd_iv = pl_iv.to_pandas()
    r_pl = pr.gmm_iv("y ~ x1 || x_end ~ z1 + z2", data=pl_iv)
    r_pd = pr.gmm_iv("y ~ x1 || x_end ~ z1 + z2", data=pd_iv)
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_panel_fe_pandas():
    rng = np.random.default_rng(99)
    n_firms, n_years = 20, 10
    n = n_firms * n_years
    pl_panel = pl.DataFrame({
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "firm": np.repeat(np.arange(n_firms), n_years),
        "year": np.tile(np.arange(n_years), n_firms),
    })
    pd_panel = pl_panel.to_pandas()
    r_pl = pr.panel_fe("y ~ x1 + x2", data=pl_panel, entity="firm", time="year")
    r_pd = pr.panel_fe("y ~ x1 + x2", data=pd_panel, entity="firm", time="year")
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_panel_re_pandas(pl_df, pd_df):
    r_pl = pr.panel_re("y ~ x1 + x2", data=pl_df, entity="firm")
    r_pd = pr.panel_re("y ~ x1 + x2", data=pd_df, entity="firm")
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)


def test_panel_fd_pandas():
    rng = np.random.default_rng(99)
    n_firms, n_years = 20, 10
    n = n_firms * n_years
    pl_panel = pl.DataFrame({
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "firm": np.repeat(np.arange(n_firms), n_years),
        "year": np.tile(np.arange(n_years), n_firms),
    })
    pd_panel = pl_panel.to_pandas()
    r_pl = pr.panel_fd("y ~ x1 + x2", data=pl_panel, entity="firm", time="year")
    r_pd = pr.panel_fd("y ~ x1 + x2", data=pd_panel, entity="firm", time="year")
    np.testing.assert_allclose(r_pd.coefficients, r_pl.coefficients)
