import numpy as np
import pytest

from polars_reg._iv import iv2sls
from polars_reg._ols import ols


def test_iv2sls_basic(iv_data):
    """2SLS should correct endogeneity bias."""
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "2SLS"
    # DGP: y = 1.0 + 2.0*x_endog + 0.5*x_exog + u
    np.testing.assert_allclose(result.coefficients[result.names.index("x_endog")], 2.0, atol=0.5)
    np.testing.assert_allclose(result.coefficients[result.names.index("x_exog")], 0.5, atol=0.5)
    assert result.n_obs == 1000


def test_iv2sls_vs_ols_bias(iv_data):
    """OLS on endogenous model should be biased; 2SLS should correct it."""
    ols_result = ols("y ~ x_exog + x_endog", data=iv_data)
    iv_result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    ols_endog = ols_result.coefficients[ols_result.names.index("x_endog")]
    iv_endog = iv_result.coefficients[iv_result.names.index("x_endog")]
    # OLS biased upward (positive corr between x_endog and u)
    assert ols_endog > iv_endog


def test_iv2sls_robust(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data, vcov="HC1")
    assert result.vcov_type == "HC1"
    assert len(result.se) > 0


def test_first_stage_f(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.first_stage_f is not None
    assert result.first_stage_f > 10  # instruments are relevant


def test_iv2sls_summary(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    s = result.summary()
    assert "2SLS" in s


def test_iv2sls_nw(iv_data_panel):
    """2SLS with Newey-West HAC standard errors."""
    result = iv2sls(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert len(result.se) > 0
    assert all(se > 0 for se in result.se)


def test_iv2sls_dk(iv_data_panel):
    """2SLS with Driscoll-Kraay standard errors."""
    result = iv2sls(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
    assert len(result.se) > 0


def test_iv2sls_nw_requires_time(iv_data):
    """NW vcov should raise without time parameter."""
    with pytest.raises(ValueError, match="requires time"):
        iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data, vcov="NW")


# ---- IV + absorbed FE tests ----


def test_iv2sls_one_fe(iv_data_panel):
    """2SLS with one-way absorbed FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.model_type == "2SLS"
    assert result.fe_absorbed == ["firm_id"]
    assert result.df_absorbed > 0
    idx = result.names.index("x_endog")
    np.testing.assert_allclose(result.coefficients[idx], 2.0, atol=0.5)


def test_iv2sls_two_fe(iv_data_panel):
    """2SLS with two-way absorbed FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id + year_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.fe_absorbed == ["firm_id", "year_id"]
    assert result.df_absorbed > 49


def test_iv2sls_fe_cluster(iv_data_panel):
    """2SLS with FE and clustered SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        cluster=["firm_id"],
    )
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_iv2sls_fe_robust(iv_data_panel):
    """2SLS with FE and robust SEs."""
    result_iid = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    result_hc1 = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="HC1",
    )
    np.testing.assert_allclose(result_iid.coefficients, result_hc1.coefficients, rtol=1e-10)
    assert not np.allclose(result_iid.se, result_hc1.se)


def test_iv2sls_fe_first_stage_f(iv_data_panel):
    """First-stage F should be computed with FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.first_stage_f is not None
    assert result.first_stage_f > 5


def test_iv2sls_fe_matches_manual_demean(iv_data_panel):
    """IV+FE should match manual demean-then-2SLS for coefficients."""
    from polars_reg._demean import demean
    from polars_reg._utils import _to_codes

    result_fe = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )

    codes = _to_codes(iv_data_panel["firm_id"])
    fe_dict = {"firm_id": codes}
    y = iv_data_panel["y"].to_numpy().astype(np.float64)
    x_exog = iv_data_panel["x_exog"].to_numpy().astype(np.float64)
    x_endog = iv_data_panel["x_endog"].to_numpy().astype(np.float64)
    z1 = iv_data_panel["z1"].to_numpy().astype(np.float64)
    z2 = iv_data_panel["z2"].to_numpy().astype(np.float64)
    all_vars = np.column_stack([y, x_exog, x_endog, z1, z2])
    dm = demean(all_vars, fe_dict)
    y_dm = dm[:, 0]
    x_exog_dm = dm[:, 1]
    x_endog_dm = dm[:, 2]
    z1_dm = dm[:, 3]
    z2_dm = dm[:, 4]

    Z = np.column_stack([x_exog_dm, z1_dm, z2_dm])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    X_endog_hat = Z @ (ZtZ_inv @ (Z.T @ x_endog_dm.reshape(-1, 1)))
    X_hat = np.column_stack([x_exog_dm, X_endog_hat])
    X = np.column_stack([x_exog_dm, x_endog_dm])
    beta_manual = np.linalg.solve(X_hat.T @ X, X_hat.T @ y_dm)
    np.testing.assert_allclose(result_fe.coefficients, beta_manual, rtol=1e-4)


def test_iv2sls_fe_nw(iv_data_panel):
    """2SLS + FE + Newey-West SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert result.fe_absorbed == ["firm_id"]


def test_iv2sls_fe_dk(iv_data_panel):
    """2SLS + FE + Driscoll-Kraay SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"


# ── Robustness edge cases ──────────────────────────────────────


def test_iv2sls_nan_dropped():
    """Null in an instrument column should be dropped automatically."""
    rng = np.random.default_rng(42)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u
    import polars as pl

    df = pl.DataFrame({
        "y": y, "x_endog": x_endog, "x_exog": x_exog,
        "z1": z1, "z2": z2,
    })
    # Set first 20 z1 values to null via Polars
    mask = pl.Series("mask", [True] * 20 + [False] * (n - 20))
    df = df.with_columns(pl.when(mask).then(None).otherwise(pl.col("z1")).alias("z1"))
    assert df["z1"].null_count() == 20
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=df)
    assert result.n_obs == n - 20
    assert np.all(np.isfinite(result.coefficients))


def test_iv2sls_lazyframe(iv_data):
    """LazyFrame input should produce same results as DataFrame."""
    result_df = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    result_lf = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data.lazy())
    np.testing.assert_allclose(result_lf.coefficients, result_df.coefficients, rtol=1e-10)


def test_iv2sls_exact_identification():
    """Just-identified IV (1 endog, 1 instrument) should work."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.8 * z1 + 0.5 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 3.0 * x_endog + 0.5 * x_exog + u
    import polars as pl

    df = pl.DataFrame({
        "y": y, "x_endog": x_endog, "x_exog": x_exog, "z1": z1,
    })
    result = iv2sls("y ~ x_exog || x_endog ~ z1", data=df)
    assert result.n_obs == n
    np.testing.assert_allclose(
        result.coefficients[result.names.index("x_endog")], 3.0, atol=0.5
    )


def test_liml_fe_raises(iv_data_panel):
    """LIML with absorbed FE should raise NotImplementedError."""
    from polars_reg._gmm import liml

    with pytest.raises(NotImplementedError, match="LIML does not yet support"):
        liml(
            "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
            data=iv_data_panel,
        )


def test_gmm_multiway_cluster_raises(iv_data):
    """GMM-IV with multi-way clustering should raise NotImplementedError."""
    from polars_reg._gmm import gmm_iv
    import polars as pl

    rng = np.random.default_rng(42)
    n = iv_data.height
    df = iv_data.with_columns([
        pl.Series("cl1", rng.integers(0, 10, size=n)),
        pl.Series("cl2", rng.integers(0, 5, size=n)),
    ])
    with pytest.raises(NotImplementedError, match="Multi-way clustered"):
        gmm_iv(
            "y ~ x_exog || x_endog ~ z1 + z2",
            data=df,
            cluster=["cl1", "cl2"],
        )
