import numpy as np

from polars_reg._results import RegressionResult


def test_result_se():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    np.testing.assert_allclose(r.se, [0.1, 0.2])


def test_result_tstat():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    np.testing.assert_allclose(r.tstat, [15.0, 10.0])


def test_result_pvalue():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    assert all(r.pvalue < 0.001)  # highly significant


def test_result_confint():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    ci = r.confint()
    assert ci.shape == (2, 2)
    assert ci[0, 0] < 1.5 < ci[0, 1]
    assert ci[1, 0] < 2.0 < ci[1, 1]


def test_result_summary_returns_string():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    s = r.summary()
    assert isinstance(s, str)
    assert "OLS Regression" in s
    assert "x1" in s
    assert "_cons" in s


def test_result_summary_with_fe():
    beta = np.array([1.5])
    vcov = np.diag([0.01])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1"], n_obs=100, k=1,
        df_r=50, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="cluster",
        fe_absorbed=["firm_id", "year_id"], df_absorbed=68,
        n_clusters={"firm_id": 50},
    )
    s = r.summary()
    assert "Absorbed FE" in s
    assert "Clusters" in s
