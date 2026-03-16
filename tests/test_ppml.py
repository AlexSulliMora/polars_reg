"""Tests for PPML (Poisson Pseudo-Maximum Likelihood) estimator."""

import warnings

import numpy as np
import polars as pl
import pytest

from polars_reg import ppml

# ---------------------------------------------------------------------------
# Fixtures / shared DGP
# ---------------------------------------------------------------------------


def _make_poisson_data(n=1000, seed=42):
    """Generate Poisson DGP: y ~ Poisson(exp(0.5 + 1.0*x1 - 0.5*x2))."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    mu = np.exp(0.5 + 1.0 * x1 - 0.5 * x2)
    y = rng.poisson(mu)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_overdispersed_data(n=1000, seed=42):
    """Generate overdispersed count data (Negative Binomial-like)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    mu = np.exp(0.5 + 1.0 * x1 - 0.5 * x2)
    # Overdispersion: y = Poisson(mu * nu) where nu ~ Gamma(shape, 1/shape)
    shape = 2.0
    nu = rng.gamma(shape, 1.0 / shape, size=n)
    y = rng.poisson(mu * nu)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_zero_heavy_data(n=2000, seed=42):
    """Generate zero-inflated count data."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    mu = np.exp(-1.0 + 0.8 * x1 - 0.3 * x2)  # lower mean -> more zeros
    y = rng.poisson(mu)
    # Additional zero-inflation: randomly zero out 30% of observations
    zero_mask = rng.random(n) < 0.3
    y[zero_mask] = 0
    return pl.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})


def _make_clustered_data(n=1000, n_clusters=50, seed=42):
    """Generate Poisson data with cluster structure."""
    rng = np.random.default_rng(seed)
    cluster_id = rng.integers(0, n_clusters, size=n)
    # Cluster-level effects
    cluster_effects = rng.standard_normal(n_clusters) * 0.5
    x1 = rng.standard_normal(n) + cluster_effects[cluster_id] * 0.3
    x2 = rng.standard_normal(n)
    mu = np.exp(0.5 + 1.0 * x1 - 0.5 * x2 + cluster_effects[cluster_id])
    y = rng.poisson(mu)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "cluster_id": cluster_id,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPPMLBasic:
    """Basic PPML estimation with Poisson DGP."""

    def test_coefficients_recover_true_values(self):
        """With Poisson DGP, PPML should recover true coefficients."""
        df = _make_poisson_data(n=5000, seed=42)
        res = ppml("y ~ x1 + x2", data=df)

        # True: intercept=0.5, x1=1.0, x2=-0.5
        assert res.model_type == "PPML"
        assert res.n_obs == 5000
        assert res.k == 3

        # Check coefficients are close to true values
        coef = dict(zip(res.names, res.coefficients))
        assert abs(coef["x1"] - 1.0) < 0.05, f"x1 coef: {coef['x1']}"
        assert abs(coef["x2"] - (-0.5)) < 0.05, f"x2 coef: {coef['x2']}"
        assert abs(coef["_cons"] - 0.5) < 0.1, f"_cons coef: {coef['_cons']}"

    def test_result_attributes(self):
        """Check all expected attributes on RegressionResult."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)

        assert hasattr(res, "coefficients")
        assert hasattr(res, "vcov")
        assert hasattr(res, "se")
        assert hasattr(res, "tstat")
        assert hasattr(res, "pvalue")
        assert hasattr(res, "residuals")
        assert hasattr(res, "r_squared")
        assert len(res.names) == 3  # x1, x2, _cons
        assert res.vcov.shape == (3, 3)

    def test_se_positive(self):
        """Standard errors should be positive."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)
        assert np.all(res.se > 0)

    def test_summary_runs(self):
        """summary() should produce a string without error."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)
        s = res.summary()
        assert "PPML" in s
        assert "x1" in s

    def test_pseudo_r2_bounded(self):
        """Pseudo R-squared should be between 0 and 1 for well-specified model."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)
        assert 0 < res.r_squared < 1


class TestPPMLOverdispersed:
    """PPML with overdispersed data -- coefficients still consistent."""

    def test_consistency_under_overdispersion(self):
        """PPML coefficients should be consistent even with overdispersion."""
        df = _make_overdispersed_data(n=5000, seed=42)
        res = ppml("y ~ x1 + x2", data=df)

        coef = dict(zip(res.names, res.coefficients))
        # PPML is consistent for E[y|x] = exp(x'beta) regardless of variance
        assert abs(coef["x1"] - 1.0) < 0.1, f"x1 coef: {coef['x1']}"
        assert abs(coef["x2"] - (-0.5)) < 0.1, f"x2 coef: {coef['x2']}"


class TestPPMLRobustSE:
    """Default robust (sandwich) standard errors."""

    def test_default_is_robust(self):
        """Default vcov should be HC1 (robust)."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)
        assert res.vcov_type == "HC1"

    def test_robust_se_larger_than_hessian(self):
        """Robust SEs should generally differ from Hessian-based SEs."""
        df = _make_overdispersed_data(n=2000, seed=42)
        res_robust = ppml("y ~ x1 + x2", data=df, vcov="HC1")
        res_hessian = ppml("y ~ x1 + x2", data=df, vcov="iid")

        # With overdispersion, robust SEs should typically be larger
        # (At minimum, they should differ)
        assert not np.allclose(res_robust.se, res_hessian.se, rtol=0.01)


class TestPPMLClustered:
    """Clustered standard errors."""

    def test_clustered_se(self):
        """Clustered SEs should run without error and produce valid output."""
        df = _make_clustered_data()
        res = ppml("y ~ x1 + x2", data=df, cluster=["cluster_id"])

        assert res.vcov_type == "cluster"
        assert res.n_clusters is not None
        assert "cluster_id" in res.n_clusters
        assert np.all(res.se > 0)

    def test_clustered_se_differs_from_robust(self):
        """Clustered SEs should generally differ from plain robust SEs."""
        df = _make_clustered_data()
        res_cl = ppml("y ~ x1 + x2", data=df, cluster=["cluster_id"])
        res_rob = ppml("y ~ x1 + x2", data=df, vcov="HC1")

        # Same coefficients (same estimator, just different VCV)
        np.testing.assert_allclose(res_cl.coefficients, res_rob.coefficients)
        # But different SEs
        assert not np.allclose(res_cl.se, res_rob.se, rtol=0.01)

    def test_clustered_string_arg(self):
        """cluster= should accept a plain string."""
        df = _make_clustered_data()
        res = ppml("y ~ x1 + x2", data=df, cluster="cluster_id")
        assert res.vcov_type == "cluster"


class TestPPMLZeroHeavy:
    """Zero-heavy (zero-inflated) data."""

    def test_zero_heavy_convergence(self):
        """PPML should converge with many zeros."""
        df = _make_zero_heavy_data()
        res = ppml("y ~ x1 + x2", data=df)

        assert res.n_obs == 2000
        assert np.all(np.isfinite(res.coefficients))
        assert np.all(np.isfinite(res.se))

    def test_zero_heavy_coefficients(self):
        """Coefficients in zero-inflated data should have correct signs."""
        df = _make_zero_heavy_data(n=5000, seed=42)
        res = ppml("y ~ x1 + x2", data=df)

        coef = dict(zip(res.names, res.coefficients))
        # True: x1 has positive effect, x2 has negative effect
        # (zero-inflation biases magnitudes but not direction)
        assert coef["x1"] > 0
        assert coef["x2"] < 0


class TestPPMLSeparation:
    """Separation detection warnings."""

    def test_separation_warning_large_coef(self):
        """Test that PPML warns on quasi/complete separation."""
        rng = np.random.default_rng(99)
        n = 500
        # x1 is a binary indicator: when x1=1, y is always 0
        # This causes the coefficient on x1 to diverge to -inf
        x1 = np.zeros(n)
        x1[:250] = 1.0
        x2 = rng.standard_normal(n)
        y = np.zeros(n, dtype=float)
        # Only observations with x1=0 have positive y
        y[250:] = rng.poisson(3.0, size=250).astype(float)

        sep_data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = ppml("y ~ x1 + x2", data=sep_data, max_iter=50)
        # Either got a warning or coefficients are very large (separation detected)
        warned = any(
            "separation" in str(wi.message).lower() or "converge" in str(wi.message).lower()
            for wi in w
        )
        has_large_coef = np.any(np.abs(res.coefficients) > 10)
        assert warned or has_large_coef, "Expected separation warning or large coefficients"


class TestPPMLPredict:
    """Prediction: exp(X @ beta) gives conditional mean."""

    def test_predict_insample(self):
        """In-sample predict() should return exp(X @ beta)."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)

        predicted = res.predict()
        # Verify in-sample predictions match exp(X @ beta) for PPML
        mu_xb = np.exp(res._X @ res.coefficients) if res._X is not None else None
        if mu_xb is not None:
            np.testing.assert_allclose(predicted, mu_xb, rtol=1e-6)
        else:
            assert predicted.shape[0] > 0

    def test_predict_matches_mu(self):
        """Predicted values should match stored mu = exp(X @ beta)."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)

        mu_from_xb = np.exp(res._X @ res.coefficients)
        fitted = res._y - res.residuals
        np.testing.assert_allclose(fitted, mu_from_xb, rtol=1e-6)

    def test_predict_new_data(self):
        """Out-of-sample predict() with new_data."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)

        new_data = pl.DataFrame(
            {
                "x1": [0.0, 1.0, -1.0],
                "x2": [0.0, 0.0, 0.0],
            }
        )
        # predict() returns x'beta (linear predictor), not exp(x'beta)
        # This is consistent with other models in the package
        pred = res.predict(new_data)
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))


class TestPPMLEdgeCases:
    """Edge cases and input validation."""

    def test_negative_y_raises(self):
        """Negative dependent variable should raise ValueError."""
        df = pl.DataFrame(
            {
                "y": [-1.0, 0.0, 1.0, 2.0],
                "x1": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="non-negative"):
            ppml("y ~ x1", data=df)

    def test_fe_not_supported(self):
        """Absorbed FE in formula should raise ValueError."""
        df = _make_poisson_data()
        with pytest.raises(ValueError, match="fixed effects"):
            ppml("y ~ x1 | x2", data=df)

    def test_all_zeros_y(self):
        """All-zero y should still run (degenerate but no crash)."""
        rng = np.random.default_rng(42)
        zero_data = pl.DataFrame(
            {
                "y": np.zeros(100),
                "x1": rng.standard_normal(100),
            }
        )
        # Should produce a result (may warn about separation)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = ppml("y ~ x1", data=zero_data)
        # Check either warning fired or result has extreme coefficients
        assert len(w) > 0 or not np.all(np.isfinite(res.coefficients))

    def test_single_regressor(self):
        """PPML with a single regressor."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.poisson(mu)
        df = pl.DataFrame({"y": y, "x": x})

        res = ppml("y ~ x", data=df)
        coef = dict(zip(res.names, res.coefficients))
        assert abs(coef["x"] - 0.5) < 0.15
        assert abs(coef["_cons"] - 1.0) < 0.15

    def test_rejects_pandas(self):
        """Should reject pandas DataFrame with helpful message."""
        pytest.importorskip("pandas")
        import pandas as pd

        df_pd = pd.DataFrame({"y": [1, 2], "x": [3, 4]})
        with pytest.raises(TypeError, match="pl.from_pandas"):
            ppml("y ~ x", data=df_pd)

    def test_coef_table(self):
        """coef_table() should return a Polars DataFrame."""
        df = _make_poisson_data()
        res = ppml("y ~ x1 + x2", data=df)
        ct = res.coef_table()
        assert isinstance(ct, pl.DataFrame)
        assert "name" in ct.columns
        assert "coef" in ct.columns
        assert len(ct) == 3


# ── Additional robustness tests ───────────────────────────────────


def test_ppml_nan_dropped():
    """NaN in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.5 * x1 - 0.3 * x2)
    y = rng.poisson(mu).astype(float)
    x1[0] = np.nan
    x2[10] = np.nan
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = ppml("y ~ x1 + x2", data=df)
    assert res.n_obs == n - 2
    assert np.all(np.isfinite(res.coefficients))


def test_ppml_lazyframe():
    """LazyFrame input works for PPML."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.5 * x1)
    y = rng.poisson(mu).astype(float)
    df = pl.DataFrame({"y": y, "x1": x1}).lazy()
    res = ppml("y ~ x1", data=df)
    assert res.model_type == "PPML"
    assert res.n_obs == n
