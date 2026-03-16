"""Tests for the SSC (small-sample corrections) dataclass and helper functions."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

import polars_reg as pr
from polars_reg._se import (
    _recode_to_contiguous,
    vcov_clustered,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
    vcov_robust,
)
from polars_reg._ssc import SSC, _compute_k_eff, _default_ssc, ssc


class TestSSCConstruction:
    """Test SSC dataclass construction."""

    def test_defaults(self):
        s = SSC()
        assert s.k_adj is True
        assert s.k_fixef == "none"
        assert s.G_adj is True
        assert s.G_df == "conventional"

    def test_custom_values(self):
        s = SSC(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")
        assert s.k_adj is False
        assert s.k_fixef == "nonnested"
        assert s.G_adj is False
        assert s.G_df == "min"

    def test_k_fixef_full(self):
        s = SSC(k_fixef="full")
        assert s.k_fixef == "full"


class TestSSCValidation:
    """Test SSC validation in __post_init__."""

    def test_invalid_k_fixef(self):
        with pytest.raises(ValueError, match="k_fixef must be"):
            SSC(k_fixef="invalid")

    def test_invalid_G_df(self):
        with pytest.raises(ValueError, match="G_df must be"):
            SSC(G_df="invalid")


class TestSSCFrozen:
    """Test that SSC is frozen (immutable)."""

    def test_cannot_modify_k_adj(self):
        s = SSC()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.k_adj = False  # type: ignore[misc]

    def test_cannot_modify_k_fixef(self):
        s = SSC()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.k_fixef = "full"  # type: ignore[misc]


class TestSSCRepr:
    """Test SSC repr is readable."""

    def test_repr_contains_field_names(self):
        s = SSC()
        r = repr(s)
        assert "k_adj" in r
        assert "k_fixef" in r
        assert "G_adj" in r
        assert "G_df" in r

    def test_repr_shows_values(self):
        s = SSC(k_adj=False, G_df="min")
        r = repr(s)
        assert "False" in r
        assert "min" in r


class TestSscFunction:
    """Test ssc() convenience function."""

    def test_returns_ssc_instance(self):
        result = ssc()
        assert isinstance(result, SSC)

    def test_defaults_match_SSC(self):
        assert ssc() == SSC()

    def test_custom_values(self):
        result = ssc(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")
        assert result == SSC(k_adj=False, k_fixef="nonnested", G_adj=False, G_df="min")

    def test_stata_preset(self):
        result = ssc(k_fixef="nonnested", G_df="min")
        assert result.k_fixef == "nonnested"
        assert result.G_df == "min"
        assert result.k_adj is True
        assert result.G_adj is True


class TestDefaultSsc:
    """Test _default_ssc() helper."""

    def test_returns_default_ssc(self):
        result = _default_ssc()
        assert isinstance(result, SSC)
        assert result == SSC()

    def test_returns_new_instance(self):
        a = _default_ssc()
        b = _default_ssc()
        # Frozen dataclasses are equal but could be same or different objects
        assert a == b


# =============================================================================
# _compute_k_eff tests
# =============================================================================


class TestComputeKEff:
    """Test _compute_k_eff helper."""

    def test_none_returns_k(self):
        assert _compute_k_eff(5, "none", df_abs=10, df_a_non_nested=8) == 5

    def test_nonnested_adds_non_nested(self):
        assert _compute_k_eff(5, "nonnested", df_abs=10, df_a_non_nested=8) == 13

    def test_nonnested_zero_nn(self):
        assert _compute_k_eff(5, "nonnested", df_abs=10, df_a_non_nested=0) == 5

    def test_nonnested_negative_nn_clamped(self):
        # Negative values are clamped to 0
        assert _compute_k_eff(5, "nonnested", df_abs=10, df_a_non_nested=-3) == 5

    def test_full_adds_df_abs(self):
        assert _compute_k_eff(5, "full", df_abs=10, df_a_non_nested=8) == 15


# =============================================================================
# SSC x VCV integration tests
# =============================================================================


def _make_test_data():
    """Generate simple OLS data with cluster structure."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    X = np.column_stack([x, np.ones(n)])
    y = 2.0 + 3.0 * x + rng.standard_normal(n)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    clusters = np.repeat(np.arange(10), 10).astype(np.int32)
    return X, y, resid, beta, clusters


class TestSSCVcovIid:
    """Test SSC effect on iid VCV."""

    def test_k_adj_false_gives_mle_variance(self):
        """k_adj=False gives sigma2 = e'e/N instead of e'e/(N-k)."""
        X, y, resid, _, _ = _make_test_data()
        n, k = X.shape

        V_default = vcov_iid(X, resid)  # k_adj=True by default
        V_no_adj = vcov_iid(X, resid, ssc=SSC(k_adj=False))

        # V_no_adj should use e'e/n, V_default uses e'e/(n-k)
        ratio = np.diag(V_no_adj) / np.diag(V_default)
        expected_ratio = (n - k) / n
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)

    def test_k_fixef_full_with_df_abs(self):
        """k_fixef='full' includes df_abs in denominator."""
        X, y, resid, _, _ = _make_test_data()
        n, k = X.shape
        df_abs = 5

        V_none = vcov_iid(X, resid, ssc=SSC(k_fixef="none"), df_abs=df_abs)
        V_full = vcov_iid(X, resid, ssc=SSC(k_fixef="full"), df_abs=df_abs)

        # V_full divides by (n - k - df_abs), V_none divides by (n - k)
        ratio = np.diag(V_full) / np.diag(V_none)
        expected_ratio = (n - k) / (n - k - df_abs)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)


class TestSSCVcovRobust:
    """Test SSC effect on robust VCV."""

    def test_hc1_k_adj_false(self):
        """HC1 with k_adj=False produces HC0 (no scaling)."""
        X, y, resid, _, _ = _make_test_data()

        V_hc0 = vcov_robust(X, resid, kind="HC0")
        V_hc1_no_adj = vcov_robust(X, resid, kind="HC1", ssc=SSC(k_adj=False))

        np.testing.assert_allclose(V_hc1_no_adj, V_hc0, rtol=1e-12)

    def test_hc2_hc3_unaffected_by_ssc(self):
        """HC2 and HC3 are leverage-based, not affected by SSC."""
        X, y, resid, _, _ = _make_test_data()

        for kind in ("HC2", "HC3"):
            V_default = vcov_robust(X, resid, kind=kind)
            V_no_adj = vcov_robust(X, resid, kind=kind, ssc=SSC(k_adj=False, G_adj=False))
            np.testing.assert_allclose(V_no_adj, V_default, rtol=1e-12)


class TestSSCVcovClustered:
    """Test SSC effect on clustered VCV."""

    def test_G_adj_false_removes_cluster_scaling(self):
        """G_adj=False removes G/(G-1) factor from clustered SEs."""
        X, _, resid, _, clusters = _make_test_data()
        n, k = X.shape
        _, G = _recode_to_contiguous(clusters)

        V_default = vcov_clustered(X, resid, clusters)  # G_adj=True
        V_no_G = vcov_clustered(X, resid, clusters, ssc=SSC(G_adj=False))

        # V_no_G should be V_default * (G-1)/G
        ratio = np.diag(V_no_G) / np.diag(V_default)
        # Default: G/(G-1) * (N-1)/(N-k). Without G_adj: 1 * (N-1)/(N-k)
        expected_ratio = (G - 1) / G
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_k_adj_false_removes_residual_df(self):
        """k_adj=False removes (N-1)/(N-k) factor from clustered SEs."""
        X, _, resid, _, clusters = _make_test_data()
        n, k = X.shape

        V_default = vcov_clustered(X, resid, clusters)  # k_adj=True, G_adj=True
        V_no_k = vcov_clustered(X, resid, clusters, ssc=SSC(k_adj=False))

        ratio = np.diag(V_no_k) / np.diag(V_default)
        expected_ratio = (n - k) / (n - 1)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_no_corrections_gives_raw_sandwich(self):
        """k_adj=False, G_adj=False gives raw sandwich (dfc=1)."""
        X, _, resid, _, clusters = _make_test_data()
        n, k = X.shape
        _, G = _recode_to_contiguous(clusters)

        V_raw = vcov_clustered(X, resid, clusters, ssc=SSC(k_adj=False, G_adj=False))
        V_default = vcov_clustered(X, resid, clusters)

        # V_raw should be V_default / (G/(G-1) * (N-1)/(N-k))
        dfc_default = (G / (G - 1)) * ((n - 1) / (n - k))
        ratio = np.diag(V_raw) / np.diag(V_default)
        np.testing.assert_allclose(ratio, 1.0 / dfc_default, rtol=1e-10)

    def test_k_fixef_nonnested_with_fe(self):
        """k_fixef='nonnested' includes non-nested FE dof in k."""
        X, _, resid, _, clusters = _make_test_data()
        n, k = X.shape
        df_a_nn = 5

        V_none = vcov_clustered(
            X, resid, clusters, ssc=SSC(k_fixef="none"), df_a_non_nested=df_a_nn
        )
        V_nn = vcov_clustered(
            X, resid, clusters, ssc=SSC(k_fixef="nonnested"), df_a_non_nested=df_a_nn
        )

        # V_nn should have larger SEs (smaller denominator in k_adj)
        ratio = np.diag(V_nn) / np.diag(V_none)
        expected_ratio = (n - k) / (n - k - df_a_nn)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)


class TestSSCVcovMultiway:
    """Test SSC effect on multiway clustered VCV."""

    def test_G_df_min_vs_conventional(self):
        """G_df='min' uses min(G) vs 'conventional' uses per-term G."""
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([rng.standard_normal(n), np.ones(n)])
        resid = rng.standard_normal(n)
        firm = np.repeat(np.arange(10), 20).astype(np.int32)
        year = np.tile(np.arange(20), 10).astype(np.int32)

        V_conv = vcov_multiway_clustered(X, resid, [firm, year], ssc=SSC(G_df="conventional"))
        V_min = vcov_multiway_clustered(X, resid, [firm, year], ssc=SSC(G_df="min"))

        # Both should produce valid VCV matrices but with different values
        assert V_conv.shape == (2, 2)
        assert V_min.shape == (2, 2)
        assert np.all(np.isfinite(V_conv))
        assert np.all(np.isfinite(V_min))
        # They should differ since min(G)=10 != G for year (20) and interaction terms
        assert not np.allclose(V_conv, V_min, rtol=1e-10)


class TestSSCBootstrap:
    """Test that SSC has no effect on bootstrap VCV."""

    def test_ssc_ignored_for_pairs_bootstrap(self):
        """SSC settings don't affect pairs bootstrap SEs."""
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([rng.standard_normal(n), np.ones(n)])
        y = 2.0 + 3.0 * X[:, 0] + rng.standard_normal(n)

        V1 = vcov_pairs_bootstrap(X, y, n_boot=199, seed=42)
        V2 = vcov_pairs_bootstrap(X, y, n_boot=199, seed=42, ssc=SSC(k_adj=False, G_adj=False))

        np.testing.assert_allclose(V1, V2, rtol=1e-12)


class TestSSCEndToEnd:
    """End-to-end tests: SSC through estimator functions.

    Note: OLS iid/HC1/cluster go through Rust fast paths which do not yet
    use SSC for VCV computation (that is Phase 2b). Tests here use NW/DK
    or panel estimators that take the Python VCV path.
    """

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n = 200
        return pl.DataFrame(
            {
                "y": 2.0 + 3.0 * rng.standard_normal(n) + rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "firm_id": np.repeat(np.arange(20), 10),
                "year_id": np.tile(np.arange(10), 20),
                "w": np.ones(n),
            }
        )

    def test_ols_iid_k_adj_false_python_path(self, sample_data):
        """OLS iid with k_adj=False gives e'e/N (Python path via weights)."""
        # Use weights=1 to force Python VCV path (Rust path skips weights)
        r_default = pr.ols("y ~ x1 + x2", data=sample_data, weights="w")
        r_no_adj = pr.ols("y ~ x1 + x2", data=sample_data, weights="w", ssc=ssc(k_adj=False))

        n, k = r_default.n_obs, r_default.k
        ratio = (r_no_adj.se / r_default.se) ** 2
        expected = (n - k) / n
        np.testing.assert_allclose(ratio, expected, rtol=1e-6)

    def test_ols_cluster_G_adj_false_python_path(self, sample_data):
        """OLS clustered with G_adj=False removes G/(G-1) (Python path)."""
        # Use weights=1 to force Python VCV path
        r_default = pr.ols("y ~ x1 + x2", data=sample_data, cluster="firm_id", weights="w")
        r_no_G = pr.ols(
            "y ~ x1 + x2",
            data=sample_data,
            cluster="firm_id",
            weights="w",
            ssc=ssc(G_adj=False),
        )

        G = len(sample_data["firm_id"].unique())
        ratio = (r_no_G.se / r_default.se) ** 2
        expected = (G - 1) / G
        np.testing.assert_allclose(ratio, expected, rtol=1e-6)

    def test_ssc_stored_on_result(self, sample_data):
        """SSC object is stored on RegressionResult."""
        custom_ssc = ssc(k_adj=False, G_adj=False)
        result = pr.ols("y ~ x1 + x2", data=sample_data, ssc=custom_ssc)
        assert result.ssc == custom_ssc
        assert result.ssc.k_adj is False
        assert result.ssc.G_adj is False

    def test_panel_fe_ssc_iid(self, sample_data):
        """panel_fe iid VCV respects SSC k_adj."""
        r_default = pr.panel_fe(
            "y ~ x1 + x2", data=sample_data, entity="firm_id", vcov="iid", cluster=[]
        )
        r_no_adj = pr.panel_fe(
            "y ~ x1 + x2",
            data=sample_data,
            entity="firm_id",
            vcov="iid",
            cluster=[],
            ssc=ssc(k_adj=False),
        )

        n = r_default.n_obs
        k = r_default.k
        k_eff_default = k  # k_fixef="none" by default
        ratio = (r_no_adj.se / r_default.se) ** 2
        # Default uses e'e/(n-k_eff), no_adj uses e'e/n
        # But k_eff = k because k_fixef="none", so default sigma2 = e'e/(n-k)
        # and with df_abs, vcov_iid uses _compute_k_eff(k, "none", df_abs, 0) = k
        expected = (n - k_eff_default) / n
        np.testing.assert_allclose(ratio, expected, rtol=1e-6)
