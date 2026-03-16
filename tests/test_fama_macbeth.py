"""Tests for Fama-MacBeth (1973) two-pass regression."""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._fama_macbeth import FamaMacBethResult, fama_macbeth
from polars_reg._group_by import GroupRegressionResult
from polars_reg._rolling import RollingRegressionResult

# ═══════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════


def _simulate_factor_model(
    rng: np.random.Generator,
    n_assets: int = 50,
    n_periods: int = 120,
    n_factors: int = 3,
    true_lambdas: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Simulate a factor model with known risk premia for testing."""
    if true_lambdas is None:
        true_lambdas = np.array([0.5, 0.3, -0.1])

    assert len(true_lambdas) == n_factors

    # Generate factor returns (common across all assets each period)
    factors = rng.normal(0, 1, (n_periods, n_factors))
    factors += true_lambdas  # shift means to create risk premia

    # Generate betas for each asset
    betas = rng.normal(1, 0.5, (n_assets, n_factors))

    # Generate returns: R_it = alpha_i + beta_i' * f_t + epsilon_it
    alphas = rng.normal(0, 0.1, n_assets)
    epsilon = rng.normal(0, 1, (n_periods, n_assets))
    returns = alphas + factors @ betas.T + epsilon

    # Build Polars DataFrame
    rows = []
    for t in range(n_periods):
        for i in range(n_assets):
            row: dict[str, object] = {
                "entity": f"asset_{i}",
                "time": t,
                "ret": returns[t, i],
            }
            for j in range(n_factors):
                row[f"f{j + 1}"] = factors[t, j]
            rows.append(row)

    df = pl.DataFrame(rows)
    return df, true_lambdas, betas


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFamaMacBethBasic:
    """Basic functionality tests."""

    def test_fm_basic(self) -> None:
        """Basic FM with simulated factor model; lambdas near true values."""
        rng = np.random.default_rng(42)
        df, true_lambdas, _ = _simulate_factor_model(rng)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        assert isinstance(result, FamaMacBethResult)
        assert result.model_type == "Fama-MacBeth"

        # Mean lambdas should be in the right ballpark (within ~0.3 for finite sample)
        mean_lam = result.mean_lambda
        # Slopes are first k entries, intercept is last
        for i in range(3):
            assert abs(mean_lam[i] - true_lambdas[i]) < 0.5, (
                f"Factor {i}: got {mean_lam[i]:.3f}, expected ~{true_lambdas[i]:.3f}"
            )

    def test_fm_full_sample(self) -> None:
        """Full-sample first pass produces GroupRegressionResult."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        assert isinstance(result.first_pass, GroupRegressionResult)

    def test_fm_rolling(self) -> None:
        """FM with window parameter produces RollingRegressionResult."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=80)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            window=30,
            stride=5,
        )

        assert isinstance(result.first_pass, RollingRegressionResult)
        # Should have fewer valid periods since rolling needs warmup
        assert result.n_periods > 0
        assert result.n_periods < 80  # can't use all periods with rolling


class TestFamaMacBethShanken:
    """Tests for Shanken correction."""

    def test_fm_shanken_increases_se(self) -> None:
        """Shanken-corrected SE >= FM SE for all coefficients."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=50, n_periods=120)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            shanken=True,
        )

        assert result.shanken_se is not None
        # Shanken SE should be >= FM SE (correction inflates variance)
        for i in range(len(result.names)):
            assert result.shanken_se[i] >= result.fm_se[i] - 1e-10, (
                f"Shanken SE ({result.shanken_se[i]:.6f}) < FM SE "
                f"({result.fm_se[i]:.6f}) for {result.names[i]}"
            )

    def test_fm_shanken_false(self) -> None:
        """shanken=False produces shanken_se=None."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            shanken=False,
        )

        assert result.shanken_se is None
        assert result.shanken_tstat is None
        assert result.shanken_pvalue is None


class TestFamaMacBethOutput:
    """Tests for output methods."""

    def test_fm_first_pass_access(self) -> None:
        """First-pass results are accessible and contain per-entity data."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        fp = result.first_pass
        assert isinstance(fp, GroupRegressionResult)
        assert len(fp) == 20  # one result per asset

    def test_fm_summary(self) -> None:
        """summary() returns a string with expected content."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        s = result.summary()
        assert isinstance(s, str)
        assert "Fama-MacBeth" in s
        assert "f1" in s
        assert "f2" in s
        assert "f3" in s
        assert "_cons" in s
        assert "Shanken" in s

    def test_fm_coef_table(self) -> None:
        """coef_table() returns DataFrame with expected columns."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        ct = result.coef_table()
        assert isinstance(ct, pl.DataFrame)
        assert "name" in ct.columns
        assert "mean_lambda" in ct.columns
        assert "fm_se" in ct.columns
        assert "fm_t" in ct.columns
        assert "fm_p" in ct.columns
        # Shanken columns present since shanken=True by default
        assert "shanken_se" in ct.columns
        assert len(ct) == 4  # 3 factors + intercept

    def test_fm_lambda_series(self) -> None:
        """lambda_series() returns DataFrame with per-period lambdas."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        ls = result.lambda_series()
        assert isinstance(ls, pl.DataFrame)
        assert "time_index" in ls.columns
        assert "f1_lambda" in ls.columns
        assert "f2_lambda" in ls.columns
        assert "f3_lambda" in ls.columns
        assert "_cons_lambda" in ls.columns
        # Should have T_total rows (including NaN periods)
        assert len(ls) == 60


class TestFamaMacBethEdgeCases:
    """Tests for edge cases and robustness."""

    def test_fm_unbalanced_panel(self) -> None:
        """Unbalanced panel: some assets missing some periods."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=30, n_periods=60)

        # Drop ~20% of rows randomly
        n_total = len(df)
        keep_mask = rng.random(n_total) > 0.2
        # Make sure we keep enough data
        keep_indices = np.where(keep_mask)[0].tolist()
        df_unbalanced = df[keep_indices]

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df_unbalanced,
            entity="entity",
            time="time",
        )

        assert isinstance(result, FamaMacBethResult)
        # Still has valid results
        assert result.n_periods > 0
        assert result.n_assets > 0

    def test_fm_single_factor(self) -> None:
        """Works with a single factor (Sigma_f is scalar)."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(
            rng, n_assets=30, n_periods=60, n_factors=1, true_lambdas=np.array([0.5])
        )

        result = fama_macbeth(
            "ret ~ f1",
            data=df,
            entity="entity",
            time="time",
        )

        assert isinstance(result, FamaMacBethResult)
        assert len(result.names) == 2  # f1 + _cons
        assert result.names[-1] == "_cons"
        assert result.shanken_se is not None

    def test_fm_rolling_no_lookahead(self) -> None:
        """Rolling FM uses betas from window ending BEFORE t, not at t.

        Create data with a structural break at period 50: factor loading
        changes sign. If look-ahead bias exists (using window ending at t
        instead of t-1), second-pass for periods just before the break
        would incorrectly use post-break betas, producing different lambdas.
        """
        rng = np.random.default_rng(42)
        n_assets = 20
        n_periods = 80
        window = 20

        # Single factor with structural break at period 50
        # Before break: beta=1, after break: beta=-1
        factor = rng.normal(0.5, 1, n_periods)

        rows = []
        for i in range(n_assets):
            beta_pre = 1.0 + rng.normal(0, 0.1)
            beta_post = -1.0 + rng.normal(0, 0.1)
            alpha = rng.normal(0, 0.05)
            for t in range(n_periods):
                beta = beta_pre if t < 50 else beta_post
                ret = alpha + beta * factor[t] + rng.normal(0, 0.5)
                rows.append(
                    {
                        "entity": f"asset_{i}",
                        "time": t,
                        "ret": ret,
                        "f1": factor[t],
                    }
                )

        df = pl.DataFrame(rows)

        result = fama_macbeth(
            "ret ~ f1",
            data=df,
            entity="entity",
            time="time",
            window=window,
            stride=1,
            shanken=False,
        )

        # The key check: with no look-ahead, the first valid second-pass
        # period should be at or after window (period >= window), because
        # the first window ends at period window-1 and can only be used
        # for periods > window-1
        valid_mask = np.all(np.isfinite(result.lambdas), axis=1)
        first_valid_idx = np.argmax(valid_mask)
        assert first_valid_idx >= window, (
            f"First valid period index is {first_valid_idx}, expected >= {window}. "
            "This suggests look-ahead bias."
        )

    def test_fm_cross_section_failure(self) -> None:
        """Period with too few assets is gracefully handled (NaN lambdas)."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=10, n_periods=30)

        # Remove most assets from period 15
        mask = ~((df["time"] == 15) & (df["entity"] != "asset_0"))
        df_sparse = df.filter(mask)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df_sparse,
            entity="entity",
            time="time",
            shanken=False,
        )

        # Period 15 should have NaN lambdas (only 1 asset)
        lam_15 = result.lambdas[15]
        assert np.all(np.isnan(lam_15)), (
            f"Period 15 lambdas should be NaN (only 1 asset), got {lam_15}"
        )

        # But overall result should still be valid
        assert result.n_periods > 0

    def test_fm_avg_r_squared(self) -> None:
        """avg_r_squared is reasonable for a well-specified model."""
        rng = np.random.default_rng(42)
        # Low noise -> high R^2
        n_assets = 50
        n_periods = 120
        n_factors = 3
        true_lambdas = np.array([0.5, 0.3, -0.1])

        factors = rng.normal(0, 1, (n_periods, n_factors))
        factors += true_lambdas
        betas = rng.normal(1, 0.5, (n_assets, n_factors))
        alphas = rng.normal(0, 0.01, n_assets)
        epsilon = rng.normal(0, 0.1, (n_periods, n_assets))  # low noise
        returns = alphas + factors @ betas.T + epsilon

        rows = []
        for t in range(n_periods):
            for i in range(n_assets):
                row: dict[str, object] = {
                    "entity": f"asset_{i}",
                    "time": t,
                    "ret": returns[t, i],
                }
                for j in range(n_factors):
                    row[f"f{j + 1}"] = factors[t, j]
                rows.append(row)
        df = pl.DataFrame(rows)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            shanken=False,
        )

        assert 0 <= result.avg_r_squared <= 1
        # With low noise, R^2 should be fairly high
        assert result.avg_r_squared > 0.5


class TestFamaMacBethDuckTyping:
    """Tests for regtable()-compatible duck-typing properties."""

    def test_fm_duck_typing(self) -> None:
        """Duck-typing properties work for regtable compatibility."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        # All these properties should work without error
        assert result.coefficients is not None
        assert len(result.coefficients) == 4
        assert result.se is not None
        assert len(result.se) == 4
        assert result.tstat is not None
        assert len(result.tstat) == 4
        assert result.pvalue is not None
        assert len(result.pvalue) == 4
        assert isinstance(result.n_obs, int)
        assert result.n_obs == result.n_assets * result.n_periods
        assert isinstance(result.r_squared, float)

    def test_fm_names_cons_last(self) -> None:
        """_cons is the last element in .names."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        assert result.names[-1] == "_cons"
        assert result.names == ["f1", "f2", "f3", "_cons"]

    def test_fm_se_uses_shanken_when_available(self) -> None:
        """The .se property returns Shanken SE when available."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=30, n_periods=60)

        result_sh = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            shanken=True,
        )
        result_no = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
            shanken=False,
        )

        # With shanken=True, .se should return shanken_se
        np.testing.assert_array_equal(result_sh.se, result_sh.shanken_se)

        # With shanken=False, .se should return fm_se
        np.testing.assert_array_equal(result_no.se, result_no.fm_se)


class TestFamaMacBethRepr:
    """Tests for string representation."""

    def test_fm_repr(self) -> None:
        """__repr__ produces a reasonable string."""
        rng = np.random.default_rng(42)
        df, _, _ = _simulate_factor_model(rng, n_assets=20, n_periods=60)

        result = fama_macbeth(
            "ret ~ f1 + f2 + f3",
            data=df,
            entity="entity",
            time="time",
        )

        r = repr(result)
        assert "FamaMacBethResult" in r
        assert "N=20" in r
        assert "T=60" in r
