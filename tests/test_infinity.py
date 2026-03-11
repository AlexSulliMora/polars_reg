"""Tests for inf / -inf handling in polars_reg.

The package converts inf/-inf to null (dropped) alongside NaN in
``extract_arrays()``.  These tests verify that behaviour end-to-end
for OLS, OLS+FE, IV, and at the ``extract_arrays`` level directly.
"""

import numpy as np
import polars as pl
import pytest

from polars_reg import iv2sls, ols
from polars_reg._formula import FormulaSpec, parse_formula
from polars_reg._utils import extract_arrays

# ── Helpers ────────────────────────────────────────────────────────


def _make_ols_data(rng, n=200):
    """Return (clean_df, dirty_df) where dirty_df has inf rows at known indices."""
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.3
    clean = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Inject inf at rows 0, 1, 2
    y_dirty = y.copy()
    x1_dirty = x1.copy()
    y_dirty[0] = np.inf
    y_dirty[1] = -np.inf
    x1_dirty[2] = np.inf
    dirty = pl.DataFrame({"y": y_dirty, "x1": x1_dirty, "x2": x2})
    # The clean reference excludes the 3 contaminated rows
    clean_ref = clean.slice(3)
    return clean_ref, dirty


# ── 1. OLS with inf in y ──────────────────────────────────────────


def test_ols_inf_in_y_dropped():
    """Rows with inf/-inf in the dependent variable are dropped; result matches clean data."""
    rng = np.random.default_rng(200)
    n = 200
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    clean_df = pl.DataFrame({"y": y, "x1": x1})

    y_dirty = y.copy()
    y_dirty[0] = np.inf
    y_dirty[5] = -np.inf
    dirty_df = pl.DataFrame({"y": y_dirty, "x1": x1})

    # Build clean reference by excluding the same rows
    keep = list(range(1, 5)) + list(range(6, n))
    ref_df = clean_df[keep]

    result_dirty = ols("y ~ x1", data=dirty_df)
    result_clean = ols("y ~ x1", data=ref_df)

    assert result_dirty.n_obs == n - 2
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients)
    np.testing.assert_allclose(result_dirty.se, result_clean.se)


# ── 2. OLS with inf in X ─────────────────────────────────────────


def test_ols_inf_in_x_dropped():
    """Rows with inf/-inf in regressors are dropped; result matches clean data."""
    rng = np.random.default_rng(201)
    n = 200
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    clean_df = pl.DataFrame({"y": y, "x1": x1})

    x1_dirty = x1.copy()
    x1_dirty[10] = np.inf
    x1_dirty[20] = -np.inf
    x1_dirty[30] = np.inf
    dirty_df = pl.DataFrame({"y": y, "x1": x1_dirty})

    keep = [i for i in range(n) if i not in (10, 20, 30)]
    ref_df = clean_df[keep]

    result_dirty = ols("y ~ x1", data=dirty_df)
    result_clean = ols("y ~ x1", data=ref_df)

    assert result_dirty.n_obs == n - 3
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients)
    np.testing.assert_allclose(result_dirty.se, result_clean.se)


# ── 3. OLS with FE + inf ─────────────────────────────────────────


def test_ols_fe_inf_dropped():
    """Inf in data with absorbed fixed effects still works correctly."""
    rng = np.random.default_rng(202)
    n = 300
    fe = rng.integers(0, 10, size=n)
    x1 = rng.standard_normal(n)
    y = 2.0 * x1 + fe * 0.5 + rng.standard_normal(n) * 0.3
    clean_df = pl.DataFrame({"y": y, "x1": x1, "fe": fe})

    y_dirty = y.copy()
    y_dirty[0] = np.inf
    y_dirty[100] = -np.inf
    dirty_df = pl.DataFrame({"y": y_dirty, "x1": x1, "fe": fe})

    keep = [i for i in range(n) if i not in (0, 100)]
    ref_df = clean_df[keep]

    result_dirty = ols("y ~ x1 | fe", data=dirty_df)
    result_clean = ols("y ~ x1 | fe", data=ref_df)

    assert result_dirty.n_obs == n - 2
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients, atol=1e-10)


# ── 4. IV with inf ───────────────────────────────────────────────


def test_iv_inf_in_endog_dropped():
    """Inf in endogenous variable columns are dropped properly."""
    rng = np.random.default_rng(203)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u

    clean_df = pl.DataFrame({"y": y, "x_endog": x_endog, "x_exog": x_exog, "z1": z1, "z2": z2})

    x_endog_dirty = x_endog.copy()
    x_endog_dirty[0] = np.inf
    x_endog_dirty[1] = -np.inf
    dirty_df = pl.DataFrame(
        {"y": y, "x_endog": x_endog_dirty, "x_exog": x_exog, "z1": z1, "z2": z2}
    )

    keep = list(range(2, n))
    ref_df = clean_df[keep]

    result_dirty = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=dirty_df)
    result_clean = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=ref_df)

    assert result_dirty.n_obs == n - 2
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients)


def test_iv_inf_in_instrument_dropped():
    """Inf in instrument columns are dropped properly."""
    rng = np.random.default_rng(204)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u

    clean_df = pl.DataFrame({"y": y, "x_endog": x_endog, "x_exog": x_exog, "z1": z1, "z2": z2})

    z1_dirty = z1.copy()
    z1_dirty[10] = np.inf
    z1_dirty[20] = -np.inf
    dirty_df = pl.DataFrame(
        {"y": y, "x_endog": x_endog, "x_exog": x_exog, "z1": z1_dirty, "z2": z2}
    )

    keep = [i for i in range(n) if i not in (10, 20)]
    ref_df = clean_df[keep]

    result_dirty = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=dirty_df)
    result_clean = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=ref_df)

    assert result_dirty.n_obs == n - 2
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients)


# ── 5. All inf → error ───────────────────────────────────────────


def test_all_inf_raises_valueerror():
    """If ALL rows have inf, should get 'No observations remain' ValueError."""
    df = pl.DataFrame(
        {
            "y": [np.inf, -np.inf, np.inf],
            "x1": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="No observations remain"):
        ols("y ~ x1", data=df)


def test_all_inf_in_x_raises_valueerror():
    """If ALL rows have inf in X, should get 'No observations remain' ValueError."""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [np.inf, -np.inf, np.inf],
        }
    )
    with pytest.raises(ValueError, match="No observations remain"):
        ols("y ~ x1", data=df)


# ── 6. Mixed NaN and inf ─────────────────────────────────────────


def test_mixed_nan_and_inf_all_dropped():
    """Rows with either NaN or inf are all dropped; result matches clean subset."""
    rng = np.random.default_rng(205)
    n = 100
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    clean_df = pl.DataFrame({"y": y, "x1": x1})

    y_dirty = y.copy()
    x1_dirty = x1.copy()
    # Rows 0,1 have NaN; rows 2,3 have inf/-inf
    y_dirty[0] = np.nan
    x1_dirty[1] = np.nan
    y_dirty[2] = np.inf
    x1_dirty[3] = -np.inf
    dirty_df = pl.DataFrame({"y": y_dirty, "x1": x1_dirty})

    keep = list(range(4, n))
    ref_df = clean_df[keep]

    result_dirty = ols("y ~ x1", data=dirty_df)
    result_clean = ols("y ~ x1", data=ref_df)

    assert result_dirty.n_obs == n - 4
    np.testing.assert_allclose(result_dirty.coefficients, result_clean.coefficients)
    np.testing.assert_allclose(result_dirty.se, result_clean.se)


# ── 7. extract_arrays directly ────────────────────────────────────


def test_extract_arrays_drops_inf_in_y():
    """extract_arrays drops rows with inf in y and returns correct n_obs."""
    rng = np.random.default_rng(206)
    n = 50
    y = rng.standard_normal(n)
    y[0] = np.inf
    y[1] = -np.inf
    df = pl.DataFrame({"y": y, "x1": rng.standard_normal(n)})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 2
    assert not np.any(np.isinf(arrays.y))
    assert not np.any(np.isnan(arrays.y))


def test_extract_arrays_drops_inf_in_x():
    """extract_arrays drops rows with inf in X and returns correct n_obs."""
    rng = np.random.default_rng(207)
    n = 50
    x1 = rng.standard_normal(n)
    x1[5] = np.inf
    x1[15] = -np.inf
    x1[25] = np.inf
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": x1})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 3
    assert not np.any(np.isinf(arrays.X))
    assert not np.any(np.isnan(arrays.X))


def test_extract_arrays_drops_inf_in_endog():
    """extract_arrays drops rows with inf in endogenous columns."""
    rng = np.random.default_rng(208)
    n = 50
    x_endog = rng.standard_normal(n)
    x_endog[0] = np.inf
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x_exog": rng.standard_normal(n),
            "x_endog": x_endog,
            "z1": rng.standard_normal(n),
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x_exog"], endog=["x_endog"], instruments=["z1"])
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 1
    assert not np.any(np.isinf(arrays.endog))


def test_extract_arrays_drops_inf_in_instruments():
    """extract_arrays drops rows with inf in instrument columns."""
    rng = np.random.default_rng(209)
    n = 50
    z1 = rng.standard_normal(n)
    z1[10] = -np.inf
    z1[20] = np.inf
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x_exog": rng.standard_normal(n),
            "x_endog": rng.standard_normal(n),
            "z1": z1,
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x_exog"], endog=["x_endog"], instruments=["z1"])
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 2
    assert not np.any(np.isinf(arrays.instruments))


def test_extract_arrays_no_inf_passthrough():
    """When data has no inf values, extract_arrays returns all rows."""
    rng = np.random.default_rng(210)
    n = 30
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n


# ── 8. Inf in cluster column ─────────────────────────────────────


def test_inf_in_float_cluster_dropped():
    """Inf in a float-typed cluster column causes those rows to be dropped."""
    rng = np.random.default_rng(211)
    n = 100
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    cl = rng.integers(0, 5, size=n).astype(float)
    cl[0] = np.inf
    cl[1] = -np.inf
    df = pl.DataFrame({"y": y, "x1": x1, "cl": cl})

    result = ols("y ~ x1", data=df, cluster=["cl"])
    assert result.n_obs == n - 2
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.se))


def test_extract_arrays_inf_in_cluster():
    """extract_arrays drops rows with inf in cluster columns."""
    rng = np.random.default_rng(212)
    n = 50
    cl = rng.integers(0, 5, size=n).astype(float)
    cl[0] = np.inf
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": rng.standard_normal(n), "cl": cl})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec, cluster=["cl"])
    assert arrays.n_obs == n - 1


# ── 9. Inf in weights column ─────────────────────────────────────


def test_inf_in_weights_dropped():
    """If weights contain inf, those rows should be dropped."""
    rng = np.random.default_rng(213)
    n = 100
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    w = np.abs(rng.standard_normal(n)) + 0.1  # positive weights
    w[0] = np.inf
    w[1] = -np.inf
    df = pl.DataFrame({"y": y, "x1": x1, "w": w})

    result = ols("y ~ x1", data=df, weights="w")
    assert result.n_obs == n - 2
    assert np.all(np.isfinite(result.coefficients))
    assert np.all(np.isfinite(result.se))


def test_extract_arrays_inf_in_weights():
    """extract_arrays drops rows with inf in weights column."""
    rng = np.random.default_rng(214)
    n = 50
    w = np.abs(rng.standard_normal(n)) + 0.1
    w[5] = np.inf
    w[10] = -np.inf
    df = pl.DataFrame({"y": rng.standard_normal(n), "x1": rng.standard_normal(n), "w": w})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec, weights="w")
    assert arrays.n_obs == n - 2
    assert arrays.weights is not None
    assert not np.any(np.isinf(arrays.weights))


# ── Edge cases ────────────────────────────────────────────────────


def test_single_inf_row_large_dataset():
    """A single inf row in a large dataset is silently dropped."""
    rng = np.random.default_rng(215)
    n = 10_000
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    y[5000] = np.inf
    df = pl.DataFrame({"y": y, "x1": x1})
    result = ols("y ~ x1", data=df)
    assert result.n_obs == n - 1
    np.testing.assert_allclose(result.coefficients[0], 2.0, atol=0.1)


def test_neg_inf_only():
    """-inf is treated the same as +inf (dropped)."""
    rng = np.random.default_rng(216)
    n = 50
    x1 = rng.standard_normal(n)
    y = rng.standard_normal(n)
    x1[0] = -np.inf
    x1[1] = -np.inf
    y[2] = -np.inf
    df = pl.DataFrame({"y": y, "x1": x1})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 3
    assert not np.any(np.isinf(arrays.y))
    assert not np.any(np.isinf(arrays.X))


def test_inf_with_parse_formula():
    """End-to-end test using parse_formula (string formula) with extract_arrays."""
    rng = np.random.default_rng(217)
    n = 60
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = rng.standard_normal(n)
    y[0] = np.inf
    x2[1] = -np.inf
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = parse_formula("y ~ x1 + x2")
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == n - 2
    assert not np.any(np.isinf(arrays.y))
    assert not np.any(np.isinf(arrays.X))
