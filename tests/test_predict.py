import numpy as np
import polars as pl
import pytest

from polars_reg._ols import ols


def test_predict_continuous_roundtrip():
    """Predict on training data should match X @ beta."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 3.0 + 1.5 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    result = ols("y ~ x1 + x2", data=df)

    # Out-of-sample predict on training data should equal X @ beta
    pred = result.predict(new_data=df)
    coefs = dict(zip(result.names, result.coefficients))
    expected = coefs["x1"] * x1 + coefs["x2"] * x2 + coefs["_cons"]
    np.testing.assert_allclose(pred, expected, atol=1e-12)

    # Also check that y - predict ~ residuals
    np.testing.assert_allclose(y - pred, result.residuals, atol=1e-12)


def test_predict_indicator_variables():
    """Predict with indicator (dummy) variables: i.industry."""
    rng = np.random.default_rng(123)
    n = 300
    industry = rng.choice([1, 2, 3], size=n)
    x1 = rng.standard_normal(n)
    y = 1.0 * x1 + 2.0 * (industry == 2) + 3.0 * (industry == 3) + rng.standard_normal(n) * 0.2
    df = pl.DataFrame({"y": y, "x1": x1, "industry": industry})

    result = ols("y ~ x1 + i.industry", data=df)

    # Coefficient names should include industry=2 and industry=3
    assert "industry=2" in result.names
    assert "industry=3" in result.names

    # Predict on training data should match y - residuals
    pred = result.predict(new_data=df)
    expected_fitted = y - result.residuals
    np.testing.assert_allclose(pred, expected_fitted, atol=1e-10)

    # Predict on new data with known industry levels
    new_df = pl.DataFrame(
        {
            "x1": [0.0, 0.0, 0.0],
            "industry": [1, 2, 3],
        }
    )
    pred_new = result.predict(new_data=new_df)
    # For industry=1 (reference), prediction = _cons + 0*x1
    # For industry=2, prediction = _cons + coef(industry=2)
    # For industry=3, prediction = _cons + coef(industry=3)
    coefs = dict(zip(result.names, result.coefficients))
    expected_1 = coefs["_cons"]
    expected_2 = coefs["_cons"] + coefs["industry=2"]
    expected_3 = coefs["_cons"] + coefs["industry=3"]
    np.testing.assert_allclose(pred_new, [expected_1, expected_2, expected_3], atol=1e-12)


def test_predict_interaction_continuous():
    """Predict with continuous interaction terms x1:x2."""
    rng = np.random.default_rng(99)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * x1 * x2 + rng.standard_normal(n) * 0.2
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    result = ols("y ~ x1 + x2 + x1:x2", data=df)
    assert "x1:x2" in result.names

    # Round-trip: predict on training data should match y - residuals
    pred = result.predict(new_data=df)
    expected_fitted = y - result.residuals
    np.testing.assert_allclose(pred, expected_fitted, atol=1e-10)


def test_predict_indicator_continuous_interaction():
    """Predict with indicator:continuous interaction i.group:x1."""
    rng = np.random.default_rng(77)
    n = 300
    group = rng.choice([1, 2, 3], size=n)
    x1 = rng.standard_normal(n)
    # Different slopes for each group
    y = (
        1.0
        + 0.5 * x1
        + 1.0 * (group == 2) * x1
        + 2.0 * (group == 3) * x1
        + rng.standard_normal(n) * 0.2
    )
    df = pl.DataFrame({"y": y, "x1": x1, "group": group})

    result = ols("y ~ x1 + i.group:x1", data=df)

    # Should have interaction terms like group=2:x1, group=3:x1
    assert any("group=2" in nm and "x1" in nm for nm in result.names)
    assert any("group=3" in nm and "x1" in nm for nm in result.names)

    # Round-trip: predict on training data should match y - residuals
    pred = result.predict(new_data=df)
    expected_fitted = y - result.residuals
    np.testing.assert_allclose(pred, expected_fitted, atol=1e-10)

    # Predict on new data
    new_df = pl.DataFrame(
        {
            "x1": [1.0, 1.0, 1.0],
            "group": [1, 2, 3],
        }
    )
    pred_new = result.predict(new_data=new_df)
    assert pred_new.shape == (3,)
    # Predictions for different groups should differ
    assert not np.allclose(pred_new[0], pred_new[1])
    assert not np.allclose(pred_new[0], pred_new[2])


def test_predict_interval_keys_and_shapes():
    """predict_interval returns dict with correct keys and 1-D arrays."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    y = 2.0 + 1.5 * x1 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1})

    result = ols("y ~ x1", data=df)
    pi = result.predict_interval(new_data=df)

    assert set(pi.keys()) == {"fit", "se", "lower", "upper"}
    for key in ("fit", "se", "lower", "upper"):
        assert pi[key].ndim == 1
        assert pi[key].shape == (n,)

    # fit should match predict()
    np.testing.assert_allclose(pi["fit"], result.predict(new_data=df), atol=1e-12)

    # lower < fit < upper
    assert np.all(pi["lower"] < pi["fit"])
    assert np.all(pi["fit"] < pi["upper"])

    # se should be positive
    assert np.all(pi["se"] > 0)


def test_predict_interval_alpha():
    """Wider intervals with higher confidence (lower alpha)."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1})

    result = ols("y ~ x1", data=df)
    pi_95 = result.predict_interval(new_data=df, alpha=0.05)
    pi_99 = result.predict_interval(new_data=df, alpha=0.01)

    # 99% interval should be wider than 95%
    width_95 = pi_95["upper"] - pi_95["lower"]
    width_99 = pi_99["upper"] - pi_99["lower"]
    assert np.all(width_99 > width_95)


def test_predict_missing_column_error():
    """predict() should raise KeyError when new_data is missing a required column."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + x1 + x2 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    result = ols("y ~ x1 + x2", data=df)

    # new_data missing x2
    bad_df = pl.DataFrame({"x1": [1.0, 2.0]})
    with pytest.raises(KeyError, match="x2"):
        result.predict(new_data=bad_df)


def test_predict_missing_indicator_column_error():
    """predict() should raise KeyError for missing indicator base column."""
    rng = np.random.default_rng(42)
    n = 100
    industry = rng.choice([1, 2, 3], size=n)
    x1 = rng.standard_normal(n)
    y = x1 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1, "industry": industry})

    result = ols("y ~ x1 + i.industry", data=df)

    # new_data missing 'industry' column
    bad_df = pl.DataFrame({"x1": [1.0, 2.0]})
    with pytest.raises(KeyError, match="industry"):
        result.predict(new_data=bad_df)


def test_predict_indicator_string_levels():
    """Indicator variables with string levels should work correctly."""
    rng = np.random.default_rng(42)
    n = 200
    color = rng.choice(["red", "green", "blue"], size=n)
    x1 = rng.standard_normal(n)
    y = (
        1.0 * x1
        + 2.0 * (color == "green").astype(float)
        + 3.0 * (color == "red").astype(float)
        + rng.standard_normal(n) * 0.2
    )
    df = pl.DataFrame({"y": y, "x1": x1, "color": color})

    result = ols("y ~ x1 + i.color", data=df)

    # Round-trip: predict on training data should match y - residuals
    pred = result.predict(new_data=df)
    expected_fitted = y - result.residuals
    np.testing.assert_allclose(pred, expected_fitted, atol=1e-10)


def test_predict_no_intercept():
    """Predict without intercept (formula with -1)."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    y = 1.5 * x1 + rng.standard_normal(n) * 0.3
    df = pl.DataFrame({"y": y, "x1": x1})

    result = ols("y ~ x1 - 1", data=df)
    assert "_cons" not in result.names

    pred = result.predict(new_data=df)
    expected = result.coefficients[0] * x1
    np.testing.assert_allclose(pred, expected, atol=1e-12)


def test_predict_with_nan_in_new_data():
    """predict() handles NaN in new_data gracefully."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"x": x, "y": y})
    result = ols("y ~ x", data=df)

    # new_data with a NaN — predict should still work (NaN propagates to prediction)
    new_df = pl.DataFrame({"x": [1.0, float("nan"), 3.0]})
    preds = result.predict(new_data=new_df)
    assert len(preds) == 3
    assert np.isfinite(preds[0])
    assert np.isfinite(preds[2])


def test_predict_with_inf_in_new_data():
    """predict() handles inf in new_data gracefully."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"x": x, "y": y})
    result = ols("y ~ x", data=df)

    # new_data with inf — predict should still work (inf propagates to prediction)
    new_df = pl.DataFrame({"x": [1.0, np.inf, -np.inf]})
    preds = result.predict(new_data=new_df)
    assert len(preds) == 3
    assert np.isfinite(preds[0])
