import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402

from polars_reg._plotting import avplot, coefplot  # noqa: E402
from polars_reg._results import RegressionResult  # noqa: E402


def _make_result(names=None, store_Xy=True):
    """Build a simple RegressionResult for testing."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    ones = np.ones(n)
    X = np.column_stack([x1, x2, ones])
    beta_true = np.array([1.5, -0.8, 3.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5

    # OLS fit
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    k = X.shape[1]
    s2 = float(resid @ resid / (n - k))
    vcov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)

    if names is None:
        names = ["x1", "x2", "_cons"]

    result = RegressionResult(
        coefficients=beta_hat,
        vcov=vcov,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=n - k,
        r_squared=r2,
        r_squared_adj=1 - (1 - r2) * (n - 1) / (n - k - 1),
        model_type="OLS",
        vcov_type="iid",
        _X=X if store_Xy else None,
        _y=y if store_Xy else None,
    )
    return result


# --- coefplot tests ---


def test_coefplot_returns_axes():
    res = _make_result()
    ax = coefplot(res)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_coefplot_excludes_cons():
    res = _make_result()
    ax = coefplot(res)
    # y-axis labels should not contain _cons (horizontal mode)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "_cons" not in labels
    assert "x1" in labels
    assert "x2" in labels


def test_coefplot_variables_filter():
    res = _make_result()
    ax = coefplot(res, variables=["x1"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "x1" in labels
    assert "x2" not in labels


def test_coefplot_exclude_filter():
    res = _make_result()
    ax = coefplot(res, exclude=["_cons", "x2"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "x1" in labels
    assert "x2" not in labels
    assert "_cons" not in labels


def test_coefplot_multi_model_with_labels():
    res1 = _make_result()
    res2 = _make_result()
    ax = coefplot(res1, res2, labels=["Model A", "Model B"])
    assert isinstance(ax, matplotlib.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Model A" in legend_texts
    assert "Model B" in legend_texts


# --- avplot tests ---


def test_avplot_single_variable_returns_axes():
    res = _make_result()
    ax = avplot(res, variable="x1")
    assert isinstance(ax, matplotlib.axes.Axes)


def test_avplot_all_variables_returns_figure():
    res = _make_result()
    fig = avplot(res)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_avplot_slope_equals_coefficient():
    """FWL theorem: regressing e_y on e_x should give the same slope as
    the full regression coefficient for that variable."""
    res = _make_result()
    X = res._X
    y = res._y
    names = res.names

    for j, name in enumerate(names):
        if name == "_cons":
            continue
        # Get partial residuals
        from polars_reg._plotting import _partial_residuals

        e_y, e_x = _partial_residuals(X, y, j)

        # Regress e_y on e_x (simple OLS, no intercept)
        slope_fwl = float(e_x @ e_y / (e_x @ e_x))
        np.testing.assert_allclose(
            slope_fwl,
            res.coefficients[j],
            atol=1e-10,
            err_msg=f"FWL slope mismatch for {name}",
        )


def test_avplot_raises_without_Xy():
    res = _make_result(store_Xy=False)
    with pytest.raises(ValueError, match="_X and _y"):
        avplot(res, variable="x1")


def test_avplot_skips_cons_in_grid():
    res = _make_result()
    fig = avplot(res)
    # Should produce a figure with 2 subplots (x1, x2) — _cons is skipped
    visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    assert len(visible_axes) == 2
    titles = [ax.get_title() for ax in visible_axes]
    assert "_cons" not in titles
    assert "x1" in titles
    assert "x2" in titles


def test_coefplot_method_on_result():
    """RegressionResult.coefplot() convenience method works."""
    res = _make_result()
    ax = res.coefplot()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_avplot_method_on_result():
    """RegressionResult.avplot() convenience method works."""
    res = _make_result()
    ax = res.avplot(variable="x1")
    assert isinstance(ax, matplotlib.axes.Axes)
