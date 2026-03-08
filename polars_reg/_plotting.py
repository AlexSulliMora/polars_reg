"""Plotting utilities for polars_reg: coefficient plots and added-variable plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import altair as alt

    from polars_reg._results import RegressionResult


def _ensure_altair():
    """Lazy-import altair; raise a helpful error if not installed."""
    try:
        import altair as alt

        return alt
    except ImportError:
        raise ImportError("Install altair for plotting: pip install altair") from None


def coefplot(
    *results: RegressionResult,
    labels: list[str] | None = None,
    alpha: float = 0.05,
    variables: list[str] | None = None,
    exclude: list[str] | None = None,
) -> alt.Chart | alt.LayerChart:
    """Coefficient plot with confidence-interval whiskers.

    Args:
        *results: One or more RegressionResult objects.
        labels: Legend labels when plotting multiple models.
        alpha: Significance level for confidence intervals (default 0.05).
        variables: If given, only plot these variable names.
        exclude: Variable names to exclude (default excludes ``_cons``).

    Returns:
        An Altair chart.
    """
    alt = _ensure_altair()

    if len(results) == 0:
        raise ValueError("At least one RegressionResult is required.")

    if exclude is None:
        exclude = ["_cons"]

    # Determine the set of variables to plot, preserving order from first result
    all_vars: list[str] = []
    seen: set[str] = set()
    for res in results:
        for name in res.names:
            if name not in seen:
                seen.add(name)
                all_vars.append(name)

    # Apply variable filters
    if variables is not None:
        all_vars = [v for v in all_vars if v in variables]
    all_vars = [v for v in all_vars if v not in exclude]

    if not all_vars:
        raise ValueError("No variables to plot after filtering.")

    # Build data for all models
    rows = []
    for m_idx, res in enumerate(results):
        ci = res.confint(alpha=alpha)
        model_label = (
            labels[m_idx] if labels is not None and m_idx < len(labels) else f"Model {m_idx + 1}"
        )
        for name in all_vars:
            if name in res.names:
                c_idx = res.names.index(name)
                rows.append(
                    {
                        "variable": name,
                        "coef": float(res.coefficients[c_idx]),
                        "ci_lower": float(ci[c_idx, 0]),
                        "ci_upper": float(ci[c_idx, 1]),
                        "model": model_label,
                    }
                )

    df = pl.DataFrame(rows)

    # Variable ordering: reverse so first variable is at top
    var_order = list(reversed(all_vars))

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("coef:Q", title="Coefficient"),
            y=alt.Y("variable:N", sort=var_order, title=None),
        )
    )
    whiskers = (
        alt.Chart(df)
        .mark_rule()
        .encode(
            x=alt.X("ci_lower:Q"),
            x2=alt.X2("ci_upper:Q"),
            y=alt.Y("variable:N", sort=var_order),
        )
    )
    zero_line = (
        alt.Chart(pl.DataFrame({"x": [0]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q")
    )

    if len(results) > 1:
        color_enc = alt.Color("model:N", title="Model")
        points = points.encode(color=color_enc)
        whiskers = whiskers.encode(color=color_enc)
        # Offset models vertically
        points = points.encode(
            yOffset=alt.YOffset("model:N"),
        )
        whiskers = whiskers.encode(
            yOffset=alt.YOffset("model:N"),
        )

    chart = (zero_line + whiskers + points).properties(
        title="Coefficient Plot", width=400, height=max(100, 30 * len(all_vars))
    )

    return chart


def avplot(
    result: RegressionResult,
    variable: str | None = None,
) -> alt.Chart | alt.ConcatChart:
    """Added-variable (partial regression) plot using Frisch-Waugh-Lovell.

    For each variable ``x_j``, the plot shows the relationship between ``y``
    and ``x_j`` after partialling out the other regressors.

    Args:
        result: A RegressionResult with ``_X`` and ``_y`` stored.
        variable: Name of the variable to plot. If None, plot all variables
            (excluding ``_cons``) in a grid.

    Returns:
        Altair Chart (single variable) or ConcatChart (all variables).

    Raises:
        ValueError: If ``_X`` or ``_y`` is not stored on the result, or if
            the named variable is not found.
    """
    alt = _ensure_altair()

    if result._X is None or result._y is None:
        raise ValueError(
            "Added-variable plot requires _X and _y stored on the result. "
            "Re-run the estimator if needed."
        )

    X = result._X
    y = result._y
    names = result.names

    def _make_avplot(j: int, var_name: str) -> alt.LayerChart:
        e_y, e_x = _partial_residuals(X, y, j)
        slope = result.coefficients[j]

        df = pl.DataFrame({"e_x": e_x, "e_y": e_y})

        scatter = (
            alt.Chart(df)
            .mark_circle(opacity=0.5, size=20)
            .encode(
                x=alt.X("e_x:Q", title=f"e({var_name} | others)"),
                y=alt.Y("e_y:Q", title="e(y | others)"),
            )
        )

        # Fitted line from FWL slope
        x_min, x_max = float(e_x.min()), float(e_x.max())
        line_df = pl.DataFrame(
            {
                "e_x": [x_min, x_max],
                "e_y": [slope * x_min, slope * x_max],
            }
        )
        line = alt.Chart(line_df).mark_line(color="red", strokeWidth=2).encode(x="e_x:Q", y="e_y:Q")

        return (scatter + line).properties(
            title=f"Added-Variable Plot: {var_name}", width=300, height=250
        )

    if variable is not None:
        if variable not in names:
            raise ValueError(f"Variable '{variable}' not found in result. Available: {names}")
        j = names.index(variable)
        return _make_avplot(j, variable)

    # All-variables mode: grid, skip _cons
    plot_vars = [(i, name) for i, name in enumerate(names) if name != "_cons"]
    if not plot_vars:
        raise ValueError("No non-constant variables to plot.")

    charts = [_make_avplot(j, name) for j, name in plot_vars]

    # Arrange in rows of 3
    n_cols = min(len(charts), 3)
    rows = []
    for i in range(0, len(charts), n_cols):
        row_charts = charts[i : i + n_cols]
        rows.append(alt.hconcat(*row_charts))

    return alt.vconcat(*rows)


def _partial_residuals(X: np.ndarray, y: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute FWL partial residuals for variable at column index j.

    Returns:
        (e_y, e_x): Residuals from regressing y and X[:,j] on all other X columns.
    """
    n, k = X.shape
    # Build X_other: all columns except j
    cols = list(range(k))
    cols.remove(j)

    if len(cols) == 0:
        # Only one regressor — residuals are just y and x_j (demeaned if _cons)
        return y - y.mean(), X[:, j] - X[:, j].mean()

    X_other = X[:, cols]

    # Regress y on X_other
    beta_y, _, _, _ = np.linalg.lstsq(X_other, y, rcond=None)
    e_y = y - X_other @ beta_y

    # Regress X[:,j] on X_other
    x_j = X[:, j]
    beta_x, _, _, _ = np.linalg.lstsq(X_other, x_j, rcond=None)
    e_x = x_j - X_other @ beta_x

    return e_y, e_x
