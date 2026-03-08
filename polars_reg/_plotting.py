"""Plotting utilities for polars_reg: coefficient plots and added-variable plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from polars_reg._results import RegressionResult


def _ensure_mpl():
    """Lazy-import matplotlib; raise a helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError("Install matplotlib for plotting: pip install matplotlib") from None


def coefplot(
    *results: RegressionResult,
    labels: list[str] | None = None,
    alpha: float = 0.05,
    variables: list[str] | None = None,
    exclude: list[str] | None = None,
    horizontal: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Coefficient plot with confidence-interval whiskers.

    Args:
        *results: One or more RegressionResult objects.
        labels: Legend labels when plotting multiple models.
        alpha: Significance level for confidence intervals (default 0.05).
        variables: If given, only plot these variable names.
        exclude: Variable names to exclude (default excludes ``_cons``).
        horizontal: If True (default), variables on y-axis, coefficients on x-axis.
        ax: Existing matplotlib Axes to draw on.
        **kwargs: Passed to ``errorbar()``.

    Returns:
        The matplotlib Axes containing the plot.
    """
    plt = _ensure_mpl()

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

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots()

    n_models = len(results)
    n_vars = len(all_vars)

    # Offsets for multi-model grouping
    if n_models > 1:
        offsets = np.linspace(-0.2, 0.2, n_models)
    else:
        offsets = np.array([0.0])

    for m_idx, res in enumerate(results):
        ci = res.confint(alpha=alpha)
        name_to_idx = {name: i for i, name in enumerate(res.names)}

        positions = []
        coefs = []
        ci_lo = []
        ci_hi = []

        for v_idx, var in enumerate(all_vars):
            if var in name_to_idx:
                c_idx = name_to_idx[var]
                positions.append(v_idx + offsets[m_idx])
                coefs.append(res.coefficients[c_idx])
                ci_lo.append(ci[c_idx, 0])
                ci_hi.append(ci[c_idx, 1])

        coefs_arr = np.array(coefs)
        lower_err = coefs_arr - np.array(ci_lo)
        upper_err = np.array(ci_hi) - coefs_arr

        label = labels[m_idx] if labels is not None and m_idx < len(labels) else None

        if horizontal:
            ax.errorbar(
                coefs_arr,
                positions,
                xerr=[lower_err, upper_err],
                fmt="o",
                label=label,
                capsize=3,
                **kwargs,
            )
        else:
            ax.errorbar(
                positions,
                coefs_arr,
                yerr=[lower_err, upper_err],
                fmt="o",
                label=label,
                capsize=3,
                **kwargs,
            )

    # Reference line at 0
    if horizontal:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(all_vars)
        ax.set_xlabel("Coefficient")
    else:
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(all_vars)
        ax.set_ylabel("Coefficient")

    # Legend for multi-model
    if n_models > 1 and labels is not None:
        ax.legend()

    return ax


def avplot(
    result: RegressionResult,
    variable: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes | matplotlib.figure.Figure:
    """Added-variable (partial regression) plot using Frisch-Waugh-Lovell.

    For each variable ``x_j``, the plot shows the relationship between ``y``
    and ``x_j`` after partialling out the other regressors.

    Args:
        result: A RegressionResult with ``_X`` and ``_y`` stored.
        variable: Name of the variable to plot. If None, plot all variables
            (excluding ``_cons``) in a grid of subplots.
        ax: Existing matplotlib Axes (only used when ``variable`` is specified).
        **kwargs: Passed to ``scatter()``.

    Returns:
        matplotlib Axes (single variable) or Figure (all variables).

    Raises:
        ValueError: If ``_X`` or ``_y`` is not stored on the result, or if
            the named variable is not found.
    """
    plt = _ensure_mpl()

    if result._X is None or result._y is None:
        raise ValueError(
            "Added-variable plot requires _X and _y stored on the result. "
            "Re-run the estimator if needed."
        )

    X = result._X
    y = result._y
    names = result.names

    if variable is not None:
        # Single variable mode
        if variable not in names:
            raise ValueError(f"Variable '{variable}' not found in result. Available: {names}")
        if ax is None:
            _, ax = plt.subplots()

        j = names.index(variable)
        e_y, e_x = _partial_residuals(X, y, j)

        ax.scatter(e_x, e_y, alpha=0.5, **kwargs)

        # Fitted line (slope = full regression coefficient for x_j)
        slope = result.coefficients[j]
        x_range = np.array([e_x.min(), e_x.max()])
        ax.plot(x_range, slope * x_range, color="red", linewidth=1.5)

        ax.set_xlabel(f"e({variable} | others)")
        ax.set_ylabel("e(y | others)")
        ax.set_title(f"Added-Variable Plot: {variable}")
        return ax

    # All-variables mode: grid of subplots, skip _cons
    plot_vars = [(i, name) for i, name in enumerate(names) if name != "_cons"]
    if not plot_vars:
        raise ValueError("No non-constant variables to plot.")

    n_plots = len(plot_vars)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.asarray(axes).flatten().tolist()

    for plot_idx, (j, var_name) in enumerate(plot_vars):
        cur_ax = axes_flat[plot_idx]
        e_y, e_x = _partial_residuals(X, y, j)

        cur_ax.scatter(e_x, e_y, alpha=0.5, **kwargs)

        slope = result.coefficients[j]
        x_range = np.array([e_x.min(), e_x.max()])
        cur_ax.plot(x_range, slope * x_range, color="red", linewidth=1.5)

        cur_ax.set_xlabel(f"e({var_name} | others)")
        cur_ax.set_ylabel("e(y | others)")
        cur_ax.set_title(var_name)

    # Hide unused subplot axes
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    return fig


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
