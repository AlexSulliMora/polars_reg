"""Rolling-window regression: run a regression over sliding time windows."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._utils import ensure_polars


@dataclass
class RollingRegressionResult:
    """Collection of RegressionResult objects keyed by window end-period.

    When ``group_by`` was used in ``rolling_reg()``, keys are
    ``(entity_value, window_end)`` tuples; otherwise they are scalar
    window-end period values.

    Supports dict-like access by key, stacked coefficient tables,
    time-series coefficient extraction, and optional Altair plotting.
    """

    results: OrderedDict[Any, RegressionResult] = field(default_factory=OrderedDict)
    failed: OrderedDict[Any, str] = field(default_factory=OrderedDict)
    window_size: int = 0
    stride: int = 1
    time_col: str = ""

    def __getitem__(self, key: Any) -> RegressionResult:
        return self.results[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.results)

    def keys(self) -> KeysView[Any]:
        return self.results.keys()

    def values(self) -> ValuesView[RegressionResult]:
        return self.results.values()

    def items(self) -> ItemsView[Any, RegressionResult]:
        return self.results.items()

    def coef_table(self) -> pl.DataFrame:
        """Stacked coefficient table with a ``window`` column."""
        tables = []
        for key, result in self.results.items():
            t = result.coef_table()
            window_val = str(key) if not isinstance(key, str) else key
            t = t.with_columns(pl.lit(window_val).alias("window"))
            tables.append(t)
        if not tables:
            return pl.DataFrame()
        return pl.concat(tables).select(
            ["window"] + [c for c in tables[0].columns if c != "window"]
        )

    def coef_series(self, alpha: float = 0.05) -> pl.DataFrame:
        """Long-format coefficient time series with confidence intervals.

        Args:
            alpha: Significance level for confidence intervals (default 0.05).

        Returns:
            Polars DataFrame with columns ``[time, variable, coefficient,
            se, ci_lower, ci_upper]``.  When ``group_by`` was used, an
            additional ``group`` column is included.
        """
        rows: list[dict[str, Any]] = []
        has_groups = self._has_tuple_keys()

        for key, result in self.results.items():
            ci = result.confint(alpha)
            if has_groups:
                group_val, time_val = key
            else:
                group_val = None
                time_val = key

            for i, name in enumerate(result.names):
                row: dict[str, Any] = {}
                if has_groups:
                    row["group"] = group_val
                row["time"] = time_val
                row["variable"] = name
                row["coefficient"] = float(result.coefficients[i])
                row["se"] = float(result.se[i])
                row["ci_lower"] = float(ci[i, 0])
                row["ci_upper"] = float(ci[i, 1])
                rows.append(row)

        if not rows:
            cols = ["time", "variable", "coefficient", "se", "ci_lower", "ci_upper"]
            if has_groups:
                cols = ["group"] + cols
            return pl.DataFrame(
                schema={c: pl.Float64 if c not in ("group", "variable") else pl.Utf8 for c in cols}
            )

        return pl.DataFrame(rows)

    def plot_coefs(
        self,
        variables: list[str] | None = None,
        alpha: float = 0.05,
    ) -> Any:
        """Altair chart of rolling coefficients with confidence bands.

        Args:
            variables: Subset of coefficient names to plot. If ``None``,
                all coefficients are plotted.
            alpha: Significance level for confidence bands (default 0.05).

        Returns:
            An Altair ``Chart`` (or ``LayerChart``) object.

        Raises:
            ImportError: If ``altair`` is not installed.
        """
        try:
            import altair as alt
        except ImportError:
            raise ImportError(
                "altair is required for plot_coefs(). Install it with: pip install altair"
            )

        cs = self.coef_series(alpha)
        if variables is not None:
            cs = cs.filter(pl.col("variable").is_in(variables))

        df_pd = cs.to_pandas()

        line = (
            alt.Chart(df_pd)
            .mark_line()
            .encode(
                x=alt.X("time:Q", title=self.time_col),
                y=alt.Y("coefficient:Q"),
                color="variable:N",
            )
        )
        band = (
            alt.Chart(df_pd)
            .mark_area(opacity=0.2)
            .encode(
                x="time:Q",
                y="ci_lower:Q",
                y2="ci_upper:Q",
                color="variable:N",
            )
        )

        return band + line

    def summary(self) -> str:
        """Compact text summary of the rolling regression."""
        n_ok = len(self.results)
        n_fail = len(self.failed)
        lines = [
            f"Rolling Regression: window={self.window_size}, stride={self.stride}, "
            f"time_col='{self.time_col}'",
            f"  {n_ok} windows succeeded" + (f", {n_fail} failed" if n_fail else ""),
        ]
        if n_ok > 0:
            # Coefficient ranges
            all_names: set[str] = set()
            for r in self.results.values():
                all_names.update(r.names)
            for name in sorted(all_names):
                coefs = [
                    float(r.coefficients[list(r.names).index(name)])
                    for r in self.results.values()
                    if name in r.names
                ]
                lines.append(f"  {name}: [{min(coefs):.4f}, {max(coefs):.4f}]")
        return "\n".join(lines)

    def by_entity(self) -> dict[Any, RollingRegressionResult]:
        """Split into per-entity RollingRegressionResult objects.

        Only valid when ``group_by`` was used in ``rolling_reg()``
        (i.e., keys are ``(entity, window_end)`` tuples).

        Returns:
            Dict mapping entity values to ``RollingRegressionResult``
            containing only that entity's windows.

        Raises:
            ValueError: If keys are not tuples (no ``group_by`` was used).
        """
        if not self._has_tuple_keys():
            raise ValueError(
                "by_entity() requires tuple keys from a group_by rolling regression. "
                "This result was created without group_by."
            )

        entity_results: dict[Any, OrderedDict[Any, RegressionResult]] = {}
        entity_failed: dict[Any, OrderedDict[Any, str]] = {}

        for key, result in self.results.items():
            entity, window_end = key
            if entity not in entity_results:
                entity_results[entity] = OrderedDict()
                entity_failed[entity] = OrderedDict()
            entity_results[entity][window_end] = result

        for key, reason in self.failed.items():
            entity, window_end = key
            if entity not in entity_failed:
                entity_results[entity] = OrderedDict()
                entity_failed[entity] = OrderedDict()
            entity_failed[entity][window_end] = reason

        out: dict[Any, RollingRegressionResult] = {}
        all_entities = set(entity_results.keys()) | set(entity_failed.keys())
        for entity in all_entities:
            out[entity] = RollingRegressionResult(
                results=entity_results.get(entity, OrderedDict()),
                failed=entity_failed.get(entity, OrderedDict()),
                window_size=self.window_size,
                stride=self.stride,
                time_col=self.time_col,
            )

        return out

    def _has_tuple_keys(self) -> bool:
        """Check whether keys are tuples (group_by was used)."""
        for key in self.results:
            return isinstance(key, tuple)
        for key in self.failed:
            return isinstance(key, tuple)
        return False

    def __repr__(self) -> str:
        return (
            f"<RollingRegressionResult windows={len(self.results)} "
            f"failed={len(self.failed)} window_size={self.window_size} "
            f"stride={self.stride}>"
        )


def rolling_reg(
    estimator_fn: Callable[..., RegressionResult],
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    time: str,
    window: int,
    stride: int = 1,
    group_by: str | list[str] | None = None,
    min_obs: int = 0,
    window_type: str = "periods",
    store_residuals: bool = False,
    **kwargs: Any,
) -> RollingRegressionResult:
    """Run a regression over sliding time windows.

    Slides a window across the time dimension and runs the given estimator
    on each window's data.  Supports per-entity rolling via ``group_by``.

    Args:
        estimator_fn: Any polars_reg estimator (``ols``, ``iv2sls``, etc.).
        formula: Formula string (e.g. ``"y ~ x1 + x2"``).
        data: Polars DataFrame or LazyFrame.
        time: Column name for the time dimension. Data is sorted by this
            column before windowing.
        window: Window size.  Interpretation depends on ``window_type``:
            ``"periods"`` counts distinct time values, ``"obs"`` counts rows.
        stride: Step size between consecutive windows (default 1).
        group_by: Column name(s) identifying entities for per-entity rolling.
            When provided, windows are formed per entity and result keys are
            ``(entity_value, window_end)`` tuples.
        min_obs: Minimum number of observations required per window.  Windows
            with fewer observations are skipped.  Defaults to ``window``
            (require full windows) when set to 0.
        window_type: ``"periods"`` (default) or ``"obs"``.  ``"periods"``
            defines the window over distinct time values; ``"obs"`` defines
            it over row counts.
        store_residuals: If ``False`` (default), residual and design-matrix
            arrays are stripped from each result to save memory.
        **kwargs: Additional arguments passed to the estimator
            (``vcov``, ``cluster``, etc.).

    Returns:
        RollingRegressionResult with results keyed by window end-period
        (or ``(entity, window_end)`` tuples when ``group_by`` is used).

    Raises:
        ValueError: If ``window <= 0``, ``stride <= 0``, ``time`` column
            is missing, or ``window_type`` is invalid.

    Example:
        >>> import polars_reg as pr
        >>> result = pr.rolling_reg(
        ...     pr.ols, "y ~ x1 + x2", data=df,
        ...     time="date", window=60, stride=12,
        ... )
        >>> result.coef_series()  # long-format coefficient time series

    Note:
        With ``group_by``, complexity is O(G * N_windows * cost_per_window)
        where G is the number of entities.  For large panels consider
        increasing ``stride`` or reducing ``window``.
    """
    # ── Validate inputs ──────────────────────────────────────────────
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if window_type not in ("periods", "obs"):
        raise ValueError(f"window_type must be 'periods' or 'obs', got {window_type!r}")

    data = ensure_polars(data)

    # ── Push column selection for LazyFrame ──────────────────────────
    if isinstance(group_by, str):
        group_by = [group_by]

    if isinstance(data, pl.LazyFrame):
        spec = parse_formula(formula)
        exog_cols: list[str] = []
        for col in spec.exog:
            if ":" in col:
                exog_cols.extend(col.split(":"))
            else:
                exog_cols.append(col)
        needed = [spec.depvar] + exog_cols + spec.fe + spec.endog + spec.instruments + [time]
        if group_by:
            needed.extend(group_by)
        cluster_kw = kwargs.get("cluster")
        if isinstance(cluster_kw, str):
            needed.append(cluster_kw)
        elif isinstance(cluster_kw, list):
            needed.extend(cluster_kw)
        entity_kw = kwargs.get("entity")
        if entity_kw:
            needed.append(entity_kw)
        time_kw = kwargs.get("time")
        if time_kw:
            needed.append(time_kw)
        needed = list(dict.fromkeys(needed))
        data = data.select(needed).collect()

    # Validate time column exists
    if time not in data.columns:
        raise ValueError(f"Time column '{time}' not found in data")

    # Default min_obs to window when 0
    if min_obs == 0:
        min_obs = window

    # ── Sort data ────────────────────────────────────────────────────
    sort_cols = (group_by or []) + [time]
    data = data.sort(sort_cols)

    result = RollingRegressionResult(
        window_size=window,
        stride=stride,
        time_col=time,
    )

    if group_by:
        _rolling_with_groups(
            result,
            estimator_fn,
            formula,
            data,
            time,
            window,
            stride,
            group_by,
            min_obs,
            window_type,
            store_residuals,
            kwargs,
        )
    else:
        _rolling_no_groups(
            result,
            estimator_fn,
            formula,
            data,
            time,
            window,
            stride,
            min_obs,
            window_type,
            store_residuals,
            kwargs,
        )

    return result


def _rolling_no_groups(
    result: RollingRegressionResult,
    estimator_fn: Callable[..., RegressionResult],
    formula: str,
    data: pl.DataFrame,
    time: str,
    window: int,
    stride: int,
    min_obs: int,
    window_type: str,
    store_residuals: bool,
    kwargs: dict[str, Any],
) -> None:
    """Slide windows over the full dataset (no grouping)."""
    if window_type == "periods":
        unique_times = data[time].unique(maintain_order=True).sort().to_list()
        n_periods = len(unique_times)

        for start_idx in range(0, n_periods - window + 1, stride):
            end_idx = start_idx + window - 1
            t_start = unique_times[start_idx]
            t_end = unique_times[end_idx]
            window_key = t_end

            window_df = data.filter((pl.col(time) >= t_start) & (pl.col(time) <= t_end))

            if len(window_df) < min_obs:
                result.failed[window_key] = f"Too few observations ({len(window_df)} < {min_obs})"
                continue

            _run_window(
                result,
                estimator_fn,
                formula,
                window_df,
                window_key,
                store_residuals,
                kwargs,
            )
    else:
        # window_type == "obs"
        n_rows = len(data)
        for start_row in range(0, n_rows - window + 1, stride):
            window_df = data.slice(start_row, window)

            # Key by the last time value in the window
            window_key = window_df[time][-1]

            if len(window_df) < min_obs:
                result.failed[window_key] = f"Too few observations ({len(window_df)} < {min_obs})"
                continue

            _run_window(
                result,
                estimator_fn,
                formula,
                window_df,
                window_key,
                store_residuals,
                kwargs,
            )


def _rolling_with_groups(
    result: RollingRegressionResult,
    estimator_fn: Callable[..., RegressionResult],
    formula: str,
    data: pl.DataFrame,
    time: str,
    window: int,
    stride: int,
    group_by: list[str],
    min_obs: int,
    window_type: str,
    store_residuals: bool,
    kwargs: dict[str, Any],
) -> None:
    """Slide windows per entity group."""
    # Extract unique time periods from the FULL dataset for consistent alignment
    global_unique_times = data[time].unique(maintain_order=True).sort().to_list()

    for group_keys, group_df in data.group_by(group_by, maintain_order=True):
        entity = group_keys[0] if len(group_keys) == 1 else group_keys

        if window_type == "periods":
            n_periods = len(global_unique_times)

            for start_idx in range(0, n_periods - window + 1, stride):
                end_idx = start_idx + window - 1
                t_start = global_unique_times[start_idx]
                t_end = global_unique_times[end_idx]
                window_key = (entity, t_end)

                window_df = group_df.filter((pl.col(time) >= t_start) & (pl.col(time) <= t_end))

                if len(window_df) < min_obs:
                    result.failed[window_key] = (
                        f"Too few observations ({len(window_df)} < {min_obs})"
                    )
                    continue

                _run_window(
                    result,
                    estimator_fn,
                    formula,
                    window_df,
                    window_key,
                    store_residuals,
                    kwargs,
                )
        else:
            # window_type == "obs"
            n_rows = len(group_df)
            for start_row in range(0, n_rows - window + 1, stride):
                window_df = group_df.slice(start_row, window)
                window_key = (entity, window_df[time][-1])

                if len(window_df) < min_obs:
                    result.failed[window_key] = (
                        f"Too few observations ({len(window_df)} < {min_obs})"
                    )
                    continue

                _run_window(
                    result,
                    estimator_fn,
                    formula,
                    window_df,
                    window_key,
                    store_residuals,
                    kwargs,
                )


def _run_window(
    result: RollingRegressionResult,
    estimator_fn: Callable[..., RegressionResult],
    formula: str,
    window_df: pl.DataFrame,
    window_key: Any,
    store_residuals: bool,
    kwargs: dict[str, Any],
) -> None:
    """Run the estimator on a single window and store the result."""
    try:
        r = estimator_fn(formula, data=window_df, **kwargs)
        if not store_residuals:
            r.residuals = np.empty(0)
            r._X = None
            r._y = None
        result.results[window_key] = r
    except Exception as e:
        result.failed[window_key] = str(e)
        warnings.warn(
            f"Window {window_key}: regression failed — {e}",
            stacklevel=4,
        )
