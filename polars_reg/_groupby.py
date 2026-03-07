"""GroupBy regression: run the same regression per group."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._utils import ensure_polars


@dataclass
class GroupRegressionResult:
    """Collection of RegressionResult objects keyed by group value(s).

    Supports dict-like access by group key, stacked coefficient tables,
    and a compact multi-group summary.
    """

    results: OrderedDict[Any, RegressionResult] = field(default_factory=OrderedDict)
    failed: OrderedDict[Any, str] = field(default_factory=OrderedDict)

    def __getitem__(self, key: Any) -> RegressionResult:
        return self.results[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def coef_table(self) -> pl.DataFrame:
        """Stacked coefficient table with a 'group' column."""
        tables = []
        for key, result in self.results.items():
            t = result.coef_table()
            group_val = str(key) if not isinstance(key, str) else key
            t = t.with_columns(pl.lit(group_val).alias("group"))
            tables.append(t)
        if not tables:
            return pl.DataFrame()
        return pl.concat(tables).select(["group"] + [c for c in tables[0].columns if c != "group"])

    def summary(self) -> str:
        """Compact summary across all groups."""
        lines = [
            f"GroupBy Regression: {len(self.results)} groups succeeded"
            + (f", {len(self.failed)} failed" if self.failed else ""),
            "=" * 60,
        ]
        for key, result in self.results.items():
            lines.append(f"\n--- Group: {key} (N={result.n_obs}) ---")
            coefs = ", ".join(f"{n}={c:.4f}" for n, c in zip(result.names, result.coefficients))
            lines.append(f"  Coefs: {coefs}")
            lines.append(f"  R²={result.r_squared:.4f}")
        if self.failed:
            lines.append("\n--- Failed groups ---")
            for key, reason in self.failed.items():
                lines.append(f"  {key}: {reason}")
        return "\n".join(lines)


def groupby_reg(
    estimator_fn: Callable[..., RegressionResult],
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    group_by: str | list[str],
    min_obs: int = 0,
    **kwargs: Any,
) -> GroupRegressionResult:
    """Run a regression per group.

    Args:
        estimator_fn: Any polars_reg estimator (ols, iv2sls, liml, etc.)
        formula: Formula string
        data: Polars DataFrame or LazyFrame
        group_by: Column name(s) to group by
        min_obs: Minimum observations per group (skip groups below this)
        **kwargs: Additional arguments passed to the estimator (vcov, cluster, etc.)

    Returns:
        GroupRegressionResult with results keyed by group value(s)
    """
    data = ensure_polars(data)

    if isinstance(group_by, str):
        group_by = [group_by]

    # Push column selection into LazyFrame before collecting
    if isinstance(data, pl.LazyFrame):
        spec = parse_formula(formula)
        exog_cols = []
        for col in spec.exog:
            if ":" in col:
                exog_cols.extend(col.split(":"))
            else:
                exog_cols.append(col)
        needed = [spec.depvar] + exog_cols + spec.fe + spec.endog + spec.instruments + group_by
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

    result = GroupRegressionResult()

    for group_keys, group_df in data.group_by(group_by, maintain_order=True):
        # group_keys is a tuple of values
        key = group_keys[0] if len(group_keys) == 1 else group_keys

        if len(group_df) < min_obs:
            result.failed[key] = f"Too few observations ({len(group_df)} < {min_obs})"
            continue

        try:
            r = estimator_fn(formula, data=group_df, **kwargs)
            result.results[key] = r
        except Exception as e:
            result.failed[key] = str(e)
            warnings.warn(
                f"Group {key}: regression failed — {e}",
                stacklevel=2,
            )

    return result
