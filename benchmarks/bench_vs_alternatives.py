"""Benchmark polars_reg against statsmodels, linearmodels, and pyfixest.

Usage:
    python benchmarks/bench_vs_alternatives.py

Compares wall-clock time for:
    1. OLS (iid SE)
    2. OLS + HC1 robust SE
    3. OLS + one-way clustered SE
    4. OLS + 1-way FE + clustered SE  (reghdfe-style)
    5. OLS + 2-way FE + clustered SE
    6. 2SLS / IV
    7. OLS on wide dataset from Parquet (LazyFrame pushdown)
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_panel(n_firms: int, n_years: int, n_extra: int = 0, rng=None):
    """Generate a balanced panel dataset."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(2000, 2000 + n_years), n_firms)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    firm_fe = rng.standard_normal(n_firms)[firm_id]
    year_fe = rng.standard_normal(n_years)[year_id - 2000]
    y = 2.0 + 1.0 * x1 - 0.5 * x2 + 1.5 * x_endog + firm_fe + year_fe + u

    data = {
        "y": y,
        "x1": x1,
        "x2": x2,
        "x_endog": x_endog,
        "z1": z1,
        "z2": z2,
        "firm_id": firm_id,
        "year_id": year_id,
    }
    for i in range(n_extra):
        data[f"extra_{i}"] = rng.standard_normal(n)

    return data


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    package: str
    n_obs: int
    time_ms: float
    coef_x1: float  # sanity check


def bench(name: str, package: str, fn, n_obs: int, warmup: int = 1, reps: int = 5) -> BenchResult:
    """Run fn() with warmup, return median time."""
    # Warmup
    for _ in range(warmup):
        result = fn()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)

    med = sorted(times)[len(times) // 2]

    # Extract x1 coefficient for sanity check
    if isinstance(result, (float, np.floating)):
        coef = float(result)
    elif hasattr(result, "coefficients"):
        # polars_reg
        idx = result.names.index("x1") if "x1" in result.names else 0
        coef = float(result.coefficients[idx])
    elif hasattr(result, "params"):
        # statsmodels / linearmodels
        if "x1" in result.params.index:
            coef = float(result.params["x1"])
        else:
            coef = float(result.params.iloc[0])
    elif hasattr(result, "coef"):
        # pyfixest
        if "x1" in result.coef().index:
            coef = float(result.coef().loc["x1"])
        else:
            coef = float(result.coef().iloc[0])
    else:
        coef = float("nan")

    return BenchResult(name=name, package=package, n_obs=n_obs, time_ms=med * 1000, coef_x1=coef)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def run_benchmarks(n_firms: int, n_years: int):
    n = n_firms * n_years
    rng = np.random.default_rng(42)
    data = make_panel(n_firms, n_years, rng=rng)
    pdf = pd.DataFrame(data)
    pldf = pl.DataFrame(data)

    results: list[BenchResult] = []

    # ── 1. OLS iid ──────────────────────────────────────────────────
    import polars_reg as pr

    results.append(bench("OLS (iid)", "polars_reg", lambda: pr.ols("y ~ x1 + x2", data=pldf), n))

    import statsmodels.api as sm

    X_sm = sm.add_constant(pdf[["x1", "x2"]])
    results.append(bench("OLS (iid)", "statsmodels", lambda: sm.OLS(pdf["y"], X_sm).fit(), n))

    import pyfixest as pf

    results.append(bench("OLS (iid)", "pyfixest", lambda: pf.feols("y ~ x1 + x2", data=pdf), n))

    # ── 2. OLS HC1 ──────────────────────────────────────────────────
    results.append(
        bench("OLS (HC1)", "polars_reg", lambda: pr.ols("y ~ x1 + x2", data=pldf, vcov="HC1"), n)
    )
    results.append(
        bench("OLS (HC1)", "statsmodels", lambda: sm.OLS(pdf["y"], X_sm).fit(cov_type="HC1"), n)
    )
    results.append(
        bench("OLS (HC1)", "pyfixest", lambda: pf.feols("y ~ x1 + x2", data=pdf, vcov="hetero"), n)
    )

    # ── 3. OLS clustered ────────────────────────────────────────────
    results.append(
        bench(
            "OLS (cluster)",
            "polars_reg",
            lambda: pr.ols("y ~ x1 + x2", data=pldf, cluster="firm_id"),
            n,
        )
    )
    results.append(
        bench(
            "OLS (cluster)",
            "statsmodels",
            lambda: sm.OLS(pdf["y"], X_sm).fit(
                cov_type="cluster", cov_kwds={"groups": pdf["firm_id"]}
            ),
            n,
        )
    )
    results.append(
        bench(
            "OLS (cluster)",
            "pyfixest",
            lambda: pf.feols("y ~ x1 + x2", data=pdf, vcov={"CRV1": "firm_id"}),
            n,
        )
    )

    # ── 4. 1-way FE + cluster ──────────────────────────────────────
    results.append(
        bench(
            "FE (1-way + cluster)",
            "polars_reg",
            lambda: pr.ols("y ~ x1 + x2 | firm_id", data=pldf, cluster="firm_id"),
            n,
        )
    )
    results.append(
        bench(
            "FE (1-way + cluster)",
            "pyfixest",
            lambda: pf.feols("y ~ x1 + x2 | firm_id", data=pdf, vcov={"CRV1": "firm_id"}),
            n,
        )
    )

    from linearmodels.panel import PanelOLS

    pdf_panel = pdf.set_index(["firm_id", "year_id"])
    results.append(
        bench(
            "FE (1-way + cluster)",
            "linearmodels",
            lambda: PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=pdf_panel).fit(
                cov_type="clustered", cluster_entity=True
            ),
            n,
        )
    )

    # ── 5. 2-way FE + cluster ──────────────────────────────────────
    results.append(
        bench(
            "FE (2-way + cluster)",
            "polars_reg",
            lambda: pr.ols("y ~ x1 + x2 | firm_id + year_id", data=pldf, cluster="firm_id"),
            n,
        )
    )
    results.append(
        bench(
            "FE (2-way + cluster)",
            "pyfixest",
            lambda: pf.feols("y ~ x1 + x2 | firm_id + year_id", data=pdf, vcov={"CRV1": "firm_id"}),
            n,
        )
    )

    # linearmodels doesn't support 2-way FE absorption directly

    # ── 6. 2SLS / IV ───────────────────────────────────────────────
    results.append(
        bench(
            "2SLS (iid)",
            "polars_reg",
            lambda: pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=pldf),
            n,
        )
    )

    from linearmodels.iv import IV2SLS

    results.append(
        bench(
            "2SLS (iid)",
            "linearmodels",
            lambda: IV2SLS.from_formula("y ~ 1 + x1 + [x_endog ~ z1 + z2]", data=pdf).fit(),
            n,
        )
    )
    results.append(
        bench("2SLS (iid)", "pyfixest", lambda: pf.feols("y ~ x1 | x_endog ~ z1 + z2", data=pdf), n)
    )

    return results


def run_parquet_benchmark(n_firms: int, n_years: int, n_extra: int = 200):
    """Benchmark LazyFrame pushdown on wide Parquet file."""
    import polars_reg as pr

    n = n_firms * n_years
    rng = np.random.default_rng(42)
    data = make_panel(n_firms, n_years, n_extra=n_extra, rng=rng)
    pldf = pl.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        parquet_path = f.name
    pldf.write_parquet(parquet_path)
    file_mb = os.path.getsize(parquet_path) / 1e6

    results = []

    # polars_reg with scan_parquet (LazyFrame pushdown)
    results.append(
        bench(
            f"Wide Parquet ({n_extra + 8} cols)",
            "polars_reg (scan)",
            lambda: pr.ols("y ~ x1 + x2", data=pl.scan_parquet(parquet_path)),
            n,
        )
    )

    # polars_reg with read_parquet (eager, no pushdown)
    results.append(
        bench(
            f"Wide Parquet ({n_extra + 8} cols)",
            "polars_reg (read)",
            lambda: pr.ols("y ~ x1 + x2", data=pl.read_parquet(parquet_path)),
            n,
        )
    )

    # pyfixest (must read all columns into pandas)
    import pyfixest as pf

    results.append(
        bench(
            f"Wide Parquet ({n_extra + 8} cols)",
            "pyfixest (read)",
            lambda: pf.feols("y ~ x1 + x2", data=pl.read_parquet(parquet_path).to_pandas()),
            n,
        )
    )

    os.unlink(parquet_path)
    return results, file_mb


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[BenchResult], title: str = ""):
    if title:
        print(f"\n{'=' * 78}")
        print(f"  {title}")
        print(f"{'=' * 78}")

    # Group by benchmark name
    from collections import OrderedDict

    groups: OrderedDict[str, list[BenchResult]] = OrderedDict()
    for r in results:
        groups.setdefault(r.name, []).append(r)

    hdr = f"  {'Benchmark':<28} {'Package':<22} {'Time (ms)':>10} {'vs best':>10} {'x1 coef':>10}"
    print(f"\n{hdr}")
    sep = f"  {'-' * 28} {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10}"
    print(sep)

    for name, group in groups.items():
        best = min(g.time_ms for g in group)
        for r in group:
            ratio = f"{r.time_ms / best:.1f}x" if r.time_ms > best * 1.01 else "fastest"
            marker = " <--" if r.time_ms <= best * 1.01 else ""
            print(
                f"  {r.name:<28} {r.package:<22}"
                f" {r.time_ms:>9.1f} {ratio:>10} {r.coef_x1:>10.4f}{marker}"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    sizes = [
        (100, 10),  # 1K
        (100, 100),  # 10K
        (1000, 100),  # 100K
        (1000, 1000),  # 1M
    ]

    for n_firms, n_years in sizes:
        n = n_firms * n_years
        print(f"\n{'#' * 78}")
        print(f"  BENCHMARK: N = {n:,} ({n_firms} firms x {n_years} years)")
        print(f"{'#' * 78}")
        results = run_benchmarks(n_firms, n_years)
        print_results(results)

    # Parquet pushdown benchmark
    print(f"\n{'#' * 78}")
    print("  BENCHMARK: Wide Parquet I/O (LazyFrame column pushdown)")
    print(f"{'#' * 78}")
    pq_results, file_mb = run_parquet_benchmark(1000, 1000, n_extra=200)
    print(f"\n  Parquet file: {file_mb:.0f} MB, 208 columns, 1M rows")
    print_results(pq_results)


if __name__ == "__main__":
    main()
