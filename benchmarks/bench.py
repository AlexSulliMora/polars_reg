"""Benchmark polars_reg performance at various data sizes.

Usage: python benchmarks/bench.py
"""

import time

import numpy as np
import polars as pl

import polars_reg as pr


def generate_data(n: int, n_firms: int = 100, n_years: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    firm_id = np.random.randint(0, n_firms, n)
    year_id = np.random.randint(0, n_years, n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(n)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


def bench(label: str, fn, warmup: int = 1, runs: int = 3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    median = sorted(times)[len(times) // 2]
    print(f"  {label:50s}  {median * 1000:8.1f} ms")
    return median


def run_benchmarks():
    sizes = [1_000, 10_000, 100_000, 1_000_000]

    for n in sizes:
        n_firms = max(10, n // 100)
        n_years = 10
        df = generate_data(n, n_firms, n_years)

        print(f"\n{'=' * 70}")
        print(f"  N = {n:,}  ({n_firms} firms x {n_years} years)")
        print(f"{'=' * 70}")

        bench("OLS (iid)", lambda: pr.ols("y ~ x1 + x2", data=df))
        bench("OLS (HC1)", lambda: pr.ols("y ~ x1 + x2", data=df, vcov="HC1"))
        bench("OLS (cluster)", lambda: pr.ols("y ~ x1 + x2", data=df, cluster=["firm_id"]))
        bench("OLS + 1-way FE", lambda: pr.ols("y ~ x1 + x2 | firm_id", data=df))
        bench("OLS + 2-way FE", lambda: pr.ols("y ~ x1 + x2 | firm_id + year_id", data=df))
        bench(
            "OLS + 2-way FE + cluster",
            lambda: pr.ols("y ~ x1 + x2 | firm_id + year_id", data=df, cluster=["firm_id"]),
        )


if __name__ == "__main__":
    run_benchmarks()
