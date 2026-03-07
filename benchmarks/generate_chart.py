"""Generate benchmark chart for README.

Runs polars_reg vs statsmodels, pyfixest, linearmodels, R/fixest, and Stata
across scales, then produces a single publication-quality figure.

Usage:
    python benchmarks/generate_chart.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

BENCH_DIR = Path(__file__).parent
RSCRIPT = os.environ.get("RSCRIPT", shutil.which("Rscript") or "")
STATA = os.environ.get(
    "STATA",
    shutil.which("stata")
    or shutil.which("StataMP-64.exe")
    or shutil.which("StataSE-64.exe")
    or shutil.which("StataBE-64.exe")
    or "",
)

# Check common WSL Stata paths
if not STATA:
    for _candidate in [
        "/mnt/c/Program Files/Stata18/StataBE-64.exe",
        "/mnt/c/Program Files/Stata18/StataSE-64.exe",
        "/mnt/c/Program Files/Stata18/StataMP-64.exe",
        "/mnt/c/Program Files/StataNow19/StataBE-64.exe",
        "/mnt/c/Program Files/StataNow19/StataSE-64.exe",
        "/mnt/c/Program Files/StataNow19/StataMP-64.exe",
    ]:
        if os.path.isfile(_candidate):
            STATA = _candidate
            break

# Check common conda R paths
if not RSCRIPT:
    for _candidate in [
        os.path.expanduser("~/miniforge3/bin/Rscript"),
        os.path.expanduser("~/miniconda3/bin/Rscript"),
        os.path.expanduser("~/anaconda3/bin/Rscript"),
    ]:
        if os.path.isfile(_candidate):
            RSCRIPT = _candidate
            break

BENCHMARKS = [
    "OLS",
    "OLS + robust SE",
    "OLS + clustered SE",
    "1-way FE + cluster",
    "2-way FE + cluster",
    "2SLS / IV",
    "High-dim FE\n(5K groups + 2-way cluster)",
]

# Map from R/Stata output names to chart benchmark names
_NAME_MAP = {
    "High-dim FE": "High-dim FE\n(5K groups + 2-way cluster)",
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_data(n: int, n_firms: int, n_industries: int = 20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    firm_id = rng.integers(0, n_firms, size=n)
    industry_id = rng.integers(0, n_industries, size=n)
    year_id = rng.integers(2000, 2020, size=n)
    firm_fe = rng.standard_normal(n_firms)[firm_id]
    y = 2.0 + x1 - 0.5 * x2 + 1.5 * x_endog + firm_fe + u
    return {
        "y": y,
        "x1": x1,
        "x2": x2,
        "x_endog": x_endog,
        "z1": z1,
        "z2": z2,
        "firm_id": firm_id,
        "industry_id": industry_id,
        "year_id": year_id,
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def timeit(fn, warmup=1, reps=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000  # median ms


def run_r_benchmarks(csv_path: str, reps: int) -> dict[str, float]:
    """Run R fixest benchmarks, return {bench_name: time_ms}."""
    if not RSCRIPT:
        return {}
    outfile = csv_path + ".r_results.csv"
    r_script = str(BENCH_DIR / "bench_r.R")
    try:
        subprocess.run(
            [RSCRIPT, r_script, csv_path, str(reps), outfile],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    R benchmark failed: {e}")
        return {}
    results = {}
    if os.path.isfile(outfile):
        with open(outfile) as f:
            for row in csv.reader(f):
                if len(row) == 2 and row[1].strip():
                    name = _NAME_MAP.get(row[0], row[0])
                    results[name] = float(row[1])
        os.unlink(outfile)
    return results


def _wslpath(posix_path: str) -> str:
    """Convert POSIX path to Windows path for WSL Stata."""
    try:
        result = subprocess.run(
            ["wslpath", "-w", posix_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return posix_path


def run_stata_benchmarks(csv_path: str, reps: int) -> dict[str, float]:
    """Run Stata benchmarks, return {bench_name: time_ms}."""
    if not STATA:
        return {}
    outfile = csv_path + ".stata_results.csv"
    do_file = str(BENCH_DIR / "bench_stata.do")

    # Stata on Windows needs Windows paths
    is_wsl = STATA.startswith("/mnt/")
    if is_wsl:
        do_path = _wslpath(do_file)
        csv_arg = _wslpath(csv_path)
        out_arg = _wslpath(outfile)
    else:
        do_path = do_file
        csv_arg = csv_path
        out_arg = outfile

    try:
        proc = subprocess.Popen(
            [STATA, "-b", "do", do_path, csv_arg, str(reps), out_arg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Stata BE on Windows/WSL may linger after exit.
        # Poll for the output file to have all 7 benchmarks, then kill.
        import time as _time

        deadline = _time.monotonic() + 1200
        expected_lines = 7
        while _time.monotonic() < deadline:
            if os.path.isfile(outfile):
                with open(outfile) as _f:
                    lines = [ln for ln in _f.readlines() if ln.strip()]
                if len(lines) >= expected_lines:
                    break
            _time.sleep(2)
        proc.kill()
        proc.wait()
    except FileNotFoundError as e:
        print(f"    Stata benchmark failed: {e}")
        return {}

    results = {}
    if os.path.isfile(outfile):
        with open(outfile) as f:
            for row in csv.reader(f):
                if len(row) == 2 and row[1].strip():
                    name = _NAME_MAP.get(row[0], row[0])
                    try:
                        results[name] = float(row[1])
                    except ValueError:
                        pass
        os.unlink(outfile)
    return results


# ---------------------------------------------------------------------------
# Python benchmarks
# ---------------------------------------------------------------------------


def run_all():
    import pyfixest as pf
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS
    from linearmodels.panel import PanelOLS

    import polars_reg as pr

    scales = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    packages = [
        "polars_reg",
        "statsmodels",
        "pyfixest",
        "linearmodels",
        "R (fixest)",
        "Stata",
    ]

    # results[bench_name][package] = list of (n, ms)
    results = {b: {p: [] for p in packages} for b in BENCHMARKS}

    for n in scales:
        n_firms = min(n // 10, 5000)
        rng = np.random.default_rng(42)
        data = make_data(n, n_firms, rng=rng)
        pdf = pd.DataFrame(data)
        pldf = pl.DataFrame(data)
        X_sm = sm.add_constant(pdf[["x1", "x2"]])

        print(f"  N = {n:>10,} ...", end="", flush=True)

        # Write CSV for R and Stata
        csv_path = os.path.join(tempfile.gettempdir(), f"bench_{n}.csv")
        pdf.to_csv(csv_path, index=False)

        # ── R / fixest ─────────────────────────────────────────
        if RSCRIPT:
            print(" R", end="", flush=True)
            r_results = run_r_benchmarks(csv_path, reps=5)
            for bname, ms in r_results.items():
                if bname in results:
                    results[bname]["R (fixest)"].append((n, ms))

        # ── Stata ──────────────────────────────────────────────
        if STATA:
            print(" Stata", end="", flush=True)
            stata_reps = 2 if n >= 500_000 else 5
            stata_results = run_stata_benchmarks(csv_path, reps=stata_reps)
            for bname, ms in stata_results.items():
                if bname in results:
                    results[bname]["Stata"].append((n, ms))

        # Clean up CSV
        if os.path.isfile(csv_path):
            os.unlink(csv_path)

        # ── Python benchmarks ─────────────────────────────────

        # OLS
        print(" py", end="", flush=True)
        results["OLS"]["polars_reg"].append((n, timeit(lambda: pr.ols("y ~ x1 + x2", data=pldf))))
        results["OLS"]["statsmodels"].append((n, timeit(lambda: sm.OLS(pdf["y"], X_sm).fit())))
        results["OLS"]["pyfixest"].append((n, timeit(lambda: pf.feols("y ~ x1 + x2", data=pdf))))

        # OLS + robust
        results["OLS + robust SE"]["polars_reg"].append(
            (n, timeit(lambda: pr.ols("y ~ x1 + x2", data=pldf, vcov="HC1")))
        )
        results["OLS + robust SE"]["statsmodels"].append(
            (n, timeit(lambda: sm.OLS(pdf["y"], X_sm).fit(cov_type="HC1")))
        )
        results["OLS + robust SE"]["pyfixest"].append(
            (n, timeit(lambda: pf.feols("y ~ x1 + x2", data=pdf, vcov="hetero")))
        )

        # OLS + cluster
        results["OLS + clustered SE"]["polars_reg"].append(
            (n, timeit(lambda: pr.ols("y ~ x1 + x2", data=pldf, cluster="firm_id")))
        )
        results["OLS + clustered SE"]["statsmodels"].append(
            (
                n,
                timeit(
                    lambda: sm.OLS(pdf["y"], X_sm).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": pdf["firm_id"]},
                    )
                ),
            )
        )
        results["OLS + clustered SE"]["pyfixest"].append(
            (
                n,
                timeit(lambda: pf.feols("y ~ x1 + x2", data=pdf, vcov={"CRV1": "firm_id"})),
            )
        )

        # 1-way FE + cluster
        results["1-way FE + cluster"]["polars_reg"].append(
            (
                n,
                timeit(lambda: pr.ols("y ~ x1 + x2 | firm_id", data=pldf, cluster="firm_id")),
            )
        )
        results["1-way FE + cluster"]["pyfixest"].append(
            (
                n,
                timeit(
                    lambda: pf.feols(
                        "y ~ x1 + x2 | firm_id",
                        data=pdf,
                        vcov={"CRV1": "firm_id"},
                    )
                ),
            )
        )
        pdf_panel = pdf.set_index(["firm_id", "year_id"])
        pdf_panel = pdf_panel[~pdf_panel.index.duplicated(keep="first")]
        results["1-way FE + cluster"]["linearmodels"].append(
            (
                n,
                timeit(
                    lambda: PanelOLS.from_formula(
                        "y ~ x1 + x2 + EntityEffects", data=pdf_panel
                    ).fit(cov_type="clustered", cluster_entity=True)
                ),
            )
        )

        # 2-way FE + cluster
        results["2-way FE + cluster"]["polars_reg"].append(
            (
                n,
                timeit(
                    lambda: pr.ols(
                        "y ~ x1 + x2 | firm_id + year_id",
                        data=pldf,
                        cluster="firm_id",
                    )
                ),
            )
        )
        results["2-way FE + cluster"]["pyfixest"].append(
            (
                n,
                timeit(
                    lambda: pf.feols(
                        "y ~ x1 + x2 | firm_id + year_id",
                        data=pdf,
                        vcov={"CRV1": "firm_id"},
                    )
                ),
            )
        )

        # 2SLS
        results["2SLS / IV"]["polars_reg"].append(
            (
                n,
                timeit(lambda: pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=pldf)),
            )
        )
        results["2SLS / IV"]["linearmodels"].append(
            (
                n,
                timeit(
                    lambda: IV2SLS.from_formula("y ~ 1 + x1 + [x_endog ~ z1 + z2]", data=pdf).fit()
                ),
            )
        )
        results["2SLS / IV"]["pyfixest"].append(
            (
                n,
                timeit(lambda: pf.feols("y ~ x1 | x_endog ~ z1 + z2", data=pdf)),
            )
        )

        # High-dim FE + 2-way cluster
        hdfe_key = "High-dim FE\n(5K groups + 2-way cluster)"
        results[hdfe_key]["polars_reg"].append(
            (
                n,
                timeit(
                    lambda: pr.ols(
                        "y ~ x1 + x2 | firm_id + industry_id",
                        data=pldf,
                        cluster=["firm_id", "industry_id"],
                    )
                ),
            )
        )
        results[hdfe_key]["pyfixest"].append(
            (
                n,
                timeit(
                    lambda: pf.feols(
                        "y ~ x1 + x2 | firm_id + industry_id",
                        data=pdf,
                        vcov={"CRV1": "firm_id + industry_id"},
                    )
                ),
            )
        )

        print(" done")

    return results, scales


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

COLORS = {
    "polars_reg": "#2563eb",
    "statsmodels": "#dc2626",
    "pyfixest": "#16a34a",
    "linearmodels": "#9333ea",
    "R (fixest)": "#f59e0b",
    "Stata": "#06b6d4",
}
MARKERS = {
    "polars_reg": "o",
    "statsmodels": "s",
    "pyfixest": "^",
    "linearmodels": "D",
    "R (fixest)": "P",
    "Stata": "X",
}
PKG_ORDER = [
    "polars_reg",
    "statsmodels",
    "pyfixest",
    "linearmodels",
    "R (fixest)",
    "Stata",
]


def make_chart(results, scales):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.flatten()

    # Collect all legend handles/labels across all panels
    all_handles = {}

    for i, bname in enumerate(BENCHMARKS):
        ax = axes[i]
        bdata = results[bname]

        for pkg in PKG_ORDER:
            pts = bdata[pkg]
            if not pts:
                continue
            ns = [p[0] for p in pts]
            ms = [p[1] for p in pts]
            lw = 2.5 if pkg == "polars_reg" else 1.5
            z = 10 if pkg == "polars_reg" else 5
            (line,) = ax.plot(
                ns,
                ms,
                color=COLORS[pkg],
                marker=MARKERS[pkg],
                markersize=6,
                linewidth=lw,
                label=pkg,
                zorder=z,
            )
            if pkg not in all_handles:
                all_handles[pkg] = line

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(bname, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("N (observations)", fontsize=9)
        ax.set_ylabel("Time (ms)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(labelsize=8)

        # Format x-axis ticks
        ax.set_xticks(scales)
        labels = []
        for s in scales:
            if s >= 1_000_000:
                labels.append(f"{s // 1_000_000}M")
            else:
                labels.append(f"{s // 1_000}K")
        ax.set_xticklabels(labels, fontsize=8)

    # Hide the unused subplot (8th position)
    axes[7].axis("off")

    # Build legend from all panels (preserving order)
    handles = [all_handles[p] for p in PKG_ORDER if p in all_handles]
    labels = [p for p in PKG_ORDER if p in all_handles]
    axes[7].legend(
        handles,
        labels,
        loc="center",
        fontsize=13,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1,
        markerscale=1.5,
    )

    title = "polars_reg benchmarks — wall-clock time (lower is better)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = str(BENCH_DIR / "benchmark_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close()


# ---------------------------------------------------------------------------


def main():
    if RSCRIPT:
        print(f"  R: {RSCRIPT}")
    else:
        print("  R: not found (skipping)")
    if STATA:
        print(f"  Stata: {STATA}")
    else:
        print("  Stata: not found (skipping)")

    print("\nRunning benchmarks...")
    results, scales = run_all()
    make_chart(results, scales)


if __name__ == "__main__":
    main()
