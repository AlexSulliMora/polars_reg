from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats


def _fmt_col(values: list[float], width: int, sig: int) -> list[str]:
    """Format a column of numbers, decimal-aligned within `width` chars.

    Uses fixed-point notation with the same number of decimal places for all
    values so decimals line up. Chooses the max decimals that fit within width.
    """
    max_int_w = max(len(f"{v:.0f}") for v in values)

    # Available space for decimal point + fraction
    avail = width - max_int_w
    if avail >= 2:  # room for "." + at least 1 digit
        dec = min(avail - 1, sig)  # sig decimal places max
        # Ensure it fits
        while dec > 0:
            strs = [f"{v:>{width}.{dec}f}" for v in values]
            if all(len(s) <= width for s in strs):
                return strs
            dec -= 1

    # No room for decimals — try integer format
    strs = [f"{v:>{width}.0f}" for v in values]
    if all(len(s) <= width for s in strs):
        return strs

    # Fallback: scientific notation, aligned by sig figs
    strs = []
    for v in values:
        for d in range(sig, 0, -1):
            s = f"{v:>{width}.{d}e}"
            if len(s) <= width:
                strs.append(s)
                break
        else:
            strs.append(f"{v:>{width}.0e}")
    return strs


def _fmt_p_col(values: list[float], width: int) -> list[str]:
    """Format p-values: 3 decimal places, <0.01 for tiny values."""
    result = []
    for v in values:
        if v < 0.01:
            result.append(f"{'<0.01':>{width}}")
        else:
            result.append(f"{v:>{width}.3f}")
    return result


def _append_wrapped(lines: list[str], label: str, text: str, width: int) -> None:
    """Append a metadata line, wrapping at commas if it exceeds table width."""
    prefix = f"  {label:<18} "
    if len(prefix + text) <= width:
        lines.append(prefix + text)
        return
    indent = " " * len(prefix)
    avail = width - len(prefix)
    parts = text.split(", ")
    current = parts[0]
    for part in parts[1:]:
        candidate = current + ", " + part
        if len(candidate) <= avail:
            current = candidate
        else:
            lines.append(prefix + current + ",")
            prefix = indent
            current = part
    lines.append(prefix + current)


@dataclass
class RegressionResult:
    coefficients: NDArray
    vcov: NDArray
    residuals: NDArray
    names: list[str]
    n_obs: int
    k: int
    df_r: int
    r_squared: float
    r_squared_adj: float
    model_type: str
    vcov_type: str
    f_stat: float | None = None
    f_pvalue: float | None = None
    n_clusters: dict[str, int] | None = None
    fe_absorbed: list[str] | None = None
    df_absorbed: int = 0
    first_stage_f: float | None = None
    j_stat: float | None = None
    j_pvalue: float | None = None

    # Store design matrix and y for predict/fitted (set by estimator)
    _X: NDArray | None = None
    _y: NDArray | None = None

    @property
    def se(self) -> NDArray:
        return np.sqrt(np.diag(self.vcov))

    @property
    def tstat(self) -> NDArray:
        return self.coefficients / self.se

    @property
    def pvalue(self) -> NDArray:
        return 2.0 * stats.t.sf(np.abs(self.tstat), df=self.df_r)

    def confint(self, alpha: float = 0.05) -> NDArray:
        """Return (k, 2) array of [lower, upper] confidence intervals."""
        t_crit = stats.t.ppf(1 - alpha / 2, df=self.df_r)
        margin = t_crit * self.se
        return np.column_stack([self.coefficients - margin, self.coefficients + margin])

    def fitted(self) -> NDArray:
        """Return fitted values (X @ beta). Only available for in-sample predictions."""
        if self._X is None or self._y is None:
            raise ValueError("Fitted values not available (model was not stored).")
        return self._y - self.residuals

    def predict(self, newdata: pl.DataFrame | None = None) -> NDArray:
        """Return predictions. Without newdata, returns in-sample fitted values."""
        if newdata is not None:
            n = len(newdata)
            x_cols = []
            for name in self.names:
                if name == "_cons":
                    x_cols.append(np.ones(n))
                elif ":" in name:
                    # Interaction term
                    parts = name.split(":")
                    arr = np.ones(n)
                    for p in parts:
                        arr = arr * newdata[p].to_numpy().astype(np.float64)
                    x_cols.append(arr)
                else:
                    x_cols.append(newdata[name].to_numpy().astype(np.float64))
            X_new = np.column_stack(x_cols)
            return X_new @ self.coefficients
        return self.fitted()

    def coef_table(self) -> pl.DataFrame:
        """Return coefficient table as a Polars DataFrame."""
        ci = self.confint()
        return pl.DataFrame(
            {
                "name": self.names,
                "coef": self.coefficients,
                "se": self.se,
                "t": self.tstat,
                "p": self.pvalue.tolist(),
                "ci_lower": ci[:, 0],
                "ci_upper": ci[:, 1],
            }
        )

    def summary(self, precision: int = 4) -> str:
        """Pretty-printed regression summary table.

        Args:
            precision: Number of significant figures to display (default 4).
        """
        w = 80
        sig = precision
        lines = [
            f"{'=' * w}",
            f"  {self.model_type} Regression Results",
            f"{'=' * w}",
        ]

        # Model info in two columns (left col 40 chars, right col 38 chars)
        lw, rw = 40, 38
        depvar = self.names[0] if len(self.names) == 1 else "y"

        def _row(l_label: str, l_val: str, r_label: str, r_val: str) -> str:
            left = f"{l_label:<18} {l_val:>6}"
            right = f"{r_label:<16} {r_val:>8}"
            return f"  {left:<{lw}}{right:>{rw}}"

        lines.append(_row("Dep. Variable:", depvar, "R-squared:", f"{self.r_squared:.4f}"))
        r2a = f"{self.r_squared_adj:.4f}"
        lines.append(_row("No. Observations:", str(self.n_obs), "Adj. R-squared:", r2a))
        lines.append(_row("Df Residuals:", str(self.df_r), "SE type:", self.vcov_type))

        if self.fe_absorbed:
            dof_suffix = f" ({self.df_absorbed} DoF)"
            fe_str = ", ".join(self.fe_absorbed) + dof_suffix
            _append_wrapped(lines, "Absorbed FE:", fe_str, w)
        if self.n_clusters:
            cl_info = ", ".join(f"{name}: {g}" for name, g in self.n_clusters.items())
            _append_wrapped(lines, "Clusters:", cl_info, w)
        if self.first_stage_f is not None:
            lines.append(f"  {'First-stage F:':<18} {self.first_stage_f:.2f}")
        if self.j_stat is not None and self.j_pvalue is not None:
            lines.append(f"  {'Hansen J:':<18} {self.j_stat:.4f} (p = {self.j_pvalue:.4f})")

        lines.append(f"{'=' * w}")

        # Column headers
        hdr = (
            f"{'':>14} {'Coef':>10} {'Std.Err.':>10} "
            f"{'t':>8} {'P>|t|':>8}   {'[95% Conf. Interval]':>23}"
        )
        lines.append(hdr)
        lines.append(f"{'-' * w}")

        # Coefficient rows — format columns then align decimals vertically
        ci = self.confint()
        n_coef = len(self.names)
        col_c = _fmt_col([self.coefficients[i] for i in range(n_coef)], 10, sig)
        col_s = _fmt_col([self.se[i] for i in range(n_coef)], 10, sig)
        col_t = [f"{self.tstat[i]:>8.2f}" for i in range(n_coef)]
        col_p = _fmt_p_col([self.pvalue[i] for i in range(n_coef)], 8)
        col_lo = _fmt_col([ci[i, 0] for i in range(n_coef)], 9, sig)
        col_hi = _fmt_col([ci[i, 1] for i in range(n_coef)], 9, sig)
        for i, name in enumerate(self.names):
            ci_str = f"[{col_lo[i]},  {col_hi[i]}]"
            lines.append(f"{name:<14} {col_c[i]} {col_s[i]} {col_t[i]} {col_p[i]}   {ci_str}")
        lines.append(f"{'=' * w}")
        return "\n".join(lines)

    def wald_test(self, R: NDArray, q: NDArray | None = None) -> dict:
        """Wald test for linear restrictions R @ beta = q.

        Args:
            R: (j, k) constraint matrix, where j = number of restrictions
            q: (j,) vector of restriction values. Defaults to zero vector.

        Returns:
            dict with 'statistic' (F or chi2), 'pvalue', 'df' (j, df_r)
        """
        R = np.atleast_2d(R)
        j = R.shape[0]
        if q is None:
            q = np.zeros(j)
        q = np.asarray(q, dtype=float)

        diff = R @ self.coefficients - q
        middle = R @ self.vcov @ R.T
        chi2_stat = float(diff @ np.linalg.solve(middle, diff))
        f_stat = chi2_stat / j
        f_pvalue = float(1.0 - stats.f.cdf(f_stat, j, self.df_r))

        return {
            "statistic": f_stat,
            "pvalue": f_pvalue,
            "df": (j, self.df_r),
            "chi2": chi2_stat,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.model_type}Result n={self.n_obs} k={self.k} "
            f"R²={self.r_squared:.4f} vcov={self.vcov_type}>"
        )
