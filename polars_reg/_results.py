from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats


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

    def summary(self) -> str:
        """Pretty-printed regression summary table."""
        w = 70
        lines = [
            f"{'=' * w}",
            f"  {self.model_type} Regression Results",
            f"{'=' * w}",
        ]

        # Model info in two columns
        left = [
            f"Dep. Variable:   {self.names[0] if len(self.names) == 1 else 'y'}",
            f"No. Observations: {self.n_obs:>6}",
            f"Df Residuals:     {self.df_r:>6}",
        ]
        right = [
            f"R-squared:     {self.r_squared:>8.4f}",
            f"Adj. R-squared:{self.r_squared_adj:>8.4f}",
            f"SE type:  {self.vcov_type:>12}",
        ]
        for l_line, r_line in zip(left, right):
            lines.append(f"  {l_line:<35} {r_line}")

        if self.fe_absorbed:
            lines.append(f"  Absorbed FE: {', '.join(self.fe_absorbed)} ({self.df_absorbed} DoF)")
        if self.n_clusters:
            cl_info = ", ".join(f"{name}: {g}" for name, g in self.n_clusters.items())
            lines.append(f"  Clusters: {cl_info}")
        if self.first_stage_f is not None:
            lines.append(f"  First-stage F: {self.first_stage_f:.2f}")
        if self.j_stat is not None and self.j_pvalue is not None:
            lines.append(f"  Hansen J: {self.j_stat:.4f} (p = {self.j_pvalue:.4f})")

        lines.append(f"{'=' * w}")

        # Column headers
        hdr = f"{'':>14} {'Coef':>10} {'Std.Err.':>10} {'t':>8} {'P>|t|':>8} {'[95% CI]':>19}"
        lines.append(hdr)
        lines.append(f"{'-' * w}")

        # Coefficient rows
        ci = self.confint()
        for i, name in enumerate(self.names):
            ci_str = f"[{ci[i, 0]:>8.4f}, {ci[i, 1]:.4f}]"
            lines.append(
                f"{name:>14} {self.coefficients[i]:>10.6f} {self.se[i]:>10.6f} "
                f"{self.tstat[i]:>8.3f} {self.pvalue[i]:>8.3f} {ci_str}"
            )
        lines.append(f"{'=' * w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<{self.model_type}Result n={self.n_obs} k={self.k} "
            f"R²={self.r_squared:.4f} vcov={self.vcov_type}>"
        )
