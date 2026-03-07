from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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

    def summary(self) -> str:
        lines = [
            f"{self.model_type} Regression",
            f"{'=' * 60}",
            f"N = {self.n_obs}    R² = {self.r_squared:.4f}    Adj. R² = {self.r_squared_adj:.4f}",
            f"SE type: {self.vcov_type}",
        ]
        if self.fe_absorbed:
            lines.append(f"Absorbed FE: {', '.join(self.fe_absorbed)} ({self.df_absorbed} DoF)")
        if self.n_clusters:
            for name, g in self.n_clusters.items():
                lines.append(f"Clusters ({name}): {g}")
        if self.first_stage_f is not None:
            lines.append(f"First-stage F: {self.first_stage_f:.2f}")
        if self.j_stat is not None:
            lines.append(f"Hansen J: {self.j_stat:.4f} (p = {self.j_pvalue:.4f})")
        lines.append(f"{'-' * 60}")
        cols = f"{'Coef':>10} {'SE':>10} {'t':>8} {'P>|t|':>8} {'[0.025':>8} {'0.975]':>8}"
        lines.append(f"{'':>12} {cols}")
        lines.append(f"{'-' * 60}")
        ci = self.confint()
        for i, name in enumerate(self.names):
            lines.append(
                f"{name:>12} {self.coefficients[i]:>10.4f} {self.se[i]:>10.4f} "
                f"{self.tstat[i]:>8.2f} {self.pvalue[i]:>8.4f} "
                f"{ci[i, 0]:>8.4f} {ci[i, 1]:>8.4f}"
            )
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
