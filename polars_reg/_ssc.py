"""Small-sample correction configuration (matches pyfixest conventions)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSC:
    """Small-sample correction specification.

    Controls degrees-of-freedom adjustments applied to variance-covariance
    matrices. Matches pyfixest's ssc() interface.

    Args:
        k_adj: If True, apply (N-1)/(N-k) residual df scaling for clustered,
            N/(N-k) for heteroskedastic. If False, no residual df scaling.
        k_fixef: How absorbed fixed effects count in k.
            "nonnested": FE not nested in any cluster dimension count in k
            (default, matches pyfixest).
            "none": FE excluded from k.
            "full": all FE parameters count in k.
        G_adj: If True, apply G/(G-1) cluster scaling. If False, no cluster scaling.
        G_df: For multiway clustering: "min" applies min(G)/(min(G)-1) to all
            terms (default, matches pyfixest). "conventional" applies
            G_i/(G_i-1) per term.
    """

    k_adj: bool = True
    k_fixef: str = "nonnested"
    G_adj: bool = True
    G_df: str = "min"

    def __post_init__(self):
        if self.k_fixef not in ("none", "nonnested", "full"):
            raise ValueError(
                f"k_fixef must be 'none', 'nonnested', or 'full', got {self.k_fixef!r}"
            )
        if self.G_df not in ("min", "conventional"):
            raise ValueError(f"G_df must be 'min' or 'conventional', got {self.G_df!r}")


def ssc(
    k_adj: bool = True,
    k_fixef: str = "nonnested",
    G_adj: bool = True,
    G_df: str = "min",
) -> SSC:
    """Configure small-sample corrections. Matches pyfixest conventions.

    Common presets:
        Default (pyfixest):    ssc()  # k_fixef="nonnested", G_df="min"
        Stata reghdfe:         ssc()  # same as default
        Stata ivregress:       ssc(k_adj=False, G_adj=False)
        R fixest:              ssc()  # same as default
        Exclude FE from k:     ssc(k_fixef="none")
        Per-term G correction: ssc(G_df="conventional")
        No corrections:        ssc(k_adj=False, G_adj=False)

    See https://pyfixest.org/ssc.html for details.
    """
    return SSC(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj, G_df=G_df)


def _default_ssc() -> SSC:
    """Return default SSC (pyfixest convention)."""
    return SSC()


def _compute_k_eff(k: int, k_fixef: str, df_abs: int, df_a_non_nested: int) -> int:
    """Compute effective k for SSC adjustment.

    Args:
        k: Number of estimated coefficients (excluding absorbed FE).
        k_fixef: How FE count in k ("none", "nonnested", "full").
        df_abs: Total absorbed FE degrees of freedom.
        df_a_non_nested: Non-nested FE degrees of freedom.
    """
    if k_fixef == "none":
        return k
    elif k_fixef == "nonnested":
        return k + max(df_a_non_nested, 0)
    elif k_fixef == "full":
        return k + df_abs
    return k  # fallback


def _backend_ssc(backend: str, estimator: str) -> SSC:
    """Return the SSC that matches a given backend's conventions.

    Args:
        backend: "pyfixest", "statsmodels", "linearmodels", "r", "stata"
        estimator: "ols", "iv2sls", "liml", "gmm_iv", "probit", "logit", "ppml", etc.
    """
    if backend == "stata":
        if estimator in ("iv2sls", "liml"):
            return SSC(k_adj=False, G_adj=False)  # Stata ivregress: asymptotic
        return SSC()  # Stata reghdfe: same as default
    elif backend == "r":
        return SSC()  # R fixest: same as default
    elif backend == "pyfixest":
        return SSC()  # Same as our defaults
    elif backend == "statsmodels":
        return SSC()  # Needs further validation
    elif backend == "linearmodels":
        return SSC()  # Needs further validation
    return SSC()
