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
            "none": FE excluded from k (default, matches pyfixest).
            "nonnested": FE not nested in any cluster dimension count in k.
            "full": all FE parameters count in k.
        G_adj: If True, apply G/(G-1) cluster scaling. If False, no cluster scaling.
        G_df: For multiway clustering: "conventional" applies G_i/(G_i-1) per term,
            "min" applies min(G)/(min(G)-1) to all terms.
    """

    k_adj: bool = True
    k_fixef: str = "none"
    G_adj: bool = True
    G_df: str = "conventional"

    def __post_init__(self):
        if self.k_fixef not in ("none", "nonnested", "full"):
            raise ValueError(
                f"k_fixef must be 'none', 'nonnested', or 'full', got {self.k_fixef!r}"
            )
        if self.G_df not in ("min", "conventional"):
            raise ValueError(f"G_df must be 'min' or 'conventional', got {self.G_df!r}")


def ssc(
    k_adj: bool = True,
    k_fixef: str = "none",
    G_adj: bool = True,
    G_df: str = "conventional",
) -> SSC:
    """Configure small-sample corrections. Matches pyfixest conventions.

    Common presets:
        Default (pyfixest):    ssc()
        Stata reghdfe:         ssc(k_fixef="nonnested", G_df="min")
        Stata ivregress:       ssc(k_adj=False, G_adj=False)
        R fixest:              ssc(k_fixef="nonnested")
        No corrections:        ssc(k_adj=False, G_adj=False)

    See https://pyfixest.org/ssc.html for details.
    """
    return SSC(k_adj=k_adj, k_fixef=k_fixef, G_adj=G_adj, G_df=G_df)


def _default_ssc() -> SSC:
    """Return default SSC (pyfixest convention)."""
    return SSC()
