"""Side-by-side regression table display (esttab-style)."""

from __future__ import annotations

from polars_reg._groupby import GroupRegressionResult
from polars_reg._results import RegressionResult


def regtable(
    *results: RegressionResult | GroupRegressionResult,
    labels: list[str] | None = None,
    precision: int = 4,
    stars: bool = True,
) -> str:
    """Display multiple regressions side-by-side in a compact table.

    Args:
        *results: RegressionResult or GroupRegressionResult objects.
            GroupRegressionResult is automatically expanded, using group
            keys as column labels.
        labels: Column labels. Defaults to group keys for GroupRegressionResult,
            (1), (2), ... for individual results.
        precision: Significant figures for coefficients/SEs (default 4).
        stars: Show significance stars (default True).
            * p<0.10, ** p<0.05, *** p<0.01

    Returns:
        Formatted string table.
    """
    if not results:
        raise ValueError("At least one RegressionResult is required.")

    # Expand GroupRegressionResult into individual results
    expanded: list[RegressionResult] = []
    auto_labels: list[str] = []
    for r in results:
        if isinstance(r, GroupRegressionResult):
            for key, val in r.items():
                expanded.append(val)
                auto_labels.append(str(key))
        else:
            expanded.append(r)
            auto_labels.append("")
    results = tuple(expanded)

    n_models = len(results)
    if labels is None:
        labels = [
            lb if lb else f"({i + 1})" for i, lb in enumerate(auto_labels)
        ]
    if len(labels) != n_models:
        raise ValueError(f"Expected {n_models} labels, got {len(labels)}.")

    col_w = max(14, max(len(lb) for lb in labels) + 4)

    # Collect all variable names in order of first appearance
    all_vars: list[str] = []
    for r in results:
        for name in r.names:
            if name not in all_vars:
                all_vars.append(name)

    # Collect all FE and cluster names for name column width
    all_row_labels = list(all_vars) + ["N", "R²", "Adj. R²", "Fixed Effects", "Clustering"]
    for r in results:
        if r.fe_absorbed:
            all_row_labels.extend(f"  {fe}" for fe in r.fe_absorbed)
        if r.n_clusters:
            all_row_labels.extend(f"  {cl}" for cl in r.n_clusters)
    name_w = max(14, max(len(lb) for lb in all_row_labels) + 2)

    # Build coefficient/SE cells per model per variable
    cells: dict[str, list[str]] = {}  # var -> list of "coef\n(se)" per model
    for var in all_vars:
        row_cells = []
        for r in results:
            if var in r.names:
                idx = r.names.index(var)
                coef = r.coefficients[idx]
                se = r.se[idx]
                pval = r.pvalue[idx]
                star = _star(pval) if stars else ""
                c_str = _fmt_g(coef, col_w - len(star), precision) + star
                se_inner = f"({_fmt_sig(se, precision)})"
                row_cells.append((c_str, se_inner))
            else:
                row_cells.append(("", ""))
        cells[var] = row_cells

    # Build the table
    lines: list[str] = []
    total_w = name_w + 1 + n_models * (col_w + 1)
    sep = "=" * total_w

    # Header
    lines.append(sep)
    hdr = f"{'':>{name_w}}"
    for lb in labels:
        hdr += f" {lb:>{col_w}}"
    lines.append(hdr)

    # Model type sub-header
    type_row = f"{'':>{name_w}}"
    for r in results:
        type_row += f" {r.model_type:>{col_w}}"
    lines.append(type_row)
    lines.append("-" * total_w)

    # Coefficient rows (coef line + SE line per variable)
    for var in all_vars:
        coef_line = f"{var:<{name_w}}"
        se_line = f"{'':>{name_w}}"
        for coef_str, se_str in cells[var]:
            coef_line += f" {coef_str:>{col_w}}"
            se_line += f" {se_str:>{col_w}}"
        lines.append(coef_line)
        lines.append(se_line)

    # Collect all FE and cluster variable names
    all_fe: list[str] = []
    for r in results:
        if r.fe_absorbed:
            for fe in r.fe_absorbed:
                if fe not in all_fe:
                    all_fe.append(fe)
    all_cl: list[str] = []
    for r in results:
        if r.n_clusters:
            for cl in r.n_clusters:
                if cl not in all_cl:
                    all_cl.append(cl)

    # FE and cluster indicator rows (group headers + indented names)
    if all_fe or all_cl:
        lines.append("-" * total_w)
    if all_fe:
        lines.append(f"{'Fixed Effects':<{name_w}}")
        for fe in all_fe:
            row = f"{'  ' + fe:<{name_w}}"
            for r in results:
                yn = "Y" if r.fe_absorbed and fe in r.fe_absorbed else "N"
                row += f" {yn:>{col_w}}"
            lines.append(row)
    if all_cl:
        lines.append(f"{'Clustering':<{name_w}}")
        for cl in all_cl:
            row = f"{'  ' + cl:<{name_w}}"
            for r in results:
                yn = "Y" if r.n_clusters and cl in r.n_clusters else "N"
                row += f" {yn:>{col_w}}"
            lines.append(row)

    # Footer: N, R²
    lines.append("-" * total_w)

    # N
    n_row = f"{'N':<{name_w}}"
    for r in results:
        n_row += f" {r.n_obs:>{col_w}}"
    lines.append(n_row)

    # R²
    r2_row = f"{'R²':<{name_w}}"
    for r in results:
        r2_row += f" {r.r_squared:>{col_w}.4f}"
    lines.append(r2_row)

    # Adj. R²
    r2a_row = f"{'Adj. R²':<{name_w}}"
    for r in results:
        r2a_row += f" {r.r_squared_adj:>{col_w}.4f}"
    lines.append(r2a_row)

    lines.append(sep)

    if stars:
        lines.append("* p<0.10, ** p<0.05, *** p<0.01")

    table = "\n".join(lines)
    return table


def _fmt_sig(x: float, sig: int) -> str:
    """Format a number with sig significant figures, no padding."""
    return f"{x:.{sig}g}"


def _fmt_g(x: float, width: int, sig: int) -> str:
    """Format a number with sig figs, right-aligned in width."""
    for s in range(sig, 0, -1):
        candidate = f"{x:.{s}g}"
        if len(candidate) <= width:
            return f"{candidate:>{width}}"
    return f"{x:>{width}.0e}"


def _star(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""
