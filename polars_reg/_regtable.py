"""Side-by-side regression table display (esttab-style)."""

from __future__ import annotations

from polars_reg._results import RegressionResult


def regtable(
    *results: RegressionResult,
    labels: list[str] | None = None,
    precision: int = 4,
    stars: bool = True,
) -> str:
    """Display multiple regressions side-by-side in a compact table.

    Args:
        *results: RegressionResult objects to compare.
        labels: Column labels. Defaults to (1), (2), ...
        precision: Significant figures for coefficients/SEs (default 4).
        stars: Show significance stars (default True).
            * p<0.10, ** p<0.05, *** p<0.01

    Returns:
        Formatted string table.
    """
    if not results:
        raise ValueError("At least one RegressionResult is required.")

    n_models = len(results)
    if labels is None:
        labels = [f"({i + 1})" for i in range(n_models)]
    if len(labels) != n_models:
        raise ValueError(f"Expected {n_models} labels, got {len(labels)}.")

    col_w = max(14, max(len(lb) for lb in labels) + 4)
    name_w = 14

    # Collect all variable names in order of first appearance
    all_vars: list[str] = []
    for r in results:
        for name in r.names:
            if name not in all_vars:
                all_vars.append(name)

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

    # Footer: N, R², FE, clusters
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

    # FE
    has_any_fe = any(r.fe_absorbed for r in results)
    if has_any_fe:
        fe_row = f"{'FE':<{name_w}}"
        for r in results:
            if r.fe_absorbed:
                fe_str = ", ".join(r.fe_absorbed)
                if len(fe_str) > col_w:
                    fe_str = "Yes"
            else:
                fe_str = "No"
            fe_row += f" {fe_str:>{col_w}}"
        lines.append(fe_row)

        # If any were truncated to "Yes", show details on separate line
        has_detail = any(
            r.fe_absorbed and len(", ".join(r.fe_absorbed)) > col_w for r in results
        )
        if has_detail:
            for r in results:
                if r.fe_absorbed and len(", ".join(r.fe_absorbed)) > col_w:
                    lines.append(f"{'':>{name_w}}   ({', '.join(r.fe_absorbed)})")
                    break  # only need to list once if all same

    # Clusters
    has_any_cl = any(r.n_clusters for r in results)
    if has_any_cl:
        cl_row = f"{'Clusters':<{name_w}}"
        for r in results:
            if r.n_clusters:
                cl_str = ", ".join(r.n_clusters.keys())
            else:
                cl_str = "No"
            cl_row += f" {cl_str:>{col_w}}"
        lines.append(cl_row)

    lines.append(sep)

    if stars:
        lines.append("* p<0.10, ** p<0.05, *** p<0.01")

    table = "\n".join(lines)
    print(table)
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
