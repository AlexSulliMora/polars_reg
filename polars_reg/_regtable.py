"""Side-by-side regression table display (esttab-style)."""

from __future__ import annotations

from dataclasses import dataclass, field

from polars_reg._groupby import GroupRegressionResult
from polars_reg._results import RegressionResult


@dataclass
class _TableData:
    """Intermediate structured data for rendering a regression table."""

    labels: list[str]
    model_types: list[str]
    all_vars: list[str]
    # cells[var] = list of (coef_str, se_str, star_str) per model
    cells: dict[str, list[tuple[str, str, str]]]
    all_fe: list[str]
    all_cl: list[str]
    # fe_flags[fe] = list of bool per model
    fe_flags: dict[str, list[bool]] = field(default_factory=dict)
    # cl_flags[cl] = list of bool per model
    cl_flags: dict[str, list[bool]] = field(default_factory=dict)
    n_obs: list[int] = field(default_factory=list)
    r_squared: list[float] = field(default_factory=list)
    r_squared_adj: list[float] = field(default_factory=list)
    stars: bool = True


def _build_table_data(
    results: tuple[RegressionResult, ...],
    labels: list[str],
    precision: int,
    stars: bool,
) -> _TableData:
    """Extract structured table data from regression results."""

    # Collect variable names in order of first appearance
    all_vars: list[str] = []
    for r in results:
        for name in r.names:
            if name not in all_vars:
                all_vars.append(name)

    # Build cells
    cells: dict[str, list[tuple[str, str, str]]] = {}
    for var in all_vars:
        row_cells = []
        for r in results:
            if var in r.names:
                idx = r.names.index(var)
                coef = r.coefficients[idx]
                se = r.se[idx]
                pval = r.pvalue[idx]
                star = _star(pval) if stars else ""
                c_str = _fmt_sig(coef, precision)
                se_str = _fmt_sig(se, precision)
                row_cells.append((c_str, se_str, star))
            else:
                row_cells.append(("", "", ""))
        cells[var] = row_cells

    # Collect FE and cluster names
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

    fe_flags = {fe: [bool(r.fe_absorbed and fe in r.fe_absorbed) for r in results] for fe in all_fe}
    cl_flags = {cl: [bool(r.n_clusters and cl in r.n_clusters) for r in results] for cl in all_cl}

    return _TableData(
        labels=labels,
        model_types=[r.model_type for r in results],
        all_vars=all_vars,
        cells=cells,
        all_fe=all_fe,
        all_cl=all_cl,
        fe_flags=fe_flags,
        cl_flags=cl_flags,
        n_obs=[r.n_obs for r in results],
        r_squared=[r.r_squared for r in results],
        r_squared_adj=[r.r_squared_adj for r in results],
        stars=stars,
    )


def _render_text(td: _TableData, precision: int) -> str:
    """Render table data as plain text."""
    n_models = len(td.labels)
    col_w = max(14, max(len(lb) for lb in td.labels) + 4)

    # Name column width
    all_row_labels = list(td.all_vars) + ["N", "R²", "Adj. R²", "Fixed Effects", "Clustering"]
    for fe in td.all_fe:
        all_row_labels.append(f"  {fe}")
    for cl in td.all_cl:
        all_row_labels.append(f"  {cl}")
    name_w = max(14, max(len(lb) for lb in all_row_labels) + 2)

    total_w = name_w + 1 + n_models * (col_w + 1)
    sep = "=" * total_w

    lines: list[str] = []

    # Header
    lines.append(sep)
    hdr = f"{'':>{name_w}}"
    for lb in td.labels:
        hdr += f" {lb:>{col_w}}"
    lines.append(hdr)

    type_row = f"{'':>{name_w}}"
    for mt in td.model_types:
        type_row += f" {mt:>{col_w}}"
    lines.append(type_row)
    lines.append("-" * total_w)

    # Coefficients
    for var in td.all_vars:
        coef_line = f"{var:<{name_w}}"
        se_line = f"{'':>{name_w}}"
        for c_str, se_str, star in td.cells[var]:
            if c_str:
                formatted = _fmt_g_pad(float(c_str), col_w - len(star), precision) + star
                coef_line += f" {formatted:>{col_w}}"
                se_line += f" {f'({se_str})':>{col_w}}"
            else:
                coef_line += f" {'':>{col_w}}"
                se_line += f" {'':>{col_w}}"
        lines.append(coef_line)
        lines.append(se_line)

    # FE / Cluster indicators
    if td.all_fe or td.all_cl:
        lines.append("-" * total_w)
    if td.all_fe:
        lines.append(f"{'Fixed Effects':<{name_w}}")
        for fe in td.all_fe:
            row = f"{'  ' + fe:<{name_w}}"
            for flag in td.fe_flags[fe]:
                row += f" {'Y' if flag else 'N':>{col_w}}"
            lines.append(row)
    if td.all_cl:
        lines.append(f"{'Clustering':<{name_w}}")
        for cl in td.all_cl:
            row = f"{'  ' + cl:<{name_w}}"
            for flag in td.cl_flags[cl]:
                row += f" {'Y' if flag else 'N':>{col_w}}"
            lines.append(row)

    # Footer
    lines.append("-" * total_w)
    n_row = f"{'N':<{name_w}}"
    for n in td.n_obs:
        n_row += f" {n:>{col_w}}"
    lines.append(n_row)

    r2_row = f"{'R²':<{name_w}}"
    for r2 in td.r_squared:
        r2_row += f" {r2:>{col_w}.4f}"
    lines.append(r2_row)

    r2a_row = f"{'Adj. R²':<{name_w}}"
    for r2a in td.r_squared_adj:
        r2a_row += f" {r2a:>{col_w}.4f}"
    lines.append(r2a_row)

    lines.append(sep)
    if td.stars:
        lines.append("* p<0.10, ** p<0.05, *** p<0.01")

    return "\n".join(lines)


def _render_latex(td: _TableData) -> str:
    """Render table data as a LaTeX tabular."""
    n_models = len(td.labels)
    col_spec = "l" + "c" * n_models

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    hdr = " & ".join([""] + [_latex_escape(lb) for lb in td.labels]) + r" \\"
    lines.append(hdr)
    type_row = " & ".join([""] + td.model_types) + r" \\"
    lines.append(type_row)
    lines.append(r"\midrule")

    # Coefficients
    for var in td.all_vars:
        coef_parts = [_latex_escape(var)]
        se_parts = [""]
        for c_str, se_str, star in td.cells[var]:
            if c_str:
                star_tex = _latex_stars(star)
                coef_parts.append(f"{c_str}{star_tex}")
                se_parts.append(f"({se_str})")
            else:
                coef_parts.append("")
                se_parts.append("")
        lines.append(" & ".join(coef_parts) + r" \\")
        lines.append(" & ".join(se_parts) + r" \\")

    # FE / Cluster indicators
    if td.all_fe or td.all_cl:
        lines.append(r"\midrule")
    if td.all_fe:
        lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Fixed Effects}} \\")
        for fe in td.all_fe:
            parts = [r"\quad " + _latex_escape(fe)]
            for flag in td.fe_flags[fe]:
                parts.append("Y" if flag else "N")
            lines.append(" & ".join(parts) + r" \\")
    if td.all_cl:
        lines.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Clustering}} \\")
        for cl in td.all_cl:
            parts = [r"\quad " + _latex_escape(cl)]
            for flag in td.cl_flags[cl]:
                parts.append("Y" if flag else "N")
            lines.append(" & ".join(parts) + r" \\")

    # Footer
    lines.append(r"\midrule")
    lines.append(" & ".join(["N"] + [str(n) for n in td.n_obs]) + r" \\")
    lines.append(" & ".join(["R$^2$"] + [f"{r2:.4f}" for r2 in td.r_squared]) + r" \\")
    lines.append(" & ".join(["Adj.\\ R$^2$"] + [f"{r2a:.4f}" for r2a in td.r_squared_adj]) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if td.stars:
        lines.append(
            r"\begin{tablenotes}\small\item "
            r"$^{*}$\,p$<$0.10, $^{**}$\,p$<$0.05, $^{***}$\,p$<$0.01"
            r"\end{tablenotes}"
        )
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _render_html(td: _TableData) -> str:
    """Render table data as an HTML table."""
    n_models = len(td.labels)

    lines: list[str] = []
    lines.append('<table class="regtable">')
    lines.append("<thead>")

    # Header
    lines.append("<tr>")
    lines.append("  <th></th>")
    for lb in td.labels:
        lines.append(f"  <th>{_html_escape(lb)}</th>")
    lines.append("</tr>")

    lines.append("<tr>")
    lines.append("  <th></th>")
    for mt in td.model_types:
        lines.append(f"  <th>{mt}</th>")
    lines.append("</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")

    # Coefficients
    for var in td.all_vars:
        # Coef row
        lines.append("<tr>")
        lines.append(f'  <td class="varname">{_html_escape(var)}</td>')
        for c_str, se_str, star in td.cells[var]:
            if c_str:
                star_html = _html_stars(star)
                lines.append(f'  <td class="coef">{c_str}{star_html}</td>')
            else:
                lines.append("  <td></td>")
        lines.append("</tr>")
        # SE row
        lines.append("<tr>")
        lines.append("  <td></td>")
        for c_str, se_str, star in td.cells[var]:
            if se_str:
                lines.append(f'  <td class="se">({se_str})</td>')
            else:
                lines.append("  <td></td>")
        lines.append("</tr>")

    # FE / Cluster indicators
    if td.all_fe:
        lines.append('<tr class="fe-header">')
        lines.append(f'  <td colspan="{n_models + 1}"><em>Fixed Effects</em></td>')
        lines.append("</tr>")
        for fe in td.all_fe:
            lines.append("<tr>")
            lines.append(f'  <td class="indent">&nbsp;&nbsp;{_html_escape(fe)}</td>')
            for flag in td.fe_flags[fe]:
                lines.append(f"  <td>{'Y' if flag else 'N'}</td>")
            lines.append("</tr>")
    if td.all_cl:
        lines.append('<tr class="cl-header">')
        lines.append(f'  <td colspan="{n_models + 1}"><em>Clustering</em></td>')
        lines.append("</tr>")
        for cl in td.all_cl:
            lines.append("<tr>")
            lines.append(f'  <td class="indent">&nbsp;&nbsp;{_html_escape(cl)}</td>')
            for flag in td.cl_flags[cl]:
                lines.append(f"  <td>{'Y' if flag else 'N'}</td>")
            lines.append("</tr>")

    # Footer
    lines.append('<tr class="footer">')
    lines.append("  <td>N</td>")
    for n in td.n_obs:
        lines.append(f"  <td>{n}</td>")
    lines.append("</tr>")

    lines.append("<tr>")
    lines.append("  <td>R&sup2;</td>")
    for r2 in td.r_squared:
        lines.append(f"  <td>{r2:.4f}</td>")
    lines.append("</tr>")

    lines.append("<tr>")
    lines.append("  <td>Adj. R&sup2;</td>")
    for r2a in td.r_squared_adj:
        lines.append(f"  <td>{r2a:.4f}</td>")
    lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")
    if td.stars:
        lines.append('<p class="regtable-note">* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01</p>')

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────


def regtable(
    *results: RegressionResult | GroupRegressionResult,
    labels: list[str] | None = None,
    precision: int = 4,
    stars: bool = True,
    format: str = "text",
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
        format: Output format — "text" (default), "latex", or "html".

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
        labels = [lb if lb else f"({i + 1})" for i, lb in enumerate(auto_labels)]
    if len(labels) != n_models:
        raise ValueError(f"Expected {n_models} labels, got {len(labels)}.")

    td = _build_table_data(results, labels, precision, stars)

    if format == "latex":
        return _render_latex(td)
    elif format == "html":
        return _render_html(td)
    else:
        return _render_text(td, precision)


# ── Helpers ───────────────────────────────────────────────────────


def _fmt_sig(x: float, sig: int) -> str:
    """Format a number with sig significant figures, no padding."""
    return f"{x:.{sig}g}"


def _fmt_g_pad(x: float, width: int, sig: int) -> str:
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


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _latex_stars(star: str) -> str:
    """Convert star string to LaTeX superscript."""
    if not star:
        return ""
    return "$^{" + star + "}$"


def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_stars(star: str) -> str:
    """Convert star string to HTML superscript."""
    if not star:
        return ""
    return f"<sup>{star}</sup>"
