"""Side-by-side regression table display (esttab-style)."""

from __future__ import annotations

from dataclasses import dataclass, field

from polars_reg._groupby import GroupRegressionResult
from polars_reg._results import RegressionResult


class RegTable(str):
    """String subclass with Jupyter-friendly _repr_html_.

    Behaves exactly like a str (print, concatenation, etc.)
    but auto-renders as HTML in Jupyter notebooks.
    """

    _html: str | None

    def __new__(cls, text: str, html: str | None = None):
        obj = super().__new__(cls, text)
        obj._html = html
        return obj

    def _repr_html_(self) -> str | None:
        return self._html


# A stat spec is (stat_key, open_bracket, close_bracket)
# e.g. ("t", "(", ")") or ("se", "[", "]")
StatSpec = tuple[str, str, str]


@dataclass
class _TableData:
    """Intermediate structured data for rendering a regression table."""

    labels: list[str]
    model_types: list[str]
    all_vars: list[str]
    # cells[var] = list of (coef_str, se_str, t_str, star_str) per model
    cells: dict[str, list[tuple[str, str, str, str]]]
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
    # Display options
    stat_specs: list[StatSpec] = field(default_factory=list)
    wide: bool = False


def _build_table_data(
    results: tuple[RegressionResult, ...],
    labels: list[str],
    precision: int,
    stars: bool,
    stat_specs: list[StatSpec],
    wide: bool,
) -> _TableData:
    """Extract structured table data from regression results."""

    # Collect variable names in order of first appearance
    all_vars: list[str] = []
    for r in results:
        for name in r.names:
            if name not in all_vars:
                all_vars.append(name)

    # Build cells — now stores (coef_str, se_str, t_str, star_str)
    cells: dict[str, list[tuple[str, str, str, str]]] = {}
    for var in all_vars:
        row_cells = []
        for r in results:
            if var in r.names:
                idx = r.names.index(var)
                coef = r.coefficients[idx]
                se = r.se[idx]
                t = r.tstat[idx]
                pval = r.pvalue[idx]
                star = _star(pval) if stars else ""
                c_str = _fmt_sig(coef, precision)
                se_str = _fmt_sig(se, precision)
                t_str = _fmt_sig(t, precision)
                row_cells.append((c_str, se_str, t_str, star))
            else:
                row_cells.append(("", "", "", ""))
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
        stat_specs=stat_specs,
        wide=wide,
    )


def _get_stat_value(cell: tuple[str, str, str, str], key: str) -> str:
    """Get the stat string from a cell tuple by key."""
    _, se_str, t_str, _ = cell
    if key == "t":
        return t_str
    elif key == "se":
        return se_str
    return ""


def _stat_footnote(stat_specs: list[StatSpec]) -> str:
    """Build footnote describing what's in parentheses/brackets."""
    _names = {"t": "t-statistics", "se": "standard errors"}
    _bracket_names = {"(": "parentheses", "[": "brackets"}
    parts = []
    for key, open_b, _close_b in stat_specs:
        name = _names.get(key, key)
        bname = _bracket_names.get(open_b, "parentheses")
        parts.append(f"{name} in {bname}")
    return ", ".join(parts).capitalize()


def _render_text(td: _TableData, precision: int) -> str:
    """Render table data as plain text."""
    n_models = len(td.labels)
    n_stats = len(td.stat_specs)

    if td.wide:
        # In wide mode, each model gets 1 + n_stats columns
        cols_per_model = 1 + n_stats
        col_w = max(14, max(len(lb) for lb in td.labels) + 4)
    else:
        cols_per_model = 1
        col_w = max(14, max(len(lb) for lb in td.labels) + 4)

    # Name column width
    all_row_labels = list(td.all_vars) + ["N", "R²", "Adj. R²", "Fixed Effects", "Clustering"]
    for fe in td.all_fe:
        all_row_labels.append(f"  {fe}")
    for cl in td.all_cl:
        all_row_labels.append(f"  {cl}")
    name_w = max(14, max(len(lb) for lb in all_row_labels) + 2)

    total_cols = n_models * cols_per_model
    total_w = name_w + 1 + total_cols * (col_w + 1)
    sep = "=" * total_w

    lines: list[str] = []

    # Header
    lines.append(sep)
    hdr = f"{'':>{name_w}}"
    for lb in td.labels:
        if td.wide:
            # Span the label across its columns
            span_w = cols_per_model * (col_w + 1) - 1
            hdr += f" {lb:^{span_w}}"
        else:
            hdr += f" {lb:>{col_w}}"
    lines.append(hdr)

    type_row = f"{'':>{name_w}}"
    for mt in td.model_types:
        if td.wide:
            span_w = cols_per_model * (col_w + 1) - 1
            type_row += f" {mt:^{span_w}}"
        else:
            type_row += f" {mt:>{col_w}}"
    lines.append(type_row)
    lines.append("-" * total_w)

    # Coefficients
    for var in td.all_vars:
        if td.wide:
            # Wide: coef and stats on same row, in separate columns
            row = f"{var:<{name_w}}"
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    formatted = _fmt_g_pad(float(c_str), col_w - len(star), precision) + star
                    row += f" {formatted:>{col_w}}"
                    for spec in td.stat_specs:
                        val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                        wrapped = f"{spec[1]}{val}{spec[2]}"
                        row += f" {wrapped:>{col_w}}"
                else:
                    row += f" {'':>{col_w}}" * cols_per_model
            lines.append(row)
        else:
            # Default: coef row, then stat rows below
            coef_line = f"{var:<{name_w}}"
            stat_lines: list[str] = [f"{'':>{name_w}}" for _ in td.stat_specs]
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    formatted = _fmt_g_pad(float(c_str), col_w - len(star), precision) + star
                    coef_line += f" {formatted:>{col_w}}"
                    for i, spec in enumerate(td.stat_specs):
                        val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                        wrapped = f"{spec[1]}{val}{spec[2]}"
                        stat_lines[i] += f" {wrapped:>{col_w}}"
                else:
                    coef_line += f" {'':>{col_w}}"
                    for i in range(n_stats):
                        stat_lines[i] += f" {'':>{col_w}}"
            lines.append(coef_line)
            for sl in stat_lines:
                lines.append(sl)

    # FE / Cluster indicators
    if td.all_fe or td.all_cl:
        lines.append("-" * total_w)
    if td.all_fe:
        lines.append(f"{'Fixed Effects':<{name_w}}")
        for fe in td.all_fe:
            row = f"{'  ' + fe:<{name_w}}"
            for flag in td.fe_flags[fe]:
                val = "Y" if flag else "N"
                if td.wide:
                    span_w = cols_per_model * (col_w + 1) - 1
                    row += f" {val:^{span_w}}"
                else:
                    row += f" {val:>{col_w}}"
            lines.append(row)
    if td.all_cl:
        lines.append(f"{'Clustering':<{name_w}}")
        for cl in td.all_cl:
            row = f"{'  ' + cl:<{name_w}}"
            for flag in td.cl_flags[cl]:
                val = "Y" if flag else "N"
                if td.wide:
                    span_w = cols_per_model * (col_w + 1) - 1
                    row += f" {val:^{span_w}}"
                else:
                    row += f" {val:>{col_w}}"
            lines.append(row)

    # Footer
    lines.append("-" * total_w)
    n_row = f"{'N':<{name_w}}"
    for n in td.n_obs:
        if td.wide:
            span_w = cols_per_model * (col_w + 1) - 1
            n_row += f" {n:^{span_w}}"
        else:
            n_row += f" {n:>{col_w}}"
    lines.append(n_row)

    r2_row = f"{'R²':<{name_w}}"
    for r2 in td.r_squared:
        if td.wide:
            span_w = cols_per_model * (col_w + 1) - 1
            r2_row += f" {r2:^{span_w}.4f}"
        else:
            r2_row += f" {r2:>{col_w}.4f}"
    lines.append(r2_row)

    r2a_row = f"{'Adj. R²':<{name_w}}"
    for r2a in td.r_squared_adj:
        if td.wide:
            span_w = cols_per_model * (col_w + 1) - 1
            r2a_row += f" {r2a:^{span_w}.4f}"
        else:
            r2a_row += f" {r2a:>{col_w}.4f}"
    lines.append(r2a_row)

    lines.append(sep)

    # Footnotes
    notes: list[str] = []
    if td.stat_specs:
        notes.append(_stat_footnote(td.stat_specs))
    if td.stars:
        notes.append("* p<0.10, ** p<0.05, *** p<0.01")
    if notes:
        lines.append(". ".join(notes))

    return "\n".join(lines)


def _render_latex(td: _TableData) -> str:
    """Render table data as a LaTeX tabular."""
    n_models = len(td.labels)
    n_stats = len(td.stat_specs)

    if td.wide:
        cols_per_model = 1 + n_stats
        col_spec = "l" + "c" * (n_models * cols_per_model)
    else:
        col_spec = "l" + "c" * n_models

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    if td.wide and n_stats > 0:
        # Use multicolumn to span each model's columns
        hdr_parts = [""]
        for lb in td.labels:
            hdr_parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{_latex_escape(lb)}}}")
        lines.append(" & ".join(hdr_parts) + r" \\")
        type_parts = [""]
        for mt in td.model_types:
            type_parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{mt}}}")
        lines.append(" & ".join(type_parts) + r" \\")
    else:
        hdr = " & ".join([""] + [_latex_escape(lb) for lb in td.labels]) + r" \\"
        lines.append(hdr)
        type_row = " & ".join([""] + td.model_types) + r" \\"
        lines.append(type_row)
    lines.append(r"\midrule")

    # Coefficients
    for var in td.all_vars:
        if td.wide:
            coef_parts = [_latex_escape(var)]
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    star_tex = _latex_stars(star)
                    coef_parts.append(f"{c_str}{star_tex}")
                    for spec in td.stat_specs:
                        val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                        coef_parts.append(f"{spec[1]}{val}{spec[2]}")
                else:
                    coef_parts.extend([""] * (1 + n_stats))
            lines.append(" & ".join(coef_parts) + r" \\")
        else:
            coef_parts = [_latex_escape(var)]
            stat_rows: list[list[str]] = [[""] for _ in td.stat_specs]
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    star_tex = _latex_stars(star)
                    coef_parts.append(f"{c_str}{star_tex}")
                    for i, spec in enumerate(td.stat_specs):
                        val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                        stat_rows[i].append(f"{spec[1]}{val}{spec[2]}")
                else:
                    coef_parts.append("")
                    for i in range(n_stats):
                        stat_rows[i].append("")
            lines.append(" & ".join(coef_parts) + r" \\")
            for sr in stat_rows:
                lines.append(" & ".join(sr) + r" \\")

    # FE / Cluster indicators
    total_cols = n_models * (1 + n_stats) if td.wide else n_models
    if td.all_fe or td.all_cl:
        lines.append(r"\midrule")
    if td.all_fe:
        lines.append(r"\multicolumn{" + str(total_cols + 1) + r"}{l}{\textit{Fixed Effects}} \\")
        for fe in td.all_fe:
            parts = [r"\quad " + _latex_escape(fe)]
            for flag in td.fe_flags[fe]:
                val = "Y" if flag else "N"
                if td.wide:
                    parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{val}}}")
                else:
                    parts.append(val)
            lines.append(" & ".join(parts) + r" \\")
    if td.all_cl:
        lines.append(r"\multicolumn{" + str(total_cols + 1) + r"}{l}{\textit{Clustering}} \\")
        for cl in td.all_cl:
            parts = [r"\quad " + _latex_escape(cl)]
            for flag in td.cl_flags[cl]:
                val = "Y" if flag else "N"
                if td.wide:
                    parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{val}}}")
                else:
                    parts.append(val)
            lines.append(" & ".join(parts) + r" \\")

    # Footer
    lines.append(r"\midrule")
    n_parts = ["N"]
    for n in td.n_obs:
        if td.wide:
            n_parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{n}}}")
        else:
            n_parts.append(str(n))
    lines.append(" & ".join(n_parts) + r" \\")

    r2_parts = ["R$^2$"]
    for r2 in td.r_squared:
        val = f"{r2:.4f}"
        if td.wide:
            r2_parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{val}}}")
        else:
            r2_parts.append(val)
    lines.append(" & ".join(r2_parts) + r" \\")

    r2a_parts = [r"Adj.\ R$^2$"]
    for r2a in td.r_squared_adj:
        val = f"{r2a:.4f}"
        if td.wide:
            r2a_parts.append(rf"\multicolumn{{{cols_per_model}}}{{c}}{{{val}}}")
        else:
            r2a_parts.append(val)
    lines.append(" & ".join(r2a_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Footnotes
    note_parts: list[str] = []
    if td.stat_specs:
        note_parts.append(_stat_footnote(td.stat_specs))
    if td.stars:
        note_parts.append(r"$^{*}$\,p$<$0.10, $^{**}$\,p$<$0.05, $^{***}$\,p$<$0.01")
    if note_parts:
        note_text = ". ".join(note_parts)
        lines.append(r"\begin{tablenotes}\small\item " + note_text + r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _render_html(td: _TableData) -> str:
    """Render table data as an HTML table."""
    n_models = len(td.labels)
    n_stats = len(td.stat_specs)

    lines: list[str] = []
    lines.append('<table class="regtable">')
    lines.append("<thead>")

    if td.wide and n_stats > 0:
        cols_per_model = 1 + n_stats
        # Header with colspan
        lines.append("<tr>")
        lines.append("  <th></th>")
        for lb in td.labels:
            lines.append(f'  <th colspan="{cols_per_model}">{_html_escape(lb)}</th>')
        lines.append("</tr>")
        lines.append("<tr>")
        lines.append("  <th></th>")
        for mt in td.model_types:
            lines.append(f'  <th colspan="{cols_per_model}">{mt}</th>')
        lines.append("</tr>")
    else:
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
        if td.wide:
            lines.append("<tr>")
            lines.append(f'  <td class="varname">{_html_escape(var)}</td>')
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    star_html = _html_stars(star)
                    lines.append(f'  <td class="coef">{c_str}{star_html}</td>')
                    for spec in td.stat_specs:
                        val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                        lines.append(f'  <td class="stat">{spec[1]}{val}{spec[2]}</td>')
                else:
                    lines.append("  <td></td>" * (1 + n_stats))
            lines.append("</tr>")
        else:
            # Coef row
            lines.append("<tr>")
            lines.append(f'  <td class="varname">{_html_escape(var)}</td>')
            for c_str, se_str, t_str, star in td.cells[var]:
                if c_str:
                    star_html = _html_stars(star)
                    lines.append(f'  <td class="coef">{c_str}{star_html}</td>')
                else:
                    lines.append("  <td></td>")
            lines.append("</tr>")
            # Stat rows
            for spec in td.stat_specs:
                lines.append("<tr>")
                lines.append("  <td></td>")
                for c_str, se_str, t_str, star in td.cells[var]:
                    val = _get_stat_value((c_str, se_str, t_str, star), spec[0])
                    if val:
                        lines.append(f'  <td class="stat">{spec[1]}{val}{spec[2]}</td>')
                    else:
                        lines.append("  <td></td>")
                lines.append("</tr>")

    # FE / Cluster indicators
    total_cols = n_models * (1 + n_stats) if td.wide else n_models
    if td.all_fe:
        lines.append('<tr class="fe-header">')
        lines.append(f'  <td colspan="{total_cols + 1}"><em>Fixed Effects</em></td>')
        lines.append("</tr>")
        for fe in td.all_fe:
            lines.append("<tr>")
            lines.append(f'  <td class="indent">&nbsp;&nbsp;{_html_escape(fe)}</td>')
            for flag in td.fe_flags[fe]:
                val = "Y" if flag else "N"
                if td.wide:
                    lines.append(f'  <td colspan="{1 + n_stats}">{val}</td>')
                else:
                    lines.append(f"  <td>{val}</td>")
            lines.append("</tr>")
    if td.all_cl:
        lines.append('<tr class="cl-header">')
        lines.append(f'  <td colspan="{total_cols + 1}"><em>Clustering</em></td>')
        lines.append("</tr>")
        for cl in td.all_cl:
            lines.append("<tr>")
            lines.append(f'  <td class="indent">&nbsp;&nbsp;{_html_escape(cl)}</td>')
            for flag in td.cl_flags[cl]:
                val = "Y" if flag else "N"
                if td.wide:
                    lines.append(f'  <td colspan="{1 + n_stats}">{val}</td>')
                else:
                    lines.append(f"  <td>{val}</td>")
            lines.append("</tr>")

    # Footer
    lines.append('<tr class="footer">')
    lines.append("  <td>N</td>")
    for n in td.n_obs:
        if td.wide:
            lines.append(f'  <td colspan="{1 + n_stats}">{n}</td>')
        else:
            lines.append(f"  <td>{n}</td>")
    lines.append("</tr>")

    lines.append("<tr>")
    lines.append("  <td>R&sup2;</td>")
    for r2 in td.r_squared:
        if td.wide:
            lines.append(f'  <td colspan="{1 + n_stats}">{r2:.4f}</td>')
        else:
            lines.append(f"  <td>{r2:.4f}</td>")
    lines.append("</tr>")

    lines.append("<tr>")
    lines.append("  <td>Adj. R&sup2;</td>")
    for r2a in td.r_squared_adj:
        if td.wide:
            lines.append(f'  <td colspan="{1 + n_stats}">{r2a:.4f}</td>')
        else:
            lines.append(f"  <td>{r2a:.4f}</td>")
    lines.append("</tr>")

    lines.append("</tbody>")
    lines.append("</table>")

    # Footnotes
    note_parts: list[str] = []
    if td.stat_specs:
        note_parts.append(_stat_footnote(td.stat_specs))
    if td.stars:
        note_parts.append("* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01")
    if note_parts:
        note_text = ". ".join(note_parts)
        lines.append(f'<p class="regtable-note">{note_text}</p>')

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────


def _normalize_stat(
    stat: str | tuple[str, ...] | None,
    brackets: str,
) -> list[StatSpec]:
    """Normalize stat parameter into a list of StatSpec tuples."""
    if stat is None:
        return []

    if brackets == "round":
        primary = ("(", ")")
        secondary = ("[", "]")
    elif brackets == "square":
        primary = ("[", "]")
        secondary = ("(", ")")
    else:
        raise ValueError(f"brackets must be 'round' or 'square', got {brackets!r}")

    valid_stats = {"t", "se"}

    if isinstance(stat, str):
        if stat not in valid_stats:
            raise ValueError(f"stat must be 't', 'se', or a tuple thereof, got {stat!r}")
        return [(stat, primary[0], primary[1])]

    # Tuple of strings
    if len(stat) < 1 or len(stat) > 2:
        raise ValueError(f"stat tuple must have 1 or 2 elements, got {len(stat)}")
    for s in stat:
        if s not in valid_stats:
            raise ValueError(f"stat must be 't' or 'se', got {s!r}")

    specs: list[StatSpec] = [(stat[0], primary[0], primary[1])]
    if len(stat) > 1:
        specs.append((stat[1], secondary[0], secondary[1]))
    return specs


def regtable(
    *results: RegressionResult | GroupRegressionResult,
    labels: list[str] | None = None,
    precision: int = 4,
    stars: bool = True,
    stat: str | tuple[str, ...] | None = "t",
    brackets: str = "round",
    wide: bool = False,
    output_format: str = "text",
    **kwargs: str,
) -> str:
    """Display multiple regressions side-by-side in a compact table.

    Args:
        *results: RegressionResult or GroupRegressionResult objects.
            GroupRegressionResult is automatically expanded, using group
            keys as column labels.
        labels: Column labels. Defaults to group keys for GroupRegressionResult,
            (1), (2), ... for individual results.
        precision: Significant figures for coefficients/stats (default 4).
        stars: Show significance stars (default True).
            * p<0.10, ** p<0.05, *** p<0.01
        stat: Which statistic(s) to display alongside coefficients.
            - ``"t"`` (default): t-statistics
            - ``"se"``: standard errors
            - ``("t", "se")``: both — first in primary brackets, second in secondary
            - ``("se", "t")``: both — reversed order
            - ``None``: coefficients only, no sub-statistics
        brackets: Primary bracket style — ``"round"`` for () or ``"square"`` for [].
            When showing two stats, the secondary stat uses the other style.
        wide: If True, stats appear as columns to the right of coefficients
            instead of rows below.
        output_format: Output format — "text" (default), "latex", or "html".

    Returns:
        Formatted string table.
    """
    # Support deprecated 'format' keyword for backwards compatibility
    if "format" in kwargs:
        output_format = kwargs.pop("format")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
    if not results:
        raise ValueError("At least one RegressionResult is required.")

    stat_specs = _normalize_stat(stat, brackets)

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

    td = _build_table_data(results, labels, precision, stars, stat_specs, wide)

    if output_format == "latex":
        return RegTable(_render_latex(td))
    elif output_format == "html":
        html = _render_html(td)
        return RegTable(html, html=html)
    else:
        text = _render_text(td, precision)
        html = _render_html(td)
        return RegTable(text, html=html)


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
