"""Side-by-side regression table display via Great Tables."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import polars as pl
from great_tables import GT, loc, style

from polars_reg._groupby import GroupRegressionResult
from polars_reg._results import RegressionResult

# A stat spec is (stat_key, open_bracket, close_bracket)
# e.g. ("t", "(", ")") or ("se", "[", "]")
StatSpec = tuple[str, str, str]


@dataclass
class _GTTableSpec:
    """Intermediate spec for constructing a GT object."""

    df: pl.DataFrame
    model_columns: list[str]
    fe_start_row: int | None = None
    summary_start_row: int | None = None
    section_header_rows: list[int] = field(default_factory=list)
    model_type_row: int | None = None
    spanners: list[tuple[str, list[str]]] = field(default_factory=list)
    footnote: str = ""


# ── DataFrame builders ───────────────────────────────────────────


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


def _build_footnote(stat_specs: list[StatSpec], stars: bool) -> str:
    """Build the full footnote string for tab_source_note."""
    parts: list[str] = []
    if stat_specs:
        parts.append(_stat_footnote(stat_specs))
    if stars:
        parts.append("* p<0.10, ** p<0.05, *** p<0.01")
    return ". ".join(parts)


def _extract_cells(
    results: tuple[RegressionResult, ...],
    rename: dict[str, str] | None,
    precision: int,
    stars: bool,
) -> tuple[list[str], dict[str, list[tuple[str, str, str, str]]]]:
    """Extract cell data from results. Returns (all_vars, cells dict)."""
    all_vars: list[str] = []
    for r in results:
        for name in r.names:
            display_name = rename.get(name, name) if rename else name
            if display_name not in all_vars:
                all_vars.append(display_name)

    _rev = {}
    if rename:
        for orig, disp in rename.items():
            _rev[disp] = orig

    cells: dict[str, list[tuple[str, str, str, str]]] = {}
    for var in all_vars:
        row_cells = []
        orig_var = _rev.get(var, var)
        for r in results:
            if orig_var in r.names:
                idx = r.names.index(orig_var)
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

    return all_vars, cells


def _extract_fe_cl(
    results: tuple[RegressionResult, ...],
) -> tuple[list[str], list[str], dict[str, list[bool]], dict[str, list[bool]]]:
    """Extract FE and cluster info from results."""
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
    return all_fe, all_cl, fe_flags, cl_flags


def _build_table_df_normal(
    results: tuple[RegressionResult, ...],
    labels: list[str],
    precision: int,
    stars_flag: bool,
    stat_specs: list[StatSpec],
    model_type: bool,
    rename: dict[str, str] | None,
) -> _GTTableSpec:
    """Build DataFrame for normal (non-transposed, non-wide) layout."""
    all_vars, cells = _extract_cells(results, rename, precision, stars_flag)
    all_fe, all_cl, fe_flags, cl_flags = _extract_fe_cl(results)

    rows: list[dict[str, str]] = []
    section_header_rows: list[int] = []
    model_type_row: int | None = None

    # Model type row
    if model_type and any(r.model_type for r in results):
        model_type_row = len(rows)
        row: dict[str, str] = {"var": ""}
        for i, lb in enumerate(labels):
            row[lb] = results[i].model_type
        rows.append(row)

    # Coefficient + sub-stat rows
    for var in all_vars:
        # Coef row
        row = {"var": var}
        for i, lb in enumerate(labels):
            c_str, _, _, star = cells[var][i]
            row[lb] = f"{c_str}{star}" if c_str else ""
        rows.append(row)

        # Sub-stat rows
        for spec in stat_specs:
            row = {"var": ""}
            for i, lb in enumerate(labels):
                val = _get_stat_value(cells[var][i], spec[0])
                row[lb] = f"{spec[1]}{val}{spec[2]}" if val else ""
            rows.append(row)

    # FE indicators
    fe_start_row: int | None = None
    if all_fe:
        fe_start_row = len(rows)
        section_header_rows.append(len(rows))
        row = {"var": "Fixed Effects"}
        for lb in labels:
            row[lb] = ""
        rows.append(row)
        for fe in all_fe:
            row = {"var": f"  {fe}"}
            for i, lb in enumerate(labels):
                row[lb] = "Y" if fe_flags[fe][i] else "N"
            rows.append(row)

    # Cluster indicators
    if all_cl:
        if fe_start_row is None:
            fe_start_row = len(rows)
        section_header_rows.append(len(rows))
        row = {"var": "Clustering"}
        for lb in labels:
            row[lb] = ""
        rows.append(row)
        for cl in all_cl:
            row = {"var": f"  {cl}"}
            for i, lb in enumerate(labels):
                row[lb] = "Y" if cl_flags[cl][i] else "N"
            rows.append(row)

    # Summary stats
    summary_start_row = len(rows)
    for stat_name, values in [
        ("N", [str(r.n_obs) for r in results]),
        ("R\u00b2", [f"{r.r_squared:.4f}" for r in results]),
        ("Adj. R\u00b2", [f"{r.r_squared_adj:.4f}" for r in results]),
    ]:
        row = {"var": stat_name}
        for i, lb in enumerate(labels):
            row[lb] = values[i]
        rows.append(row)

    df = pl.DataFrame(rows, schema={"var": pl.Utf8} | {lb: pl.Utf8 for lb in labels})

    return _GTTableSpec(
        df=df,
        model_columns=labels,
        fe_start_row=fe_start_row,
        summary_start_row=summary_start_row,
        section_header_rows=section_header_rows,
        model_type_row=model_type_row,
        footnote=_build_footnote(stat_specs, stars_flag),
    )


def _build_table_df_wide(
    results: tuple[RegressionResult, ...],
    labels: list[str],
    precision: int,
    stars_flag: bool,
    stat_specs: list[StatSpec],
    model_type: bool,
    rename: dict[str, str] | None,
) -> _GTTableSpec:
    """Build DataFrame for wide layout (stats as columns beside coefficients)."""
    all_vars, cells = _extract_cells(results, rename, precision, stars_flag)
    all_fe, all_cl, fe_flags, cl_flags = _extract_fe_cl(results)

    # Build column names: each model gets 1 + n_stats columns
    col_groups: list[tuple[str, list[str]]] = []
    all_model_cols: list[str] = []
    for lb in labels:
        sub_cols = [f"{lb}__coef"]
        for spec in stat_specs:
            sub_cols.append(f"{lb}__{spec[0]}")
        col_groups.append((lb, sub_cols))
        all_model_cols.extend(sub_cols)

    rows: list[dict[str, str]] = []
    section_header_rows: list[int] = []
    model_type_row: int | None = None

    # Model type row
    if model_type and any(r.model_type for r in results):
        model_type_row = len(rows)
        row: dict[str, str] = {"var": ""}
        for i, (lb, sub_cols) in enumerate(col_groups):
            row[sub_cols[0]] = results[i].model_type
            for sc in sub_cols[1:]:
                row[sc] = ""
        rows.append(row)

    # Coefficient rows (no sub-stat rows — stats are in columns)
    for var in all_vars:
        row = {"var": var}
        for i, (lb, sub_cols) in enumerate(col_groups):
            c_str, se_str, t_str, star = cells[var][i]
            if c_str:
                row[sub_cols[0]] = f"{c_str}{star}"
                for j, spec in enumerate(stat_specs):
                    val = _get_stat_value(cells[var][i], spec[0])
                    row[sub_cols[1 + j]] = f"{spec[1]}{val}{spec[2]}"

            else:
                for sc in sub_cols:
                    row[sc] = ""
        rows.append(row)

    # FE indicators
    fe_start_row: int | None = None
    if all_fe:
        fe_start_row = len(rows)
        section_header_rows.append(len(rows))
        row = {"var": "Fixed Effects"}
        for sc in all_model_cols:
            row[sc] = ""
        rows.append(row)
        for fe in all_fe:
            row = {"var": f"  {fe}"}
            for i, (lb, sub_cols) in enumerate(col_groups):
                row[sub_cols[0]] = "Y" if fe_flags[fe][i] else "N"
                for sc in sub_cols[1:]:
                    row[sc] = ""
            rows.append(row)

    # Cluster indicators
    if all_cl:
        if fe_start_row is None:
            fe_start_row = len(rows)
        section_header_rows.append(len(rows))
        row = {"var": "Clustering"}
        for sc in all_model_cols:
            row[sc] = ""
        rows.append(row)
        for cl in all_cl:
            row = {"var": f"  {cl}"}
            for i, (lb, sub_cols) in enumerate(col_groups):
                row[sub_cols[0]] = "Y" if cl_flags[cl][i] else "N"
                for sc in sub_cols[1:]:
                    row[sc] = ""
            rows.append(row)

    # Summary stats
    summary_start_row = len(rows)
    for stat_name, values in [
        ("N", [str(r.n_obs) for r in results]),
        ("R\u00b2", [f"{r.r_squared:.4f}" for r in results]),
        ("Adj. R\u00b2", [f"{r.r_squared_adj:.4f}" for r in results]),
    ]:
        row = {"var": stat_name}
        for i, (lb, sub_cols) in enumerate(col_groups):
            row[sub_cols[0]] = values[i]
            for sc in sub_cols[1:]:
                row[sc] = ""
        rows.append(row)

    schema = {"var": pl.Utf8} | {sc: pl.Utf8 for sc in all_model_cols}
    df = pl.DataFrame(rows, schema=schema)

    spanners = [(lb, sub_cols) for lb, sub_cols in col_groups]

    return _GTTableSpec(
        df=df,
        model_columns=all_model_cols,
        fe_start_row=fe_start_row,
        summary_start_row=summary_start_row,
        section_header_rows=section_header_rows,
        model_type_row=model_type_row,
        spanners=spanners,
        footnote=_build_footnote(stat_specs, stars_flag),
    )


def _build_table_df_transposed(
    results: tuple[RegressionResult, ...],
    labels: list[str],
    precision: int,
    stars_flag: bool,
    stat_specs: list[StatSpec],
    model_type: bool,
    rename: dict[str, str] | None,
) -> _GTTableSpec:
    """Build DataFrame for transposed layout (models as rows, variables as columns)."""
    n_models = len(results)
    all_vars, cells = _extract_cells(results, rename, precision, stars_flag)
    all_fe, all_cl, fe_flags, cl_flags = _extract_fe_cl(results)

    # Build column list: var + variables + FE/cluster + summary
    summary_cols: list[tuple[str, list[str]]] = []
    for fe in all_fe:
        summary_cols.append((f"FE:{fe}", ["Y" if f else "N" for f in fe_flags[fe]]))
    for cl in all_cl:
        summary_cols.append((f"Cl:{cl}", ["Y" if f else "N" for f in cl_flags[cl]]))
    summary_cols.append(("N", [str(r.n_obs) for r in results]))
    summary_cols.append(("R\u00b2", [f"{r.r_squared:.4f}" for r in results]))
    summary_cols.append(("Adj. R\u00b2", [f"{r.r_squared_adj:.4f}" for r in results]))

    model_columns = list(all_vars) + [name for name, _ in summary_cols]

    rows: list[dict[str, str]] = []
    for i in range(n_models):
        # Model label with optional type suffix
        if model_type:
            model_label = f"{labels[i]} {results[i].model_type}".rstrip()
        else:
            model_label = labels[i]

        # Coef row
        row: dict[str, str] = {"var": model_label}
        for var in all_vars:
            c_str, _, _, star = cells[var][i]
            row[var] = f"{c_str}{star}" if c_str else ""
        for name, vals in summary_cols:
            row[name] = vals[i]
        rows.append(row)

        # Sub-stat rows
        for spec in stat_specs:
            row = {"var": ""}
            for var in all_vars:
                val = _get_stat_value(cells[var][i], spec[0])
                row[var] = f"{spec[1]}{val}{spec[2]}" if val else ""
            for name, _ in summary_cols:
                row[name] = ""
            rows.append(row)

    schema = {"var": pl.Utf8} | {col: pl.Utf8 for col in model_columns}
    df = pl.DataFrame(rows, schema=schema)

    return _GTTableSpec(
        df=df,
        model_columns=model_columns,
        footnote=_build_footnote(stat_specs, stars_flag),
    )


# ── GT construction ──────────────────────────────────────────────


def _build_gt(spec: _GTTableSpec) -> GT:
    """Construct a GT object from a _GTTableSpec."""
    gt = GT(spec.df)

    # Rename "var" column to empty label
    gt = gt.cols_label(var="")

    # Alignment
    gt = gt.cols_align(align="left", columns="var")
    if spec.model_columns:
        gt = gt.cols_align(align="center", columns=spec.model_columns)

    # Spanners for wide mode
    for label, sub_cols in spec.spanners:
        gt = gt.tab_spanner(label=label, columns=sub_cols)
        # Hide sub-column names
        gt = gt.cols_label(**{sc: "" for sc in sub_cols})

    # Footnote
    if spec.footnote:
        gt = gt.tab_source_note(source_note=spec.footnote)

    # HTML-only styling (silently no-ops in LaTeX)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Styles are not yet supported")

        # Section borders
        if spec.fe_start_row is not None:
            gt = gt.tab_style(
                style.borders(sides="top", weight="1px", color="#D3D3D3"),
                loc.body(rows=[spec.fe_start_row]),
            )
        if spec.summary_start_row is not None:
            gt = gt.tab_style(
                style.borders(sides="top", weight="1px", color="#D3D3D3"),
                loc.body(rows=[spec.summary_start_row]),
            )

        # Italic section headers
        if spec.section_header_rows:
            gt = gt.tab_style(
                style.text(style="italic"),
                loc.body(columns="var", rows=spec.section_header_rows),
            )

        # Italic model type row
        if spec.model_type_row is not None:
            gt = gt.tab_style(
                style.text(style="italic"),
                loc.body(rows=[spec.model_type_row]),
            )

    # Table options
    gt = gt.tab_options(
        table_body_hlines_style="none",
        column_labels_border_bottom_width="2px",
    )

    return gt


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
    transpose: bool = False,
    rename: dict[str, str] | None = None,
    model_type: bool = True,
) -> GT:
    """Display multiple regressions side-by-side in a compact table.

    Returns a ``great_tables.GT`` object that renders natively in Jupyter
    notebooks and can be exported to HTML or LaTeX.

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
        transpose: If True, models are presented as rows and coefficients as
            columns (inverted from the default layout). N, R², and FE/cluster
            indicators appear as additional columns.
        rename: Dict mapping original variable names to display names.
            E.g. ``{"_cons": "Constant"}``
        model_type: If False, suppress the model type row (e.g. "OLS")
            in the default layout, or the type suffix in transposed layout.

    Returns:
        great_tables.GT: A GT table object. Use ``.as_raw_html()`` for HTML
        string output, ``.as_latex()`` for LaTeX output. Renders automatically
        in Jupyter notebooks.

    Examples:
        >>> table = regtable(r1, r2, r3)
        >>> table                          # renders in Jupyter
        >>> table.as_latex()               # LaTeX string
        >>> table.as_raw_html()            # HTML string
        >>> table.tab_header(title="Table 1")  # further GT customization
    """
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
    results_tuple = tuple(expanded)

    n_models = len(results_tuple)
    if labels is None:
        labels = [lb if lb else f"({i + 1})" for i, lb in enumerate(auto_labels)]
    if len(labels) != n_models:
        raise ValueError(f"Expected {n_models} labels, got {len(labels)}.")

    if transpose:
        spec = _build_table_df_transposed(
            results_tuple, labels, precision, stars, stat_specs, model_type, rename
        )
    elif wide:
        spec = _build_table_df_wide(
            results_tuple, labels, precision, stars, stat_specs, model_type, rename
        )
    else:
        spec = _build_table_df_normal(
            results_tuple, labels, precision, stars, stat_specs, model_type, rename
        )

    return _build_gt(spec)


# ── Helpers ───────────────────────────────────────────────────────


def _fmt_sig(x: float, sig: int) -> str:
    """Format a number with sig significant figures, no padding."""
    return f"{x:.{sig}g}"


def _star(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""
