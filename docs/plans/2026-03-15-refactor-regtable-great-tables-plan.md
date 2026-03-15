---
title: "refactor: Replace regtable renderers with Great Tables"
type: refactor
date: 2026-03-15
---

# Refactor: Replace regtable Renderers with Great Tables

## Overview

Replace the hand-built text/HTML/LaTeX rendering in `_regtable.py` (~1050 lines, 6 renderers) with Great Tables (GT). `regtable()` will return a `GT` object that natively renders in Jupyter notebooks and exports to HTML/LaTeX via GT methods.

**Breaking change**: Return type changes from `RegTable` (str subclass) to `great_tables.GT`. Version bump to `0.2.0`.

## Problem Statement / Motivation

The current `_regtable.py` contains 6 hand-built renderers (text, HTML, LaTeX × normal + transposed) totaling ~700 lines of string-building code. This is:
- Maintenance-heavy: every new feature (stat display, wide mode, transpose) must be implemented 6 times
- Not Polars-native: output is a string, not composable with Polars workflows
- Not customizable: users can't adjust styling without modifying source code

Great Tables is the Polars-recommended display library. It provides professional HTML rendering, LaTeX export, and a fluent customization API.

### Research Insights

**Prior art — pyfixest/maketables:**
- pyfixest extracted table logic into [maketables](https://github.com/py-econometrics/maketables), which uses GT for HTML and has separate LaTeX/Word renderers
- Their `ETable` class accepts fitted model objects, extracts stats, and renders via GT
- They replaced their `great_tables` dependency with `maketables` to decouple table generation
- Our approach is simpler: we stay with GT directly since we own the full pipeline

**statsmodels proposal:**
- [statsmodels #9563](https://github.com/statsmodels/statsmodels/issues/9563) proposes GT integration — still in proposal stage
- Confirms the ecosystem is moving toward GT for regression tables

## Proposed Solution

1. **Build a Polars DataFrame** as the intermediate representation (replaces `_TableData`)
2. **Wrap in GT** with styling applied via `tab_style`, `tab_spanner`, `tab_source_note`
3. **Return the GT object** — users get native Jupyter rendering + `.as_latex()` + `.as_raw_html()` + full GT customization API
4. **Drop plain text rendering** — terminal users use `.as_raw_html()` or Jupyter
5. **Drop `RegTable` class** — GT replaces it
6. **Drop `output_format` parameter** — users call `.as_latex()` or `.as_raw_html()` on the result

## Technical Approach

### Architecture: Flat DataFrame Strategy

GT's LaTeX export does NOT support `rowname_col`, `groupname_col`, `md()`, `sub_missing()`, or `tab_style()` (all raise `NotImplementedError` or silently no-op). Therefore, we use a **flat DataFrame** where all structure is encoded in row ordering and cell text, not GT metadata.

#### GT v0.21.0 Feature Support Matrix (verified by testing)

| Feature | HTML | LaTeX | Notes |
|---------|------|-------|-------|
| `GT(df)` | ✅ | ✅ | No `rowname_col` or `groupname_col` |
| `cols_label()` | ✅ | ✅ | Can set label to empty string `""` |
| `cols_align()` | ✅ | ✅ | Maps to `l`/`c`/`r` in tabular spec |
| `tab_spanner()` | ✅ | ✅ | Generates `\multicolumn` + `\cmidrule` in LaTeX |
| `tab_source_note()` | ✅ | ✅ | Plain text only — `md()` raises NotImplementedError in LaTeX |
| `tab_header()` | ✅ | ✅ | `\caption*{}` in LaTeX |
| `tab_options()` | ✅ | ⚠️ | Some options affect LaTeX (font size), most HTML-only |
| `tab_style()` | ✅ | ❌ | **No-op in LaTeX** — warns "Styles are not yet supported" |
| `sub_missing()` | ✅ | ❌ | NotImplementedError in LaTeX — use empty strings instead |
| `fmt_number()` | ✅ | ✅ | But we pre-format cells, so not needed |

#### DataFrame Structure — Normal Mode

The DataFrame has columns: `["var", "(1)", "(2)", ...]` where `"var"` is the label column (renamed to `""` via `cols_label`).

```
Row type        | var column       | Model columns
─────────────────────────────────────────────────
coef            | "x1"             | "1.234***"
sub-stat        | ""               | "(3.45)"
coef            | "x2"             | "0.567**"
sub-stat        | ""               | "(2.01)"
fe_header       | "Fixed Effects"  | ""
fe_indicator    | "  firm_id"      | "Y"
cl_header       | "Clustering"     | ""
cl_indicator    | "  firm_id"      | "Y"
summary         | "N"              | "1000"
summary         | "R²"             | "0.456"
summary         | "Adj. R²"        | "0.451"
```

**Note:** No explicit separator rows. Use `tab_style(style.borders(...))` for HTML visual separation. LaTeX output is structurally clean without separators — the `\bottomrule` at the end and content labels ("Fixed Effects", "N") provide sufficient visual structure.

**Model type handling:** When `model_type=True` (default), include model type as a second row in the DataFrame, directly after a header-like structure. Since GT column headers only support a single label row (no multi-line headers cleanly), encode the model type as the first data row:

```
var column       | Model columns
─────────────────────────────────
""               | "OLS"          ← model type row (italicized via tab_style)
"x1"             | "1.234***"
```

When `model_type=False`, omit this row entirely.

#### Verified LaTeX output (GT v0.21.0):
```latex
\begin{table}[!t]
\fontsize{12.0pt}{14.4pt}\selectfont
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc}
\toprule
 & M1 & M2 \\
\midrule\addlinespace[2.5pt]
x1 & 1.234*** & 0.891* \\
 & (3.45) & (1.78) \\
x2 & 0.567** & 1.112*** \\
 & (2.01) & (4.56) \\
Fixed Effects &  &  \\
  firm\_id & N & Y \\
N & 1000 & 1000 \\
R² & 0.456 & 0.789 \\
\bottomrule
\end{tabular*}
\begin{minipage}{\linewidth}
T-statistics in parentheses. * p<0.10, ** p<0.05, *** p<0.01\\
\end{minipage}
\end{table}
```

Key observations:
- GT auto-escapes LaTeX special chars (`_cons` → `\_cons`, `firm_id` → `firm\_id`)
- `\toprule`/`\midrule`/`\bottomrule` from booktabs auto-generated
- Unicode `²` in `R²` passes through (works with modern LaTeX + utf8 inputenc)
- `tab_source_note` renders inside `\begin{minipage}` block
- Full-width table via `\begin{tabular*}{\linewidth}` with `@{\extracolsep{\fill}}`

#### Wide Mode

Each model gets `1 + n_stats` columns grouped under a `tab_spanner`:
```
Columns: ["var", "(1)__coef", "(1)__t", "(2)__coef", "(2)__t"]
Spanner: "(1)" spans ["(1)__coef", "(1)__t"]
cols_label: all sub-columns labeled "" (hidden)
```

**Note:** Double underscore `__` as separator to avoid collision with user labels containing single `_`. The `cols_label` sets all sub-column headers to empty string, but GT still renders an empty sub-header row. This is a cosmetic issue — acceptable for now.

Verified LaTeX wide mode output:
```latex
\multicolumn{2}{c}{(1)} & \multicolumn{2}{c}{(2)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 &  &  &  \\   % empty sub-header row
```

#### Transposed Mode

Models are rows and variables are columns. FE/cluster indicators and summary stats become additional columns:
```
Row type   | var column         | x1         | x2         | FE:g | N    | R²
───────────────────────────────────────────────────────────────────────────────
coef       | "(1) OLS"          | "1.234***" | "0.567**"  | "N"  | "1000" | "0.456"
sub-stat   | ""                 | "(3.45)"   | "(2.01)"   | ""   | ""     | ""
coef       | "(2) OLS"          | "0.891*"   | "1.112***" | "Y"  | "1000" | "0.789"
sub-stat   | ""                 | "(1.78)"   | "(4.56)"   | ""   | ""     | ""
```

When `model_type=False`, the var column shows just `"(1)"` / `"(2)"` without the model type suffix.

### GT Construction Pattern

```python
from great_tables import GT, style, loc

gt = (
    GT(df)
    .cols_label(var="")
    .cols_align(align="center", columns=model_columns)
    .cols_align(align="left", columns="var")
    .tab_source_note(footnote_text)
    # HTML-only styling (no-op in LaTeX, which is fine)
    .tab_style(
        style.borders(sides="top", weight="1px", color="#D3D3D3"),
        loc.body(rows=[first_fe_row])
    )
    .tab_style(
        style.borders(sides="top", weight="1px", color="#D3D3D3"),
        loc.body(rows=[first_summary_row])
    )
    .tab_style(
        style.text(style="italic"),
        loc.body(columns="var", rows=section_header_rows)
    )
    .tab_options(
        table_body_hlines_style="none",  # remove default row lines
        column_labels_border_bottom_width="2px",
    )
)
```

### Key Constraint: LaTeX Compatibility

All GT features used must produce usable output in BOTH `.as_raw_html()` AND `.as_latex()`. Strategy:
- Use only universally-supported features for structure (`cols_label`, `cols_align`, `tab_spanner`, `tab_source_note`)
- Apply `tab_style` for HTML enhancement — it silently no-ops in LaTeX with a warning
- Pre-format all cell values as strings (don't rely on `fmt_number` or `sub_missing`)

### Internal Column Naming Convention

Use a `"var"` column (renamed to `""` via `cols_label`) instead of an empty string column name. This avoids potential issues with Polars handling of empty string column names.

- Normal mode: `"var"` (label), `"(1)"`, `"(2)"`, etc. (or user-provided labels)
- Wide mode: `"var"`, `"{label}__coef"`, `"{label}__t"`, `"{label}__se"` (hidden via `cols_label`)
- Transposed mode: `"var"`, variable names, summary column names

## Implementation Plan

### Phase 1: Dependencies and Setup

- [x] Add `great-tables>=0.15` to `pyproject.toml` `dependencies`
- [x] Bump version to `0.2.0` in `pyproject.toml`
- [x] Verify GT installs correctly in dev environment: `python -c "from great_tables import GT; print(GT.__module__)"`

### Phase 2: Core DataFrame Builder

Replace `_build_table_data` → `_build_table_df`. Keep `_normalize_stat`, `_star`, `_fmt_sig` helpers.

- [x] `_build_table_df()` → returns `pl.DataFrame` with all rows (coef, sub-stat, FE, cluster, summary)
- [x] Use `"var"` as the label column name (renamed to `""` via `cols_label` later)
- [x] Use empty strings `""` for missing cells (not None/null — avoids `sub_missing` dependency)
- [x] Track row indices for each section (for `tab_style` application in HTML)
- [x] Handle `rename`, `precision`, `stars`, `stat_specs` during DataFrame construction
- [x] Handle `wide=True` variant (columns instead of sub-rows, `__coef`/`__t`/`__se` suffixes)
- [x] Handle `transpose=True` variant (models as rows)
- [x] Handle `model_type`: when True, insert a model type row as the first data row; when False, omit it
- [x] Return a `_GTTableSpec` dataclass:

```python
@dataclass
class _GTTableSpec:
    df: pl.DataFrame
    model_columns: list[str]     # column names for model data
    fe_start_row: int | None     # first FE row index (for border)
    summary_start_row: int | None  # first summary row index (for border)
    section_header_rows: list[int]  # rows to italicize ("Fixed Effects", "Clustering")
    spanners: list[tuple[str, list[str]]]  # (label, columns) for wide mode
    footnote: str                # stat description + stars legend
```

### Phase 3: GT Construction

New function `_build_gt(spec: _GTTableSpec) -> GT`:

- [x] Create `GT(spec.df)` — no `rowname_col` or `groupname_col`
- [x] Apply `cols_label(var="")` to hide the label column header
- [x] Apply `cols_align(align="center", columns=spec.model_columns)`
- [x] Apply `cols_align(align="left", columns="var")`
- [x] Apply `tab_spanner` for each entry in `spec.spanners` (wide mode)
- [x] Apply `tab_style` borders at `spec.fe_start_row` and `spec.summary_start_row` (HTML-only, no-op in LaTeX)
- [x] Apply `tab_style` italic for `spec.section_header_rows` (HTML-only)
- [x] Apply `tab_source_note(spec.footnote)` — plain text only
- [x] Apply `tab_options(table_body_hlines_style="none", column_labels_border_bottom_width="2px")`
- [x] In wide mode, apply `cols_label` to set all `__coef`/`__t`/`__se` sub-column headers to `""`
- [x] Suppress GT's `tab_style` LaTeX warning via `warnings.filterwarnings("ignore", message="Styles are not yet supported")` scoped to `_build_gt`

### Phase 4: Update Public API

- [x] `regtable()` returns `GT` instead of `RegTable`
- [x] Remove `output_format` parameter (and deprecated `format` kwarg)
- [x] Keep all other parameters: `stat`, `brackets`, `wide`, `transpose`, `rename`, `model_type`, `precision`, `stars`, `labels`
- [x] `GroupRegressionResult` expansion logic stays the same (pre-GT, unchanged)
- [x] Remove `RegTable` class
- [x] Remove all 6 old renderers (`_render_text`, `_render_html`, `_render_latex` × normal + transposed)
- [x] Remove `_TableData` dataclass
- [x] Remove renderer-only helpers: `_fmt_g_pad`, `_latex_escape`, `_latex_stars`, `_html_escape`, `_html_stars`, `_transposed_columns`
- [x] Keep `_stat_footnote` (used to build `tab_source_note` text) and `_get_stat_value` (used during DataFrame construction)
- [x] Update `__init__.py`: remove `RegTable` from exports and `__all__`, keep `regtable`

### Phase 5: Update Tests

58 existing tests. Categorization:

**Keep unchanged (10 tests):** `_normalize_stat` unit tests — internal logic, no output format dependency.

**Rewrite content checks (30+ tests):** Tests that assert on string content (`"x1" in table`, `"N" in table`) need to check `table.as_raw_html()` instead. Pattern:
```python
# Before
table = regtable(r1, r2)
assert "x1" in table

# After
table = regtable(r1, r2)
html = table.as_raw_html()
assert "x1" in html
```

**Delete (5 tests):** `RegTable`-specific tests:
- `test_regtable_returns_regtable_type` → replace with `isinstance(table, GT)` check
- `test_regtable_str_operations` → delete (GT is not a string)
- `test_regtable_repr_html_text_mode` → replace with GT `_repr_html_` check
- `test_regtable_repr_html_html_mode` → replace with GT `as_raw_html` check
- `test_regtable_repr_html_latex_mode` → replace with GT `as_latex` check

**Rewrite format-specific (8 tests):** LaTeX tests use `table.as_latex()`, HTML tests use `table.as_raw_html()`. Remove `format=` parameter from all calls.

**Add new tests:**
- [x] `test_regtable_returns_gt_object` — `isinstance(table, GT)`
- [x] `test_regtable_as_latex_valid` — LaTeX output contains `\begin{tabular*}`, `\toprule`, etc.
- [x] `test_regtable_as_raw_html_valid` — HTML output contains `<table`, `</table>`
- [x] `test_regtable_repr_html` — `table._repr_html_()` returns non-None HTML string
- [x] `test_regtable_gt_chainable` — `table.tab_header(title="Table 1")` doesn't error
- [x] `test_regtable_latex_auto_escapes` — underscores in var names escaped in LaTeX

### Phase 6: Documentation

- [x] Update CLAUDE.md: remove `RegTable` mention, note GT dependency
- [x] Update `regtable()` docstring: return type is `GT`, remove `output_format` docs, add examples of `.as_latex()` and `.as_raw_html()`
- [x] Update memory: note the breaking change and new architecture

## Acceptance Criteria

- [x] `regtable(r1, r2)` returns a `great_tables.GT` object
- [x] GT renders correctly in Jupyter (via `_repr_html_`)
- [x] `.as_latex()` produces valid LaTeX with booktabs (`\toprule`, `\midrule`, `\bottomrule`)
- [x] `.as_raw_html()` produces valid HTML with `<table>` structure
- [x] All existing features work: stat, brackets, wide, transpose, stars, precision, rename, model_type, labels
- [x] `GroupRegressionResult` auto-expansion works
- [x] FE/cluster Y/N indicators display correctly
- [x] N, R², Adj. R² summary rows display correctly
- [x] Footnotes (stat description + stars legend) display in both HTML and LaTeX
- [x] Section separators visible in HTML (borders via `tab_style`)
- [x] LaTeX output is usable without separators (content structure is sufficient)
- [x] Users can chain GT methods on the result for further customization
- [x] All tests pass
- [x] `great-tables>=0.15` is a required dependency

## Dependencies & Risks

**Dependencies:**
- `great-tables>=0.15` (adds ~7 transitive deps: Babel, commonmark, faicons, htmltools, importlib-metadata, importlib-resources, typing_extensions)
- GT's LaTeX support is partial — must use flat DataFrame (no stub/groups/styles)

**Risks:**
- **Breaking change**: Return type `str` → `GT`. Existing code using string operations on `regtable()` output will break.
- **`print()` regression**: `print(regtable(r1, r2))` in terminal won't produce readable output (GT has no plain text mode). Users must use Jupyter or `.as_raw_html()`.
- **GT LaTeX limitations**: `tab_style` is a no-op in LaTeX — visual separators are HTML-only. This is acceptable because LaTeX output is typically post-processed by users and the content structure (labels, booktabs rules) provides sufficient visual organization.
- **Wide mode empty sub-header**: When `cols_label` hides sub-column names, GT renders an empty sub-header row. Cosmetic issue.
- **Dependency weight**: ~7 new transitive dependencies for all users, even those who never call `regtable()`.

**Mitigations:**
- Bump to `0.2.0` to signal breaking change
- Document migration path in changelog
- Flat DataFrame approach is robust against GT version changes
- `tab_style` gracefully degrades (warns, doesn't error) in LaTeX

### Edge Cases

1. **Single model**: `regtable(r1)` — column is `"(1)"`, no comparison. Works fine.
2. **10+ models**: Columns may overflow horizontally. GT handles this via HTML scrolling; LaTeX may need landscape mode (user responsibility).
3. **Model type heterogeneity**: OLS + IV + GMM in same table — model type appears in column header. Verified to work with multi-line labels.
4. **Empty FE/cluster**: Models with no FE or clusters — skip FE/cluster sections entirely. No separator borders needed.
5. **All variables disjoint**: Models share zero variables — each var row has blanks for non-participating models. Works with empty strings.
6. **Unicode in variable names**: GT handles Unicode in both HTML and LaTeX (with utf8 encoding).
7. **Very long variable names**: May cause column width issues. GT auto-sizes in HTML; LaTeX uses `@{\extracolsep{\fill}}`.

## References

- [Great Tables docs](https://posit-dev.github.io/great-tables/)
- [GT LaTeX blog post](https://posit-dev.github.io/great-tables/blog/latex-output-tables/)
- [GT + Polars blog post](https://posit-dev.github.io/great-tables/blog/polars-styling/)
- [Polars ecosystem page](https://docs.pola.rs/user-guide/ecosystem/)
- [pyfixest / maketables](https://github.com/py-econometrics/maketables) — prior art for GT regression tables
- [statsmodels GT proposal](https://github.com/statsmodels/statsmodels/issues/9563)
- Current implementation: `polars_reg/_regtable.py`
- Current tests: `tests/test_regtable.py`
- Existing brainstorm (display options): `docs/brainstorms/2026-03-11-regtable-display-options-brainstorm.md`
