# Brainstorm: regtable Display Options

**Date:** 2026-03-11

## What We're Building

Enhanced `regtable()` with flexible stat display options, matching Stata esttab/estout flexibility.

### New kwargs:

```python
regtable(*results,
    stat="t",           # "t" | "se" | ("t","se") | ("se","t") | None
    brackets="round",   # "round" (default) or "square" — sets primary bracket style
    wide=False,         # stats beside coefficient instead of below
    # existing: labels, precision, stars, output_format
)
```

### Behavior:

- **`stat`** (default `"t"`): What statistic to show alongside coefficients.
  - `"t"` — t-statistics (BREAKING: was SEs before)
  - `"se"` — standard errors
  - `("t", "se")` — both; first element gets primary brackets, second gets secondary
  - `("se", "t")` — both, reversed order
  - `None` — coefficients only

- **`brackets`** (default `"round"`): Primary bracket style.
  - `"round"` — primary stat in `()`, secondary in `[]`
  - `"square"` — primary stat in `[]`, secondary in `()`

- **`wide`** (default `False`): Layout direction.
  - `False` — stats appear as rows below coefficients (current behavior)
  - `True` — stats appear as unlabeled columns to the right of coefficient

- **Footnote**: Always print what's in parentheses/brackets, e.g. `t-statistics in parentheses` or `t-statistics in parentheses, standard errors in brackets`

## Key Decisions

1. **Default changed to t-stats** — breaking change, acceptable at v0.1.x
2. **`stat` parameter is polymorphic** — string or tuple, tuple order controls display order
3. **First tuple element = primary brackets** — `()` by default, flippable with `brackets`
4. **Wide columns are unlabeled** — consistent with how sub-rows are unlabeled
5. **Footnote always shown** — describes what the brackets contain

## Scope

- Affects `_build_table_data`, `_render_text`, `_render_latex`, `_render_html`
- `_TableData` needs t-stat strings added (currently only stores SE strings)
- All three renderers need wide mode support
- Tests need updating (current tests expect SEs)
