# Decompose `generative_ui.py` into a package

**Date:** 2026-07-15
**Status:** Approved (design)

## Goal

`app/frontend/components/generative_ui.py` is 1,429 lines — too large and doing
too much for a single module. Split it into a cohesive `generative_ui/` package
of ~6–7 small, single-purpose modules, **preserving all existing behavior** and
**keeping the public API unchanged** so no callers need edits.

Decisions confirmed with the user:
- **No tests** — refactor only. Behavior is preserved by moving code verbatim
  and re-exporting the public API; verification is via import check + lint.
- **Cohesive package** (~6–7 modules grouped by responsibility, not ultra-fine).

## Non-goals

- No behavior, signature, or output changes.
- No new abstractions, naming changes, or internal cleanups. The move is a pure
  relocation so the diff is reviewable as "code moved, nothing changed."
- No edits to `text_renderer.py` or any other consumer.

## Public API (unchanged)

`text_renderer.py` imports:
```python
from .generative_ui import has_ui_blocks, render_mixed_content, auto_enhance_content
```
The new package `__init__.py` re-exports those three names, plus
`render_ui_component` and `GENERATIVE_UI_SYSTEM_PROMPT` for completeness. No
other file imports from `generative_ui` (no tests reference it either).

## Module layout

Dependency arrow (`→`) means "imports from". The graph is acyclic.

```
app/frontend/components/generative_ui/
├── __init__.py        # module docstring + re-exports public API
├── parsing.py         # LEAF: fence + HTML-document parsing, JSON helpers (no streamlit)
├── security.py        # LEAF: _sanitize_html, _sanitize_url
├── constants.py       # LEAF: _RAW_HTML_COMPONENT_TYPES, CSS/JS strings, regexes, system prompt
├── renderers.py       # → security
├── webapp.py          # → parsing, security, constants
├── auto_enhance.py    # → parsing, webapp
└── dispatch.py        # → parsing, constants, renderers, webapp
```

### `parsing.py` (leaf, stdlib-only)
`_is_valid_fence_info`, `_is_valid_ui_component_type`, `_iter_fenced_blocks`,
`_extract_fenced_body`, `_match_single_fenced_block`, `_find_html_code_fence`,
`_try_parse_json_object`, `_is_html_tag_boundary`, `_parse_doctype_html_tag_end`,
`_parse_html_open_tag_end`, `_parse_html_close_tag_end`,
`_find_primary_html_document_span`, `_extract_primary_html_document`.

### `security.py` (leaf)
`_sanitize_html`, `_sanitize_url`.

### `constants.py` (leaf)
`_RAW_HTML_COMPONENT_TYPES`, `_WEBAPP_CSS`, `_IFRAME_APP_SHELL_CSS`,
`_IFRAME_APP_BOOTSTRAP_SCRIPT`, regexes (`_INTERACTIVE_HTML_RE`,
`_HTML_HEAD_CLOSE_RE`, `_HTML_BODY_CLOSE_RE`), `GENERATIVE_UI_SYSTEM_PROMPT`.

### `renderers.py`
The 13 simple `_render_*` components: info_card, data_table, metric, chart,
button_group, progress, alert, columns, json_viewer, link_cards, steps, tabs,
callout. Imports `security`.

### `webapp.py`
All webapp/iframe rendering & normalization: `_inject_webapp_css`,
`_clamp_webapp_height`, `_extract_webapp_height`, `_wrap_webapp_html`,
`_wrap_iframe_app_html`, `_inject_html_fragment`, `_compose_iframe_app_markup`,
`_component_type_for_html`, `_normalise_webapp_payload`,
`_normalise_iframe_app_payload`, `_build_iframe_app_iframe_src`,
`_build_webapp_iframe_src`, `_render_webapp`, `_render_iframe_app`.
Imports `parsing`, `security`, `constants`.

### `auto_enhance.py`
`_match_bold_kv_line`, `_split_md_row`, `_looks_like_md_table_row`,
`_is_md_table_separator_cell`, `_is_md_table_separator_row`, `_parse_md_table`,
`_auto_convert_tables`, `_auto_convert_metrics`, `_wrap_html_markup_as_ui_block`,
`_auto_convert_html_markup`, `auto_enhance_content`.
Imports `parsing`, `webapp` (for `_component_type_for_html`).

### `dispatch.py`
`_COMPONENT_REGISTRY` (component_type → renderer fn), `render_ui_component`,
`render_mixed_content` (keeps its lazy `from ..utils.text_processing import ...`),
`has_ui_blocks`. Imports `parsing`, `constants`, `renderers`, `webapp`.

## Implementation steps

1. Delete `app/frontend/components/generative_ui.py`.
2. Create `app/frontend/components/generative_ui/` package with the modules
   above, moving each function verbatim to its target module.
3. Build `_COMPONENT_REGISTRY` in `dispatch.py` exactly as it was (same keys/order).
4. `generative_ui/__init__.py` re-exports the public API and carries the
   original module docstring.

## Verification

- `uv run ruff check .` passes.
- `uv run python -c "from app.frontend.components.generative_ui import has_ui_blocks, render_mixed_content, auto_enhance_content"` imports cleanly.
- (Optional manual) Run the Streamlit app and confirm a response with `ui:*`
  blocks and the auto-enhance path still render.
