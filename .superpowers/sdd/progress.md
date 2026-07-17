# UI Redesign — SDD Progress Ledger

Branch: `feat/nextjs-frontend`
Plan: `docs/superpowers/plans/2026-07-17-ui-redesign.md`

All tasks complete. Verified 2026-07-17.

## Tasks

- T1: Conversation data model (types.ts) — complete
- T2: Zustand store rewrite (store.ts) — complete
- T3: Extract Knowledge Base component (knowledge-base.tsx) — complete
- T4: Settings panel (consolidated slide-over) — complete
- T5: Sidebar rewrite (conversation hub) — complete
- T6: Remove global header from layout — complete
- T7: Chat view (active-conversation driven) — complete
- T8: RAG view (remove file mgmt, add web toggle) — complete
- T9: Delete web-search and settings pages — complete (pages do not exist)
- T10: API client forward use_web_search — complete (api.ts RagQueryRequest + streamRagQuery)
- T11: Backend RAG + web search combine — complete
  - `app/backend/models/api_models.py`: added `use_web_search` to `RAGQueryRequestEnhanced`
  - `app/backend/api/rag.py`: `_fetch_web_search_section()` + wired into `_handle_rag_query`
    event_generator; augments `request.system_prompt` with a delimited "Web Search
    Results" section; surfaces friendly `⚠️` notice when SERP_API_KEY missing/fails.
- T12: Verification gates — complete
  - `npm run lint` (nextjs-frontend): 0 errors, 0 warnings
  - `npm run build` (nextjs-frontend): success
  - `uv run ruff check` on changed Python: passed
  - `uv run ty check` on changed Python: passed

## Notes

- Lint fixes for the new `react-hooks/set-state-in-effect` rule (React 19): replaced
  mounted/isSupported effect patterns with `useSyncExternalStore` in
  `theme-toggle.tsx` and `use-voice-input.ts`; added justified `eslint-disable` block
  directives for genuine UI-sync effects (sidebar route-change close; knowledge-base
  fetch-on-mount).
- The entire `nextjs-frontend/` directory is untracked in git (was never committed on
  this branch). Work lives on disk; no commit made (user did not request one).
- Manual UI checklist (create/switch/persist conversations, web search in chat + RAG,
  upload/reset KB, delete conversation fallback, light/dark themes) NOT executed —
  requires a running backend + browser. Covered by lint/build/ruff/ty only.
