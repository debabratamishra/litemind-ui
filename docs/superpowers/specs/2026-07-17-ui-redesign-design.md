# UI Redesign — Conversations Sidebar, Unified Chat/RAG, Settings Panel

**Date:** 2026-07-17
**Branch:** `feat/nextjs-frontend`
**Scope:** `nextjs-frontend/` (TypeScript/React) + a small backend change in `app/backend/` (Python).

## Goal

Improve the LiteMindUI frontend so that:

1. The **left pane** is a conversation hub — it lists conversation history and creates new
   conversations.
2. There are **just two mode buttons: Chat and RAG** (the standalone Web Search page is
   removed).
3. **Web search is integrable across both Chat and RAG** (a per-conversation toggle), not a
   separate page.
4. **Settings lives at the bottom-left** of the sidebar as a slide-over panel.

## Decisions (confirmed with user)

| Decision | Choice |
|----------|--------|
| Conversation history model | **Unified list** — one list of all conversations, each tagged Chat (💬) or RAG (📚) with a badge |
| Knowledge-base / document management | **Inside the Settings panel** (bottom-left) as a "Knowledge Base" section |
| Web search in RAG | **Backend combines** — RAG query optionally retrieves docs AND searches the web, merging both into the answer |
| Structure / routing | **Route-driven** — keep `/chat` and `/rag`; sidebar navigates and loads the selected conversation |

## 1. Architecture & Routing

- `app/page.tsx` → `redirect('/chat')` (unchanged behavior).
- `app/chat/page.tsx` → `ChatView` (renders the active Chat conversation).
- `app/rag/page.tsx` → `RagView` (renders the active RAG conversation).
- **Remove** `app/web-search/page.tsx` and `app/settings/page.tsx` (settings becomes a panel).
- `app/layout.tsx` → remove the redundant global `<header>` strip (the sidebar owns branding; its
  existing mobile hamburger handles small screens). Keep the flex shell, `ThemeProvider`,
  `TooltipProvider`, and `<Sidebar/>`. Each view renders its own top toolbar.
- Settings is a **client-side slide-over panel** opened from the sidebar footer — not a route.

## 2. Data Model (`src/lib/types.ts` + `src/lib/store.ts`)

Replace the single `chatMessages: UIMessage[]` + `sessionId` with a conversations collection.

```ts
// src/lib/types.ts
export type ConversationMode = 'chat' | 'rag';

export interface Conversation {
  id: string;
  title: string;            // auto-derived from the first user message
  mode: ConversationMode;
  messages: UIMessage[];
  webSearch: boolean;       // per-conversation web-search toggle (used in Chat & RAG)
  createdAt: string;        // ISO
  updatedAt: string;        // ISO
}
```

Store shape (`AppStore`):

- State: `settings` (unchanged `AppSettings`), `conversations: Conversation[]`,
  `activeId: string | null`.
- Actions:
  - `createConversation(mode: ConversationMode): string` — appends a new empty conversation,
    sets it active, returns its id. Title defaults to `"New conversation"`.
  - `selectConversation(id: string)` — sets `activeId`.
  - `deleteConversation(id: string)` — removes it; if it was active, activates the next most
    recent conversation of either mode (or `null`).
  - `renameConversation(id: string, title: string)`.
  - `addMessage(convId, msg)` — appends a `UIMessage` (generates id + `createdAt`).
  - `updateLastMessage(convId, content, isStreaming?)` — replaces the last assistant message
    content (streaming helper, unchanged logic).
  - `setWebSearch(convId, value)` — toggles the per-conversation web-search flag.
  - `clearConversation(convId)` — empties `messages`.
  - `setSettings(partial)` — unchanged.
- **Persistence:** `persist` middleware keeps `settings`, `conversations`, and `activeId` in
  `localStorage` (key `litemind-store`). Chat history is now persisted (previously session-only).
- **Migration note:** the previous persisted `chatMessages` key shape is replaced; existing
  browser history is not migrated (fresh start). No server data is affected.

RAG query-time options (`multiAgent`, `hybridSearch`, `nResults`) remain **local component state**
in `RagView` (they are per-query, not per-conversation); they reset when a conversation is
loaded.

## 3. Sidebar (`src/components/layout/sidebar.tsx`, rewrite)

Layout, top → bottom, in a `flex flex-col` panel (existing sidebar tokens/colors reused):

1. **Brand row** — `BrainCircuit` icon + "LiteMind" / "AI Workspace" (unchanged).
2. **New conversation** button (primary) — calls `createConversation(activeMode)` where
   `activeMode` is derived from the current route (`/rag` → `'rag'`, else `'chat'`), then
   navigates to that route and selects the new conversation.
3. **Mode switch** — segmented `Chat | RAG` control (`MessageSquare` / `Database` icons).
   Active state follows `usePathname()` (`/chat` vs `/rag`). Clicking navigates to the route.
4. **Conversation list** (scrollable, `flex-1`) — unified list of all conversations sorted by
   `updatedAt` desc. Each item is a `<button>` showing:
   - mode badge (💬 Chat / 📚 RAG),
   - `title` (truncated),
   - relative time (e.g. "2h", "Mar 3"),
   - hover-revealed delete (`Trash2`) with a confirmation step.
   Active item uses `bg-sidebar-primary` + `aria-current="page"`. Clicking selects the
   conversation and navigates to its mode route.
5. **Footer** — `Settings` button (gear + "Settings" label, bottom-left) opening the Settings
   slide-over, plus the existing `ThemeToggle`.

Mobile behavior: keep the existing fixed hamburger + overlay slide-in; the conversation list and
footer scroll within the panel.

## 4. Settings Panel (`src/components/settings-panel.tsx`, new)

A left-anchored slide-over (built on Radix `Dialog` for focus trapping + Esc-to-close), opened
from the sidebar footer. Scrollable, with sections (reusing existing logic from `chat/page.tsx`
`SettingsPanel` and `settings/page.tsx`):

1. **Models** — backend select (Ollama/OpenRouter/Nvidia NIM); model dropdown for Ollama
   (local + cloud catalog) or free-text for cloud providers (reuse `getEnhancedModels`).
2. **Generation** — temperature, max tokens, top-p, frequency penalty, repetition penalty
   (sliders, from the settings page).
3. **Backend keys** — Ollama URL, OpenRouter API key, Nvidia NIM API key.
4. **Knowledge Base** — moved from the RAG page: upload zone (click + drag/drop, same accepted
   types), file list with delete, KB status card, and reset dialog. Reuses the existing
   `RagStatusCard` / `FileItem` components (extract them into shared components if not already
   shared). Reads/writes `ragFiles` and calls `getRagFiles`, `uploadRagFile`, `deleteRagFile`,
   `getRagStatus`, `resetRag`, `checkRagDuplicate`.

**Dropped:** the old per-request SerpAPI key input on the Web Search page — web search relies on
the backend's server-side `SERP_API_KEY` (the old field was never sent in the request).

## 5. Chat View (`src/app/chat/page.tsx`, adapt to `ChatView`)

- Reads the active conversation where `mode === 'chat'` (falls back to the most recent chat
  conversation, or prompts to create one).
- Keeps the existing message rendering: avatars, GenUI renderer, markdown fallback, streaming
  cursor, empty state.
- Keeps the top **GenUI mode selector** bar (Rendered/Code) when Generative UI is enabled.
- Composer controls: **web search toggle** (🌐, bound to the conversation's `webSearch`),
  **voice toggle**, **Send**. Web search on sends the request to `/api/chat/web-search` with
  `use_web_search: true` (existing behavior preserved).
- **Removes** the in-page right-side `SettingsPanel`.
- Adds a **Clear chat** action in the toolbar (calls `clearConversation`). "New conversation"
  lives in the sidebar.

## 6. RAG View (`src/app/rag/page.tsx`, adapt to `RagView`)

- **Removes** all file-management UI (now in Settings → Knowledge Base).
- Reads the active `mode === 'rag'` conversation; renders the Q&A message list + composer.
- Composer has a **new web search toggle** (🌐) bound to the conversation's `webSearch`. When on,
  the query is sent to `/api/rag/query` with `use_web_search: true` (new backend support).
- Keeps a compact options row: **multi-agent**, **hybrid search**, **n results** (local state).
- Empty state prompts the user to upload documents via Settings.
- Uses `streamRagQuery` (updated to forward `use_web_search`).

## 7. Backend Change — RAG + Web Search (`app/backend/`)

1. **`app/backend/models/api_models.py`** — add to `RAGQueryRequestEnhanced`:
   ```python
   use_web_search: Optional[bool] = False
   ```
2. **`app/backend/api/rag.py`** — in `_handle_rag_query` (or the point where the RAG skill
   builds its context), when `request.use_web_search` is true:
   - call `WebSearchService` to fetch web results for the query,
   - format the snippets and **merge them with the retrieved document context** (e.g. prepend a
     clearly-delimited "Web results" section to the context/prompt the skill receives),
   - then stream the answer via the existing RAG skill.
   The exact wiring point is finalized during implementation after inspecting `rag_service.py` and
   the RAG skill's `stream()` signature, to avoid deep skill changes. The goal is unchanged:
   retrieve documents AND search the web, merge both, and let the LLM answer.
3. No new endpoint; `/api/rag/query` keeps its `text/plain` streaming contract.

## 8. API Client (`src/lib/api.ts` + `src/lib/types.ts`)

- `RagQueryRequest` gains `use_web_search?: boolean`.
- `streamRagQuery(request, signal)` forwards `use_web_search` unchanged.

## 9. Error Handling & Accessibility

- Empty states: no conversations (prompt to create), no RAG documents (prompt to upload in
  Settings).
- Conversation deletion: confirmation (inline confirm or Radix `AlertDialog`).
- Web search unavailable (no `SERP_API_KEY`): backend error surfaces a friendly
  "⚠️ Error: …" message in the message stream (existing pattern).
- ARIA: conversation items are `<button>` with labels and `aria-current`; Settings panel uses
  Radix `Dialog` (focus trap, Esc close, `aria-modal`); mode switch uses `role="group"` with
  `aria-pressed`.
- Tested in **light and dark** themes (existing tokens reused).

## 10. Testing & Gates

- **Mandatory (per `nextjs-frontend/AGENTS.md`):** `npm run lint` and `npm run build` must
  pass with zero errors.
- **Backend (per root `CLAUDE.md`):** `uv run ruff check .` and `uv run ty check …` on the
  changed Python files.
- **Manual verification checklist:**
  - Create a Chat and a RAG conversation; switch between them; reload the page and confirm
    history persists.
  - Web search toggle works in Chat (hits `/api/chat/web-search`) and in RAG (hits
    `/api/rag/query` with `use_web_search: true`, and the answer reflects both docs + web).
  - Upload/delete documents and reset the knowledge base from Settings → Knowledge Base.
  - Delete a conversation; confirm active conversation falls back correctly.
  - Settings (model, generation, backend keys) persist across reload.
  - Verify light + dark themes.
- Frontend unit tests are out of scope unless a test runner is already configured (none is set
  up in `nextjs-frontend/`); correctness is covered by lint + build + manual checks.

## 11. Files Touched

**Create**
- `src/components/settings-panel.tsx` — consolidated Settings slide-over.
- (Optional) `src/components/knowledge-base.tsx` — extracted KB management UI for Settings.

**Rewrite**
- `src/components/layout/sidebar.tsx` — conversation list, mode switch, settings button.
- `src/lib/store.ts` — conversations model + actions + persistence.
- `src/lib/types.ts` — `Conversation`, `ConversationMode`, `RagQueryRequest.use_web_search`.
- `src/app/chat/page.tsx` — `ChatView`.
- `src/app/rag/page.tsx` — `RagView`.

**Delete**
- `src/app/web-search/page.tsx`
- `src/app/settings/page.tsx`

**Modify**
- `src/app/layout.tsx` — remove global header.
- `src/lib/api.ts` — `RagQueryRequest` + `streamRagQuery` forward `use_web_search`.
- `app/backend/models/api_models.py` — add `use_web_search` to `RAGQueryRequestEnhanced`.
- `app/backend/api/rag.py` — wire web search into RAG query.
