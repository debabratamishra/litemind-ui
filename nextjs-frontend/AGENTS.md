# AGENTS.md — Next.js Frontend

> Rules specific to the `nextjs-frontend/` sub-project.  
> These extend (and never override) the root `AGENTS.md` and `CONSTITUTION.md`.  
> Always read those files first.

---

## Heads-up: this is not the Next.js you know

This project uses **Next.js 16** with **React 19**. APIs, conventions, and file conventions
differ from older versions in your training data. Before writing any code:

1. Check `node_modules/next/dist/docs/` for applicable guides.
2. Heed deprecation notices — do not use Pages Router patterns in this App Router project.
3. When in doubt about a Next.js API, search the installed source rather than relying on memory.

---

## 1. Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Framework | Next.js 16 App Router | Server components, streaming, layouts |
| Language | TypeScript 5, strict mode | No implicit `any` |
| UI components | shadcn/ui + Radix UI primitives | Check `src/components/` before building new ones |
| Styling | Tailwind CSS v4 | No plain CSS files; no inline `style={{}}` for static values |
| State | Zustand | Cross-component / persistent client state only |
| Charts | Recharts | Already a dependency; use it for all chart needs |
| Icons | lucide-react | Import individual icons — never the whole library |
| Markdown | react-markdown + remark-gfm + rehype-highlight | Use for all AI response rendering |
| Theme | next-themes | Dark / light mode; always test both |
| HTTP to backend | Native `fetch` | SSE / streaming via `ReadableStream` |
| Realtime voice | WebRTC + Pipecat | `useRealtimeVoice` hook; `voice-activity.tsx`, `voice-input-button.tsx` |

---

## 2. Directory layout

```
src/
  app/              Next.js App Router — pages, layouts, route segments
  components/       Shared React components (shadcn/ui based)
    ui/             shadcn/ui primitives (auto-generated; do not hand-edit)
    voice-activity.tsx, voice-input-button.tsx   realtime-voice UI
  hooks/            Custom React hooks (prefix: use*): use-realtime-voice.ts, use-voice-input.ts
  lib/              Utilities, API client helpers, type definitions
public/             Static assets
```

### Rules
- Business logic belongs in `src/lib/` or `src/hooks/`, not in components.
- shadcn/ui primitives in `src/components/ui/` are auto-generated — **do not hand-edit** them.
  Re-generate with `npx shadcn add <component>` if you need changes.
- New pages go in `src/app/<route>/page.tsx`.
- New layouts go in `src/app/<route>/layout.tsx`.
- Shared components go in `src/components/<Name>.tsx` (PascalCase, one component per file).

---

## 3. Mandatory checks

Run these before marking any frontend task complete:

```bash
# from nextjs-frontend/
npm run lint      # ESLint — zero errors required
npm run build     # TypeScript + Next.js build — must succeed
```

- `npm run build` is the type-check gate; it catches errors the IDE might miss.
- Fix all lint errors; do not suppress with `eslint-disable` comments without a justification.
- Never use `// @ts-ignore` or `// @ts-expect-error` without an explanatory comment.

---

## 4. Backend communication

The Next.js frontend talks to the FastAPI backend at the URL configured in:

```
nextjs-frontend/.env.local   → NEXT_PUBLIC_API_URL (default: http://localhost:8000)
```

### Rules
- All backend calls go through helpers in `src/lib/` — never call `fetch` directly in components.
- Streaming responses (chat, RAG, web search) use **SSE** or plain chunked encoding consumed via
  `ReadableStream`. Do not buffer streaming responses before rendering.
- Never expose `OPENROUTER_API_KEY`, `NVIDIA_NIM_API_KEY`, or any backend secret in a
  `NEXT_PUBLIC_*` env variable.
- If the backend returns a non-2xx status, surface a user-friendly error message — do not
  expose raw server error text to the UI.

---

## 5. Generative UI

The backend can instruct the frontend to render rich components via fenced code blocks in the
AI response stream:

```
```ui:component_name
{ ...json payload }
```
```

Supported component names and their expected payloads are defined in `src/lib/` (search for
`ui:` handling code). When adding a new generative UI component:

1. Add the payload type in `src/lib/` (TypeScript interface).
2. Add the renderer component in `src/components/`.
3. Add the case to the generative UI dispatcher.
4. Update the backend system prompt instructions in `app/backend/api/chat.py` to document
   the new component name and payload schema.

---

## 6. Streaming and performance

- Render streaming tokens incrementally — do not wait for the full response.
- Use `dynamic(() => import(...), { ssr: false })` for heavy client-only components (charts,
  audio visualisers, WebRTC widgets).
- Do not import entire libraries in component files. Import only what is used.
- Use `React.memo` or `useMemo` only when profiling confirms a measurable benefit — do not
  premature-optimise.

---

## 7. Accessibility

- Every interactive element must have an accessible label.
- Buttons without visible text must have `aria-label`.
- Form fields must have associated `<label>` or `aria-labelledby`.
- Focus management must be correct for modals and dialogs (shadcn/ui Dialog handles this
  automatically — do not override `onOpenAutoFocus` without good reason).
- Colour contrast must meet WCAG 2.1 AA (4.5:1 normal text, 3:1 large text / UI components).
- Test new UI in both **light** and **dark** mode before marking complete.

---

## 8. Dependency rules

- Do not add npm packages without checking whether an existing dependency covers the need.
- New packages must be added with an **exact version** (no `^` or `~`).
- Do not add React state management libraries (Redux, MobX, Jotai, Recoil) — use Zustand.
- Do not add CSS-in-JS libraries (styled-components, emotion) — use Tailwind.
- Do not add a charting library other than Recharts.
- Do not add a component library other than shadcn/ui + Radix UI.

---

## 9. File naming conventions

| Artifact | Convention | Example |
|----------|-----------|---------|
| Page components | `page.tsx` (App Router) | `src/app/chat/page.tsx` |
| Layout components | `layout.tsx` | `src/app/layout.tsx` |
| Shared components | `PascalCase.tsx` | `ChatMessage.tsx` |
| Client components | `PascalCase.tsx` with `'use client'` at top | `StreamingChat.tsx` |
| Custom hooks | `useCamelCase.ts` | `useStreamingChat.ts` |
| Utilities | `kebab-case.ts` | `format-bytes.ts` |
| Type-only files | `kebab-case.types.ts` | `chat.types.ts` |

---

## 10. Environment variables

```
# nextjs-frontend/.env.local (gitignored)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- `NEXT_PUBLIC_*` variables are baked into the client bundle at build time — **never put secrets here**.
- Server-only secrets (if ever needed) go in non-prefixed vars and are accessed only in
  Server Components or Route Handlers, never in Client Components.
- Document any new variable in a comment in `nextjs-frontend/.env.local` and in the root
  `.env.example` if it affects the Docker compose setup.

---

## 11. What frontend agents must NOT do

- Do not hand-edit files in `src/components/ui/` (shadcn/ui auto-generated primitives).
- Do not add Pages Router files (`pages/` directory) — this is an App Router project.
- Do not use `getServerSideProps`, `getStaticProps`, or `getInitialProps` — they are Pages Router only.
- Do not add `useEffect` for data fetching that could be done in a Server Component.
- Do not call the backend directly from Server Components using hard-coded `localhost` — use
  the `NEXT_PUBLIC_API_URL` / internal Docker URL pattern.
- Do not suppress TypeScript errors with blanket `any` casts — fix the type properly.
- Do not ship `console.log()` statements in committed code.

---

## 12. Realtime voice mode

Realtime voice uses **WebRTC**, not the Web Speech API (the latter only powers the
non-realtime `use-voice-input.ts` dictation button).

- Entry point: `useRealtimeVoice(settings, callbacks)` hook in `src/hooks/use-realtime-voice.ts`.
- It POSTs an SDP offer to `${NEXT_PUBLIC_API_URL}/api/voice/offer` and streams events
  over the WebRTC data channel: `ready`, `user_transcript`, `assistant_text`,
  `assistant_end`, `error`, `ended`.
- UI: `voice-activity.tsx` (speaking indicator), `voice-input-button.tsx` (toggle).
- Full event contract: `docs/superpowers/specs/2026-07-18-realtime-voice-mode-design.md`.
- Do **not** call the backend LLM gateway from the frontend; the server pipeline drives TTS.
