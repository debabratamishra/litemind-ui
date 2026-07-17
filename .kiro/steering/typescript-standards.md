---
inclusion: fileMatch
fileMatchPattern: "nextjs-frontend/**/*.{ts,tsx,mts}"
---

# TypeScript / Next.js Coding Standards

Applies whenever a TypeScript or TSX file inside `nextjs-frontend/` is in context.

## Toolchain
- **Next.js 16** App Router — no Pages Router patterns
- **React 19** functional components only — no class components
- **TypeScript 5** strict mode (`"strict": true`) — no implicit `any`
- **npm** for packages — not yarn or pnpm
- Run before finishing any frontend change:
  ```bash
  cd nextjs-frontend
  npm run lint    # must be zero errors
  npm run build   # must succeed — this is the type-check gate
  ```

## Directory rules
| What | Where |
|------|-------|
| Pages | `src/app/<route>/page.tsx` |
| Layouts | `src/app/<route>/layout.tsx` |
| Shared components | `src/components/PascalCase.tsx` |
| shadcn/ui primitives | `src/components/ui/` — **do not hand-edit** |
| Custom hooks | `src/hooks/useCamelCase.ts` |
| Utilities + API clients | `src/lib/kebab-case.ts` |

## Component rules
- `'use client'` directive only when the component uses browser APIs, event handlers, or hooks
- Server Components by default — they cannot use `useState`, `useEffect`, or browser APIs
- Do not use `getServerSideProps`, `getStaticProps`, or `getInitialProps` (Pages Router only)
- Use `dynamic(() => import(...), { ssr: false })` for heavy client-only components

## Styling
- **Tailwind CSS v4** utilities only — no plain CSS files, no inline `style={{}}` for static values
- Dark/light theme via `next-themes` — test both modes for every new UI component
- Re-generate shadcn/ui components with `npx shadcn add <component>` — never hand-edit `src/components/ui/`

## State management
- Local UI state: `useState` / `useReducer`
- Cross-component / persistent client state: **Zustand** only
- Do not add Redux, MobX, Jotai, Recoil, React Query, or SWR

## Data fetching and backend calls
- All backend calls go through helpers in `src/lib/` — never call `fetch` directly in components
- Backend URL comes from `NEXT_PUBLIC_API_URL` env var — never hard-code `localhost:8000`
- Streaming responses (chat, RAG): consume via `ReadableStream` — do not buffer before rendering
- Non-2xx responses: show a user-friendly error — do not expose raw server error text

## Environment variables
- `NEXT_PUBLIC_*` vars are baked into the client bundle — **never put secrets here**
- Document new vars in `nextjs-frontend/.env.local` comments and in root `.env.example`

## Accessibility
- Interactive elements without visible text must have `aria-label`
- Form fields must have `<label>` or `aria-labelledby`
- WCAG 2.1 AA contrast: 4.5:1 normal text, 3:1 large text / UI components

## Forbidden patterns
- `// @ts-ignore` or `// @ts-expect-error` without an explanatory comment
- `eslint-disable` without a justification comment
- `console.log()` in committed code
- Importing an entire icon library — import individual icons from `lucide-react`
- Adding charting libraries other than **Recharts**
- Adding component libraries other than **shadcn/ui + Radix UI**
- New packages with version ranges (`^` or `~`) — use exact versions
