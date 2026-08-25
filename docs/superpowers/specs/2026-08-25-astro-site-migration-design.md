# LiteMindUI Pages site: Jekyll → Astro + React islands

**Date:** 2026-08-25
**Status:** Approved design (brainstorming session, same day)
**Scope:** `site/` directory and `.github/workflows/pages.yml`. No changes to the app (`app/`, `nextjs-frontend/`).

## Goals

1. Migrate the GitHub Pages marketing site from Jekyll to **Astro with React islands**, keeping it deployed on GitHub Pages via the existing Actions workflow.
2. Preserve the entire current experience: both pages' content, URLs, visual design (CSS), light/dark theming, and interactions.
3. Make future edits data-driven: cards/steps/tiles are structured data rendered by components; a new page is one dropped file.
4. Small, safe improvements: per-page Open Graph tags and type-checked build output.

Non-goals: runtime-fetched data (GitHub API etc.), SSR/hosting move, sitemap/RSS, view transitions, redesigning any page.

## Current state (baseline being preserved)

- Source lives in `site/` on `main`: `_config.yml`, `_layouts/default.html`, `index.md`, `developer.md`, `assets/{demo.gif,favicon.svg,css/style.css}`. A stale committed `site/_site/` build output exists and must be removed from git and ignored.
- Deploy: `.github/workflows/pages.yml` — push to `main` touching `site/**` → `actions/jekyll-build-pages` → `actions/deploy-pages`. Site settings use the "workflow" build type; no `gh-pages` branch is involved despite what `_config.yml`'s comment says.
- Site URL `https://debabratamishra.github.io`, baseurl `/litemind-ui`.
- Two pages:
  - `/litemind-ui/` — hero ("Ethereal Glass" dark backdrop, animated orbs, mouse parallax), 4 feature cards, demo GIF figure, 4 benefit cards, 3 numbered steps, callout, developer-guide teaser, enterprise section (intro + 4 tiles + CTA).
  - `/litemind-ui/developer/` — prose guide: intro, process/port table, quick-start callout, directory tree (syntax-highlighted spans), design-pattern h3 sections, request-travel walkthrough, dev tips list, extension notes, provider/env tables, ingestion formats, commands `<pre>`, CI summary, API-contract callout.
- Layout: sticky nav (brand logo tile, Home/Developer links with active state, GitHub link, theme toggle), footer (brand blurb + Project/Resources columns + copyright bar), inline head script setting `data-theme` from `localStorage['lm-theme']` or OS preference before first paint, toggle button syncing icon/`aria-pressed`, hero mouse-parallax rAF loop.
- Styling: single `assets/css/style.css` (~540 lines), plain CSS custom properties, `[data-theme="dark"]` overrides, responsive breakpoint at 640px, focus-visible outlines. No framework, no external fonts.

## Target architecture

```
site/
  astro.config.mjs        # site: https://debabratamishra.github.io, base: '/litemind-ui'
  package.json            # astro, @astrojs/react, react, react-dom, typescript
  tsconfig.json
  public/
    favicon.svg
    assets/demo.gif
    assets/favicon.svg    # keep legacy path working
  src/
    styles/global.css     # style.css ported verbatim (class names unchanged)
    layouts/Base.astro    # <head>, FOUC guard script, Header, Footer, slot
    components/
      Header.astro       # nav + active-link logic + ThemeToggle island
      Footer.astro
      SectionHead.astro  # centered h2 + lead paragraph
      CardGrid.astro     # renders CardData[] (icon/title/body)
      Steps.astro        # numbered steps (CSS counter, unchanged)
      Callout.astro      # teal left-border callout
    components/ThemeToggle.tsx   # React island (client:load)
    data/home.ts         # typed arrays: features, benefits, steps, enterpriseTiles
    pages/index.astro    # hero + sections composed from data/home.ts
    pages/developer.md   # markdown page using Base layout (frontmatter layout field)
  dist/                  # build output (gitignored)
```

### Decisions

- **Astro project root is `site/` itself** so the workflow's `paths: site/**` filter and mental model stay intact. The Astro action receives `path: site`.
- **npm** as package manager, matching `nextjs-frontend/`; lockfile committed.
- **Content model:** card groups live in `src/data/home.ts` as typed objects (`{icon, title, body}`); bespoke sections (hero, enterprise intro, CTA) remain template markup in `index.astro` — no generic "section renderer" abstraction for a two-page site.
- **Developer guide stays markdown.** `src/pages/developer.md` keeps frontmatter (`layout`, `title`, `description`) and its HTML fragments (tables/tree/callouts) which Astro markdown renders natively. Mechanical substitutions only: `{{ site.baseurl }}` → `/litemind-ui`, kramdown-flavored bits normalized. Prose copy untouched.
- **Theming:** the pre-paint inline script moves into `Base.astro`'s `<head>` unchanged in behaviour (reads `lm-theme` else OS preference). `ThemeToggle.tsx` hydrates on `client:load`, owns click handling, persists to localStorage, syncs icon + `aria-pressed`. Same key name, same semantics.
- **Parallax stays vanilla JS** inside `Base.astro` or `index.astro` scoped `<script>` (rAF damped background shift on `.hero`). Not a React component; islands only where state justifies them.
- **Styling:** `global.css` imported once by `Base.astro`; contents byte-equivalent to today's `style.css` except the file header comment. No class renames, no utility frameworks.
- **URLs:** Astro defaults (`build.format: 'directory'`, trailing slash) reproduce `/litemind-ui/` and `/litemind-ui/developer/`. Assets under `public/assets/` serve at `/litemind-ui/assets/…` exactly as today.
- **Head improvements:** per-page title/description already exist as frontmatter; add Open Graph tags (`og:title`, `og:description`, `og:type`, `og:url`) derived from the same props in `Base.astro`. Nothing else.

### Removed

- `site/_config.yml`, `site/_layouts/`
- Committed `site/_site/` output (added to `.gitignore`)
- Jekyll step in `pages.yml` → replaced by `withastro/action@v3` with `path: site`; upload path `site/dist`

## Error handling / edge cases

- FOUC guard wrapped in try/catch as today (localStorage may throw in hardened browsers).
- ThemeToggle reads/writes localStorage defensively; initial render must not mismatch server HTML beyond the icon glyph (icon text set in effect, not render, to avoid hydration warnings).
- Workflow path filter updated to also fire on `.github/workflows/pages.yml` changes (already does).
- Node version pinned via action default; lockfile ensures reproducible builds.

## Professional visual-element QA

Derived from harness design skills (`design-taste-frontend`, `high-end-visual-design`, `frontend-design`). Mode: **Redesign–Preserve** — the site's own design language is the standard, judged for fidelity and professionalism, not redesigned.

**Locked brand tokens (must survive the migration untouched):**
- Accent system: indigo + teal, one accent family across every section, both themes (Color Consistency Lock)
- Shape system: radius scale `14px` cards / `10px` small / pill buttons+nav+toggle, applied consistently (Shape Consistency Lock)
- Signature element: the "Ethereal Glass" hero (OLED backdrop, purple/emerald radial orbs, gradient headline) — the one place the site spends its boldness; all other sections stay quiet around it
- System font stack, emoji icon tiles, footer/nav structure, all visible copy

**Quality-floor checks applied to the migrated build:**

| Check | Standard |
|---|---|
| Theme integrity | One theme per page state; dark/light each complete and consistent; no half-migrated tokens |
| Contrast | WCAG AA for all text/buttons/callouts in BOTH themes (incl. `.callout` teal-on-teal, `.btn-solid` in dark) |
| Motion | New: gate the hero orb `float` animation and mouse parallax behind `prefers-reduced-motion` (today they ignore it); transitions stay transform/opacity only |
| Focus | `:focus-visible` outlines preserved on links/buttons after componentisation |
| Hero discipline | Eyebrow + H1 (≤2 lines) + lead + CTA pair, fits first viewport at desktop and 375px |
| Copy tells | Zero em-dashes anywhere rendered; no decorative status dots; single copy register; no version labels/scroll cues |
| Asset reality | Demo GIF renders at native paths; favicon resolves at both `/favicon.svg` and `/assets/favicon.svg` |

**Process:** after build, screenshot both pages in light and dark at 1280px and 375px, compare against the live https://debabratamishra.github.io/litemind-ui/, and walk the table above before declaring done. Any intentional divergence gets listed here first.

Note: `vercel-react-best-practices` was consulted and skipped deliberately — the React surface is one ~30-line theme-toggle island with no data-fetching or server-component concerns.

## Verification

1. `cd site && npm run build && npm run astro check` clean.
2. Structural parity: built `dist/index.html` + `dist/developer/index.html` contain every invariant selector and content block present today (hero, eyebrow, 4 feature cards, demo figure, 4 benefit cards, 3 steps, callout, enterprise tiles ×4, ent-cta mailto link; developer: both tables, tree block, all h2/h3 headings, commands `<pre>`, both back-links, API-contract link).
3. URL parity: `dist/` yields `index.html` at root and `developer/index.html`; asset paths resolve under `/litemind-ui/assets/`.
4. Visual check in browser against https://debabratamishra.github.io/litemind-ui/ at desktop + 375px mobile, light and dark themes: nav, hero orbs/gradient text, cards, steps numbering, tables, tree colors, footer.
5. Theme toggle round-trips and persists across reload; no flash of wrong theme.
6. Visual QA table in "Professional visual-element QA" fully passes, including `prefers-reduced-motion` collapsing the orb float and parallax to static.
6. Workflow dry-run logic reviewed locally (`npm run build` mirrors what the action runs).

## Risks

- Markdown renderer differences (kramdown vs remark) could subtly alter raw-HTML blocks in `developer.md` — mitigated by structural parity check #2.
- Emoji/icon fonts render identically since system font stack is unchanged.
- GitHub Actions cache/Node setup differences handled by official `withastro/action`.
