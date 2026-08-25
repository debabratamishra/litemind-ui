# LiteMindUI Pages Site: Jekyll → Astro Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Jekyll build in `site/` with an Astro + React-islands project that reproduces today's two pages byte-faithfully (URLs, DOM classes, copy, theming) and deploys through the existing `pages.yml` workflow.

**Architecture:** Astro 5 static output rooted at `site/` (base `/litemind-ui`). Content cards/steps/tiles live as typed data in `src/data/home.ts` rendered by shared `.astro` components. One React island (`ThemeToggle.tsx`) owns theme state; everything else is zero-JS static HTML. The developer guide remains a markdown page rendered through the base layout. Hero parallax stays vanilla JS in a scoped script, now gated behind `prefers-reduced-motion`.

**Tech Stack:** Astro 5, @astrojs/react, React 19, TypeScript, npm. GitHub Actions `withastro/action@v3`.

**Spec:** `docs/superpowers/specs/2026-08-25-astro-site-migration-design.md` (read it first — it defines preservation guarantees and the visual-QA table this plan verifies).

## Global Constraints

- Work happens on the current worktree branch (`worktree-stateless-skipping-treehouse`); never push, never touch `version.json`.
- All paths below are relative to the repo root unless prefixed `site/`.
- **Preservation rule:** class names, copy strings, emoji icons, link targets, and URLs are copied verbatim from the legacy files (`site/index.md`, `site/developer.md`, `site/_layouts/default.html`, `site/assets/css/style.css`). Do not "improve" wording, spacing, or structure while porting.
- Base URL is `/litemind-ui` (no trailing slash in config value; Astro appends one).
- Inside `.astro`/`.tsx` files, build URLs through `u()` from `src/lib/urls.ts`. Inside markdown files, write `/litemind-ui/...` literally (markdown cannot call JS).
- The only CSS additions allowed beyond the verbatim port: the `prefers-reduced-motion` block in Task 3. Nothing else changes `style.css`.
- Zero em-dashes in any rendered string (spec visual-QA table).
- Package manager is npm; commit `package-lock.json`. Never commit `node_modules/` or `dist/`.
- Every task ends with `cd site && npm run build` green before committing.

---

### Task 1: Astro scaffold replaces Jekyll tooling (pages untouched)

Legacy *pages* (`index.md`, `developer.md`) stay in place until their replacements land in Tasks 3–4. Only build tooling, layouts, and committed `_site/` output are removed now.

**Files:**
- Create: `site/package.json`, `site/astro.config.mjs`, `site/tsconfig.json`, `site/src/lib/urls.ts`
- Create: `site/public/favicon.svg`, `site/public/assets/favicon.svg`, `site/public/assets/demo.gif` (copies)
- Create: `site/src/styles/global.css` (verbatim copy of `site/assets/css/style.css`)
- Delete: `site/_config.yml`, `site/_layouts/default.html`, `site/assets/css/style.css`, `site/assets/demo.gif`, `site/assets/favicon.svg`, committed `site/_site/` directory
- Modify: `.gitignore` (root)

**Interfaces:**
- Produces: `src/lib/urls.ts` exports `u(path: string): string` — strips trailing slash off `import.meta.env.BASE_URL` and appends `path`. Every later task imports this.

- [ ] **Step 1: Init npm project and install dependencies**

```bash
cd site
rm -rf _site                      # stale committed Jekyll output (also git rm below)
npm init -y
npm pkg set type=module name="litemind-ui-site" private=true
npm pkg set scripts.dev="astro dev" scripts.build="astro build" \
  scripts.preview="astro preview" scripts.check="astro check" scripts.astro="astro"
npm install astro @astrojs/react react react-dom
npm install -D @astrojs/check typescript @types/react @types/react-dom
```

Expected: installs succeed; `package-lock.json` created. If `npm init -y` prints an existing-package warning, ignore it.

- [ ] **Step 2: Write `site/astro.config.mjs`**

```js
// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  site: 'https://debabratamishra.github.io',
  base: '/litemind-ui',
  output: 'static',
  integrations: [react()],
});
```

- [ ] **Step 3: Write `site/tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/base",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist", "node_modules"]
}
```

- [ ] **Step 4: Write `site/src/lib/urls.ts`**

```ts
/** Prefix a root-relative path with the configured site base (/litemind-ui). */
export function u(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path}`;
}
```

Note for executors: `import.meta.env` typing under `astro/tsconfigs/base` works because Astro injects env types via `.astro/types.d.ts`; if the editor complains before first build, run `npm run build` once to generate them.

- [ ] **Step 5: Port styles verbatim and copy assets**

```bash
mkdir -p src/styles public/assets
cp assets/css/style.css src/styles/global.css
cp assets/demo.gif public/assets/demo.gif
cp assets/favicon.svg public/assets/favicon.svg
cp assets/favicon.svg public/favicon.svg
```

Then edit `src/styles/global.css`: change only the 5-line header comment block to say `LiteMindUI — Astro site stylesheet` (keeps provenance clear). No other character changes.

- [ ] **Step 6: Remove Jekyll tooling from git and disk**

```bash
git rm -rq site/_site
git rm -q site/_config.yml site/_layouts/default.html \
  site/assets/css/style.css site/assets/demo.gif site/assets/favicon.svg
```

Do NOT `git rm` `site/index.md` or `site/developer.md`.

- [ ] **Step 7: Update root `.gitignore`**

Append (skip lines already present):

```
site/node_modules/
site/dist/
site/.astro/
```

- [ ] **Step 8: Placeholder page + first build**

Create `site/src/pages/index.astro` (temporary; replaced in Task 3):

```astro
---
---
<html lang="en"><head><meta charset="utf-8"><title>scaffold</title></head>
<body><h1>scaffold</h1></body></html>
```

Run: `cd site && npm run build && npm run check`
Expected: build writes `dist/index.html`; check reports no errors. Verify base handling: `grep -o '/litemind-ui/[^"]*' dist/index.html | head -1` returns nothing yet (placeholder has no base-relative URLs), and `ls dist` shows `index.html` at root.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(site): scaffold Astro project, port styles/assets, drop Jekyll tooling"
```

---

### Task 2: Base layout, Header, Footer, ThemeToggle island

**Files:**
- Create: `site/src/layouts/Base.astro`, `site/src/components/Header.astro`, `site/src/components/Footer.astro`, `site/src/components/ThemeToggle.tsx`
- Modify: `site/src/pages/index.astro` (use the layout so the shell is verifiable)

**Interfaces:**
- Produces: `Base.astro` accepts EITHER direct props (`title: string`, `description: string`) OR Astro-markdown mode (`frontmatter: { title: string; description: string }`) — markdown pages set `layout:` in frontmatter and Astro passes their frontmatter as a prop. Resolution line is in Step 1. Later tasks rely on exactly this contract.
- Produces: `ThemeToggle.tsx` default export, no props, hydrates via `client:load`.
- Consumes: `u()` from Task 1.

- [ ] **Step 1: Write `site/src/layouts/Base.astro`**

Port of `_layouts/default.html` (deleted in Task 1 — recover its exact text with `git show HEAD~1:site/_layouts/default.html` if needed; every attribute and string below is already transcribed):

```astro
---
import '../styles/global.css';
import Header from '../components/Header.astro';
import Footer from '../components/Footer.astro';

interface Props {
  title?: string;
  description?: string;
  frontmatter?: { title?: string; description?: string };
}
const fm = Astro.props.frontmatter ?? Astro.props;
const title = fm.title ?? 'LiteMindUI';
const description =
  fm.description ??
  'LiteMindUI is a private, local-first AI workspace for chat, document Q&A, web search, and realtime voice. Your data stays with you.';
const ogUrl = new URL(Astro.url.pathname, Astro.site);
---

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} | LiteMindUI</title>
    <meta name="description" content={description} />
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:type" content="website" />
    <meta property="og:url" content={ogUrl} />
    <link rel="icon" href={u('/assets/favicon.svg')} type="image/svg+xml" />
    <script is:inline>
      (function () {
        try {
          var t = localStorage.getItem('lm-theme');
          if (!t) {
            t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
          }
          document.documentElement.setAttribute('data-theme', t);
        } catch (e) {}
      })();
    </script>
  </head>
  <body>
    <Header />
    <main>
      <slot />
    </main>
    <Footer />
  </body>
</html>
```

Note: the global stylesheet is imported in frontmatter (Astro injects the built CSS link); the legacy `<link rel="stylesheet">` tag is intentionally gone.

- [ ] **Step 2: Write `site/src/components/Header.astro`**

```astro
---
import ThemeToggle from './ThemeToggle';
import { u } from '../lib/urls';

const pathname = Astro.url.pathname;
const onHome = pathname === u('/') || pathname === u('');
const onDev = pathname.startsWith(u('/developer'));
---

<header class="site-nav">
  <div class="container">
    <a class="brand" href={u('/')}>
      <span class="logo">L</span>
      <span>LiteMindUI</span>
    </a>
    <nav class="nav-links" aria-label="Primary">
      <a href={u('/')} class={onHome ? 'active' : undefined}>Home</a>
      <a href={u('/developer/')} class={onDev ? 'active' : undefined}>Developer</a>
      <a class="github" href="https://github.com/debabratamishra/litemind-ui" target="_blank" rel="noopener">GitHub ↗</a>
      <ThemeToggle client:load />
    </nav>
  </div>
</header>
```

- [ ] **Step 3: Write `site/src/components/ThemeToggle.tsx`**

```tsx
import { useEffect, useState } from 'react';

const KEY = 'lm-theme';

export default function ThemeToggle() {
  // Server render assumes light; the effect syncs to reality right after
  // hydration, so the pre-hydration glyph matches the FOUC script's flash-free
  // paint and aria/icon never mismatch for more than a frame.
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');
  }, []);

  function toggle() {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage unavailable (hardened browsers) — theme still applies for session */
    }
    setTheme(next);
  }

  return (
    <button
      id="theme-toggle"
      className="theme-toggle"
      type="button"
      aria-label="Switch between light and dark theme"
      aria-pressed={theme === 'dark'}
      title="Switch theme"
    >
      <span className="t-icon" aria-hidden="true">
        {theme === 'dark' ? '☀️' : '🌙'}
      </span>
    </button>
  );
}
```

- [ ] **Step 4: Write `site/src/components/Footer.astro`**

```astro
---
import { u } from '../lib/urls';
const year = new Date().getFullYear();
---

<footer class="site-footer">
  <div class="container">
    <div>
      <div class="brand"><span class="logo">L</span><span>LiteMindUI</span></div>
      <p style="max-width:320px;color:#94a3b8;font-size:.92rem;margin:12px 0 0;">
        A private, local-first AI workspace. Chat, ask your documents,
        search the web, and talk out loud, with your data staying with you.
      </p>
    </div>
    <div class="footer-cols">
      <div class="footer-col">
        <h4>Project</h4>
        <a href={u('/')}>Home</a>
        <a href={u('/developer/')}>Developer guide</a>
        <a href="https://github.com/debabratamishra/litemind-ui" target="_blank" rel="noopener">Source code</a>
      </div>
      <div class="footer-col">
        <h4>Resources</h4>
        <a href="https://github.com/debabratamishra/litemind-ui#readme" target="_blank" rel="noopener">README</a>
        <a href="https://github.com/debabratamishra/litemind-ui/blob/main/docs/api-contract.md" target="_blank" rel="noopener">API contract</a>
        <a href="https://github.com/debabratamishra/litemind-ui/blob/main/LICENSE" target="_blank" rel="noopener">License</a>
      </div>
    </div>
  </div>
  <div class="container footer-bottom">
    <span>© {year} LiteMindUI. Free and open source.</span>
    <span>Built with Astro &amp; GitHub Pages.</span>
  </div>
</footer>
```

(The one intentional copy change: "Built with Jekyll" → "Built with Astro". Everything else verbatim.)

- [ ] **Step 5: Wire the scaffold page through the layout**

Replace the entire content of `site/src/pages/index.astro` with:

```astro
---
import Base from '../layouts/Base.astro';
---
<Base title="LiteMindUI: your private AI workspace" description="Layout shell verification page.">
  <section class="block"><div class="container"><p>shell ok</p></div></section>
</Base>
```

- [ ] **Step 6: Build and verify the shell**

Run: `cd site && npm run build`
Then verify, expecting success on every line:

```bash
grep -c 'site-nav\|site-footer\|theme-toggle' dist/index.html   # ≥ 3
grep -o '<title>[^<]*</title>' dist/index.html                  # <title>LiteMindUI: your private AI workspace | LiteMindUI</title>
grep -c 'og:' dist/index.html                                   # ≥ 4
grep -o 'href="/litemind-ui/assets/favicon.svg"' dist/index.html
grep -o 'Built with Astro' dist/index.html
node -e "const h=require('fs').readFileSync('dist/index.html','utf8'); console.log(/data-theme=\"(light|dark)\"/.test(h)?'theme attr set':'MISSING')" # may print MISSING — the FOUC script runs client-side, absence in static HTML is fine; what must be present:
grep -c "localStorage.getItem('lm-theme')" dist/index.html      # 1 (the inline FOUC script survived bundling)
```

Functional check (toggle logic ships as a bundled island): `grep -rl "lm-theme" dist/_astro/ | head -1` finds the chunk.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(site): base layout, header/footer, React theme-toggle island"
```

---

### Task 3: Data model, section components, full home page (incl. parallax + reduced motion)

**Files:**
- Create: `site/src/data/home.ts`, `site/src/components/CardGrid.astro`, `site/src/components/Steps.astro`, `site/src/components/Callout.astro`, `site/src/components/SectionHead.astro`
- Modify: `site/src/pages/index.astro` (full replacement), `site/src/styles/global.css` (append reduced-motion block ONLY)
- Delete: `site/index.md` (legacy home page, after this task's page supersedes it)

**Interfaces:**
- Consumes: `Base.astro` (direct-props mode), `u()` from Task 1.
- Produces: `src/data/home.ts` exports:

```ts
export interface CardData { icon: string; title: string; body: string }
export interface StepData { title: string; body: string }
export declare const features: CardData[];   // length 4
export declare const benefits: CardData[];   // length 4
export declare const steps: StepData[];      // length 3
export declare const enterpriseTiles: CardData[]; // length 4
```

and components `CardGrid.astro {cards: CardData[]}`, `Steps.astro {steps: StepData[]}`, `Callout.astro` (slot), `SectionHead.astro {heading: string; lead?: string}`.

- [ ] **Step 1: Write `site/src/data/home.ts`**

All strings transcribed verbatim from legacy `site/index.md` (still on disk until Step 6 — cross-check against it):

```ts
export interface CardData {
  icon: string;
  title: string;
  body: string;
}

export interface StepData {
  title: string;
  body: string;
}

export const features: CardData[] = [
  {
    icon: '💬',
    title: 'Chat',
    body: 'Have a conversation. Ask questions, brainstorm ideas, or get help writing an email, a story, or a plan.',
  },
  {
    icon: '📄',
    title: 'Ask your documents',
    body: 'Drop in your PDFs, notes, or reports and ask questions about them. Ask "What did we decide in the meeting notes?" and it will tell you.',
  },
  {
    icon: '🌐',
    title: 'Search the web',
    body: 'Let it look things up online and bring back a clear, sourced answer instead of a wall of links.',
  },
  {
    icon: '🎙️',
    title: 'Talk out loud',
    body: 'Turn on voice mode and just speak. It listens, thinks, and replies with a natural voice, like a phone call with your assistant.',
  },
];

export const benefits: CardData[] = [
  {
    icon: '🔒',
    title: 'Private by design',
    body: 'Your files and conversations can stay entirely on your own computer. Nothing leaves your machine unless you choose to use a cloud service.',
  },
  {
    icon: '🖥️',
    title: 'Works offline',
    body: 'With a local AI model, LiteMindUI keeps working even without the internet. No connection? No problem.',
  },
  {
    icon: '🤝',
    title: 'Friendly for everyone',
    body: 'You do not need to be a programmer. If you can open a web page, you can use it.',
  },
  {
    icon: '🌱',
    title: 'Open source',
    body: 'The code is free and open for anyone to read, improve, and trust. No lock-in, no surprises.',
  },
];

export const steps: StepData[] = [
  { title: 'Get a copy', body: 'Download or clone the project from GitHub to your computer. It is free.' },
  { title: 'Run it', body: 'One command starts everything (Docker does the heavy lifting). No manual setup needed.' },
  { title: 'Open & chat', body: "Open the address it shows you in your browser, and start talking to your AI. That's it." },
];

export const enterpriseTiles: CardData[] = [
  {
    icon: '🛡️',
    title: 'Private, on-prem deployment',
    body: 'Run entirely inside your own network, with no data leaving your perimeter and no third-party APIs required.',
  },
  {
    icon: '🔐',
    title: 'SSO & access control',
    body: 'Connect your identity provider and decide exactly who can see what, down to workspace and document level.',
  },
  {
    icon: '📈',
    title: 'SLAs & priority support',
    body: 'Guaranteed response times and a direct line to me when something business-critical is on the line.',
  },
  {
    icon: '🧩',
    title: 'Custom integrations',
    body: 'Wire up your internal tools, private models, and data sources, plus features built to your spec.',
  },
];
```

- [ ] **Step 2: Write the four section components**

`site/src/components/CardGrid.astro`:

```astro
---
import type { CardData } from '../data/home';
interface Props { cards: CardData[] }
const { cards } = Astro.props;
---

<div class="cards">
  {cards.map((card) => (
    <div class="card">
      <div class="icon">{card.icon}</div>
      <h3>{card.title}</h3>
      <p>{card.body}</p>
    </div>
  ))}
</div>
```

`site/src/components/Steps.astro` (numbers come from the CSS counter on `.step .num`, which stays empty):

```astro
---
import type { StepData } from '../data/home';
interface Props { steps: StepData[] }
const { steps } = Astro.props;
---

<div class="steps">
  {steps.map((step) => (
    <div class="step">
      <div class="num"></div>
      <h3>{step.title}</h3>
      <p>{step.body}</p>
    </div>
  ))}
</div>
```

`site/src/components/Callout.astro`:

```astro
---
---

<div class="callout"><slot /></div>
```

`site/src/components/SectionHead.astro`:

```astro
---
interface Props { heading: string; lead?: string }
const { heading, lead } = Astro.props;
---

<div class="section-head">
  <h2>{heading}</h2>
  {lead && <p>{lead}</p>}
</div>
```

- [ ] **Step 3: Write the full `site/src/pages/index.astro`**

Section order and every literal string match legacy `site/index.md`. Inline `style=""` attributes are preserved deliberately (pixel fidelity beats purity).

```astro
---
import Base from '../layouts/Base.astro';
import SectionHead from '../components/SectionHead.astro';
import CardGrid from '../components/CardGrid.astro';
import Steps from '../components/Steps.astro';
import Callout from '../components/Callout.astro';
import { features, benefits, steps, enterpriseTiles } from '../data/home';
import { u } from '../lib/urls';
---

<Base
  title="LiteMindUI: your private AI workspace"
  description="LiteMindUI is a friendly, private AI workspace you run on your own computer: chat, ask questions about your files, search the web, and talk out loud."
>
  <section class="hero">
    <div class="container">
      <p class="eyebrow">Private · Local-first · Open source</p>
      <h1>Your own AI workspace<br />private, friendly, and easy</h1>
      <p class="lead">
        LiteMindUI lets you chat, ask questions about <em>your own files</em>,
        search the web, and even talk out loud with an AI that runs on your
        computer. Your data stays with you.
      </p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="https://github.com/debabratamishra/litemind-ui" target="_blank" rel="noopener">Get LiteMindUI</a>
        <a class="btn btn-ghost" href={u('/developer/')}>For Developers&nbsp;→</a>
      </div>
    </div>
  </section>

  <section class="block">
    <div class="container">
      <SectionHead
        heading="What is LiteMindUI?"
        lead="Think of it as a friendly helper that lives on your computer. You type or speak to it, and it answers, writes, and finds things for you, using either a smart AI model on your machine or one from the internet, your choice."
      />

      <CardGrid cards={features} />

      <figure class="demo">
        <img src={u('/assets/demo.gif')} alt="Short demo of LiteMindUI in action" />
        <figcaption>A quick look at LiteMindUI: chatting, asking documents, and more.</figcaption>
      </figure>
    </div>
  </section>

  <section class="block soft">
    <div class="container">
      <SectionHead
        heading="Why people choose LiteMindUI"
        lead="It is built to be calm, capable, and respectful of your privacy."
      />
      <CardGrid cards={benefits} />
    </div>
  </section>

  <section class="block">
    <div class="container">
      <SectionHead
        heading="How do I get started?"
        lead="Three simple steps. The details live in the project README, and here is the friendly version."
      />
      <Steps steps={steps} />

      <Callout>
        <strong>Good to know:</strong> You can start with a small AI model that
        runs privately on your computer, or connect a more powerful cloud model
        later. Either way, you stay in control of your data.
      </Callout>

      <div style="text-align:center;margin-top:28px;">
        <a class="btn btn-outline" href="https://github.com/debabratamishra/litemind-ui" target="_blank" rel="noopener">Read the setup guide on GitHub</a>
      </div>
    </div>
  </section>

  <section class="block soft">
    <div class="container prose" style="margin:0 auto;text-align:center;">
      <h2>Curious about how it works?</h2>
      <p style="max-width:640px;margin:0 auto;">
        If you like computers and want to see the engines under the hood,
        including the architecture, the APIs, and the design, we wrote a
        <a href={u('/developer/')}>Developer guide</a> just for you.
      </p>
    </div>
  </section>

  <section class="block enterprise">
    <div class="container prose" style="margin:0 auto;text-align:center;">
      <h2>Building something bigger?</h2>
      <p style="max-width:680px;margin:0 auto;">
        LiteMindUI is free and open source, and a great fit for personal use and small
        teams. If you are rolling it out across a company, or need it to meet stricter
        requirements, I offer <strong>enterprise support</strong> shaped around how your
        organisation actually works.
      </p>

      <div class="cards" style="margin-top:30px;text-align:left;">
        {enterpriseTiles.map((tile) => (
          <div class="card">
            <div class="icon">{tile.icon}</div>
            <h3>{tile.title}</h3>
            <p>{tile.body}</p>
          </div>
        ))}
      </div>

      <div class="ent-cta">
        <p>
          Have a use case in mind, or want to see what is possible for your team? Reach
          out and we can talk through how LiteMindUI can support your business.
        </p>
        <a class="btn btn-solid" href="mailto:debabrata.mishra641@gmail.com?subject=LiteMindUI%20enterprise%20support">Email me about enterprise support</a>
      </div>
    </div>
  </section>

  <script>
    // Subtle mouse parallax for the hero background. Vanilla on purpose: a raw
    // rAF loop over one element; React adds nothing. Skipped entirely when the
    // visitor prefers reduced motion.
    const hero = document.querySelector<HTMLElement>('.hero');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (hero && !reduceMotion) {
      let mouseX = 0;
      let mouseY = 0;
      let heroX = 0;
      let heroY = 0;

      const updateMousePosition = (e: MouseEvent) => {
        const rect = hero.getBoundingClientRect();
        mouseX = e.clientX - rect.left - rect.width / 2;
        mouseY = e.clientY - rect.top - rect.height / 2;
      };

      const animate = () => {
        heroX += (mouseX - heroX) * 0.05;
        heroY += (mouseY - heroY) * 0.05;
        hero.style.backgroundPosition = `${heroX * 0.1}px ${heroY * 0.1}px`;
        requestAnimationFrame(animate);
      };

      document.addEventListener('mousemove', updateMousePosition);
      requestAnimationFrame(animate);

      document.addEventListener(
        'touchmove',
        (e: TouchEvent) => {
          if (e.touches.length > 0) {
            const rect = hero.getBoundingClientRect();
            mouseX = e.touches[0].clientX - rect.left - rect.width / 2;
            mouseY = e.touches[0].clientY - rect.top - rect.height / 2;
          }
        },
        { passive: true },
      );
    }
  </script>
</Base>
```

- [ ] **Step 4: Append the reduced-motion CSS gate**

Add to the END of `site/src/styles/global.css` (this is the single sanctioned CSS change; see Global Constraints):

```css

/* ---------- Reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
  .hero::before {
    animation: none;
  }

  html {
    scroll-behavior: auto;
  }
}
```

- [ ] **Step 5: Remove the superseded legacy home page**

```bash
git rm -q site/index.md
```

- [ ] **Step 6: Build and run home-page parity checks**

Run: `cd site && npm run build && npm run check`
Both green. Then, from `site/`:

```bash
test -f dist/index.html && echo OK
echo "dev page not built yet (expected)"
test -f dist/assets/demo.gif && echo GIF-OK
grep -c 'class="card"' dist/index.html            # 12  (4 features + 4 benefits + 4 enterprise)
grep -c 'class="step"' dist/index.html            # 3
grep -c 'class="callout"' dist/index.html         # 1
grep -c 'section class="block' dist/index.html    # 5
grep -o 'mailto:debabrata.mishra641@gmail.com[^"]*' dist/index.html
grep -o 'Private · Local-first · Open source' dist/index.html
grep -o '/litemind-ui/assets/demo.gif' dist/index.html
grep -o 'Your own AI workspace' dist/index.html
grep -o 'prefers-reduced-motion' dist/_astro/*.js | head -2   # parallax guard bundled
grep -c 'animation: none' dist/_astro/*.css       # ≥ 1
grep -o '—' dist/index.html                       # expect NO MATCH (exit 1) — em-dash ban
```

Every assertion line must match its comment. On any mismatch, diff against `git show HEAD~2:site/index.md` and fix before proceeding.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(site): data-driven home page with hero, cards, steps, enterprise section"
```

---

### Task 4: Developer guide markdown port

The legacy `site/developer.md` stays on disk during this task as the porting source; it is removed in Step 4.

**Files:**
- Create: `site/src/pages/developer.md`
- Delete: `site/developer.md`

**Interfaces:**
- Consumes: `Base.astro` markdown mode (layout receives `frontmatter.{title,description}` — implemented in Task 2 Step 1).

- [ ] **Step 1: Create `site/src/pages/developer.md` with new frontmatter**

Astro markdown pages take their layout via frontmatter (path relative to the markdown file):

```markdown
---
layout: ../layouts/Base.astro
title: "Developer guide: LiteMindUI"
description: >-
  The technical backbone of LiteMindUI: architecture, processes and ports,
  directory layout, design patterns, LLM backends, configuration, and how to run it.
---
```

- [ ] **Step 2: Port the body verbatim with exactly three mechanical substitutions**

Copy the body of legacy `site/developer.md` (everything after its old frontmatter, i.e. from `<section class="block">` through the final `</section>`) unchanged EXCEPT:

1. `{{ site.baseurl }}` → `/litemind-ui` (occurs twice: the two `← Back to home` links become `href="/litemind-ui/"`)
2. Old frontmatter block (lines 1–8) is dropped, replaced by Step 1's frontmatter
3. Nothing else — kramdown and Astro/remark both render these raw-HTML tables, spans, and `<pre>` blocks natively; preserve indentation of the `.tree` block exactly (it is whitespace-sensitive inside `<div class="tree">`)

Sanity-check the port: `diff <(sed -n '10,$p' site/developer.md) <(tail -n +9 src/pages/developer.md)` should show only the two href substitutions. (Line offsets approximate; adjust the sed/tail ranges so both sides start at `<section class="block">`.)

- [ ] **Step 3: Build and run developer-page parity checks**

Run: `cd site && npm run build`
Expect `dist/developer/index.html` (directory-format URL). Then:

```bash
grep -c '<table>' dist/developer/index.html       # 3  (process/ports, providers, env vars)
grep -c '<h2>' dist/developer/index.html          # 11
grep -c '<h3>' dist/developer/index.html          # 7
grep -c 'class="tree"' dist/developer/index.html  # 1
grep -c 'llm_gateway.py' dist/developer/index.html # ≥ 1
grep -o 'docs/api-contract.md' dist/developer/index.html
grep -o 'href="/litemind-ui/"' dist/developer/index.html   # ≥ 2 (back-links)
grep -o 'uv run pytest' dist/developer/index.html
grep -o '<span class="k">' dist/developer/index.html       # ≥ 1 (tree syntax colors intact)
grep -c 'GoTrue' dist/developer/index.html        # ≥ 1 (auth section survived)
grep -c 'class="callout"' dist/developer/index.html # 2
```

If remark wrapped or reordered any raw HTML unexpectedly, compare rendering side-by-side with `git show HEAD:site/developer.md` and adjust only formatting whitespace, never content.

- [ ] **Step 4: Remove the legacy page and commit**

```bash
git rm -q site/developer.md
git add -A
git commit -m "feat(site): port developer guide to Astro markdown page"
```

---

### Task 5: Deploy workflow swap

**Files:**
- Modify: `.github/workflows/pages.yml` (build job only; triggers, permissions, concurrency, deploy job untouched)

**Interfaces:**
- Consumes: `site/package.json` scripts `build` (Task 1), lockfile committed.
- Produces: Pages artifact from `site/dist` instead of Jekyll's `./_site`.

- [ ] **Step 1: Replace the Jekyll build step**

In the `build` job, replace:

```yaml
      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Build with Jekyll
        uses: actions/jekyll-build-pages@v1
        with:
          source: site
          destination: ./_site

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./_site
```

with:

```yaml
      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Install, build, and upload site
        uses: withastro/action@v3
        with:
          path: site

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/dist
```

(`withastro/action@v3` reads `site/package-lock.json`, runs `npm ci && npm run build`, and sets up Node automatically.)

Also update the file-top comment: replace "Build with Jekyll" wording with "Build with Astro"; keep everything else.

- [ ] **Step 2: Local mirror of what CI runs**

```bash
cd site && npm ci && npm run build && npm run check
```

Expected: clean install from lockfile, build green, check green.

- [ ] **Step 3: Actionlint-style review + commit**

Read the final workflow top-to-bottom; confirm triggers (`paths: site/**` still correct — all site source lives under `site/`), permissions, concurrency group, and deploy job are untouched.

```bash
git add .github/workflows/pages.yml
git commit -m "ci(site): build Pages site with withastro/action instead of Jekyll"
```

---

### Task 6: Full parity sweep + visual QA (spec sign-off)

No new files. This task executes the spec's Verification section and Professional visual-element QA table end to end.

**Files:**
- Modify: none (fix-ups in earlier-task files if any check fails)

- [ ] **Step 1: Clean-room build**

```bash
cd site && rm -rf dist .astro && npm run build && npm run check
```

- [ ] **Step 2: URL + asset parity**

```bash
ls dist/index.html dist/developer/index.html dist/assets/demo.gif dist/assets/favicon.svg dist/favicon.svg
```

All five must exist. Confirm internal links in `dist/**/*.html` resolve to files within `dist/`:

```bash
grep -rhoP 'href="/litemind-ui/[^"#]*"' dist --include='*.html' | sort -u
```

Each result maps to `dist/<path>/index.html` or a static asset.

- [ ] **Step 3: Content parity sweep (both pages)**

Re-run every grep from Task 3 Step 6 and Task 4 Step 3 against the clean build; all counts hold.

- [ ] **Step 4: Serve and visually verify against the live site**

```bash
cd site && npm run preview   # serves dist at localhost:4321 under /litemind-ui
```

Using the agent-browser skill, capture screenshots of both pages and compare with https://debabratamishra.github.io/litemind-ui/ :
- Desktop 1280px AND mobile 375px, light AND dark themes (four captures per page)
- Walk the spec's visual-QA table: nav active state per page, Ethereal Glass hero orbs/gradient headline, card grid wrapping (4-up desktop, stacked mobile), numbered steps, teal callouts, table/tree styling on the developer page, footer columns, focus-visible outlines via keyboard Tab, theme-toggle round-trip persisting across reload, and `prefers-reduced-motion` emulation collapsing orb float + parallax
- Record any divergence; fix and rebuild until zero

- [ ] **Step 5: Final commit (if fix-ups occurred) and report**

```bash
git status --short                 # confirm nothing stray
git log --oneline main..HEAD       # summarize the migration commits
```

Report to the user: checks passed, divergences found/fixed, and that merge + push to `main` will trigger the new workflow (do NOT push yourself).

---

## Self-Review Notes

- Spec coverage: scaffold/cleanup (T1), layout+theming+islands (T2), data-driven home + parallax/reduced-motion (T3), markdown dev guide (T4), workflow swap (T5), verification incl. visual-QA table (T6). OG tags in T2; favicon dual-path in T1/T6. All spec sections mapped.
- Type consistency: `CardData`/`StepData` shapes identical across `home.ts`, `CardGrid`, `Steps`, `index.astro`; `Base.astro` dual-mode contract defined once (T2) and consumed in T3/T4 as specified.
- Known deliberate divergences (documented, not drift): "Built with Astro" footer string, reduced-motion CSS/script gates, OG tags added, favicon served additionally from `/favicon.svg`.
