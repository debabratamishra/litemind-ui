---
layout: default
title: Developer guide — LiteMindUI
permalink: /developer/
description: >-
  The technical backbone of LiteMindUI: architecture, processes and ports,
  directory layout, design patterns, LLM backends, configuration, and how to run it.
---

<section class="block">
  <div class="container prose" style="margin:0 auto;">
    <p><a href="{{ site.baseurl }}/">← Back to home</a></p>
    <h2>Developer guide</h2>
    <p>
      LiteMindUI is a <strong>local-first AI workspace</strong> supporting chat,
      retrieval-augmented generation (RAG), web search, and realtime voice. It is
      composed of a FastAPI backend and a Next.js frontend that talk to each other
      exclusively over HTTP — they share no code imports.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Process</th><th>Entry point</th><th>Default port</th></tr>
        </thead>
        <tbody>
          <tr><td>FastAPI backend</td><td><code>main.py</code></td><td>8000</td></tr>
          <tr><td>Next.js frontend (primary)</td><td><code>nextjs-frontend/</code></td><td>3000</td></tr>
        </tbody>
      </table>
    </div>

    <div class="callout">
      <strong>Quick start.</strong> Install Python deps with <code>uv sync --group all</code>,
      start the backend with <code>uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload</code>,
      then <code>cd nextjs-frontend &amp;&amp; npm install &amp;&amp; npm run dev</code>.
      Prefer containers? <code>make up</code> brings up the whole stack.
    </div>

    <h2>Directory layout</h2>
    <p>The backend lives in <code>app/</code>; the UI in <code>nextjs-frontend/src/</code>.</p>
    <div class="tree">
<span class="c">app/</span>
  <span class="k">backend/</span>
    api/          <span class="c"># routes: chat.py, rag.py, models.py, health.py, voice.py (WebRTC SDP)</span>
    core/         <span class="c"># backend config, embedding helpers, DEFAULT_RAG_CONFIG</span>
    models/       <span class="c"># Pydantic request/response models</span>
  <span class="k">core/</span>           <span class="c"># shared utils: env detection, RAG formats, text markup</span>
  <span class="k">services/</span>       <span class="c"># business logic: llm_gateway, rag_service, voice_pipeline, …</span>
  <span class="k">ingestion/</span>     <span class="c"># file_ingest, document processors, OCR extractors</span>
  <span class="k">skills/</span>         <span class="c"># pluggable chat &amp; RAG skill routing</span>
<span class="k">nextjs-frontend/src/</span>
  app/           <span class="c"># App Router pages &amp; layouts</span>
  components/    <span class="c"># shadcn/ui components</span>
  hooks/         <span class="c"># custom React hooks</span>
  lib/           <span class="c"># API clients &amp; utilities</span>
<span class="c">main.py</span>            <span class="c"># FastAPI entry (lifespan, route registration)</span>
<span class="c">config.py</span>          <span class="c"># global Config (env vars, paths, tuning)</span>
    </div>

    <h2>Key design patterns</h2>

    <h3>LiteLLM Gateway <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>app/services/llm_gateway.py</code>)</span></h3>
    <p>
      A unified transport for <code>ollama</code>, <code>openrouter</code>, and
      <code>nvidia_nim</code>. For Ollama it calls the native <code>ollama</code>
      Python client directly (bypassing LiteLLM streaming) to avoid a known upstream
      bug. <code>resolve_backend_config()</code> normalises provider names, API bases,
      and keys from request params or environment variables.
    </p>

    <h3>Pluggable Skill Layer <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>app/skills/</code>)</span></h3>
    <p>
      Chat and RAG requests route through <code>ChatSkillRegistry</code> /
      <code>RAGSkillRegistry</code>. Each skill implements <code>supports()</code>,
      <code>validate()</code>, and <code>stream()</code> — add new capabilities
      without touching the API routes.
    </p>

    <h3>RAG system <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>app/services/rag_service.py</code>)</span></h3>
    <p>
      ChromaDB vector store plus BM25 keyword retrieval for <strong>hybrid search</strong>.
      Configurable embedding providers (sentence-transformers, Ollama, OpenRouter,
      Nvidia NIM). Pipeline: format detection → extraction → chunking → embedding → indexing.
    </p>

    <h3>Conversation memory <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>app/services/conversation_memory.py</code>)</span></h3>
    <p>
      Session-based multi-turn context. Older messages are auto-summarised once token
      usage passes 75% of the 24&nbsp;K context limit. Persisted in SQLite
      (<code>conversation_db.py</code>).
    </p>

    <h3>Generative UI <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>app/backend/api/chat.py</code>)</span></h3>
    <p>
      When <code>enable_generative_ui</code> is set, the LLM emits
      <code>```ui:component_name</code> fenced blocks. The Next.js frontend renders
      these as charts, tables, metrics, progress bars, and iframe apps.
    </p>

    <h3>Realtime voice mode <span style="font-weight:400;font-size:.9rem;color:var(--muted)">(<code>voice.py</code>, <code>voice_pipeline.py</code>)</span></h3>
    <p>
      Browser and server establish a WebRTC peer connection; the browser POSTs an SDP
      offer to <code>POST /api/voice/offer</code>. A Pipecat pipeline runs in the
      background — Whisper STT + Kokoro TTS + LLM. Transcripts and control events flow
      back over the WebRTC data channel. <strong>Voice is a separate pipeline, not a
      Skill</strong> — do not route it through the skill layer.
    </p>

    <h2>LLM provider backends</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Backend</th><th>Key env var</th><th>Default model</th></tr>
        </thead>
        <tbody>
          <tr><td>Ollama (local)</td><td><code>OLLAMA_API_URL</code></td><td><code>gemma3:1b</code></td></tr>
          <tr><td>OpenRouter</td><td><code>OPENROUTER_API_KEY</code></td><td><code>meta-llama/llama-3.3-70b-instruct</code></td></tr>
          <tr><td>Nvidia NIM</td><td><code>NVIDIA_NIM_API_KEY</code></td><td><code>meta/llama3-70b-instruct</code></td></tr>
        </tbody>
      </table>
    </div>

    <h2>Configuration &amp; environment</h2>
    <p>
      Copy <code>.env.example</code> → <code>.env</code> and fill in secrets. Critical
      variables:
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Variable</th><th>Purpose</th></tr>
        </thead>
        <tbody>
          <tr><td><code>OLLAMA_API_URL</code></td><td>Ollama server URL (default <code>http://localhost:11434</code>)</td></tr>
          <tr><td><code>OPENROUTER_API_KEY</code></td><td>OpenRouter API key</td></tr>
          <tr><td><code>NVIDIA_NIM_API_KEY</code></td><td>Nvidia NIM API key</td></tr>
          <tr><td><code>SERP_API_KEY</code></td><td>SerpAPI key for web search</td></tr>
          <tr><td><code>SECRET_KEY</code></td><td>App secret (change in production)</td></tr>
          <tr><td><code>CHROMA_DB_PATH</code></td><td>ChromaDB storage path</td></tr>
          <tr><td><code>UPLOAD_FOLDER</code></td><td>Document upload directory</td></tr>
          <tr><td><code>LOG_LEVEL</code></td><td>Logging verbosity (<code>INFO</code> / <code>DEBUG</code>)</td></tr>
        </tbody>
      </table>
    </div>

    <h2>Document ingestion</h2>
    <p>
      Supported formats: PDF (PyMuPDF + pdfplumber + Camelot tables), DOCX, PPTX,
      XLSX, EPUB, RTF, ODF, HTML, CSV, images (EasyOCR fallback), and plain text.
    </p>

    <h2>Useful commands</h2>
    <pre><code># Python / backend
uv sync --group all              # install all dependency groups
uv run uvicorn main:app --reload # start backend on :8000
uv run pytest                    # run tests
uv run ruff check .              # lint
uv run ty check app              # type-check

# Next.js frontend
cd nextjs-frontend
npm install
npm run dev                      # dev server on :3000
npm run build                    # production build
npm run lint                     # eslint

# Docker (primary workflow)
make up                          # build &amp; run the stack
make dev / make prod             # dev (hot-reload) / production
make logs / make health          # tail logs / health check</code></pre>

    <h2>Quality &amp; CI</h2>
    <p>
      Pull requests run <code>pr-checks.yml</code> (Python compile, ruff, ty); Docker
      images build via <code>docker-publish.yml</code>; releases bump
      <code>version.json</code> and tag via <code>release.yml</code>. PRs are labelled
      <code>patch</code> (default), <code>minor</code>, or <code>major</code>.
    </p>

    <div class="callout">
      <strong>Want the contract?</strong> Frontend developers should read the
      <a href="https://github.com/debabratamishra/litemind-ui/blob/main/docs/api-contract.md" target="_blank" rel="noopener">HTTP API contract</a>
      in the repository for request/response shapes.
    </div>

    <p style="margin-top:28px;"><a href="{{ site.baseurl }}/">← Back to home</a></p>
  </div>
</section>
