# Repository Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository layout enterprise-grade and scalable while preserving existing public entrypoints, keeping the Astro site separate, and cleaning up the backend/Docker/CI/docs drift discovered during exploration.

**Architecture:** Keep `backend/` as the canonical Python backend root and `infra/docker/` as the canonical deployment root. Preserve thin root shims for compatibility, move the Astro site to `docs/site/`, and update Makefile, CI, docs, tests, and Docker references to match the new structure. The refactor also fixes the most urgent correctness risks found in the current partial state: duplicated route handlers in `backend/main.py`, unsafe upload cleanup, generic 500 errors, and optional dependency import failures.

**Tech Stack:** Python 3.13, FastAPI, uv, pytest, ruff, ty, Next.js 16, React 19, TypeScript, Astro, Docker, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-23-repository-architecture-refactor.md`

## Global Constraints

- Preserve existing public entrypoints and deployment names.
- Keep root `main.py`, `config.py`, and `logging_config.py` as compatibility shims.
- Keep the Astro site separate from the Next.js app.
- Use `backend/` as the canonical Python backend root.
- Use `infra/docker/` as the canonical Docker/Compose root.
- Do not add new dependencies.
- Do not touch `version.json` manually.
- Use `uv run` for Python commands.
- Run `npm run lint` and `npm run build` inside `nextjs-frontend/` for frontend checks.
- Do not discard existing uncommitted work in the current worktree.
- Do not remove root-level compatibility shims in this pass.

---

### Task 1: Align docs, Makefile, CI, and install scripts to the new layout

**Files:**
- Modify: `README.md`
- Modify: `DOCKER.md`
- Modify: `docs/README.md`
- Modify: `docs/docker/README.md`
- Modify: `Makefile`
- Modify: `.github/workflows/docker-publish.yml`
- Modify: `.github/workflows/pr-checks.yml`
- Modify: `install.sh`
- Modify: `.dockerignore`
- Modify: `AGENTS.md`
- Modify: `CONSTITUTION.md`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: current Makefile, CI workflows, install script, docs, and test fixtures.
- Produces: consistent references to `backend/`, `infra/docker/`, and `docs/site/`.

- [ ] **Step 1: Write the failing layout test**

```python
from pathlib import Path

def test_canonical_layout_references_exist():
    root = Path(".")
    assert (root / "backend" / "main.py").exists()
    assert (root / "infra" / "docker" / "Dockerfile").exists()
    assert (root / "infra" / "docker" / "compose" / "docker-compose.yml").exists()
    assert (root / "docs" / "site" / "package.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repository_layout.py -v`

Expected: FAIL because `docs/site/` and the canonical Docker paths do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Update the docs and scripts to use the canonical paths:

- `Makefile` should point to `infra/docker/compose/docker-compose.yml` and keep any root wrapper as a thin delegation layer.
- `docker-publish.yml` should build from `infra/docker/Dockerfile` and `infra/docker/Dockerfile.nextjs`.
- `install.sh` should reference `infra/docker/compose/docker-compose.hub.yml`.
- `.dockerignore` should stop ignoring the canonical Dockerfiles and Compose files.
- `README.md`, `DOCKER.md`, and `docs/docker/README.md` should describe the new layout.
- `AGENTS.md` and `CONSTITUTION.md` should reference `backend/...` instead of `app/...`.
- `tests/conftest.py` should import `backend.config` instead of `app.backend.config`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repository_layout.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md DOCKER.md docs/README.md docs/docker/README.md Makefile .github/workflows/docker-publish.yml .github/workflows/pr-checks.yml install.sh .dockerignore AGENTS.md CONSTITUTION.md tests/conftest.py tests/test_repository_layout.py
git commit -m "chore: align repo layout references"
```

---

### Task 2: Move the Astro site to `docs/site/`

**Files:**
- Create: `docs/site/package.json`
- Create: `docs/site/package-lock.json`
- Create: `docs/site/astro.config.mjs`
- Create: `docs/site/tsconfig.json`
- Create: `docs/site/src/`
- Create: `docs/site/public/`
- Delete: `site/package.json`
- Delete: `site/package-lock.json`
- Delete: `site/astro.config.mjs`
- Delete: `site/tsconfig.json`
- Delete: `site/src/`
- Delete: `site/public/`

**Interfaces:**
- Consumes: current `site/` Astro app.
- Produces: same Astro site under `docs/site/` with unchanged behavior.

- [ ] **Step 1: Write the failing site-location test**

```python
from pathlib import Path

def test_astro_site_is_under_docs_site():
    assert (Path("docs/site/package.json")).exists()
    assert not Path("site").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repository_layout.py::test_astro_site_is_under_docs_site -v`

Expected: FAIL because `docs/site/` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Move the Astro site contents from `site/` to `docs/site/` without changing package scripts or source behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repository_layout.py::test_astro_site_is_under_docs_site -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/site site
git commit -m "chore: move Astro site under docs"
```

---

### Task 3: Consolidate backend routes into mounted routers

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/app/backend/api/rag.py`
- Modify: `backend/app/backend/api/health.py`
- Modify: `backend/app/backend/api/models.py`
- Modify: `backend/app/backend/api/speech_service.py`
- Modify: `backend/app/backend/api/tts_service.py`
- Modify: `backend/app/backend/api/voice.py`
- Modify: `tests/backend/api/test_chat.py`
- Modify: `tests/backend/api/test_health.py`
- Modify: `tests/backend/api/test_rag.py`
- Modify: `tests/backend/api/test_voice.py`
- Modify: `tests/backend/api/test_voice_api_legacy.py`
- Modify: `tests/backend/api/test_voice_pipeline_legacy.py`
- Modify: `tests/backend/api/test_models.py`

**Interfaces:**
- Consumes: existing route handlers in `backend/main.py`.
- Produces: a single mounted router layer with no duplicated inline handlers.

- [ ] **Step 1: Write the failing route-ownership test**

```python
from backend.main import app

def test_health_is_mounted_from_router():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/health/ready" in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/api/test_health.py -v`

Expected: FAIL if the route is still only inline in `backend/main.py` or mounted incorrectly.

- [ ] **Step 3: Write minimal implementation**

Move the inline handlers from `backend/main.py` into the existing router modules:

- `health_check` and `readiness_check` → `backend/app/backend/api/health.py`
- RAG endpoints → `backend/app/backend/api/rag.py`
- models endpoints → `backend/app/backend/api/models.py`
- STT/TTS endpoints → `backend/app/backend/api/speech_service.py` and `backend/app/backend/api/tts_service.py`
- voice endpoints remain in `backend/app/backend/api/voice.py`

Then mount each router once in `backend/main.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/api/test_health.py tests/backend/api/test_rag.py tests/backend/api/test_voice.py tests/backend/api/test_voice_api_legacy.py tests/backend/api/test_voice_pipeline_legacy.py tests/backend/api/test_models.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/app/backend/api tests/backend/api
git commit -m "refactor: consolidate backend route ownership"
```

---

### Task 4: Fix unsafe upload cleanup and generic 500 responses

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/app/backend/api/rag.py`
- Modify: `tests/backend/api/test_rag.py`
- Modify: `tests/backend/api/test_health.py`

**Interfaces:**
- Consumes: current upload cleanup logic and 500 handlers.
- Produces: safe upload preservation and client-safe error responses.

- [ ] **Step 1: Write the failing safety tests**

```python
from pathlib import Path

def test_upload_folder_is_not_deleted_on_startup(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    marker = upload_dir / "keep-me.txt"
    marker.write_text("keep me")

    from backend.main import lifespan

    async def run_lifespan():
        async with lifespan(None):
            pass

    import asyncio
    asyncio.run(run_lifespan())

    assert marker.exists()
```

```python
def test_internal_errors_are_generic():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    @app.get("/__test__/boom")
    async def boom():
        raise RuntimeError("secret detail")

    response = client.get("/__test__/boom")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "secret detail" not in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/api/test_rag.py tests/backend/api/test_health.py -v`

Expected: FAIL because startup cleanup deletes uploads and the 500 handler leaks details.

- [ ] **Step 3: Write minimal implementation**

- Replace unconditional upload directory deletion with mount-aware safe behavior.
- Keep uploaded files intact across restarts unless the user explicitly asks for cleanup.
- Replace the 500 handler with a generic response that logs the exception server-side.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/api/test_rag.py tests/backend/api/test_health.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/app/backend/api/rag.py tests/backend/api/test_rag.py tests/backend/api/test_health.py
git commit -m "fix: protect uploads and hide internal errors"
```

---

### Task 5: Make optional backend imports lazy and fix import-time config issues

**Files:**
- Modify: `backend/app/backend/api/voice.py`
- Modify: `backend/app/backend/core/config.py`
- Modify: `backend/app/backend/api/chat.py`
- Modify: `tests/services/test_llm_gateway.py`
- Modify: `tests/backend/api/test_voice.py`
- Modify: `tests/backend/api/test_voice_api_legacy.py`
- Modify: `tests/backend/api/test_voice_pipeline_legacy.py`

**Interfaces:**
- Consumes: current import-time optional dependency loading and import-time config instantiation.
- Produces: backend startup that does not fail when optional voice dependencies are absent.

- [ ] **Step 1: Write the failing import test**

```python
def test_backend_imports_without_optional_voice_deps(monkeypatch):
    monkeypatch.delenv("PIPECAT_AVAILABLE", raising=False)

    import backend.main

    assert backend.main.app is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/api/test_voice.py tests/backend/api/test_voice_api_legacy.py tests/backend/api/test_voice_pipeline_legacy.py -v`

Expected: FAIL if optional voice imports break startup.

- [ ] **Step 3: Write minimal implementation**

- Move Pipecat imports inside the functions that need them.
- Move config object construction out of module import time.
- Keep the public API surface unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/api/test_voice.py tests/backend/api/test_voice_api_legacy.py tests/backend/api/test_voice_pipeline_legacy.py tests/services/test_llm_gateway.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/backend/api/voice.py backend/app/backend/core/config.py backend/app/backend/api/chat.py tests/backend/api/test_voice.py tests/backend/api/test_voice_api_legacy.py tests/backend/api/test_voice_pipeline_legacy.py tests/services/test_llm_gateway.py
git commit -m "fix: make optional backend imports lazy"
```

---

### Task 6: Clean Docker, CI, and stale root references

**Files:**
- Modify: `.github/workflows/docker-publish.yml`
- Modify: `.github/workflows/pr-checks.yml`
- Modify: `Makefile`
- Modify: `.dockerignore`
- Modify: `install.sh`
- Modify: `docs/docker/README.md`
- Modify: `DOCKER.md`
- Modify: `README.md`
- Modify: `scripts/startup-validation.py`
- Modify: `scripts/health-check.py`
- Modify: `scripts/graceful-shutdown.py`
- Modify: `scripts/start_app.sh`
- Modify: `scripts/prepare-release.py`
- Modify: `scripts/docker-env-setup.sh`
- Modify: `scripts/generate-docker-env.py`
- Modify: `scripts/validate-cache-setup.py`

**Interfaces:**
- Consumes: current stale Docker/CI/docs references.
- Produces: a repo where all deployment and validation commands point at canonical paths.

- [ ] **Step 1: Write the failing reference test**

```python
from pathlib import Path

def test_no_stale_root_docker_references():
    text = Path(".github/workflows/docker-publish.yml").read_text()
    assert "infra/docker/Dockerfile" in text
    assert "./Dockerfile" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_repository_layout.py::test_no_stale_root_docker_references -v`

Expected: FAIL because CI still points at the deleted root Dockerfile.

- [ ] **Step 3: Write minimal implementation**

- Update CI to build from `infra/docker/Dockerfile` and `infra/docker/Dockerfile.nextjs`.
- Update Makefile to use `infra/docker/compose/`.
- Update `.dockerignore` so canonical Dockerfiles are not ignored.
- Update install/docs/scripts to use the new paths.
- Remove stale `localhost:8501` references if still present.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_repository_layout.py::test_no_stale_root_docker_references -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/docker-publish.yml .github/workflows/pr-checks.yml Makefile .dockerignore install.sh docs/docker/README.md DOCKER.md README.md scripts/startup-validation.py scripts/health-check.py scripts/graceful-shutdown.py scripts/start_app.sh scripts/prepare-release.py scripts/docker-env-setup.sh scripts/generate-docker-env.py scripts/validate-cache-setup.py tests/test_repository_layout.py
git commit -m "chore: clean docker and ci references"
```

---

### Task 7: Final verification and cleanup

**Files:**
- Modify: `tests/test_repository_layout.py`
- Modify: any remaining stale references found during verification.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified repository layout and documentation.

- [ ] **Step 1: Write the final verification checklist**

```python
def test_repository_layout_is_consistent():
    assert Path("backend/main.py").exists()
    assert Path("infra/docker/Dockerfile").exists()
    assert Path("infra/docker/compose/docker-compose.yml").exists()
    assert Path("docs/site/package.json").exists()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_repository_layout.py -v`

Expected: PASS.

- [ ] **Step 3: Run repository-wide checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check backend/app/backend backend/app/services backend/app/core backend/app/ingestion backend/app/skills backend/main.py backend/config.py backend/logging_config.py main.py config.py logging_config.py
uv run pytest -x -q
cd nextjs-frontend
npm run lint
npm run build
```

- [ ] **Step 4: Commit final cleanup**

```bash
git add .
git commit -m "chore: verify repository architecture refactor"
```

---

## Verification Summary

- Python: `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check ...`, `uv run pytest -x -q`
- Frontend: `npm run lint`, `npm run build` inside `nextjs-frontend/`
- Docker/Compose: validate Makefile and CI references against `infra/docker/`
- Docs/layout: confirm README, DOCKER.md, docs, and install scripts all match the new layout
- Compatibility: confirm root shims and existing local commands still work

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-23-repository-architecture-refactor.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?