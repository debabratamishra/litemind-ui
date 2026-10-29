# Repository Architecture Refactor — Design

**Date:** 2026-09-23  
**Status:** Approved approach: Option 1 — compatibility-preserving cleanup  
**Scope:** Clean up the repository layout, preserve existing public entrypoints, and make the backend, frontend, site, Docker, CI, and docs structure more enterprise-grade and scalable.

## Problem

The repository currently mixes several concerns at the root and in legacy paths:

- Python backend code lives partly under `app/...` in the committed layout, while the current worktree already contains a partial move to `backend/...`.
- Dockerfiles and Compose files are split between root-level files and `infra/docker/`.
- The Astro marketing/site code sits at the repository root next to the backend, frontend, docs, scripts, and tests.
- Root compatibility shims (`main.py`, `config.py`, `logging_config.py`) still exist, but the docs and CI still describe the older `app/...` layout.
- The frontend is already a separate Next.js app, but the repo-wide structure does not yet clearly communicate the product boundaries.

This makes the repository harder to read, harder to onboard into, and harder to scale without accidentally coupling backend, frontend, site, and deployment concerns.

## Goals

1. Preserve existing public entrypoints and deployment names so current users, scripts, docs, and integrations keep working.
2. Make `backend/` the single source of truth for Python backend code.
3. Make `infra/docker/` the single source of truth for Dockerfiles and Compose workflows.
4. Keep the Astro site as a separate product surface, but move it out of the root clutter.
5. Keep the Next.js frontend as its own first-class runtime under `nextjs-frontend/`.
6. Update CI, docs, and Makefile references so the repository reads like a maintained open-source project.
7. Avoid unrelated product refactors in this pass.

## Non-goals

- Do not merge the Astro site into the Next.js app in this pass.
- Do not remove the root compatibility shims in this pass.
- Do not redesign API contracts, auth behavior, RAG behavior, or voice behavior.
- Do not add new dependencies.
- Do not rewrite the backend service layer beyond what is required to make the new layout coherent.
- Do not touch `version.json` directly.

## Target Repository Layout

```text
.github/
  workflows/
  CODEOWNERS

.kiro/                     # harness/project metadata; leave out of product layout
.pdm-build/                # generated cache; keep ignored

backend/
  main.py                  # backend package entrypoint
  config.py                # backend config
  logging_config.py        # backend logging
  app/
    backend/
      api/                 # FastAPI routers and route-level models
      core/                # backend configuration and core helpers
      models/              # Pydantic API models
      user_memory_store.py # persistent user memory store
    core/                  # shared backend utilities
    ingestion/             # document ingestion and extraction
    services/              # business services and integrations
    skills/                # pluggable chat/RAG skills

docs/
  README.md
  api-contract.md
  docker/
  superpowers/
  site/                    # Astro site source moved out of the root

infra/
  docker/
    Dockerfile
    Dockerfile.nextjs
    compose/
      docker-compose.yml
      docker-compose.auth.yml
      docker-compose.dev.yml
      docker-compose.prod.yml
      docker-compose.hub.yml

nextjs-frontend/
  package.json
  package-lock.json
  src/
  AGENTS.md
  CLAUDE.md

scripts/
  version.py
  health-check.py
  docker-setup.sh
  docker-entrypoint.sh
  docker-env-setup.sh
  generate-docker-env.py
  validate-cache-setup.py
  graceful-shutdown.py
  prepare-release.py
  start_app.sh
  db/
    init-auth-schema.sql

tests/
  backend/
  core/
  ingestion/
  services/
  skills/

main.py                    # root shim for backward compatibility
config.py                  # root shim for backward compatibility
logging_config.py          # root shim for backward compatibility

pyproject.toml
uv.lock
Makefile
README.md
DOCKER.md
LICENSE
.env.example
.dockerignore
.gitignore
.python-version
version.json
docker-compose.yml              # optional thin compatibility wrapper for local compose commands
```

## Design

### 1. Backend package becomes the canonical Python layout

`backend/` becomes the canonical Python backend root. The committed `app/...` layout is treated as legacy and should not be reintroduced.

The backend package keeps its current public module shape:

- `backend/main.py` exposes `app`.
- `backend/config.py` exposes `Config`.
- `backend/logging_config.py` exposes the existing logging helpers.
- `backend/app/backend/...` keeps the existing internal package boundaries.

The root-level `main.py`, `config.py`, and `logging_config.py` remain as thin compatibility shims:

```python
from backend.main import app

__all__ = ["app"]
```

```python
from backend.config import Config

__all__ = ["Config"]
```

```python
from backend.logging_config import (
    CleanFormatter,
    ContainerLogFilter,
    add_container_context,
    get_logger,
    get_logging_config,
    setup_health_check_logging,
    setup_logging,
)

__all__ = [
    "CleanFormatter",
    "ContainerLogFilter",
    "add_container_context",
    "get_logger",
    "get_logging_config",
    "setup_health_check_logging",
    "setup_logging",
]
```

This avoids breaking existing imports, scripts, docs, and local development commands while making the real source of truth unambiguous.

### 2. Docker and Compose live under `infra/docker/`

The root-level `Dockerfile`, `Dockerfile.nextjs`, and `docker-compose*.yml` files are removed from the canonical layout.

The canonical files move under:

```text
infra/docker/Dockerfile
infra/docker/Dockerfile.nextjs
infra/docker/compose/docker-compose.yml
infra/docker/compose/docker-compose.auth.yml
infra/docker/compose/docker-compose.dev.yml
infra/docker/compose/docker-compose.prod.yml
infra/docker/compose/docker-compose.hub.yml
```

The Makefile and CI reference these canonical paths.

If root-level `docker-compose.yml` is still needed for backward compatibility, it should be a thin wrapper that delegates to `infra/docker/compose/docker-compose.yml`, rather than carrying duplicated service definitions.

The recommendation is to keep that root Compose wrapper because it preserves existing local commands while removing duplicated service definitions.

The recommendation is to keep that root Compose wrapper because it preserves existing local commands while removing duplicated service definitions.

### 3. Astro site moves out of the root

The Astro site remains a separate product surface, but it moves from the repository root into `docs/site/`.

This keeps the marketing/site source visible and documented without making the root look like a flat mix of backend, frontend, docs, and site code.

The site package remains self-contained:

```text
docs/site/package.json
docs/site/package-lock.json
docs/site/astro.config.mjs
docs/site/tsconfig.json
docs/site/src/
docs/site/public/
```

The README and docs should link to the site from its new location.

### 4. Frontend remains a first-class runtime

The Next.js frontend remains under `nextjs-frontend/` with its own package manager, lint, build, and test commands.

No frontend product behavior changes in this pass. The refactor only clarifies the boundary and updates references that assume the old root layout.

### 5. CI and docs follow the new layout

CI should check the actual source paths:

- Python syntax/type/lint checks should target `backend/`, `main.py`, `config.py`, and `logging_config.py`.
- Frontend checks should continue to run inside `nextjs-frontend/`.
- Docker build paths should use `infra/docker/Dockerfile` and `infra/docker/Dockerfile.nextjs`.
- Docs should describe `backend/` as the backend source of truth and `infra/docker/` as the deployment source of truth.

The README repository layout section should be rewritten to match the target layout.

### 6. Root clutter policy

The root should contain only:

- repository metadata and policy files;
- dependency manifests;
- top-level runtime shims;
- Docker/Makefile entrypoints;
- environment examples;
- version metadata.

Generated or local-only directories should remain ignored and should not be treated as product structure.

## Migration Plan

1. Confirm the current `backend/` and `infra/docker/` files are the intended source of truth.
2. Move or create canonical Docker/Compose files under `infra/docker/`.
3. Move the Astro site from the root to `docs/site/`.
4. Update Makefile, CI, README, DOCKER.md, and docs references to the new paths.
5. Keep root Python shims intact.
6. Remove only clearly obsolete root Docker/Compose files after confirming no reference still needs them.
7. Run repository-wide checks after the layout change.

## Compatibility Risks

### Existing imports

Some scripts or local workflows may still import root `main.py`, `config.py`, or `logging_config.py`. The root shims prevent a breaking change.

### Docker commands

Existing users may run `make up`, `make dev`, `make prod`, or direct `docker compose -f docker-compose.yml`. If root Compose files are removed, the Makefile should continue to provide the same commands by delegating to `infra/docker/compose/`.

### Docs and CI

Any stale references to `app/...`, root Dockerfiles, or root Compose files should be updated in the same pass to avoid documenting a layout that no longer exists.

### Site links

Moving the Astro site changes local paths. README links and docs references should be updated together.

## Related Implementation Risks

The exploration also surfaced several issues that should be handled during implementation, even though they are not pure layout changes:

- `backend/main.py` duplicates several API handlers that already exist under `backend/app/backend/api/`; those should be mounted once and removed from the entrypoint.
- Startup currently clears the configured upload directory, which is a persistence/data-loss risk and should be replaced with safe mount-aware behavior.
- The 500 handler exposes internal exception text and should return a generic client-safe error.
- Some backend modules import optional dependencies at module import time, which can make the backend fail to start when optional features are unavailable.
- Docker/CI/docs still reference deleted root-level Compose files and Dockerfiles, so the refactor must update those references in the same pass.
- The frontend/backend boundary is already mostly correct, but docs and references should be updated to match the new layout.

## Verification Strategy

### Python

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check backend/app/backend backend/app/services backend/app/core backend/app/ingestion backend/app/skills backend/main.py backend/config.py backend/logging_config.py main.py config.py logging_config.py
uv run pytest -x -q
```

### Frontend

From `nextjs-frontend/`:

```bash
npm run lint
npm run build
```

### Docker/Compose

Validate that the Makefile still resolves the same high-level commands and that the canonical Compose files are referenced consistently.

### Docs/layout

Confirm:

- README layout matches the target layout.
- CI references the new backend and Docker paths.
- Makefile commands still work or clearly delegate to canonical files.
- The Astro site is linked from its new path.
- Root-level product clutter is reduced without deleting compatibility shims.

## Decisions

- Use `docs/site/` for the Astro site. This keeps the site separate from the root product layout while keeping it close to the repo's documentation surface.
- Keep a thin root `docker-compose.yml` wrapper only if needed for existing local commands. The canonical service definitions stay under `infra/docker/compose/`.
