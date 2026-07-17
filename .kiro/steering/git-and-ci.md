---
inclusion: always
---

# Git Workflow & CI/CD

## Branch strategy
- `main` — production-ready; protected; releases are cut from here
- `develop` — integration branch; PRs target this or `main` directly
- Feature branches: `feat/<short-description>`, `fix/<short-description>`, `chore/<short-description>`
- **Never push directly to `main` or `develop`** — always open a PR

## Commit messages — Conventional Commits
```
type(scope): short description under 72 chars

Optional longer body explaining WHY (not what).
```
Types: `feat` `fix` `refactor` `docs` `test` `chore` `ci` `perf`  
Scope examples: `rag`, `llm`, `frontend`, `docker`, `deps`, `auth`

## PR labels → version bump
| Label | Effect on merge to `main` |
|-------|--------------------------|
| `patch` (default) | 0.0.x bump |
| `minor` | 0.x.0 bump |
| `major` | x.0.0 bump |

The release workflow (`release.yml`) bumps `version.json`, creates a git tag, and publishes a GitHub release automatically on merge.

## CI gates — all must pass before merge
| Check | Command | Workflow |
|-------|---------|----------|
| Python syntax | `python -m compileall -q app main.py` | `pr-checks.yml` |
| Ruff lint | `uv run ruff check .` | `pr-checks.yml` |
| ty type-check | `uv run ty check <backend paths>` | `pr-checks.yml` |

After every Python change run locally:
```bash
uv run ruff check . && uv run ty check app/backend app/services app/core app/ingestion app/skills main.py config.py logging_config.py
```

After every TypeScript change run locally:
```bash
cd nextjs-frontend && npm run lint && npm run build
```

## What the Docker publish workflow does
On push to `main` or a semver tag, `docker-publish.yml` builds and pushes:
- `litemindui/backend:<tag>` — from `Dockerfile`
- `litemindui/frontend:<tag>` — from `Dockerfile.nextjs`

Do not modify workflow files unless the task is explicitly about CI/CD.

## Hard git rules
- Never force-push (`--force` or `--force-with-lease`) without explicit user approval
- Never skip hooks (`--no-verify`) without explicit user approval
- Never commit `.env` (git-ignored — contains real secrets)
- Stage specific files — avoid `git add .` to prevent accidental secret commits
- Amend only your own unpushed commits; prefer new commits otherwise
