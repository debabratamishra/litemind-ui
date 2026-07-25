# Shared Auth Compose File for Standalone + Docker Modes

**Date:** 2026-07-25
**Status:** Approved

## Problem

The `litemind-gotrue` container panics on startup in standalone mode
(`make gotrue-up`):

```
panic: runtime error: invalid memory address or nil pointer dereference
net/url.(*URL).Query(...)
github.com/supabase/auth/cmd.migrate(...)  migrate_cmd.go:55
```

### Root cause

In the Makefile `gotrue-up` target, the DB URL is wrapped in single quotes:

```make
-e GOTRUE_DB_DATABASE_URL='postgresql://postgres:$${POSTGRES_PASSWORD:-postgres}@litemind-postgres:5432/...'
```

Make expands `$$` to `$`, but the single quotes prevent the shell from
expanding `${POSTGRES_PASSWORD:-postgres}`. GoTrue receives the literal string
`${POSTGRES_PASSWORD:-postgres}` inside the URL. The `{`/`}` characters make
`url.Parse` fail; GoTrue v2.143.0's `migrate_cmd.go:55` ignores the parse
error and calls `u.Query()` on a nil URL, producing the SIGSEGV.

### Adjacent defects found during exploration

1. **Port inconsistency.** `.env.example` documents
   `GOTRUE_API_URL=http://localhost:9999`, but `gotrue-up` runs GoTrue with
   `PORT=8081`. `docker-compose.yml` has the reverse bug: GoTrue listens on
   9999 but the port mapping is `8081:8081`, which maps nothing.
2. **Silent empty JWT secret.** `gotrue-up` passes
   `GOTRUE_JWT_SECRET=$${GOTRUE_JWT_SECRET}`, but Make does not read `.env`,
   so the secret is empty unless exported in the invoking shell.
3. **Duplicated config.** GoTrue/Postgres settings exist twice (Makefile
   `docker run` block and `docker-compose.yml`) and have already drifted.

## Decisions

- **Port:** 9999 everywhere, both modes.
- **Secrets:** compose auto-loads `.env`; `gotrue-up` fails fast if
  `GOTRUE_JWT_SECRET` is missing or empty.
- **Structure (Approach C):** a dedicated `docker-compose.auth.yml` holding
  only `db` + `gotrue`, layered into docker mode via multiple `-f` flags.

## Design

### 1. New file: `docker-compose.auth.yml`

Single source of truth for auth infrastructure.

- `db` service: `postgres:15-alpine`
  - `pgdata` named volume; publishes `5432:5432`.
  - Healthcheck: `pg_isready -U postgres`.
  - Mounts `scripts/db/init-auth-schema.sql` into
    `/docker-entrypoint-initdb.d/` to create the `auth` schema on first boot
    of the volume (replaces the Makefile's `docker exec psql` calls).
- `gotrue` service: `supabase/gotrue:v2.143.0`
  - `PORT=9999`; publishes `9999:9999`.
  - `depends_on: db: condition: service_healthy` so migrations never race
    Postgres.
  - `GOTRUE_DB_DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD:-postgres}@db:5432/postgres?options=-csearch_path=auth`
    — expanded by compose from `.env`, eliminating the quoting bug.
  - Retains existing settings: `GOTRUE_DB_SCHEMA=auth`, site URL/allow list,
    `GOTRUE_MAILER_AUTOCONFIRM=true`, SMTP passthrough vars.
  - `API_EXTERNAL_URL=http://localhost:9999`.

### 2. `docker-compose.yml` changes

- Remove the `gotrue` and `db` service definitions (including the broken
  `8081:8081` mapping) and the `pgdata` volume (moves to the auth file).
- Docker mode runs with layered files:
  `-f docker-compose.yml -f docker-compose.auth.yml`.
- Backend keeps `depends_on: [gotrue, db]`, `GOTRUE_API_URL=http://gotrue:9999`,
  and `DATABASE_URL=...@db:5432/...` — included services join the same
  project and network.
- Two `-f` flags are used instead of compose `include:` for portability with
  older compose versions (the Makefile's `COMPOSE_CMD` supports both
  `docker-compose` and `docker compose`).

### 3. Makefile changes

- `gotrue-up`:
  1. Guard: error with a clear message if `.env` is missing or
     `GOTRUE_JWT_SECRET` is empty/`change-me-...` placeholder.
  2. `$(COMPOSE_CMD) -f docker-compose.auth.yml up -d`.
  3. Echo corrected to `http://localhost:9999`.
- `gotrue-down`: `$(COMPOSE_CMD) -f docker-compose.auth.yml down`, plus
  one-time cleanup of legacy `litemind-gotrue` / `litemind-postgres`
  containers and the `litemind-auth` network.
- `up` / `down` / `logs`: pass both `-f` flags so docker mode includes the
  auth services.

### 4. Config alignment

- All `8081` references removed; 9999 is the GoTrue port in both modes.
- `.env.example` (`GOTRUE_API_URL=http://localhost:9999`) becomes accurate
  as-is; comments updated where they describe the old flow.

### 5. Error handling

- GoTrue starts only after Postgres reports healthy and the `auth` schema
  exists, so the migrate step cannot hit a half-up database.
- `gotrue-up` fails fast on missing secrets instead of starting GoTrue with
  an empty JWT secret.

### 6. Data migration note

Existing data in the legacy standalone `litemind-postgres` container does
**not** carry over; the new `db` service uses the compose `pgdata` volume.
Accepted: auth was crashing, so there is no data worth preserving.

### 7. Testing

- **Standalone:** `make gotrue-up` → `curl http://localhost:9999/health` →
  run backend natively (`uv run uvicorn main:app ...`, `AUTH_MODE=standalone`)
  → register/login via the frontend.
- **Docker:** `make up` → backend reaches `http://gotrue:9999` → same
  register/login flow.
- Existing lint gates: `uv run ruff check .` (no Python changes expected) and
  YAML validity via `docker compose config` on both file combinations.

## Out of scope

- `docker-compose.dev.yml` / `docker-compose.prod.yml` / `docker-compose.hub.yml`
  integration of auth services (can layer the same auth file later).
- Email verification / SMTP configuration.
- Upgrading the GoTrue image version.
