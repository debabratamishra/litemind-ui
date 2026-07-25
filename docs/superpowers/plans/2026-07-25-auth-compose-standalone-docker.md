# Shared Auth Compose File (Standalone + Docker) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the litemind-gotrue startup panic and make GoTrue + Postgres run identically in standalone mode (`make gotrue-up`) and docker mode (`make up`) from a single `docker-compose.auth.yml`.

**Architecture:** Extract the `db` and `gotrue` services out of `docker-compose.yml` into a new `docker-compose.auth.yml` (single source of truth). Docker mode layers both files with two `-f` flags; standalone mode runs only the auth file. Compose interpolates `${VAR}` from `.env`, eliminating the Makefile shell-quoting bug that produced the nil-pointer panic in GoTrue's migrate command.

**Tech Stack:** Docker Compose (v2 or docker-compose v1 via `COMPOSE_CMD`), `supabase/gotrue:v2.143.0`, `postgres:15-alpine`, GNU Make.

**Spec:** `docs/superpowers/specs/2026-07-25-auth-compose-standalone-docker-design.md`

## Global Constraints

- GoTrue port is **9999** everywhere (container `PORT`, published port, `API_EXTERNAL_URL`, docs). No `8081` references may remain.
- GoTrue image stays `supabase/gotrue:v2.143.0`; Postgres stays `postgres:15-alpine`.
- Use multiple `-f` flags for layering, NOT compose `include:` (portability with docker-compose v1).
- `gotrue-up` must fail fast with a clear error if `.env` is missing or `GOTRUE_JWT_SECRET` is empty or still the placeholder.
- Do not modify `version.json`. Do not commit `.env`.
- Container names keep the existing `litemindui-` prefix (`litemindui-gotrue`, `litemindui-db`).
- Data in the legacy standalone `litemind-postgres` container is intentionally NOT migrated (spec §6).
- Note: the repo has pre-existing uncommitted changes in `.env.example`, `Makefile`, `docker-compose.yml`, and `nextjs-frontend/src/app/(auth)/register/page.tsx`. Commit only the files each task touches; leave the register page change alone.

---

### Task 1: Create `docker-compose.auth.yml` + Postgres init script

**Files:**
- Create: `scripts/db/init-auth-schema.sql`
- Create: `docker-compose.auth.yml`

**Interfaces:**
- Produces: compose services named `db` and `gotrue`, network-addressable as `db:5432` and `gotrue:9999` by any service layered into the same compose project; named volume `pgdata`. Task 2 relies on these exact service names.

- [ ] **Step 1: Create the Postgres init script**

Create `scripts/db/init-auth-schema.sql`:

```sql
-- Runs automatically on first initialization of the pgdata volume
-- (via /docker-entrypoint-initdb.d). Creates the schema GoTrue migrates into.
CREATE SCHEMA IF NOT EXISTS auth;
ALTER ROLE postgres SET search_path TO auth, public;
```

- [ ] **Step 2: Create `docker-compose.auth.yml`**

```yaml
# Auth infrastructure: PostgreSQL + Supabase GoTrue.
# Single source of truth for BOTH deployment modes:
#   Docker mode     — layered with the main file:
#                     docker compose -f docker-compose.yml -f docker-compose.auth.yml up -d
#   Standalone mode — run alone (backend runs natively):
#                     docker compose -f docker-compose.auth.yml up -d   (or `make gotrue-up`)
# Compose reads ${VARS} from .env in this directory.

services:
  # PostgreSQL — stores GoTrue auth data plus LiteMindUI users/conversations.
  db:
    image: postgres:15-alpine
    container_name: litemindui-db
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
      - POSTGRES_DB=${POSTGRES_DB:-postgres}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/db/init-auth-schema.sql:/docker-entrypoint-initdb.d/10-init-auth-schema.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 12
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Supabase Auth (GoTrue) — self-hosted email/password authentication.
  # Backend (docker mode) reaches it at http://gotrue:9999; the host (standalone
  # mode) at http://localhost:9999.
  gotrue:
    image: supabase/gotrue:v2.143.0
    container_name: litemindui-gotrue
    environment:
      - PORT=9999
      - GOTRUE_DB_DRIVER=postgres
      - GOTRUE_DB_DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD:-postgres}@db:5432/postgres?options=-csearch_path=auth
      - GOTRUE_DB_SCHEMA=auth
      - API_EXTERNAL_URL=http://localhost:9999
      - GOTRUE_JWT_SECRET=${GOTRUE_JWT_SECRET:-}
      - GOTRUE_SITE_URL=http://localhost:3000
      - GOTRUE_URI_ALLOW_LIST=http://localhost:3000,http://localhost:8501
      - GOTRUE_DISABLE_SIGNUP=false
      - GOTRUE_SMTP_HOST=${SMTP_HOST:-}
      - GOTRUE_SMTP_PORT=${SMTP_PORT:-587}
      - GOTRUE_SMTP_USER=${SMTP_USER:-}
      - GOTRUE_SMTP_PASS=${SMTP_PASS:-}
      - GOTRUE_SMTP_ADMIN_EMAIL=${SMTP_ADMIN_EMAIL:-admin@litemind.local}
      # Auto-confirm signups so dev login works without a configured SMTP server.
      - GOTRUE_MAILER_AUTOCONFIRM=true
    ports:
      - "9999:9999"
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  pgdata:
```

- [ ] **Step 3: Validate compose file syntax and interpolation**

Run: `docker compose -f docker-compose.auth.yml config --quiet && echo VALID`
Expected: `VALID` (no output before it; warnings about unset SMTP vars are OK)

- [ ] **Step 4: Boot the auth stack from a clean slate and verify GoTrue is healthy**

The old `pgdata` volume (if any) predates the init script and lacks the `auth`
schema, and legacy standalone containers hold ports 5432/8081. Clean slate
(data loss accepted per spec §6):

```bash
docker rm -f litemind-gotrue litemind-postgres 2>/dev/null || true
docker network rm litemind-auth 2>/dev/null || true
docker compose -f docker-compose.auth.yml down -v 2>/dev/null || true
docker compose -f docker-compose.auth.yml up -d
```

Wait ~10s, then:

Run: `docker compose -f docker-compose.auth.yml ps --format '{{.Name}} {{.Status}}'`
Expected: `litemindui-db` shows `Up ... (healthy)`, `litemindui-gotrue` shows `Up` (NOT `Restarting`)

Run: `curl -s http://localhost:9999/health`
Expected: JSON like `{"version":"v2.143.0","name":"GoTrue","description":"..."}`

Run: `docker logs litemindui-gotrue 2>&1 | grep -i panic || echo NO_PANIC`
Expected: `NO_PANIC`

- [ ] **Step 5: Commit**

```bash
git add docker-compose.auth.yml scripts/db/init-auth-schema.sql
git commit -m "feat(auth): add docker-compose.auth.yml as single source of truth for GoTrue+Postgres"
```

---

### Task 2: Remove `db`/`gotrue` from `docker-compose.yml`

**Files:**
- Modify: `docker-compose.yml` (delete lines 129–183 region: the `gotrue` service, the `db` service, and the `volumes: pgdata:` block; keep the trailing `networks:` block)

**Interfaces:**
- Consumes: service names `db` and `gotrue` from Task 1's `docker-compose.auth.yml`.
- Produces: a `docker-compose.yml` that is only valid when layered with the auth file (backend's `depends_on` references `gotrue` and `db`).

- [ ] **Step 1: Delete the `gotrue` service, `db` service, and `pgdata` volume from `docker-compose.yml`**

Remove the entire block starting at the comment `# Supabase Auth (GoTrue) — self-hosted email/password authentication.` (line ~129) through the end of the `db` service definition (line ~180), and remove:

```yaml
volumes:
  pgdata:
```

Keep the final block exactly:

```yaml
networks:
  default:
    driver: bridge
```

In the backend service, keep `depends_on: [gotrue, db]` and all `GOTRUE_*`/`DATABASE_URL` environment lines unchanged. Add a comment above the backend's `depends_on` noting the services come from the auth file:

```yaml
    # gotrue and db are defined in docker-compose.auth.yml — always launch with:
    #   docker compose -f docker-compose.yml -f docker-compose.auth.yml ...
    depends_on:
      - gotrue
      - db
```

- [ ] **Step 2: Verify the layered configuration is valid and complete**

Run: `docker compose -f docker-compose.yml -f docker-compose.auth.yml config --services | sort`
Expected output (exactly these five):

```
backend
coturn
db
frontend
gotrue
```

Run: `docker compose -f docker-compose.yml -f docker-compose.auth.yml config | grep -c "8081" || echo ZERO`
Expected: `ZERO` (broken 8081 mapping is gone)

- [ ] **Step 3: Verify the main file alone now fails loudly (expected behavior)**

Run: `docker compose -f docker-compose.yml config --quiet; echo "exit=$?"`
Expected: an error mentioning undefined service `gotrue` (or `db`) in `depends_on`, `exit=` non-zero. This is intentional — the Makefile (Task 3) always passes both files.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "refactor(compose): move gotrue and db services to docker-compose.auth.yml"
```

---

### Task 3: Makefile — layered compose files, rewritten `gotrue-up`/`gotrue-down`

**Files:**
- Modify: `Makefile` (lines 5–6 area for the new variable; targets `build`, `up`, `down`, `logs`, `clean`, `status`; auth comment block lines ~137–145; targets `gotrue-up`/`gotrue-down` lines ~146–190)

**Interfaces:**
- Consumes: `docker-compose.auth.yml` from Task 1; layered validity from Task 2.
- Produces: `make up`, `make down`, `make logs`, `make clean`, `make status`, `make gotrue-up`, `make gotrue-down` — behavior documented below; README (Task 4) references these.

- [ ] **Step 1: Add the `COMPOSE_FILES` variable**

Directly below the existing `COMPOSE_CMD` definition (line 5), add:

```make
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.auth.yml
```

- [ ] **Step 2: Update default-stack targets to use both files**

Replace the compose invocations that use the default file (do NOT touch dev/prod/hub lines):

```make
build:
	@echo "🏗️  Building Docker images..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) build

up: setup
	@echo "🚀 Starting LiteMindUI services..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) up -d
	@echo "✅ Services started. Run 'make logs' to see output or 'make health' to check status."

down:
	@echo "🛑 Stopping services..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) down
	$(COMPOSE_CMD) -f docker-compose.dev.yml down 2>/dev/null || true
	$(COMPOSE_CMD) -f docker-compose.prod.yml down 2>/dev/null || true

logs:
	$(COMPOSE_CMD) $(COMPOSE_FILES) logs -f
```

In `clean`, change only the first compose line:

```make
	$(COMPOSE_CMD) $(COMPOSE_FILES) down -v --rmi all --remove-orphans 2>/dev/null || true
```

In `status`, change only the first compose line:

```make
	@$(COMPOSE_CMD) $(COMPOSE_FILES) ps 2>/dev/null || echo "No services running with default compose file"
```

- [ ] **Step 3: Rewrite the auth comment block and `gotrue-up`/`gotrue-down`**

Replace everything from `# ── Authentication (GoTrue) ─────...` (line ~137) to the end of `gotrue-down` with:

```make
# ── Authentication (GoTrue) ───────────────────────────────────────
# Auth infrastructure (Postgres + GoTrue) lives in docker-compose.auth.yml —
# the single source of truth for BOTH modes:
#   Docker mode     — `make up` layers docker-compose.yml + docker-compose.auth.yml.
#                     Backend uses AUTH_MODE=docker and reaches GoTrue at
#                     http://gotrue:9999 and Postgres at db:5432.
#   Standalone mode — backend runs natively (`uv run uvicorn ...`) with
#                     AUTH_MODE=standalone. `make gotrue-up` starts only the
#                     auth services; backend reaches GoTrue at
#                     http://localhost:9999 and Postgres at localhost:5432.
# Set GOTRUE_JWT_SECRET and POSTGRES_PASSWORD in .env (see .env.example).

# Start auth services (Postgres + GoTrue) for standalone/native backend mode.
gotrue-up:
	@test -f .env || { echo "❌ .env not found. Copy .env.example to .env and set GOTRUE_JWT_SECRET."; exit 1; }
	@secret=$$(grep -E '^GOTRUE_JWT_SECRET=' .env | head -1 | cut -d= -f2-); \
	if [ -z "$$secret" ] || [ "$$secret" = "change-me-to-a-long-random-string" ]; then \
		echo "❌ GOTRUE_JWT_SECRET is empty or still the placeholder in .env."; \
		echo "   Set it to a long random string, e.g.: openssl rand -hex 32"; \
		exit 1; \
	fi
	@echo "🔐 Starting auth services (Postgres + GoTrue)..."
	$(COMPOSE_CMD) -f docker-compose.auth.yml up -d
	@echo "✅ GoTrue running at http://localhost:9999 (Postgres at localhost:5432)."
	@echo "   Set AUTH_MODE=standalone in .env for the native backend."

# Stop auth services; also removes legacy pre-compose containers if present.
gotrue-down:
	@echo "🛑 Stopping auth services..."
	$(COMPOSE_CMD) -f docker-compose.auth.yml down
	@docker rm -f litemind-gotrue litemind-postgres 2>/dev/null || true
	@docker network rm litemind-auth 2>/dev/null || true
	@echo "✅ Auth services stopped."
```

- [ ] **Step 4: Verify the fail-fast guard**

```bash
mv .env /tmp/litemind.env.bak 2>/dev/null || true
make gotrue-up; echo "exit=$?"
```

Expected: `❌ .env not found...` and `exit=2` (non-zero).

```bash
printf 'GOTRUE_JWT_SECRET=\n' > .env
make gotrue-up; echo "exit=$?"
rm .env && mv /tmp/litemind.env.bak .env 2>/dev/null || true
```

Expected: `❌ GOTRUE_JWT_SECRET is empty or still the placeholder...` and non-zero exit. (If you had no real `.env`, create one from `.env.example` with a real secret before Step 5.)

- [ ] **Step 5: Verify `make gotrue-up` and `make gotrue-down` work end to end**

Run: `make gotrue-down && make gotrue-up`
Expected: guard passes, compose starts, `✅ GoTrue running at http://localhost:9999 ...`

Run: `curl -s http://localhost:9999/health`
Expected: GoTrue health JSON (`"name":"GoTrue"`)

Run: `make gotrue-down`
Expected: containers stopped/removed without errors (legacy cleanup lines are no-ops if nothing legacy exists).

- [ ] **Step 6: Commit**

```bash
git add Makefile
git commit -m "feat(make): layer auth compose file; fail-fast gotrue-up on missing JWT secret"
```

---

### Task 4: Align docs — README and `.env.example`

**Files:**
- Modify: `README.md:86` (auth setup bullet)
- Modify: `.env.example` (auth section comments, lines ~34–43)

**Interfaces:**
- Consumes: `make gotrue-up` / layered-compose behavior from Task 3.
- Produces: documentation only.

- [ ] **Step 1: Update README docker/standalone auth bullets**

Replace the bullet at `README.md:86` (starts with `- **Docker (recommended with \`make up\`):**`) with:

```markdown
- **Docker (recommended with `make up`):** auth infrastructure (`gotrue` + `db` PostgreSQL) lives in `docker-compose.auth.yml`, which `make up` layers automatically with `docker-compose.yml`. Set `GOTRUE_JWT_SECRET` (a long random string, e.g. `openssl rand -hex 32`) and `POSTGRES_PASSWORD` in `.env`, then `make up`. The backend reaches GoTrue at `http://gotrue:9999` and Postgres at `db:5432` inside the compose network.
- **Standalone (native backend):** run `make gotrue-up` to start only Postgres + GoTrue in Docker, and run the backend natively with `AUTH_MODE=standalone`. The backend reaches GoTrue at `http://localhost:9999` and Postgres at `localhost:5432`.
```

If an existing standalone bullet follows line 86, replace it with the standalone bullet above instead of duplicating it.

- [ ] **Step 2: Update `.env.example` auth comments**

In the auth section (~lines 34–43), update the `AUTH_MODE` comment to:

```bash
# "docker" (backend runs in compose; make up layers docker-compose.auth.yml) or
# "standalone" (backend runs natively; start auth services with `make gotrue-up`).
AUTH_MODE=standalone
```

Leave `GOTRUE_API_URL=http://localhost:9999` and the other values as they are.

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -rn "8081" Makefile docker-compose.yml docker-compose.auth.yml README.md .env.example || echo CLEAN`
Expected: `CLEAN`

Run: `grep -rn "litemind-postgres\|litemind-auth" Makefile | grep -v "docker rm -f\|docker network rm" || echo CLEAN`
Expected: `CLEAN` (legacy names appear only in `gotrue-down` cleanup)

- [ ] **Step 4: Commit**

```bash
git add README.md .env.example
git commit -m "docs: document layered auth compose file and standalone auth flow"
```

---

### Task 5: End-to-end verification (both modes)

**Files:**
- None created/modified; verification only.

**Interfaces:**
- Consumes: everything from Tasks 1–4.

- [ ] **Step 1: Standalone mode — auth stack + native backend**

```bash
make gotrue-up
curl -s http://localhost:9999/health
```

Expected: GoTrue health JSON, no panic in `docker logs litemindui-gotrue`.

Start the backend natively (needs `AUTH_MODE=standalone`, `GOTRUE_API_URL=http://localhost:9999`, `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres` in `.env`):

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 8
curl -s http://localhost:8000/health
```

Expected: backend health OK.

- [ ] **Step 2: Standalone mode — register + login round-trip**

```bash
curl -s -X POST http://localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"e2e-test@litemind.local","password":"Str0ngPass!234"}'
curl -s -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"e2e-test@litemind.local","password":"Str0ngPass!234"}'
```

Expected: register returns a user/session JSON (autoconfirm is on); login returns an access token. (Routes verified: `app/backend/api/auth.py` uses `APIRouter(prefix="/api/auth")` with `@router.post("/register")` and `@router.post("/login")`.) Then stop the native backend (`kill %1`).

- [ ] **Step 3: Docker mode — full stack**

```bash
make up
sleep 30
docker compose -f docker-compose.yml -f docker-compose.auth.yml ps --format '{{.Name}} {{.Status}}'
```

Expected: `litemindui-backend`, `litemindui-frontend`, `litemindui-coturn`, `litemindui-db (healthy)`, `litemindui-gotrue` all `Up`.

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:9999/health
docker logs litemindui-gotrue 2>&1 | grep -i panic || echo NO_PANIC
```

Expected: both health checks OK, `NO_PANIC`.

- [ ] **Step 4: Return to the user's working mode**

The user is currently developing in standalone mode:

```bash
make down
make gotrue-up
```

Expected: full stack down; only `litemindui-db` + `litemindui-gotrue` running.

- [ ] **Step 5: Final lint gate (per CLAUDE.md)**

Run: `uv run ruff check .`
Expected: no new errors (no Python files were changed).
