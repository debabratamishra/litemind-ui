# LiteMindUI Docker Makefile

.PHONY: help setup build up down logs clean health dev prod hub-up hub-down version tag-release test-docker-local create-docker-repos gotrue-up gotrue-down

COMPOSE_CMD ?= $(shell if command -v docker-compose >/dev/null 2>&1; then echo docker-compose; elif docker compose version >/dev/null 2>&1; then echo "docker compose"; fi)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.auth.yml

# Default target
help:
	@echo "LiteMindUI Docker Commands:"
	@echo "  setup     - Run initial setup (create directories, .env file)"
	@echo "  build     - Build Docker images"
	@echo "  up        - Start services (default configuration)"
	@echo "  dev       - Start services in development mode"
	@echo "  prod      - Start services in production mode"
	@echo "  hub-up    - Start services using Docker Hub images"
	@echo "  hub-down  - Stop Docker Hub services"
	@echo "  down      - Stop and remove containers"
	@echo "  logs      - Show container logs"
	@echo "  health    - Check service health"
	@echo "  clean     - Remove containers, images, and volumes"
	@echo "  restart   - Restart services"
	@echo "  version   - Show current version and version commands"
	@echo "  tag-release - Create a new release tag"
	@echo "  test-docker-local - Test Docker build and push locally"
	@echo "  create-docker-repos - Create required Docker Hub repositories"
	@echo "  gotrue-up - Start standalone GoTrue auth container (non-Docker/native mode)"
	@echo "  gotrue-down - Stop and remove the standalone GoTrue container"

# Setup directories and environment
setup:
	@echo "🔧 Running Docker setup..."
	@./scripts/docker-setup.sh

# Build Docker images
build:
	@echo "🏗️  Building Docker images..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) build

# Start services (default)
up: setup
	@echo "🚀 Starting LiteMindUI services..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) up -d
	@echo "✅ Services started. Run 'make logs' to see output or 'make health' to check status."

# Development mode
dev: setup
	@echo "🛠️  Starting LiteMindUI in development mode..."
	$(COMPOSE_CMD) -f docker-compose.dev.yml up -d
	@echo "✅ Development services started."

# Production mode  
prod: setup
	@echo "🏭 Starting LiteMindUI in production mode..."
	$(COMPOSE_CMD) -f docker-compose.prod.yml up -d
	@echo "✅ Production services started."

# Stop services
down:
	@echo "🛑 Stopping services..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) down
	$(COMPOSE_CMD) -f docker-compose.dev.yml down 2>/dev/null || true
	$(COMPOSE_CMD) -f docker-compose.prod.yml down 2>/dev/null || true

# Show logs
logs:
	$(COMPOSE_CMD) $(COMPOSE_FILES) logs -f

# Health check
health:
	@python3 scripts/health-check.py

# Clean up everything
clean: down
	@echo "🧹 Cleaning up Docker resources..."
	$(COMPOSE_CMD) $(COMPOSE_FILES) down -v --rmi all --remove-orphans 2>/dev/null || true
	$(COMPOSE_CMD) -f docker-compose.dev.yml down -v --rmi all --remove-orphans 2>/dev/null || true
	$(COMPOSE_CMD) -f docker-compose.prod.yml down -v --rmi all --remove-orphans 2>/dev/null || true
	docker system prune -f
	@echo "✅ Cleanup complete."

# Restart services
restart: down up

# Quick development workflow
dev-restart:
	$(COMPOSE_CMD) -f docker-compose.dev.yml down
	$(COMPOSE_CMD) -f docker-compose.dev.yml up -d
	@echo "🔄 Development services restarted."

# Docker Hub deployment
hub-up:
	@echo "🐳 Starting LiteMindUI using Docker Hub images..."
	@./scripts/docker-setup.sh
	$(COMPOSE_CMD) -f docker-compose.hub.yml pull
	$(COMPOSE_CMD) -f docker-compose.hub.yml up -d
	@echo "✅ Docker Hub services started."

hub-down:
	@echo "🛑 Stopping Docker Hub services..."
	$(COMPOSE_CMD) -f docker-compose.hub.yml down

# Version management
version:
	@echo "📋 Version Management:"
	@echo "Current version: $$(python3 scripts/version.py current 2>/dev/null || echo 'version.json not found')"
	@echo ""
	@echo "Commands:"
	@echo "  python3 scripts/version.py current     - Show current version"
	@echo "  python3 scripts/version.py bump patch  - Bump patch version (0.0.1 -> 0.0.2)"
	@echo "  python3 scripts/version.py bump minor  - Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  python3 scripts/version.py bump major  - Bump major version (1.0.0 -> 2.0.0)"
	@echo "  python3 scripts/version.py tag         - Create git tag for current version"
	@echo "  make tag-release                       - Interactive release creation"

tag-release:
	@echo "🏷️  Creating new release..."
	@python3 scripts/version.py bump patch

# View service status
status:
	@echo "📊 Service Status:"
	@$(COMPOSE_CMD) $(COMPOSE_FILES) ps 2>/dev/null || echo "No services running with default compose file"
	@$(COMPOSE_CMD) -f docker-compose.dev.yml ps 2>/dev/null || echo "No development services running"
	@$(COMPOSE_CMD) -f docker-compose.prod.yml ps 2>/dev/null || echo "No production services running"

# Test Docker build and push locally
test-docker-local:
	@echo "🧪 Testing Docker build and push locally..."
	@python3 scripts/test-docker-publish.py

# Create Docker Hub repositories
create-docker-repos:
	@echo "🐳 Creating Docker Hub repositories..."
	@python3 scripts/create-docker-repos.py

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