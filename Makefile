# LLMWebUI Docker Makefile

.PHONY: help setup build up down logs clean health dev prod

# Default target
help:
	@echo "LLMWebUI Docker Commands:"
	@echo "  setup     - Run initial setup (create directories, .env file)"
	@echo "  build     - Build Docker images"
	@echo "  up        - Start services (default configuration)"
	@echo "  dev       - Start services in development mode"
	@echo "  prod      - Start services in production mode"
	@echo "  down      - Stop and remove containers"
	@echo "  logs      - Show container logs"
	@echo "  health    - Check service health"
	@echo "  clean     - Remove containers, images, and volumes"
	@echo "  restart   - Restart services"

# Setup directories and environment
setup:
	@echo "🔧 Running Docker setup..."
	@./scripts/docker-setup.sh

# Build Docker images
build:
	@echo "🏗️  Building Docker images..."
	docker-compose build

# Start services (default)
up: setup
	@echo "🚀 Starting LLMWebUI services..."
	docker-compose up -d
	@echo "✅ Services started. Run 'make logs' to see output or 'make health' to check status."

# Development mode
dev: setup
	@echo "🛠️  Starting LLMWebUI in development mode..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development services started."

# Production mode  
prod: setup
	@echo "🏭 Starting LLMWebUI in production mode..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production services started."

# Stop services
down:
	@echo "🛑 Stopping services..."
	docker-compose down
	docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
	docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

# Show logs
logs:
	docker-compose logs -f

# Health check
health:
	@python3 scripts/health-check.py

# Clean up everything
clean: down
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --rmi all --remove-orphans 2>/dev/null || true
	docker-compose -f docker-compose.dev.yml down -v --rmi all --remove-orphans 2>/dev/null || true
	docker-compose -f docker-compose.prod.yml down -v --rmi all --remove-orphans 2>/dev/null || true
	docker system prune -f
	@echo "✅ Cleanup complete."

# Restart services
restart: down up

# Quick development workflow
dev-restart:
	docker-compose -f docker-compose.dev.yml down
	docker-compose -f docker-compose.dev.yml up -d
	@echo "🔄 Development services restarted."

# View service status
status:
	@echo "📊 Service Status:"
	@docker-compose ps 2>/dev/null || echo "No services running with default compose file"
	@docker-compose -f docker-compose.dev.yml ps 2>/dev/null || echo "No development services running"
	@docker-compose -f docker-compose.prod.yml ps 2>/dev/null || echo "No production services running"