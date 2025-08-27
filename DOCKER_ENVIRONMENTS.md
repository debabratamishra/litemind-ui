# Docker Environment Configurations

This document describes the development and production Docker configurations for LLMWebUI, including logging, security settings, and environment-specific optimizations.

## Overview

The application supports two distinct Docker environments:

- **Development**: Optimized for debugging, development, and testing
- **Production**: Optimized for performance, security, and stability

## Quick Start

### Development Environment

```bash
# Start development environment
./scripts/docker-env-setup.sh up

# Or manually with docker-compose
docker-compose --env-file .env.development -f docker-compose.dev.yml up -d
```

### Production Environment

```bash
# Start production environment
./scripts/docker-env-setup.sh -e production up

# Or manually with docker-compose
docker-compose --env-file .env.production -f docker-compose.prod.yml up -d
```

## Environment Configurations

### Development Environment (`docker-compose.dev.yml`)

**Features:**
- Source code mounting for live reload
- Verbose logging (DEBUG level)
- Lower resource limits for development machines
- Frequent health checks for faster feedback
- Development-specific environment variables

**Optimizations:**
- Reduced thread counts for easier debugging
- Extended log retention for troubleshooting
- Hot reload enabled for both backend and frontend
- Access logs enabled for request debugging

**Security:**
- Relaxed security settings for development convenience
- Source code mounted as volumes for live editing

### Production Environment (`docker-compose.prod.yml`)

**Features:**
- Optimized for performance and stability
- Structured JSON logging
- Resource limits and reservations
- Enhanced security settings
- Production-grade health checks

**Optimizations:**
- Higher thread counts for better performance
- Multiple worker processes for backend
- Optimized memory and CPU limits
- Log rotation and structured logging

**Security:**
- Read-only mounts where possible
- Security options (`no-new-privileges`)
- Temporary filesystem with restrictions
- Environment variable validation

## Logging Configuration

### Development Logging
- **Level**: DEBUG
- **Format**: Detailed with timestamps and function names
- **Output**: Console + rotating files
- **Files**:
  - `/app/logs/debug.log` - All debug information
  - `/app/logs/error.log` - Error-specific logs

### Production Logging
- **Level**: INFO
- **Format**: Structured JSON for log aggregation
- **Output**: Console + rotating files
- **Files**:
  - `/app/logs/application.log` - Application logs
  - `/app/logs/error.log` - Error logs
  - `/app/logs/access.log` - Access logs (separate)

### Log Rotation
- **Development**: 50MB files, 5 backups
- **Production**: 100MB files, 10 backups
- **Error logs**: 10-50MB files, 3-5 backups

## Environment Variables

### Development (`.env.development`)
```bash
ENVIRONMENT=development
DEBUG=1
LOG_LEVEL=DEBUG
PROD_WORKERS=1
BACKEND_MEMORY_LIMIT=2G
FRONTEND_MEMORY_LIMIT=512M
```

### Production (`.env.production`)
```bash
ENVIRONMENT=production
DEBUG=0
LOG_LEVEL=INFO
PROD_WORKERS=4
BACKEND_MEMORY_LIMIT=4G
FRONTEND_MEMORY_LIMIT=1G
SECRET_KEY=your-production-secret-key
```

## Resource Management

### Development Resources
- **Backend**: 2GB RAM, 1 CPU core
- **Frontend**: 512MB RAM, 0.5 CPU core
- **Workers**: 1 (for easier debugging)
- **Threads**: 2 per type (OMP, MKL, NUMEXPR)

### Production Resources
- **Backend**: 4GB RAM limit, 2 CPU cores
- **Frontend**: 1GB RAM limit, 1 CPU core
- **Workers**: 4 (configurable via `PROD_WORKERS`)
- **Threads**: 8 per type (configurable)

## Security Settings

### Development Security
- Relaxed settings for development convenience
- Source code mounting enabled
- Debug information exposed

### Production Security
- `no-new-privileges` security option
- Read-only mounts where possible
- Restricted temporary filesystem
- Environment variable validation
- Hash seed randomization

## Health Checks

### Development Health Checks
- **Interval**: 15 seconds (faster feedback)
- **Timeout**: 5 seconds
- **Retries**: 3
- **Start Period**: 5-10 seconds

### Production Health Checks
- **Interval**: 30 seconds (stability focused)
- **Timeout**: 15 seconds
- **Retries**: 5
- **Start Period**: 45-60 seconds

## Volume Mounts

### Common Volumes
```yaml
# Cache directories (OS-independent)
- ${HOST_HF_CACHE}:${CONTAINER_HF_CACHE}
- ${HOST_OLLAMA_CACHE}:${CONTAINER_OLLAMA_CACHE}

# Application data
- ./uploads:/app/uploads
- ./chroma_db:/app/chroma_db
- ./storage:/app/storage
- ./logs:/app/logs
```

### Development-Specific Volumes
```yaml
# Source code for live reload
- .:/app
```

### Production-Specific Volumes
```yaml
# Read-only configuration
- ./.streamlit:/app/.streamlit:ro
```

## Management Scripts

### Docker Environment Setup Script

The `scripts/docker-env-setup.sh` script provides unified management:

```bash
# Available commands
./scripts/docker-env-setup.sh [OPTIONS] COMMAND

# Commands
up          # Start the application
down        # Stop the application
restart     # Restart the application
logs        # Show application logs
build       # Build Docker images
clean       # Clean up containers and images
status      # Show container status

# Options
-e, --env ENV       # Environment (development|production)
-f, --force         # Force rebuild of images
-p, --pull          # Pull latest base images
-h, --help          # Show help
```

### Examples

```bash
# Development workflow
./scripts/docker-env-setup.sh up                    # Start dev environment
./scripts/docker-env-setup.sh logs                  # View logs
./scripts/docker-env-setup.sh restart               # Restart services

# Production deployment
./scripts/docker-env-setup.sh -e production up      # Start production
./scripts/docker-env-setup.sh -e production status  # Check status
./scripts/docker-env-setup.sh -e production logs    # View production logs

# Maintenance
./scripts/docker-env-setup.sh -f build              # Force rebuild
./scripts/docker-env-setup.sh clean                 # Clean up resources
```

## Monitoring and Debugging

### Development Debugging
- Source code changes trigger automatic reload
- Detailed logs available in console and files
- Debug endpoints enabled
- Verbose error messages

### Production Monitoring
- Structured JSON logs for log aggregation
- Health check endpoints for monitoring
- Resource usage limits for stability
- Error tracking and alerting ready

### Log Analysis

```bash
# View real-time logs
docker-compose -f docker-compose.dev.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs backend

# Access log files directly
tail -f logs/application.log
tail -f logs/error.log
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Fix directory permissions
   chmod 755 logs uploads chroma_db storage .streamlit
   ```

2. **Memory Issues**
   ```bash
   # Adjust memory limits in environment files
   BACKEND_MEMORY_LIMIT=2G  # Reduce if needed
   ```

3. **Port Conflicts**
   ```bash
   # Check if ports are in use
   lsof -i :8000  # Backend port
   lsof -i :8501  # Frontend port
   ```

4. **Cache Directory Issues**
   ```bash
   # Verify cache directories exist and are accessible
   ls -la ~/.cache/huggingface
   ls -la ~/.ollama
   ```

### Log Debugging

```bash
# Check container logs
docker logs llmwebui-backend-dev
docker logs llmwebui-frontend-dev

# Check application logs
tail -f logs/debug.log      # Development
tail -f logs/application.log # Production
tail -f logs/error.log      # Errors
```

## Best Practices

### Development
- Use development environment for coding and testing
- Monitor debug logs for detailed information
- Utilize hot reload for faster development cycles
- Keep resource limits reasonable for development machines

### Production
- Always use production environment for deployment
- Monitor structured logs with log aggregation tools
- Set appropriate resource limits based on server capacity
- Implement proper backup strategies for persistent data
- Use environment variables for sensitive configuration
- Regularly update base images and dependencies

### Security
- Change default secret keys in production
- Use read-only mounts where possible
- Implement proper network security
- Monitor logs for security events
- Keep Docker images updated

## Integration with External Services

### Log Aggregation
The production configuration supports integration with external logging services:

```bash
# Example: Fluentd integration
FLUENTD_HOST=localhost
FLUENTD_PORT=24224

# Example: Syslog integration
SYSLOG_ADDRESS=localhost:514
```

### Monitoring
Health check endpoints are available for monitoring integration:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:8501/_stcore/health`

### Backup and Recovery
Persistent data locations for backup:
- Application uploads: `./uploads`
- Vector database: `./chroma_db`
- Configuration: `./storage`
- Logs: `./logs`