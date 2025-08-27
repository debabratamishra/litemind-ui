# Docker Health Checks and Startup Validation

This document describes the comprehensive health check and startup validation system implemented for the LLMWebUI Docker containers.

## Overview

The Docker integration includes three main components for container lifecycle management:

1. **Startup Validation** - Validates container environment and host service connectivity before starting the application
2. **Health Check Endpoints** - Provides multiple health check endpoints for different container states
3. **Graceful Shutdown** - Ensures proper cleanup of resources and host processes during container shutdown

## Components

### 1. Startup Validation Script (`scripts/startup-validation.py`)

Performs comprehensive validation before the application starts:

- **Environment Variables**: Validates required and optional environment variables
- **Volume Mounts**: Checks accessibility and permissions of mounted directories
- **Host Services**: Tests connectivity to Ollama and vLLM services
- **Critical Requirements**: Ensures all critical components are ready

#### Usage

```bash
# Run startup validation manually
python3 scripts/startup-validation.py

# The script is automatically run by the Docker entrypoint
# Control with environment variables:
VALIDATION_ENABLED=true    # Enable/disable validation
VALIDATION_REQUIRED=true   # Fail startup if validation fails
STARTUP_TIMEOUT=60         # Timeout for validation in seconds
```

#### Exit Codes

- `0`: Validation passed, container ready to start
- `1`: Validation failed, critical errors prevent startup

### 2. Health Check Endpoints

The FastAPI application provides multiple health check endpoints:

#### `/health` - Basic Health Check
Simple endpoint that returns if the service is running.

```json
{
  "status": "healthy",
  "service": "LLM WebUI API"
}
```

#### `/health/ready` - Readiness Check
Comprehensive check for container readiness to serve requests:

- RAG service initialization
- Critical directory accessibility
- Host service connectivity (non-blocking)

```json
{
  "status": "ready",
  "timestamp": 1703123456.789,
  "checks": {
    "rag_service": {"status": "ready"},
    "directories": {"uploads": {"status": "ready"}},
    "ollama": {"status": "ready", "response_time_ms": 45.2},
    "vllm": {"status": "unavailable", "optional": true}
  }
}
```

#### `/health/live` - Liveness Check
Basic check to determine if the container should be restarted:

- FastAPI responsiveness
- Filesystem accessibility
- Memory usage (if psutil available)

```json
{
  "status": "alive",
  "timestamp": 1703123456.789,
  "process_id": 1234,
  "checks": {
    "fastapi": {"status": "alive"},
    "filesystem": {"status": "alive"},
    "memory": {"status": "alive", "usage_percent": 45.2}
  }
}
```

#### `/health/startup` - Startup Check
Indicates when the container has completed startup:

- Service initialization status
- Configuration validation
- Directory creation status

```json
{
  "status": "started",
  "timestamp": 1703123456.789,
  "startup_time": 12.5,
  "checks": {
    "rag_service": {"status": "started"},
    "configuration": {"status": "started"},
    "directories": {"status": "started"}
  }
}
```

### 3. Health Check Script (`scripts/health-check.py`)

Comprehensive health checking tool that can be used standalone or by Docker:

#### Usage

```bash
# Different types of health checks
python3 scripts/health-check.py --type startup      # Startup validation
python3 scripts/health-check.py --type readiness    # Readiness check
python3 scripts/health-check.py --type liveness     # Liveness check
python3 scripts/health-check.py --type comprehensive # Full check

# Options
--timeout 10        # HTTP request timeout
--verbose          # Detailed output
--output results.json # Save results to file
--quiet            # Suppress output (for Docker)
```

#### Check Types

- **Startup**: Validates filesystem and process health
- **Readiness**: Checks service responsiveness and filesystem
- **Liveness**: Basic process and service health
- **Comprehensive**: All checks including host services

### 4. Graceful Shutdown Handler (`scripts/graceful-shutdown.py`)

Handles proper cleanup during container shutdown:

- **Host Process Cleanup**: Terminates vLLM and other spawned processes
- **Temporary File Cleanup**: Removes temporary files and caches
- **State Persistence**: Saves shutdown state for diagnostics
- **Log Flushing**: Ensures all logs are written

#### Usage

```bash
# Manual graceful shutdown
python3 scripts/graceful-shutdown.py --timeout 30

# Register signal handlers (for long-running processes)
python3 scripts/graceful-shutdown.py --register-signals

# Cleanup only (no signal handling)
python3 scripts/graceful-shutdown.py --cleanup-only
```

### 5. Docker Entrypoint (`scripts/docker-entrypoint.sh`)

Orchestrates container startup with validation and shutdown handling:

- Runs startup validation
- Starts the application with monitoring
- Handles graceful shutdown signals
- Provides logging and diagnostics

#### Environment Variables

```bash
VALIDATION_ENABLED=true          # Enable startup validation
VALIDATION_REQUIRED=true         # Require validation to pass
GRACEFUL_SHUTDOWN_ENABLED=true   # Enable graceful shutdown
STARTUP_TIMEOUT=60               # Startup validation timeout
LOG_LEVEL=INFO                   # Logging level
```

## Docker Integration

### Dockerfile Configuration

The Dockerfile is configured with:

```dockerfile
# Health check using custom script
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD /app/scripts/docker-healthcheck.sh

# Custom entrypoint for lifecycle management
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
```

### Docker Compose Integration

```yaml
services:
  backend:
    build: .
    environment:
      - VALIDATION_ENABLED=true
      - GRACEFUL_SHUTDOWN_ENABLED=true
      - STARTUP_TIMEOUT=60
    healthcheck:
      test: ["CMD", "/app/scripts/docker-healthcheck.sh"]
      interval: 30s
      timeout: 15s
      retries: 3
      start_period: 10s
```

### Kubernetes Integration

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: llmwebui
    image: llmwebui:latest
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 5
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
      failureThreshold: 12
```

## Monitoring and Diagnostics

### Health Check Results

All health check results are structured JSON that can be consumed by monitoring systems:

```bash
# Get detailed health status
curl http://localhost:8000/health/ready | jq .

# Monitor startup progress
curl http://localhost:8000/health/startup | jq .checks

# Check liveness
curl http://localhost:8000/health/live | jq .
```

### Log Analysis

The scripts provide structured logging for easy analysis:

```bash
# Container logs show health check activity
docker logs <container_id> | grep "Health Check"

# Startup validation logs
docker logs <container_id> | grep "Startup validation"

# Graceful shutdown logs
docker logs <container_id> | grep "Graceful shutdown"
```

### Troubleshooting

#### Common Issues

1. **Startup Validation Fails**
   ```bash
   # Check validation details
   docker exec <container> python3 /app/scripts/startup-validation.py
   
   # Check volume mounts
   docker exec <container> ls -la /app/uploads /app/chroma_db
   
   # Check host services
   curl http://localhost:11434/api/tags  # Ollama
   curl http://localhost:8001/v1/models  # vLLM
   ```

2. **Health Checks Failing**
   ```bash
   # Run comprehensive health check
   docker exec <container> python3 /app/scripts/health-check.py --verbose
   
   # Check specific endpoint
   curl -v http://localhost:8000/health/ready
   ```

3. **Graceful Shutdown Issues**
   ```bash
   # Check shutdown state
   docker exec <container> cat /tmp/shutdown_state.json
   
   # Manual cleanup
   docker exec <container> python3 /app/scripts/graceful-shutdown.py --cleanup-only
   ```

#### Debug Mode

Enable debug mode for detailed diagnostics:

```bash
docker run -e LOG_LEVEL=DEBUG -e VALIDATION_ENABLED=true llmwebui:latest
```

## Best Practices

1. **Always enable startup validation** in production environments
2. **Monitor health check endpoints** with your orchestration platform
3. **Set appropriate timeouts** based on your infrastructure
4. **Use graceful shutdown** to prevent data loss
5. **Monitor logs** for health check patterns and issues
6. **Test health checks** in your CI/CD pipeline

## Performance Impact

The health check system is designed to be lightweight:

- **Startup validation**: Runs once at container start
- **Health endpoints**: Minimal overhead, cached results where appropriate
- **Liveness checks**: Basic checks only, ~1-2ms response time
- **Readiness checks**: More comprehensive, ~10-50ms response time
- **Graceful shutdown**: Only runs during container termination

## Security Considerations

- Health check endpoints expose minimal system information
- No sensitive data is included in health check responses
- Graceful shutdown properly cleans up temporary files
- Volume mount validation ensures proper permissions