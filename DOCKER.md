# Docker Deployment Guide

This guide covers how to run LLMWebUI using Docker containers while maintaining integration with host system services like Ollama and vLLM.

## Quick Start

1. **Setup the environment:**
   ```bash
   make setup
   # or manually: ./scripts/docker-setup.sh
   ```

2. **Start the application:**
   ```bash
   make up
   # or: docker-compose up -d
   ```

3. **Access the application:**
   - Frontend (Streamlit): http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Prerequisites

### Required Host Services

#### Ollama Service (Required)
- **Port:** `localhost:11434`
- **Purpose:** Primary LLM inference engine
- **Setup:**
  ```bash
  # Install Ollama (if not already installed)
  curl -fsSL https://ollama.ai/install.sh | sh
  
  # Start Ollama service
  ollama serve
  
  # Verify it's running
  curl http://localhost:11434/api/tags
  
  # Pull a model for testing
  ollama pull llama3.1
  ```

#### vLLM Service (Optional)
- **Port:** `localhost:8001` 
- **Purpose:** Alternative high-performance inference engine
- **Requirements:**
  - Conda environment named `llm_ui`
  - GPU support (recommended)
- **Management:** Controlled by the containerized application

#### Host Service Dependencies

**Network Requirements:**
- Containers use **host networking mode** for direct service access
- No port forwarding needed
- Host services must bind to `localhost` or `0.0.0.0`

**Service Validation:**
```bash
# Test Ollama connectivity
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1", "prompt": "Hello", "stream": false}'

# Test vLLM connectivity (if running)
curl http://localhost:8001/health

# Check from container perspective
docker exec llmwebui-backend curl http://localhost:11434/api/tags
```

**Startup Order:**
1. Start Ollama service first
2. Verify Ollama is accessible
3. Start Docker containers
4. vLLM can be started on-demand through the UI

### System Requirements

- Docker and Docker Compose
- At least 4GB RAM (8GB+ recommended)
- Sufficient disk space for model caches

## Configuration Files

### Docker Compose Variants

- `docker-compose.yml` - Default configuration
- `docker-compose.dev.yml` - Development with live reload
- `docker-compose.prod.yml` - Production optimized

### Environment Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key environment variables:
- `OLLAMA_API_URL` - Ollama service URL (default: http://localhost:11434)
- `VLLM_API_URL` - vLLM service URL (default: http://localhost:8001)
- `OMP_NUM_THREADS` - CPU thread count for ML operations

## Volume Mounts

The application uses several volume mounts for persistence and host integration:

### Cache Directories (OS-Independent)
- `${HOME}/.cache/huggingface` → `/root/.cache/huggingface` - HuggingFace model cache
- `${HOME}/.ollama` → `/root/.ollama` - Ollama model cache

### Application Data
- `./uploads` → `/app/uploads` - Uploaded documents
- `./chroma_db` → `/app/chroma_db` - Vector database
- `./storage` → `/app/storage` - Application storage
- `./.streamlit` → `/app/.streamlit` - Streamlit configuration

### Volume Mount Requirements

**Critical Requirements:**
1. **Host cache directories must exist** before starting containers
2. **Proper permissions** are required for read/write access
3. **Sufficient disk space** for model downloads (10GB+ recommended)
4. **OS-independent paths** are automatically resolved by setup scripts

**Setup Process:**
```bash
# Automatic setup (recommended)
make setup

# Manual setup if needed
python scripts/cache-setup.py
python scripts/generate-docker-env.py
```

📖 **For detailed cache management information, see [CACHE_MANAGEMENT.md](CACHE_MANAGEMENT.md)**

**Verification:**
```bash
# Check mount points after container start
docker inspect llmwebui-backend | grep -A 20 "Mounts"

# Verify cache access
docker exec llmwebui-backend ls -la /root/.cache/huggingface
docker exec llmwebui-backend ls -la /root/.ollama
```

## Deployment Modes

### Development Mode

Features:
- Live code reload
- Debug logging
- Source code mounted as volume

```bash
make dev
# or: docker-compose -f docker-compose.dev.yml up -d
```

### Production Mode

Features:
- Optimized performance settings
- Multiple workers
- Structured logging
- Automatic restarts

```bash
make prod
# or: docker-compose -f docker-compose.prod.yml up -d
```

## Network Configuration

The containers use **host networking mode** to communicate with host services:

- Containers can access `localhost:11434` (Ollama)
- Containers can access `localhost:8001` (vLLM)
- No port mapping needed
- Services exposed directly on host ports

## Health Checks

### Automated Health Checks

Both containers include health checks:
- Backend: `curl -f http://localhost:8000/health`
- Frontend: `curl -f http://localhost:8501/_stcore/health`

### Manual Health Check

```bash
make health
# or: python3 scripts/health-check.py
```

## Common Operations

### View Logs
```bash
make logs
# or: docker-compose logs -f
```

### Restart Services
```bash
make restart
# or: make down && make up
```

### Check Status
```bash
make status
# or: docker-compose ps
```

### Clean Up
```bash
make clean  # Removes containers, images, and volumes
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   
   **Symptoms:** Backend shows "Ollama service unavailable" or model list is empty
   
   **Solutions:**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check if accessible from host
   curl http://localhost:11434/api/tags
   
   # Test from within container
   docker exec llmwebui-backend curl http://localhost:11434/api/tags
   
   # Restart Ollama if needed
   pkill ollama && ollama serve
   ```

2. **Permission Errors on Cache Directories**
   
   **Symptoms:** "Permission denied" errors when accessing models or cache
   
   **Solutions:**
   ```bash
   # Fix cache directory permissions
   chmod 755 ~/.cache/huggingface ~/.ollama
   
   # Ensure directories exist
   mkdir -p ~/.cache/huggingface ~/.ollama
   
   # Fix ownership if needed (Linux/macOS)
   sudo chown -R $USER:$USER ~/.cache/huggingface ~/.ollama
   ```

3. **Port Already in Use**
   
   **Symptoms:** "Port 8000/8501 already in use" errors
   
   **Solutions:**
   ```bash
   # Check what's using the ports
   lsof -i :8000  # Backend
   lsof -i :8501  # Frontend
   
   # Kill existing processes
   pkill -f "uvicorn.*main:app"
   pkill -f "streamlit run"
   
   # Or use different ports in docker-compose.yml
   ```

4. **Container Build Failures**
   
   **Symptoms:** Docker build fails with dependency or network errors
   
   **Solutions:**
   ```bash
   # Clean build cache and rebuild
   docker system prune -f
   make build
   
   # Build with no cache
   docker-compose build --no-cache
   
   # Check disk space
   df -h
   ```

5. **vLLM Process Management Issues**
   
   **Symptoms:** Cannot start/stop vLLM server from containerized app
   
   **Solutions:**
   ```bash
   # Check if vLLM conda environment exists
   conda env list | grep llm_ui
   
   # Verify host network access
   docker exec llmwebui-backend curl http://localhost:8001/health
   
   # Check process permissions
   ps aux | grep vllm
   ```

6. **Volume Mount Issues**
   
   **Symptoms:** Models not persisting, cache not working
   
   **Solutions:**
   ```bash
   # Verify mount points
   docker inspect llmwebui-backend | grep -A 10 "Mounts"
   
   # Check host directory permissions
   ls -la ~/.cache/huggingface ~/.ollama
   
   # Recreate containers with proper mounts
   make down && make up
   ```

7. **Memory/Resource Issues**
   
   **Symptoms:** Container crashes, out of memory errors
   
   **Solutions:**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase Docker memory limits
   # Docker Desktop: Settings > Resources > Memory
   
   # Reduce model size or concurrent operations
   # Edit .env file to adjust OMP_NUM_THREADS
   ```

8. **Network Connectivity Issues**
   
   **Symptoms:** Services can't communicate, API calls fail
   
   **Solutions:**
   ```bash
   # Test network connectivity
   docker exec llmwebui-backend ping host.docker.internal
   
   # Check host networking mode
   docker inspect llmwebui-backend | grep NetworkMode
   
   # Verify firewall settings (macOS/Linux)
   sudo ufw status  # Linux
   # System Preferences > Security & Privacy > Firewall (macOS)
   ```

### Debug Mode

For detailed debugging:

```bash
# Run with debug output
docker-compose -f docker-compose.dev.yml up --build

# Check container logs
docker logs llmwebui-backend-dev
docker logs llmwebui-frontend-dev
```

### Host Service Validation

```bash
# Test Ollama connectivity
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "Hello", "stream": false}'

# Test vLLM connectivity (if running)
curl http://localhost:8001/health
```

## Performance Tuning

### CPU Optimization

Adjust thread counts in `.env`:
```bash
OMP_NUM_THREADS=6      # Number of CPU cores - 1
MKL_NUM_THREADS=6      # Same as OMP_NUM_THREADS
NUMEXPR_NUM_THREADS=6  # Same as OMP_NUM_THREADS
```

### Memory Management

For systems with limited RAM:
- Reduce `gpu_memory_utilization` for vLLM
- Use smaller embedding models
- Limit concurrent document processing

### Storage Optimization

- Use SSD storage for cache directories
- Monitor disk usage: `du -sh ~/.cache/huggingface ~/.ollama`
- Clean unused models periodically

## Security Considerations

### Container Security

- Containers run as non-root user (appuser:1000)
- Host networking provides direct access to host services
- Volume mounts are read/write for cache persistence

### Production Hardening

1. **Use specific image tags** instead of `latest`
2. **Limit resource usage** with Docker resource constraints
3. **Regular security updates** for base images
4. **Monitor container logs** for suspicious activity

## Backup and Recovery

### Important Data

Backup these directories:
- `./uploads` - User uploaded documents
- `./chroma_db` - Vector database
- `./storage` - Application configuration
- `~/.ollama` - Ollama models (optional, can re-download)

### Backup Script Example

```bash
#!/bin/bash
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp -r ./uploads "$BACKUP_DIR/"
cp -r ./chroma_db "$BACKUP_DIR/"
cp -r ./storage "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"

echo "Backup created: $BACKUP_DIR"
```

## Migration from Non-Docker Setup

1. **Stop existing services**
2. **Backup data directories**
3. **Run Docker setup**: `make setup`
4. **Copy data to new locations** if needed
5. **Start Docker services**: `make up`
6. **Verify functionality**: `make health`

## Quick Troubleshooting Checklist

When experiencing issues, follow this checklist:

### 1. Basic Health Check
```bash
# Check if containers are running
make status

# Run comprehensive health check
make health
# or: python3 scripts/health-check.py --type comprehensive --verbose

# Check specific health endpoints
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/live

# Check logs for errors
make logs
```

📖 **For detailed health check information, see [DOCKER_HEALTH_CHECKS.md](DOCKER_HEALTH_CHECKS.md)**

### 2. Host Service Verification
```bash
# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Check if ports are available
lsof -i :8000 :8501 :11434
```

### 3. Volume Mount Verification
```bash
# Check if cache directories exist
ls -la ~/.cache/huggingface ~/.ollama

# Verify container mounts
docker inspect llmwebui-backend | grep -A 10 "Mounts"
```

### 4. Container Diagnostics
```bash
# Check container resource usage
docker stats

# Inspect container configuration
docker inspect llmwebui-backend llmwebui-frontend

# Test network connectivity from container
docker exec llmwebui-backend curl http://localhost:11434/api/tags
```

### 5. Clean Restart
```bash
# Complete restart with cleanup
make clean
make setup
make up
```

## Support

For issues specific to Docker deployment:
1. Follow the troubleshooting checklist above
2. Check the detailed troubleshooting section
3. Run `make health` to diagnose service issues
4. Check container logs with `make logs`
5. Verify host service connectivity manually

**Common Resolution Steps:**
- Restart Ollama service: `pkill ollama && ollama serve`
- Rebuild containers: `make clean && make build && make up`
- Reset cache permissions: `chmod 755 ~/.cache/huggingface ~/.ollama`
- Check available disk space and memory