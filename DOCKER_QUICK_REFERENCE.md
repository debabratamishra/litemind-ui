# Quick Reference: Docker Builds with UV

## TL;DR

Your Docker containers now use **dependency groups** to stay optimized. Each container only installs what it needs.

### Build Locally
```bash
# Full install (all groups)
uv sync

# Just backend deps
uv sync --only-group backend

# Just frontend deps
uv sync --only-group frontend
```

### Build Docker
```bash
# Build both containers (optimized)
docker-compose build

# Build only backend
docker-compose build backend

# Build only frontend
docker-compose build frontend
```

### Check Sizes
```bash
# View image sizes
docker images | grep litemind

# Expected sizes:
# - Backend: ~2.5 GB (includes ML/RAG)
# - Frontend: ~1.8 GB (lighter)
```

## File Structure

```
pyproject.toml
├── [project]
│   └── dependencies: shared (python-dotenv, psutil)
├── [dependency-groups]
│   ├── backend: FastAPI, ML, RAG, document processing
│   ├── frontend: Streamlit, audio, WebRTC
│   └── dev: pytest, black, ruff

Dockerfile
└── RUN uv sync --frozen --no-dev --only-group backend

Dockerfile.streamlit
└── RUN uv sync --frozen --no-dev --only-group frontend

uv.lock
└── All 448 packages locked (both groups included)
```

## Add/Remove Dependencies

```bash
# Add to backend
uv add --group backend package-name

# Add to frontend
uv add --group frontend package-name

# Add to dev
uv add --group dev package-name

# Add to shared (both containers)
uv add package-name

# Remove
uv remove package-name

# Always update lock after changes
uv lock --upgrade
```

## Documentation Files

- **DOCKER_OPTIMIZATION.md** - Detailed explanation and benefits
- **DOCKER_OPTIMIZATION_IMPLEMENTATION.md** - Implementation summary
- **UV_MIGRATION.md** - Migration from pip to uv
- **UV_MIGRATION_SUMMARY.md** - Quick summary of migration

## Old Files (Can Delete)

```bash
# These are no longer needed (all in pyproject.toml now)
git rm requirements.txt requirements-backend.txt requirements-frontend.txt
```

## Common Tasks

### Update all dependencies
```bash
uv lock --upgrade
uv sync
```

### Check what's installed
```bash
source .venv/bin/activate
python -m pip list | head -20
```

### Test specific group
```bash
# Run backend app
uv run --group backend python main.py

# Run frontend
uv run --group frontend streamlit run streamlit_app.py
```

### Docker build troubleshooting
```bash
# Rebuild without cache
docker-compose build --no-cache

# View detailed build output
docker-compose build --progress=plain backend

# Check what was installed
docker-compose run backend pip list | head -20
```

## Key Difference from Before

| Aspect | Before | After |
|--------|--------|-------|
| Config files | 3 (requirements.txt, -backend, -frontend) | 1 (pyproject.toml) |
| Dependency separation | Manual (separate files) | Automatic (dependency groups) |
| Version control | 3 files to track | 1 file + uv.lock |
| Maintenance | Sync 3 files | Edit pyproject.toml |
| Docker builds | 2 separate file copies | 1 file copy + group selection |
| IDE support | ❌ Limited | ✅ Full (pyproject.toml standard) |

## Benefits

✅ Single source of truth  
✅ Same build optimization as before  
✅ Easier to maintain  
✅ Better IDE support  
✅ Standard Python packaging (PEP 735)  
✅ Faster dependency resolution (uv)  

