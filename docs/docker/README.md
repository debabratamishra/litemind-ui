# Docker workflows

LiteMindUI supports Docker in four practical ways: source builds, development mode, production-style compose, and Docker Hub images.

## Recommended commands

| Workflow | Command | Use when |
| --- | --- | --- |
| Source build | `make up` | Default local Docker run from the checked-out repo |
| Development | `make dev` | Iterating with the development compose file |
| Production-style | `make prod` | Validating the production compose setup locally |
| Docker Hub images | `make hub-up` | Pulling prebuilt images instead of building locally |
| macOS workaround | `./scripts/fix_macos.sh` | Switching to the macOS compose file if Docker Desktop networking needs it |

`make setup` runs `scripts/docker-setup.sh`, which creates the runtime directories and `.streamlit` config expected by the compose files.

## Quick start

```bash
git clone https://github.com/debabratamishra/litemind-ui.git
cd litemind-ui
make up
```

If you want to customize providers or ports first:

```bash
cp .env.example .env
```

After startup:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Compose file matrix

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | default source build for local Docker use |
| `docker-compose.dev.yml` | development-oriented stack |
| `docker-compose.prod.yml` | production-style stack |
| `docker-compose.hub.yml` | prebuilt Docker Hub images |
| `docker-compose.macos.yml` | macOS-specific networking fallback |

## Day-to-day operations

```bash
make logs
make health
make down
make status
```

Useful direct compose commands:

```bash
docker compose ps
docker compose logs -f backend frontend
curl http://localhost:8000/health
```

## Runtime data and mounts

The Docker setup uses repository-local runtime directories so data survives container restarts:

- `uploads/`
- `chroma_db/`
- `storage/`
- `.streamlit/`
- `logs/`

## Provider notes

- Native Ollama should usually be exposed to containers through `http://host.docker.internal:11434`.
- Override provider settings in `.env` when you want to switch from Ollama to OpenRouter or another compatible endpoint.
- The Streamlit frontend depends on the FastAPI backend, so both services should stay up together.

## Troubleshooting

1. If containers start but the UI cannot reach Ollama on macOS, try `./scripts/fix_macos.sh`.
2. If health checks fail, inspect logs with `make logs` or `docker compose logs -f backend frontend`.
3. If you need a clean rebuild, run `make clean` and then `make up`.

## Publishing

Release and image-publishing notes live in [`publishing.md`](publishing.md).
