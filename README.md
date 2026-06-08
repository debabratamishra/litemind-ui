# LiteMindUI

LiteMindUI is a local-first AI workspace with a FastAPI backend and a Streamlit frontend for chat, RAG, web search, and voice-enabled workflows. It supports Ollama for local models and OpenRouter for hosted models.

![LiteMindUI demo](docs/assets/litemindui-demo.gif)

## Quick start

| Mode | Best for | Command |
| --- | --- | --- |
| Quick install | Fastest path with prebuilt Docker images | `curl -fsSL https://raw.githubusercontent.com/debabratamishra/litemind-ui/main/install.sh \| bash` |
| Docker from source | Running the repo locally with the checked-out source | `make up` |
| Native install | Development on the Python codebase | `uv sync --group all` |

### 1. Quick install with prebuilt Docker images

```bash
curl -fsSL https://raw.githubusercontent.com/debabratamishra/litemind-ui/main/install.sh | bash
```

This downloads `docker-compose.hub.yml`, prepares the required runtime directories, and starts the app with prebuilt images.

### 2. Run from source with Docker

```bash
git clone https://github.com/debabratamishra/litemind-ui.git
cd litemind-ui
make up
```

Useful variants:

- `make dev` - development compose stack
- `make prod` - production-style compose stack
- `make hub-up` - pull and run Docker Hub images from this repo

### 3. Run natively for development

```bash
git clone https://github.com/debabratamishra/litemind-ui.git
cd litemind-ui
uv sync --group all
mkdir -p uploads .streamlit
```

If you want to override defaults, copy `.env.example` to `.env` before starting the services.

Backend:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

Native execution is supported, but Docker remains the simplest setup for most users.

## Default URLs

| Surface | URL |
| --- | --- |
| Streamlit frontend | `http://localhost:8501` |
| FastAPI backend | `http://localhost:8000` |
| API docs | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/health` |

## Repository layout

```text
app/
  backend/      FastAPI routes, schemas, and API logic
  core/         shared configuration and application wiring
  frontend/     Streamlit pages, components, and UI helpers
  ingestion/    document ingestion and knowledge-processing flow
  services/     model, RAG, speech, and web-search integrations
  skills/       pluggable chat and RAG capability layer (web search and RAG strategies use it)
scripts/        setup, Docker, release, and health-check helpers
docs/           deeper documentation and docs assets
main.py         backend entrypoint
streamlit_app.py frontend entrypoint
install.sh      quick Docker Hub installer
Dockerfile*     backend and frontend container builds
docker-compose*.yml supported container workflows
```

## Documentation

- [`docs/README.md`](docs/README.md) - documentation index
- [`DOCKER.md`](DOCKER.md) - Docker guide shortcut
- [`docs/docker/README.md`](docs/docker/README.md) - Docker workflows, health checks, and compose files
- [`docs/docker/publishing.md`](docs/docker/publishing.md) - image publishing and release automation

## Notes

- Ollama should be reachable at `http://localhost:11434` for native runs and `http://host.docker.internal:11434` from containers unless you override `OLLAMA_API_URL`.
- The frontend depends on the FastAPI backend for chat, RAG, and web search features.
- RAG embeddings can use direct Ollama/HuggingFace integrations or OpenRouter via the existing LiteLLM transport.
- For production-like deployments, prefer the compose workflows over ad hoc process startup.
