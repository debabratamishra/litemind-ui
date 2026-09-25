# Docker and Compose

Docker definitions live under `infra/docker/`:

```text
infra/docker/
  Dockerfile          FastAPI backend image
  Dockerfile.nextjs  Next.js frontend image
  compose/
    docker-compose.yml
    docker-compose.auth.yml
    docker-compose.dev.yml
    docker-compose.prod.yml
    docker-compose.hub.yml
```

## Common workflows

The repository `Makefile` is the supported entry point:

```bash
make up          # build/start the default stack with auth services
make dev         # start the development stack
make prod        # start the production stack
make hub-up      # pull and start Docker Hub images
make down        # stop the active stacks
make logs        # follow logs
make health      # run the health-check helper
```

The underlying Compose files are referenced with `-f infra/docker/compose/<file>` when running Docker Compose directly. Pass `--project-directory .` as well: the Compose project directory (which drives the `.env` lookup, the project/volume names, and every relative path below) must stay the repository root. All Compose files use the repository root as their build context (`context: .`) and the canonical Dockerfiles in `infra/docker/`.

## Environment and persistence

Copy `.env.example` to `.env` and fill in the required provider and authentication values. The default stack persists `uploads/`, `chroma_db/`, `storage/`, and `logs/` through bind mounts. Do not commit `.env` or provider credentials.

`make gotrue-up` starts the standalone GoTrue/PostgreSQL services for a natively running backend. `make gotrue-down` stops them and removes the legacy standalone containers if present.
