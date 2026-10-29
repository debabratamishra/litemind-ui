# Docker guide

The canonical Docker definitions live under `infra/docker/`:

- `infra/docker/Dockerfile` — FastAPI backend image
- `infra/docker/Dockerfile.nextjs` — Next.js frontend image
- `infra/docker/compose/` — supported Compose workflows

Most users only need one of these commands:

```bash
make up
make dev
make prod
make hub-up
```

See [`docs/docker/README.md`](docs/docker/README.md) for the Compose layout, environment setup, persistence, and direct Compose examples.
