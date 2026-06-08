# Docker image publishing

LiteMindUI already includes both local helper commands and GitHub Actions workflows for versioned image releases.

## Local helpers

```bash
make test-docker-local
make create-docker-repos
```

- `make test-docker-local` runs `scripts/test-docker-publish.py`
- `make create-docker-repos` runs `scripts/create-docker-repos.py`

## Automated workflows

### Release workflow

`.github/workflows/release.yml`:

- runs when a pull request is merged into `main`, or by manual dispatch
- bumps `version.json`
- creates and pushes the corresponding `v*.*.*` git tag
- creates the GitHub release

Version bump behavior:

- `major` label -> major bump
- `minor` label -> minor bump
- no label -> patch bump

### Docker publish workflow

`.github/workflows/docker-publish.yml`:

- runs on pushes to `main` and `develop`
- runs on version tags
- can also be started manually with `workflow_dispatch`
- builds `Dockerfile` and `Dockerfile.streamlit`
- publishes backend and frontend images to Docker Hub
- pushes the semver image tags from the `v*.*.*` tag build
- uploads `docker-compose.release.yml` to the matching GitHub release
- skips the automated `chore: bump version to ...` branch push so the tag run is the only release publish

Current image names:

- `debabratamishra1/litemindui-backend`
- `debabratamishra1/litemindui-frontend`
- `debabratamishra1/litemindui`

## Required secrets

The GitHub workflows expect these repository secrets:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `RELEASE_PAT`

## Release checklist

1. Ensure the Docker Hub repositories exist.
2. Merge the release PR into `main` with the desired version label, or trigger the release workflow manually.
3. Confirm the release workflow created the new git tag and GitHub release.
4. Confirm the Docker publish workflow pushed the matching backend and frontend images.
