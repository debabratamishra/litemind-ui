# Multi-stage build for LiteMindUI FastAPI Backend.
#
# T6 split the dependencies into uv dependency-groups (core / ml / voice / ocr /
# dev / all) and made every heavy import lazy, so a core-only image boots a
# healthy API. This Dockerfile installs exactly the requested profiles in a
# builder stage and ships only the virtualenv + app code in the runtime stage,
# so the image no longer carries build-essential, the apt toolchain or the uv
# installer.
#
# Build args:
#   PROFILE  uv dependency groups to install, comma-separated (default core,ml).
#            Add `voice` for realtime voice mode, `ocr` for document ingestion,
#            `dev` for a test-capable image.
#   VOICE    set to 1 to add the native WebRTC/TTS libraries (aiortc dlopen()s
#            the ffmpeg -dev .so symlinks at runtime; kokoro/espeak need
#            libsndfile1 + espeak).
ARG PROFILE=core,ml
ARG VOICE=0

# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Installs the requested profiles into /app/.venv. UV_LINK_MODE=copy (not
# hardlink) so the venv survives being copied into the runtime stage — hardlinks
# would break across the layer/copy boundary.
FROM python:3.13-slim AS builder

ARG PROFILE
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.13

# curl is only needed here to fetch the uv installer; the runtime stage gets
# its own (smaller) apt set.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency metadata first: this layer only rebuilds when the lockfile moves.
COPY pyproject.toml uv.lock ./

# --no-install-project: the runtime stage imports `backend` from /app
# (WORKDIR) directly, so installing the project here would only add churn.
# uv takes one --only-group per flag, so the comma-separated PROFILE build arg
# is expanded into repeated flags by shell word splitting.
RUN --mount=type=cache,target=/root/.cache/uv \
    PROFILE_ARGS=""; for g in $(echo "${PROFILE}" | tr ',' ' '); do \
        PROFILE_ARGS="${PROFILE_ARGS} --only-group ${g}"; \
    done; \
    uv sync --no-dev ${PROFILE_ARGS} --no-install-project

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# `base` is the full runtime image; docker-compose.dev.yml targets it for its
# source-mounted live-reload container. `runtime` is the same image re-tagged
# as the default/final target (the FROM line costs nothing and reuses base's
# cache) so `--target runtime` and the plain default build agree.
FROM python:3.13-slim AS base

ARG VOICE

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VALIDATION_ENABLED=true \
    GRACEFUL_SHUTDOWN_ENABLED=true \
    STARTUP_TIMEOUT=60 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}"

# Base runtime apt set: curl (healthchecks), procps (ps for graceful-shutdown
# + diagnostics), libsndfile1 (soundfile/kokoro TTS). VOICE=1 adds the ffmpeg
# -dev family + espeak: aiortc (via Pipecat small-webrtc) dlopen()s the
# unversioned .so symlinks only the -dev packages provide, and espeak backs the
# pyttsx3 TTS fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    libsndfile1 \
    && if [ "$VOICE" = "1" ]; then \
        apt-get install -y --no-install-recommends \
            espeak \
            espeak-ng \
            pkg-config \
            libavformat-dev \
            libavcodec-dev \
            libavdevice-dev \
            libavutil-dev \
            libswscale-dev \
            libswresample-dev \
            libopus-dev \
            libvpx-dev; \
    fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# venv from the builder, application code, and nothing else.
COPY --from=builder /app/.venv /app/.venv
COPY . .

# The `backend` package is imported from /app (WORKDIR) directly, as in the
# previous single-stage image — the runtime stage has no uv binary, so a
# project reinstall would need a second builder round-trip for no benefit.
RUN mkdir -p /app/uploads /app/chroma_db /app/storage /app/logs /tmp/litemind_tts_cache \
    && chmod 755 /app/uploads /app/chroma_db /app/storage /app/logs /tmp/litemind_tts_cache \
    && chmod +x /app/scripts/*.py /app/scripts/*.sh \
    && mkdir -p /root/.cache/huggingface /root/.ollama \
    && chmod 755 /root/.cache/huggingface /root/.ollama

# Enhanced health check using our custom script
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD /app/scripts/docker-healthcheck.sh

# Expose port
EXPOSE 8000

# Use our custom entrypoint for startup validation and graceful shutdown
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Default target: the runtime image (see comment on the `base` stage).
FROM base AS runtime
