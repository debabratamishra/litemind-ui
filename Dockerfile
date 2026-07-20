# Multi-stage build for LiteMindUI FastAPI Backend
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VALIDATION_ENABLED=true \
    GRACEFUL_SHUTDOWN_ENABLED=true \
    STARTUP_TIMEOUT=60

# Install system dependencies including health check tools, OpenCV requirements, and TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    procps \
    psmisc \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libfreetype6 \
    libsndfile1 \
    espeak \
    espeak-ng \
    # Realtime voice mode: aiortc (via Pipecat small-webrtc) is a cffi wrapper
    # that dlopen()s these shared libraries at runtime. The -dev packages provide
    # the unversioned .so symlinks aiortc's cffi loader expects.
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libopus-dev \
    libvpx-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.13

# Create app directory
WORKDIR /app
 
# Copy dependency metadata first for better caching
COPY pyproject.toml uv.lock ./

# Install backend + voice dependencies (not frontend, not dev).
# The voice group pulls in pipecat-ai and aiortc, which main.py imports at
# startup (app/backend/api/voice.py). Without it the app fails with
# "ModuleNotFoundError: No module named 'pipecat'".
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --only-group backend --only-group voice --no-install-project

# Copy application code
COPY . .

# Install the project itself (backend + voice groups)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --only-group backend --only-group voice

# Ensure the container uses the project's virtual environment for runtime
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:${PATH}"

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/chroma_db /app/storage /app/logs /tmp/litemind_tts_cache \
    && chmod 755 /app/uploads /app/chroma_db /app/storage /app/logs /tmp/litemind_tts_cache

# Make scripts executable
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh

# Create cache directories for volume mounts
RUN mkdir -p /root/.cache/huggingface /root/.ollama \
    && chmod 755 /root/.cache/huggingface /root/.ollama

# Enhanced health check using our custom script
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD /app/scripts/docker-healthcheck.sh

# Expose port
EXPOSE 8000

# Use our custom entrypoint for startup validation and graceful shutdown
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
