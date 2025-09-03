# Multi-stage build for LiteMindUI FastAPI Backend
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VALIDATION_ENABLED=true \
    GRACEFUL_SHUTDOWN_ENABLED=true \
    STARTUP_TIMEOUT=60

# Install system dependencies including health check tools and minimal OpenCV requirements
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
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with uv + build cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/chroma_db /app/storage /app/.streamlit /app/logs \
    && chmod 755 /app/uploads /app/chroma_db /app/storage /app/.streamlit /app/logs

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
