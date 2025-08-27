#!/bin/bash
# Docker HEALTHCHECK script for LLMWebUI containers
# This script is designed to be used as a Docker HEALTHCHECK command

# Configuration
HEALTH_CHECK_TYPE=${HEALTH_CHECK_TYPE:-liveness}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}

# Run the health check
exec python3 /app/scripts/health-check.py \
    --type "$HEALTH_CHECK_TYPE" \
    --timeout "$HEALTH_CHECK_TIMEOUT" \
    --quiet