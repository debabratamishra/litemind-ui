#!/bin/bash
set -e

# Docker Entrypoint Script for LLMWebUI
# This script handles container startup, validation, and graceful shutdown

# Configuration
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-60}
VALIDATION_ENABLED=${VALIDATION_ENABLED:-true}
GRACEFUL_SHUTDOWN_ENABLED=${GRACEFUL_SHUTDOWN_ENABLED:-true}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    log_info "🛑 Container shutdown initiated (exit code: $exit_code)"
    
    if [ "$GRACEFUL_SHUTDOWN_ENABLED" = "true" ]; then
        log_info "Running graceful shutdown handler..."
        
        # Run graceful shutdown script if available
        if [ -f "/app/scripts/graceful-shutdown.py" ]; then
            python3 /app/scripts/graceful-shutdown.py --timeout 30 || {
                log_warn "Graceful shutdown script failed, continuing with normal shutdown"
            }
        else
            log_warn "Graceful shutdown script not found at /app/scripts/graceful-shutdown.py"
        fi
    else
        log_info "Graceful shutdown disabled, performing immediate shutdown"
    fi
    
    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log_info "Container shutdown complete"
    exit $exit_code
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT SIGQUIT

# Function to wait for a condition with timeout
wait_for_condition() {
    local condition_cmd="$1"
    local timeout="$2"
    local description="$3"
    local interval="${4:-2}"
    
    log_info "⏳ Waiting for: $description (timeout: ${timeout}s)"
    
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if eval "$condition_cmd" >/dev/null 2>&1; then
            log_success "✅ Condition met: $description"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        
        if [ $((elapsed % 10)) -eq 0 ]; then
            log_info "Still waiting for: $description (${elapsed}s elapsed)"
        fi
    done
    
    log_error "❌ Timeout waiting for: $description"
    return 1
}

# Function to check if a service is responding
check_service() {
    local service_name="$1"
    local url="$2"
    local timeout="${3:-5}"
    
    if command -v curl >/dev/null 2>&1; then
        curl -s -f --max-time "$timeout" "$url" >/dev/null 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget -q --timeout="$timeout" --tries=1 -O /dev/null "$url" >/dev/null 2>&1
    else
        # Fallback using Python
        python3 -c "
import sys
import urllib.request
import socket
socket.setdefaulttimeout($timeout)
try:
    urllib.request.urlopen('$url')
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null
    fi
}

# Pre-startup validation
run_startup_validation() {
    log_info "🚀 Running container startup validation..."
    
    # Check if validation script exists
    if [ ! -f "/app/scripts/startup-validation.py" ]; then
        log_warn "Startup validation script not found, skipping validation"
        return 0
    fi
    
    # Run startup validation with timeout
    if timeout "$STARTUP_TIMEOUT" python3 /app/scripts/startup-validation.py; then
        log_success "✅ Startup validation passed"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "❌ Startup validation timed out after ${STARTUP_TIMEOUT}s"
        else
            log_error "❌ Startup validation failed (exit code: $exit_code)"
        fi
        
        # Check if we should continue despite validation failure
        if [ "$VALIDATION_REQUIRED" = "false" ]; then
            log_warn "⚠️  Continuing startup despite validation failure (VALIDATION_REQUIRED=false)"
            return 0
        else
            return $exit_code
        fi
    fi
}

# Function to start the main application
start_application() {
    local app_command="$1"
    shift
    local app_args="$@"
    
    log_info "🚀 Starting main application: $app_command $app_args"
    
    # Start the application in the background
    exec "$app_command" $app_args &
    local app_pid=$!
    
    log_info "Application started with PID: $app_pid"
    
    # Wait for the application to be ready
    if wait_for_condition "check_service 'FastAPI' 'http://localhost:8000/health'" 30 "FastAPI backend to be ready"; then
        log_success "🎉 Application is ready and serving requests"
    else
        log_error "❌ Application failed to become ready"
        kill $app_pid 2>/dev/null || true
        return 1
    fi
    
    # Wait for the application process
    wait $app_pid
    local exit_code=$?
    
    log_info "Application process exited with code: $exit_code"
    return $exit_code
}

# Function to handle different application types
handle_application_startup() {
    local command="$1"
    shift
    local args="$@"
    
    case "$command" in
        "uvicorn"|"python"|"python3")
            # FastAPI backend
            log_info "🔧 Detected FastAPI backend startup"
            start_application "$command" $args
            ;;
        "streamlit")
            # Streamlit frontend
            log_info "🔧 Detected Streamlit frontend startup"
            start_application "$command" $args
            ;;
        *)
            # Generic application
            log_info "🔧 Generic application startup"
            start_application "$command" $args
            ;;
    esac
}

# Main entrypoint logic
main() {
    log_info "🐳 LLMWebUI Docker Container Starting"
    log_info "Container ID: $(hostname)"
    log_info "Startup timeout: ${STARTUP_TIMEOUT}s"
    log_info "Validation enabled: $VALIDATION_ENABLED"
    log_info "Graceful shutdown enabled: $GRACEFUL_SHUTDOWN_ENABLED"
    
    # Print environment info
    log_info "Environment: $(uname -a)"
    log_info "Python version: $(python3 --version 2>/dev/null || echo 'Not available')"
    log_info "Working directory: $(pwd)"
    
    # Ensure we're in the correct directory
    if [ -d "/app" ]; then
        cd /app
        log_info "Changed to application directory: /app"
    fi
    
    # Run startup validation if enabled
    if [ "$VALIDATION_ENABLED" = "true" ]; then
        if ! run_startup_validation; then
            log_error "❌ Startup validation failed, container cannot start"
            exit 1
        fi
    else
        log_info "Startup validation disabled"
    fi
    
    # Check if we have arguments (command to run)
    if [ $# -eq 0 ]; then
        log_error "❌ No command provided to entrypoint"
        log_info "Usage: docker run <image> <command> [args...]"
        exit 1
    fi
    
    # Handle special commands
    case "$1" in
        "health-check")
            log_info "Running health check..."
            exec python3 /app/scripts/health-check.py "${@:2}"
            ;;
        "validation")
            log_info "Running startup validation..."
            exec python3 /app/scripts/startup-validation.py "${@:2}"
            ;;
        "shutdown")
            log_info "Running graceful shutdown..."
            exec python3 /app/scripts/graceful-shutdown.py "${@:2}"
            ;;
        "bash"|"sh")
            log_info "Starting interactive shell..."
            exec "$@"
            ;;
        *)
            # Normal application startup
            handle_application_startup "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"