#!/bin/bash

# Docker Environment Setup Script
# Handles development and production Docker configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
COMPOSE_FILE=""
ENV_FILE=""
FORCE_REBUILD=false
PULL_IMAGES=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Docker Environment Setup Script for LLMWebUI

COMMANDS:
    up          Start the application in specified environment
    down        Stop the application
    restart     Restart the application
    logs        Show application logs
    build       Build Docker images
    clean       Clean up containers and images
    status      Show container status

OPTIONS:
    -e, --env ENV       Environment (development|production) [default: development]
    -f, --force         Force rebuild of images
    -p, --pull          Pull latest base images before building
    -h, --help          Show this help message

EXAMPLES:
    $0 up                           # Start in development mode
    $0 -e production up             # Start in production mode
    $0 -f build                     # Force rebuild images
    $0 -e production -p build       # Build production with latest base images
    $0 logs                         # Show logs for current environment
    $0 down                         # Stop all containers

EOF
}

# Function to validate environment
validate_environment() {
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
        print_error "Invalid environment: $ENVIRONMENT. Must be 'development' or 'production'"
        exit 1
    fi
}

# Function to set compose file and env file based on environment
set_environment_files() {
    case "$ENVIRONMENT" in
        "development")
            COMPOSE_FILE="docker-compose.dev.yml"
            ENV_FILE=".env.development"
            ;;
        "production")
            COMPOSE_FILE="docker-compose.prod.yml"
            ENV_FILE=".env.production"
            ;;
    esac
    
    print_status "Using environment: $ENVIRONMENT"
    print_status "Compose file: $COMPOSE_FILE"
    print_status "Environment file: $ENV_FILE"
}

# Function to ensure required directories exist
ensure_directories() {
    print_status "Ensuring required directories exist..."
    
    directories=(
        "logs"
        "uploads"
        "chroma_db"
        "storage"
        ".streamlit"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 logs uploads chroma_db storage .streamlit 2>/dev/null || true
}

# Function to validate Docker setup
validate_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
}

# Function to check if files exist
check_files() {
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    if [[ ! -f "$ENV_FILE" ]]; then
        print_warning "Environment file not found: $ENV_FILE"
        print_status "Using default environment variables"
    fi
}

# Function to build images
build_images() {
    print_status "Building Docker images for $ENVIRONMENT environment..."
    
    build_args=""
    if [[ "$PULL_IMAGES" == "true" ]]; then
        build_args="--pull"
    fi
    
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi
    
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build $build_args
    else
        docker-compose -f "$COMPOSE_FILE" build $build_args
    fi
    
    print_success "Images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting services in $ENVIRONMENT mode..."
    
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
    else
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
    
    print_success "Services started successfully"
    print_status "Backend available at: http://localhost:8000"
    print_status "Frontend available at: http://localhost:8501"
    
    # Show service status
    sleep 2
    show_status
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
    else
        docker-compose -f "$COMPOSE_FILE" down
    fi
    
    print_success "Services stopped successfully"
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    stop_services
    start_services
}

# Function to show logs
show_logs() {
    print_status "Showing logs for $ENVIRONMENT environment..."
    
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs -f
    else
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Function to show status
show_status() {
    print_status "Container status:"
    
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
    else
        docker-compose -f "$COMPOSE_FILE" ps
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and images..."
    
    # Stop and remove containers
    if [[ -f "$ENV_FILE" ]]; then
        docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans
    else
        docker-compose -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans
    fi
    
    # Clean up unused Docker resources
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_REBUILD=true
            shift
            ;;
        -p|--pull)
            PULL_IMAGES=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        up|down|restart|logs|build|clean|status)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$COMMAND" ]]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

validate_environment
validate_docker
set_environment_files
check_files
ensure_directories

# Execute command
case "$COMMAND" in
    "up")
        build_images
        start_services
        ;;
    "down")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        show_logs
        ;;
    "build")
        build_images
        ;;
    "clean")
        cleanup
        ;;
    "status")
        show_status
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac