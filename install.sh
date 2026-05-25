# LiteMindUI - Quick Install Script for Docker Hub
# This script helps users quickly get started with LiteMindUI using Docker Hub images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="litemind-ui"
COMPOSE_FILE="docker-compose.hub.yml"
SETUP_SCRIPT="scripts/docker-setup.sh"

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║                   LiteMindUI                         ║"
    echo "║              Quick Docker Install                    ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_dependencies() {
    echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        echo -e "${YELLOW}Visit: https://docs.docker.com/get-docker/${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        echo -e "${YELLOW}Visit: https://docs.docker.com/compose/install/${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
}

detect_compose_cmd() {
    # Prefer docker-compose binary if present, otherwise fallback to `docker compose` plugin
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}❌ Neither 'docker-compose' nor 'docker compose' is available. Please install Docker Compose or enable the Docker Compose CLI plugin.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Using compose command: ${COMPOSE_CMD}${NC}"
}

maybe_set_default_platform() {
    arch=$(uname -m || true)
    # Detect Apple Silicon / arm64 hosts and set default platform to amd64 to allow emulation
    if [[ "$arch" == "arm64" || "$arch" == "aarch64" ]]; then
        echo -e "${YELLOW}⚠️  Detected arm64 host. Setting DOCKER_DEFAULT_PLATFORM=linux/amd64 to pull amd64 images via emulation.${NC}"
        export DOCKER_DEFAULT_PLATFORM="linux/amd64"
    fi
}

platform_guidance() {
    osname=$(uname -s || echo "unknown")
    case "$osname" in
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            echo -e "${YELLOW}Note: On Windows, run this installer from WSL2 (recommended) or Git Bash which provides bash. If running on plain PowerShell/cmd, use WSL or install Git Bash.${NC}"
            ;;
        *)
            # no-op for Linux/macOS
            ;;
    esac
}

download_files() {
    echo -e "${YELLOW}📥 Downloading configuration files...${NC}"
    
    # Create directory
    mkdir -p "$REPO_NAME"
    cd "$REPO_NAME"
    
    # Download docker-compose file
    if ! curl -fsSL "https://raw.githubusercontent.com/debabratamishra/litemind-ui/main/docker-compose.hub.yml" -o "$COMPOSE_FILE"; then
        echo -e "${RED}❌ Failed to download docker-compose file${NC}"
        exit 1
    fi
    
    # Create basic setup script locally
    mkdir -p scripts
    cat > "$SETUP_SCRIPT" << 'EOF'
#!/bin/bash
# Basic setup script for Docker Hub installation

# Create necessary directories
mkdir -p uploads chroma_db storage .streamlit logs

# Create basic .streamlit config
mkdir -p .streamlit
cat > .streamlit/config.toml << 'CONFIG_EOF'
[server]
address = "localhost"
port = 8501

[browser]
serverAddress = "localhost"
CONFIG_EOF

echo "✅ Basic setup completed"
EOF
    
    chmod +x "$SETUP_SCRIPT"
    
    echo -e "${GREEN}✅ Configuration files downloaded${NC}"
}

run_setup() {
    echo -e "${YELLOW}🔧 Running initial setup...${NC}"
    
    if [[ -f "$SETUP_SCRIPT" ]]; then
        ./"$SETUP_SCRIPT"
    else
        # Fallback setup
        mkdir -p uploads chroma_db storage .streamlit logs
        echo "[server]" > .streamlit/config.toml
        echo "address = \"localhost\"" >> .streamlit/config.toml
        echo "port = 8501" >> .streamlit/config.toml
    fi
    
    echo -e "${GREEN}✅ Setup completed${NC}"
}

start_services() {
    echo -e "${YELLOW}🚀 Starting LiteMindUI services...${NC}"
    echo -e "${BLUE}This will download the Docker images (may take a few minutes)${NC}"
    
    if docker-compose -f "$COMPOSE_FILE" pull && docker-compose -f "$COMPOSE_FILE" up -d; then
        echo -e "${GREEN}✅ Services started successfully!${NC}"
        
        # Wait a moment for services to fully start
        echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
        sleep 10
        
        # Show access information
        echo -e "${GREEN}"
        echo "🎉 LiteMindUI is now running!"
        echo ""
        echo "Access Points:"
        echo "  • Frontend (Streamlit): http://localhost:8501"
        echo "  • Backend API: http://localhost:8000"
        echo "  • API Documentation: http://localhost:8000/docs"
        echo ""
        echo "Useful Commands:"
        echo "  • View logs: docker-compose -f $COMPOSE_FILE logs -f"
        echo "  • Stop services: docker-compose -f $COMPOSE_FILE down"
        echo "  • Restart services: docker-compose -f $COMPOSE_FILE restart"
        echo -e "${NC}"
        
        # Check if services are responding
        echo -e "${YELLOW}🔍 Checking service health...${NC}"
        sleep 5
        
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Backend is healthy${NC}"
        else
            echo -e "${YELLOW}⚠️  Backend may still be starting up${NC}"
        fi
        
        if curl -f http://localhost:8501 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Frontend is healthy${NC}"
        else
            echo -e "${YELLOW}⚠️  Frontend may still be starting up${NC}"
        fi
        
    else
        echo -e "${RED}❌ Failed to start services${NC}"
        exit 1
    fi
}

show_next_steps() {
    echo -e "${BLUE}"
    echo "📖 Next Steps:"
    echo ""
    echo "1. Open your browser and go to http://localhost:8501"
    echo "2. Explore the Chat and RAG features"
    echo "3. For Ollama support, install Ollama locally and ensure it's running"
    echo ""
    echo "Need help?"
    echo "  • Documentation: https://github.com/debabratamishra/litemind-ui"
    echo "  • Issues: https://github.com/debabratamishra/litemind-ui/issues"
    echo -e "${NC}"
}

cleanup_on_error() {
    echo -e "${RED}❌ Installation failed. Cleaning up...${NC}"
    if [[ -f "$COMPOSE_FILE" ]]; then
        docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    fi
    cd ..
    rm -rf "$REPO_NAME" 2>/dev/null || true
    exit 1
}

main() {
    trap cleanup_on_error ERR
    
    print_banner
    check_dependencies
    detect_compose_cmd
    maybe_set_default_platform
    platform_guidance
    download_files
    run_setup
    start_services
    show_next_steps
}

# Run main function
main "$@"
