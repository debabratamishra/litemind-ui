# macOS Fix Script for LLM WebUI
# This script switches to macOS-compatible networking and restarts the application

detect_compose_cmd() {
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        echo "❌ Neither 'docker-compose' nor 'docker compose' is available."
        exit 1
    fi
}

detect_compose_cmd

echo "🔧 Fixing LLM WebUI for macOS Docker Desktop..."
echo "==============================================="

# Stop existing containers
echo "🛑 Stopping existing containers..."
$COMPOSE_CMD down 2>/dev/null || true

# Backup original docker-compose.yml
if [[ ! -f "docker-compose.yml.backup" ]]; then
    echo "💾 Backing up original docker-compose.yml..."
    cp docker-compose.yml docker-compose.yml.backup
fi

# Use macOS-compatible configuration
echo "🔄 Switching to macOS-compatible networking..."
cp docker-compose.macos.yml docker-compose.yml

# Rebuild and start with new configuration
echo "🏗️  Building with new configuration..."
$COMPOSE_CMD build --no-cache

echo "🚀 Starting services with proper port mapping..."
$COMPOSE_CMD up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check container status
echo "📊 Container Status:"
$COMPOSE_CMD ps

# Test services
echo ""
echo "🧪 Testing Services:"

# Test backend
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Backend: http://localhost:8000 - ACCESSIBLE"
else
    echo "❌ Backend: http://localhost:8000 - NOT ACCESSIBLE"
    echo "   Trying to test from container..."
    BACKEND_HEALTH=$(docker exec litemindui-backend curl -s http://localhost:8000/health 2>/dev/null || echo "Failed")
    if [[ "$BACKEND_HEALTH" == *"healthy"* ]]; then
        echo "   ✅ Backend running inside container but port mapping may need time"
    fi
fi

# Test frontend
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo "✅ Frontend: http://localhost:8501 - ACCESSIBLE"
else
    echo "❌ Frontend: http://localhost:8501 - NOT ACCESSIBLE"
    echo "   Give it a few more seconds, then try: http://localhost:8501"
fi

echo ""
echo "🎉 macOS Configuration Applied!"
echo "==============================="
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:8501"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🔄 If ports are still not accessible, wait 30 seconds and try again."
echo "   Docker Desktop on macOS sometimes takes time to bind ports."
echo ""
echo "📋 Troubleshooting:"
echo "   Check logs: $COMPOSE_CMD logs -f"
echo "   Restart:    $COMPOSE_CMD restart"
echo "   Stop:       $COMPOSE_CMD down"
echo ""
echo "🔙 To restore original networking:"
echo "   mv docker-compose.yml.backup docker-compose.yml"
