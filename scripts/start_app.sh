#!/usr/bin/env bash

# LiteMindUI Application Startup Script
# This script ensures proper application startup and provides helpful information

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

cd "$REPO_ROOT"

echo "🚀 Starting LiteMindUI..."
echo "========================="

# Stop any existing containers
echo "🛑 Stopping existing containers..."
$COMPOSE_CMD down 2>/dev/null || true

# Build the application
echo "🏗️  Building application..."
$COMPOSE_CMD build --no-cache

# Start the application
echo "🚀 Starting services..."
$COMPOSE_CMD up -d

# Wait a moment for containers to start
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check container status
echo "📊 Container Status:"
$COMPOSE_CMD ps

# Test backend health
echo ""
echo "🩺 Testing Backend Health..."
BACKEND_HEALTH=$(docker exec litemindui-backend curl -s http://localhost:8000/health 2>/dev/null || echo "Backend not ready")
if [[ "$BACKEND_HEALTH" == *"healthy"* ]]; then
    echo "✅ Backend: HEALTHY"
else
    echo "❌ Backend: NOT READY - Check logs with: $COMPOSE_CMD logs backend"
fi

# Test frontend
echo "🖥️  Testing Frontend..."
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo "✅ Frontend: ACCESSIBLE"
else
    echo "❌ Frontend: NOT ACCESSIBLE - Check logs with: $COMPOSE_CMD logs frontend"
fi

echo ""
echo "🎉 Application startup complete!"
echo ""
echo "🌐 Access your application at:"
echo "   📱 Frontend (Streamlit UI): http://localhost:8501"
echo "   🔧 Backend API Documentation: Available via container"
echo ""
echo "📋 Management Commands:"
echo "   View all logs:      $COMPOSE_CMD logs -f"
echo "   View backend logs:  $COMPOSE_CMD logs backend"
echo "   View frontend logs: $COMPOSE_CMD logs frontend"
echo "   Stop application:   $COMPOSE_CMD down"
echo "   Restart:            $COMPOSE_CMD restart"
echo "   Container status:   $COMPOSE_CMD ps"
echo ""
echo "🔍 Optional Services (for enhanced functionality):"
echo "   • Ollama (for local LLM models): https://ollama.ai/"
echo "     - Install: brew install ollama (macOS) or visit ollama.ai"
echo "     - Run: ollama serve"
echo "     - Pull models: ollama pull llama2"
echo ""
echo "✨ LiteMindUI is ready to use!"
echo "   Start by visiting: http://localhost:8501"
