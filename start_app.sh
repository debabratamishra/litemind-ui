#!/bin/bash

#!/bin/bash

# LLMWebUI Application Startup Script
# This script ensures proper application startup and provides helpful information

echo "🚀 Starting LLMWebUI Application..."
echo "=================================="

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build the application
echo "🏗️  Building application..."
docker-compose build --no-cache

# Start the application
echo "🚀 Starting services..."
docker-compose up -d

# Wait a moment for containers to start
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check container status
echo "📊 Container Status:"
docker-compose ps

# Test backend health
echo ""
echo "🩺 Testing Backend Health..."
BACKEND_HEALTH=$(docker exec llmwebui-backend curl -s http://localhost:8000/health 2>/dev/null || echo "Backend not ready")
if [[ "$BACKEND_HEALTH" == *"healthy"* ]]; then
    echo "✅ Backend: HEALTHY"
else
    echo "❌ Backend: NOT READY - Check logs with: docker-compose logs backend"
fi

# Test frontend
echo "🖥️  Testing Frontend..."
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo "✅ Frontend: ACCESSIBLE"
else
    echo "❌ Frontend: NOT ACCESSIBLE - Check logs with: docker-compose logs frontend"
fi

echo ""
echo "🎉 Application startup complete!"
echo ""
echo "🌐 Access your application at:"
echo "   📱 Frontend (Streamlit UI): http://localhost:8501"
echo "   🔧 Backend API Documentation: Available via container"
echo ""
echo "📋 Management Commands:"
echo "   View all logs:      docker-compose logs -f"
echo "   View backend logs:  docker-compose logs backend"
echo "   View frontend logs: docker-compose logs frontend"
echo "   Stop application:   docker-compose down"
echo "   Restart:            docker-compose restart"
echo "   Container status:   docker-compose ps"
echo ""
echo "🔍 Optional Services (for enhanced functionality):"
echo "   • Ollama (for local LLM models): https://ollama.ai/"
echo "     - Install: brew install ollama (macOS) or visit ollama.ai"
echo "     - Run: ollama serve"
echo "     - Pull models: ollama pull llama2"
echo ""
echo "✨ Your LLM WebUI is ready to use!"
echo "   Start by visiting: http://localhost:8501"
