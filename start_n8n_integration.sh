#!/bin/bash

# LiteMind UI with n8n Workflow Integration - Startup Script

set -e

echo "🚀 Starting LiteMind UI with n8n Workflow Integration..."

# Check if conda environment exists
if ! conda env list | grep -q "llm_ui"; then
    echo "❌ Conda environment 'llm_ui' not found. Please create it first."
    exit 1
fi

# Activate conda environment
echo "🔄 Activating conda environment 'llm_ui'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_ui

# Check Python packages
echo "📦 Checking Python dependencies..."
python -c "import streamlit, fastapi, httpx, asyncio; print('✅ Core dependencies available')"

# Check n8n installation
if ! command -v n8n &> /dev/null; then
    echo "❌ n8n is not installed. Please install it with: npm install -g n8n"
    exit 1
fi

echo "✅ n8n found: $(n8n --version)"

# Check if Ollama is running (optional)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running and accessible"
else
    echo "⚠️  Ollama is not running on localhost:11434. You may need to start it separately."
fi

# Function to start n8n in background
start_n8n() {
    echo "🔧 Starting n8n server..."
    # Set n8n environment variables
    export N8N_HOST=localhost
    export N8N_PORT=5678
    export N8N_PROTOCOL=http
    
    # Start n8n in background
    nohup n8n start > n8n.log 2>&1 &
    N8N_PID=$!
    echo $N8N_PID > n8n.pid
    
    # Wait for n8n to start
    echo "⏳ Waiting for n8n to start..."
    for i in {1..30}; do
        if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
            echo "✅ n8n server is running on http://localhost:5678"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ n8n failed to start within 30 seconds"
    return 1
}

# Function to stop n8n
stop_n8n() {
    if [ -f n8n.pid ]; then
        N8N_PID=$(cat n8n.pid)
        if kill -0 $N8N_PID 2>/dev/null; then
            echo "🛑 Stopping n8n server (PID: $N8N_PID)..."
            kill $N8N_PID
            rm -f n8n.pid
        fi
    fi
}

# Function to start FastAPI backend
start_backend() {
    echo "🔧 Starting FastAPI backend..."
    python main.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    
    # Wait for backend to start
    echo "⏳ Waiting for FastAPI backend to start..."
    for i in {1..20}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ FastAPI backend is running on http://localhost:8000"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ FastAPI backend failed to start within 20 seconds"
    return 1
}

# Function to stop backend
stop_backend() {
    if [ -f backend.pid ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo "🛑 Stopping FastAPI backend (PID: $BACKEND_PID)..."
            kill $BACKEND_PID
            rm -f backend.pid
        fi
    fi
}

# Function to start Streamlit
start_streamlit() {
    echo "🔧 Starting Streamlit frontend..."
    streamlit run streamlit_app.py --server.port 8501 &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > streamlit.pid
    
    echo "✅ Streamlit is starting on http://localhost:8501"
}

# Function to stop streamlit
stop_streamlit() {
    if [ -f streamlit.pid ]; then
        STREAMLIT_PID=$(cat streamlit.pid)
        if kill -0 $STREAMLIT_PID 2>/dev/null; then
            echo "🛑 Stopping Streamlit (PID: $STREAMLIT_PID)..."
            kill $STREAMLIT_PID
            rm -f streamlit.pid
        fi
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    stop_streamlit
    stop_backend
    stop_n8n
    echo "✅ Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Parse command line arguments
case "${1:-start}" in
    "start")
        echo "🚀 Starting all services..."
        
        # Start n8n first
        if ! start_n8n; then
            echo "❌ Failed to start n8n. Exiting."
            exit 1
        fi
        
        # Start FastAPI backend
        if ! start_backend; then
            echo "❌ Failed to start FastAPI backend. Exiting."
            exit 1
        fi
        
        # Give backend a moment to fully initialize
        sleep 2
        
        # Start Streamlit frontend
        start_streamlit
        
        echo ""
        echo "🎉 LiteMind UI with n8n Workflow Integration is now running!"
        echo ""
        echo "📱 Frontend:    http://localhost:8501"
        echo "🔧 Backend:     http://localhost:8000"
        echo "⚙️  n8n Server:  http://localhost:5678"
        echo ""
        echo "🔧 Available Tools:"
        echo "   • Web Search (DuckDuckGo)"
        echo "   • File Operations (Read/Write)"
        echo "   • Data Processing"
        echo "   • Email Tool (configurable)"
        echo ""
        echo "📖 To use tools in chat:"
        echo "   1. Go to http://localhost:8501"
        echo "   2. Navigate to 'Workflows' tab"
        echo "   3. Enable 'Tool Use' in chat"
        echo "   4. Ask questions like:"
        echo "      - 'Search for latest AI news'"
        echo "      - 'Read the file README.md'"
        echo "      - 'Process this data: [1,2,3,4,5]'"
        echo ""
        echo "Press Ctrl+C to stop all services..."
        
        # Wait for interrupt
        while true; do
            sleep 1
        done
        ;;
    
    "stop")
        echo "🛑 Stopping all services..."
        cleanup
        ;;
    
    "status")
        echo "📊 Service Status:"
        
        # Check n8n
        if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
            echo "✅ n8n: Running (http://localhost:5678)"
        else
            echo "❌ n8n: Not running"
        fi
        
        # Check backend
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Backend: Running (http://localhost:8000)"
        else
            echo "❌ Backend: Not running"
        fi
        
        # Check streamlit (harder to detect, just check if port is open)
        if netstat -an | grep -q ":8501.*LISTEN"; then
            echo "✅ Streamlit: Running (http://localhost:8501)"
        else
            echo "❌ Streamlit: Not running"
        fi
        
        # Check Ollama
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama: Running (http://localhost:11434)"
        else
            echo "⚠️  Ollama: Not running"
        fi
        ;;
    
    "test")
        echo "🧪 Testing n8n workflow integration..."
        
        # Activate environment
        conda activate llm_ui
        
        # Test n8n connectivity
        if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
            echo "✅ n8n server is accessible"
            
            # Test workflow creation via Python
            python -c "
import asyncio
from app.services.n8n_service import n8n_service

async def test():
    status = await n8n_service.get_workflow_status()
    print(f'✅ n8n Status: {status}')
    
    # Test tool availability
    tools = n8n_service.get_available_tools()
    print(f'✅ Available tools: {list(tools.keys())}')

asyncio.run(test())
"
        else
            echo "❌ n8n server is not accessible. Please start it first."
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|status|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services (n8n, backend, frontend)"
        echo "  stop    - Stop all services"
        echo "  status  - Check service status"
        echo "  test    - Test n8n integration"
        exit 1
        ;;
esac
