#!/bin/bash

# Docker setup script for LLMWebUI
# This script ensures proper directory structure and permissions for Docker deployment
# Uses Python-based cache setup for OS-independent directory management

set -e

echo "🐳 Setting up LLMWebUI Docker environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is required but not found. Please install Python 3.7+ and try again."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

# Generate OS-independent Docker environment file
echo "🔧 Generating Docker environment configuration..."
if $PYTHON_CMD scripts/generate-docker-env.py --output .env.docker; then
    echo "✅ Docker environment file created: .env.docker"
else
    echo "❌ Failed to generate Docker environment file"
    exit 1
fi

# Set up cache directories using Python script
echo "💾 Setting up OS-independent cache directories..."
if $PYTHON_CMD scripts/cache-setup.py --verbose; then
    echo "✅ Cache directories configured successfully"
else
    echo "❌ Failed to set up cache directories"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your specific configuration"
else
    echo "✅ .env file already exists"
fi

# Check if Ollama is running
echo "🔍 Checking host services..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama service is running on localhost:11434"
else
    echo "⚠️  Ollama service not detected on localhost:11434"
    echo "   Please ensure Ollama is installed and running before starting containers"
fi

echo ""
echo "🎉 Docker setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file if needed"
echo "2. Ensure Ollama is running: ollama serve"
echo "3. Start the application with the generated environment:"
echo "   - Development: docker-compose --env-file .env.docker -f docker-compose.dev.yml up"
echo "   - Production:  docker-compose --env-file .env.docker -f docker-compose.prod.yml up"
echo "   - Default:     docker-compose --env-file .env.docker up"
echo ""
echo "💡 The .env.docker file contains OS-specific cache directory paths"
echo "💡 You can also source it manually: export \$(cat .env.docker | xargs)"
echo ""
echo "The application will be available at:"
echo "- Backend API: http://localhost:8000"
echo "- Frontend UI: http://localhost:8501"