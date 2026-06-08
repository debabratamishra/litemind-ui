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
