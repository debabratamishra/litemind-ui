# n8n Workflow Integration for LiteMind UI

This document describes the comprehensive integration of n8n workflows with LiteMind UI to provide tool use and MCP (Model Context Protocol) capabilities for local LLMs.

## 🌟 Overview

The n8n integration adds powerful workflow automation and tool calling capabilities to LiteMind UI, enabling your local LLMs (via Ollama or vLLM) to:

- **Search the web** using DuckDuckGo
- **Read and write files** on the local system  
- **Process and transform data** using JavaScript
- **Send emails** (configurable)
- **Execute custom workflows** for any automation task
- **Call external APIs** and services

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │      n8n        │
│   Frontend      │◄──►│   Backend       │◄──►│   Workflows     │
│                 │    │                 │    │                 │
│ • Chat UI       │    │ • Tool API      │    │ • Web Search    │
│ • Workflow Page │    │ • n8n Client    │    │ • File Ops      │
│ • Tool Results  │    │ • LLM Router    │    │ • Data Process  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Local LLMs    │
                    │                 │
                    │ • Ollama        │
                    │ • vLLM          │
                    │ • Tool-aware    │
                    └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

1. **Conda environment** `llm_ui` (already created)
2. **Node.js** and npm installed
3. **n8n** installed globally: `npm install -g n8n`
4. **Ollama** running (optional but recommended)

### Quick Start

1. **Start all services**:
   ```bash
   ./start_n8n_integration.sh start
   ```

2. **Access the application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - n8n Interface: http://localhost:5678

3. **Enable tool use**:
   - Go to the Workflows tab
   - Enable "🔧 Enable Tools" in chat
   - Start asking questions that require tools!

### Manual Setup

If you prefer to start services manually:

1. **Start n8n**:
   ```bash
   n8n start
   ```

2. **Start FastAPI backend**:
   ```bash
   conda activate llm_ui
   python main.py
   ```

3. **Start Streamlit frontend**:
   ```bash
   conda activate llm_ui
   streamlit run streamlit_app.py
   ```

## 🔧 Available Tools

### 1. Web Search Tool
- **Purpose**: Search the internet using DuckDuckGo
- **Parameters**:
  - `query` (string): Search query
  - `num_results` (integer): Number of results (default: 5)
- **Example**: "Search for latest AI developments"

### 2. File Operations Tool  
- **Purpose**: Read, write, and manipulate files
- **Parameters**:
  - `operation` (string): "read", "write", or "list"
  - `file_path` (string): Path to the file
  - `content` (string): Content to write (for write operation)
- **Example**: "Read the contents of README.md"

### 3. Data Processing Tool
- **Purpose**: Process and transform data using JavaScript
- **Parameters**:
  - `data` (object): Data to process
  - `operation` (string): "filter", "transform", or "aggregate"
- **Example**: "Process this list and remove duplicates: [1,2,2,3,4,4,5]"

### 4. Email Tool (Configurable)
- **Purpose**: Send emails via SMTP
- **Parameters**:
  - `to` (string): Recipient email
  - `subject` (string): Email subject
  - `message` (string): Email content
  - `smtp_host` (string): SMTP server (default: smtp.gmail.com)
  - `smtp_port` (integer): SMTP port (default: 587)

## 💬 Using Tools in Chat

### Basic Tool Use

1. Enable tools in the chat interface
2. Ask natural language questions that require external data or actions
3. The LLM will automatically call appropriate tools and incorporate results

### Example Conversations

**Web Search**:
```
User: What are the latest developments in AI?
