# LLM WebUI

A robust, production-ready web interface for Large Language Models (LLMs) featuring a hybrid architecture with FastAPI backend and Streamlit frontend. Built for developers, researchers, and AI enthusiasts who need a comprehensive platform for LLM interaction, document processing, and API integration.

## 🚀 Architecture

**Hybrid Design** - Combines the best of both worlds:

- **FastAPI Backend** (`localhost:8000`) - Entry point at `main.py` with high-performance async API and comprehensive endpoints suitable for asynchronous workload
- **Streamlit Frontend** (`localhost:8501`) - Entry point at `streamlit_app.py` with intuitive web interface and automatic backend detection
- **Modular Services** - Includes `rag_service.py`, `ollama.py`, `file_ingest.py`, `enhanced_extractors.py`, and `enhanced_document_processor.py` for specialized functionalities
- **Intelligent Fallback** - Seamlessly switches between FastAPI and local processing based on backend availability

---

## ✨ Key Features

### 🔧 **Core Capabilities**

- ⚡ **High-Performance API** - Async FastAPI backend for scalable LLM processing
- 🧠 **Multi-Model Support** - Dynamic model detection from Ollama backend handled via `ollama.py`
- 📚 **RAG Integration** - Upload documents (PDFs, DOCX, TXT) using enhanced extraction and preprocessing modules (`enhanced_extractors.py`, `enhanced_document_processor.py`) and query with context-aware responses
- 🔄 **Auto-Failover** - Intelligent backend detection with graceful fallbacks

### 🛠 **Developer Experience**

- 📖 **Auto-Generated API Docs** - Interactive Swagger UI at `/docs`
- 🌐 **RESTful Endpoints** - Complete API for chat, RAG, and model management
- 🐍 **Pure Python Stack** - Easy to extend, customize, and deploy with modular Python files for RAG and LLM interaction
- 📦 **Dependency Management** - Reproducible installs with [`uv`](https://github.com/astral-sh/uv)

### 🔒 **Production Ready**

- 🏠 **Local-First** - Runs entirely on localhost, no external dependencies
- 🔐 **CORS Configured** - Proper cross-origin resource sharing setup
- ⚙️ **Health Monitoring** - Built-in health checks and status monitoring
- 📊 **Streaming Support** - Real-time response streaming capabilities

---

## 📋 Prerequisites

- Python 3.12+ (Should support earlier versions of Python, but it has been developed and tested with this version onwards)
- Ollama - Running locally on `localhost:11434`
- UV Package Manager - For dependency management

---

## 🛠 Installation

1. **Clone the repository**

```bash
git clone https://github.com/debabratamishra/llm-webui
cd llm-webui
```

2. **Install dependencies**

```bash
uv pip install -r requirements.txt
```

3. **Create required directories**

```bash
mkdir -p uploads .streamlit
```

- The `UPLOAD_FOLDER` can be customized via environment variables.

## 🚀 Quick Start

### 1. Start the FastAPI Backend

```bash
python main.py
```

- The main API routes are registered in `main.py`.

**Expected Output:**

```bash
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [^3687] using WatchFiles
INFO:     Started server process [^3707]
INFO:     Waiting for application startup.
🚀 LLM WebUI API starting up...
📁 Upload folder: uploads
📚 RAG service ready
💬 Chat service ready
INFO:     Application startup complete.
```

### 2. Launch the Streamlit Frontend

```bash
streamlit run streamlit_app.py --server.address localhost --server.port 8501
```

- `streamlit_app.py` integrates backend detection logic for FastAPI.

**Expected Output:**

```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

### 3. Access Your Application

- **🖥 Web Interface:** http://localhost:8501
- **📖 API Documentation:** http://localhost:8000/docs
- **🔍 Alternative Docs:** http://localhost:8000/redoc

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :-- | :-- | :-- |
| `/health` | GET | Backend health check |
| `/models` | GET | Available Ollama models |
| `/api/chat` | POST | Process chat messages leveraging `ollama.py` for LLM interactions |
| `/api/chat/stream` | POST | Streaming chat responses leveraging `ollama.py` for LLM interactions |
| `/api/rag/upload` | POST | Upload documents for RAG using `file_ingest.py` and `enhanced_document_processor.py` internally |
| `/api/rag/query` | POST | Query uploaded documents routed through `rag_service.py` for vector search and contextual response |
| `/api/rag/documents` | GET | List uploaded documents |

## 💡 Usage Examples

### Chat Interface

1. Navigate to the **Chat** tab
2. Select your preferred model from the dropdown
3. Enter your message and receive AI responses

### Document Q\&A (RAG)

1. Switch to the **RAG** tab
2. Upload PDF, TXT, or DOCX files
3. Query your documents with natural language
4. Get contextually relevant answers

### API Integration

```python
import requests

# Chat with the API
response = requests.post("http://localhost:8000/api/chat", 
    json={"message": "Hello, world!", "model": "llama3.1"})
print(response.json()["response"])
```

## 🔧 Configuration

### Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[server]
address = "localhost"
port = 8501
```

### Environment Variables

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export UPLOAD_FOLDER="./uploads"
```

## 🎯 Advanced Features

- **Backend Detection:** Automatic FastAPI availability checking with local fallback
- **Dynamic Models:** Real-time model list fetching from Ollama backend
- **Streaming Responses:** Real-time token streaming for better UX
- **Document Processing:** Multi-format document ingestion and vectorization performed at ingestion for faster retrieval
- **Enhanced Extraction:** Enhanced text extraction for PDFs/DOCX via `enhanced_extractors.py`
- **Error Handling:** Comprehensive error handling with user-friendly messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request
