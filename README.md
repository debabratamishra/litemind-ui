# LLM WebUI

A simple and efficient web-based interface for working with Large Language Models (LLMs), built entirely on the Python stack. Designed for developers, researchers, and AI enthusiasts seeking a minimal yet powerful front-end to interact with LLMs locally.

---

## ✨ Features

- ⚡ Lightweight and fast interface for LLM interaction  
- 🧩 Pythonic architecture—easy to extend or customize  
- 📦 Dependency-managed with [`uv`](https://github.com/astral-sh/uv) for reproducible installs  
- 💻 Runs locally, no external API or cloud dependency  
- 🌐 Web-based interface accessible at `localhost:5100`

---

## Installation

1. git clone https://github.com/debabratamishra/llm-webui
2. cd llm-webui
3. uv pip install -r requirements.txt
4. Launch the WebUI by running the application :
```python
python run.py
```
5. Then navigate to http://localhost:5100

---

To Do : 
1. Advanced RAG implementation using hybrid search handling images + text using local LLM
2. Support for Local LLM and Third party models(e.g. OpenAI GPT series models, Google Gemini series models, etc.)
