# LiteMindUI Backend — API Contract

This document is the single source of truth for the LiteMindUI FastAPI backend
HTTP interface. Any frontend (Next.js, TUI, VS Code extension, mobile app, …)
that targets this backend should implement its own thin HTTP client against this
contract rather than sharing Python code.

**Base URL:** `http://localhost:8000` (configurable via `FASTAPI_URL` env var)  
**Auth:** None required for local deployments. Pass provider API keys per-request in the body (see below).  
**Content-Type:** `application/json` for all request bodies unless noted.

---

## Table of contents

1. [Health](#1-health)
2. [Models](#2-models)
3. [Chat](#3-chat)
4. [RAG](#4-rag)
5. [Inference backends](#5-inference-backends)
6. [Streaming response formats](#6-streaming-response-formats)
7. [Error handling](#7-error-handling)
8. [Common field reference](#8-common-field-reference)

---

## 1. Health

### `GET /health`

Basic liveness check. Returns immediately with no external dependency checks.

**Response `200`**
```json
{ "status": "healthy" }
```

### `GET /health/ready`

Readiness check. Verifies RAG service initialisation and required directories.

**Response `200`** — ready
```json
{
  "status": "ready",
  "timestamp": 1720000000.0,
  "checks": {
    "rag_service": { "status": "ready" },
    "uploads":     { "status": "ready", "path": "/app/uploads" },
    "storage":     { "status": "ready", "path": "/app/storage" }
  }
}
```

**Response `503`** — not ready (same shape, `status` is `"not_ready"` or `"error"`)

---

## 2. Models

### `GET /models`

Returns the names of all locally installed Ollama models.

**Response `200`**
```json
{ "models": ["llama3.2:latest", "mistral:7b", "gemma2:9b"] }
```

> Falls back to `["default"]` if Ollama is unreachable.

---

### `GET /models/enhanced`

Returns local models with metadata **plus** a curated cloud model catalog.
Cloud models are those in the catalog that are **not** already installed locally.

**Response `200`**
```json
{
  "local_models": [
    {
      "name": "llama3.2:latest",
      "parameter_size": "3B",
      "quantization": "Q4_K_M",
      "family": "llama",
      "is_local": true,
      "description": null
    }
  ],
  "cloud_models": [
    {
      "name": "deepseek-v4-flash:cloud",
      "parameter_size": "284B",
      "family": "deepseek-v4-flash",
      "is_local": false,
      "description": "DeepSeek-V4-Flash — MoE, 1M token context"
    }
  ]
}
```

> Cloud model names follow Ollama's naming convention and require an Ollama
> cloud subscription to pull and run.

---

## 3. Chat

All chat endpoints share the same request body shape (`ChatRequest`).

### ChatRequest fields

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `string` | **required** | The user's message |
| `model` | `string` | `"default"` | Model name (see §5 for per-backend naming) |
| `backend` | `string` | `"ollama"` | Provider: `ollama` · `openrouter` · `nvidia_nim` |
| `api_base` | `string\|null` | `null` | Override provider base URL |
| `api_key` | `string\|null` | `null` | Provider API key (required for OpenRouter/NIM) |
| `temperature` | `float` | `0.7` | Sampling temperature `[0.0, 2.0]` |
| `max_tokens` | `int` | `2048` | Max tokens to generate |
| `top_p` | `float` | `0.9` | Nucleus sampling `[0.0, 1.0]` |
| `frequency_penalty` | `float` | `0.0` | Penalise frequent tokens `[-2.0, 2.0]` |
| `repetition_penalty` | `float` | `1.0` | Penalise repeated tokens `[0.0, 2.0]` |
| `session_id` | `string\|null` | `null` | UUID for conversation memory tracking |
| `conversation_history` | `ChatMessage[]\|null` | `null` | Previous turns `[{role, content}]` |
| `conversation_summary` | `string\|null` | `null` | Summarised older context |
| `is_voice_mode` | `bool` | `false` | Short responses optimised for TTS |
| `enable_generative_ui` | `bool` | `false` | Allow `ui:*` fenced blocks in response |
| `use_web_search` | `bool` | `false` | Ground response with web search (requires SerpAPI key) |

**ChatMessage shape**
```json
{ "role": "user", "content": "Hello" }
```
`role` is one of `"user"`, `"assistant"`, `"system"`.

---

### `POST /api/chat/stream`

Streaming chat response.

**Request body:** `ChatRequest`

**Response:** `text/event-stream` — SSE stream

Each event:
```
data: {"chunk": "Hello"}\n\n
data: {"chunk": " world"}\n\n
```

Clients must strip `data: `, JSON-parse the object, and extract `chunk`.
See [§6 Streaming formats](#6-streaming-response-formats) for a reference parser.

---

### `POST /api/chat`

Non-streaming chat. Returns the complete response in one go.

**Request body:** `ChatRequest`

**Response `200`**
```json
{ "response": "Hello, how can I help?", "model": "llama3.2:latest" }
```

---

### `POST /api/chat/web-search`

Streaming chat grounded with live web search results (SerpAPI).
Falls back to standard chat if SerpAPI is unavailable or key is missing.

**Request body:** `ChatRequest` (set `use_web_search: true`)

**Response:** `text/plain` — plain-text stream (no SSE envelope)

---

## 4. RAG

### RAGQueryRequest fields

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `string` | **required** | The user's question |
| `messages` | `dict[]` | `[]` | Conversation history `[{role, content}]` |
| `model` | `string` | `"default"` | LLM model name |
| `backend` | `string` | `"ollama"` | Provider (same values as chat) |
| `api_base` | `string\|null` | `null` | Provider base URL override |
| `api_key` | `string\|null` | `null` | Provider API key |
| `system_prompt` | `string` | `"You are a helpful assistant."` | System instruction |
| `n_results` | `int` | `3` | Number of document chunks to retrieve |
| `use_multi_agent` | `bool` | `false` | Multi-agent RAG pipeline |
| `use_hybrid_search` | `bool` | `false` | BM25 + vector hybrid retrieval |
| `temperature` | `float` | `0.7` | |
| `max_tokens` | `int` | `2048` | |
| `top_p` | `float` | `0.9` | |
| `frequency_penalty` | `float` | `0.0` | |
| `repetition_penalty` | `float` | `1.0` | |
| `session_id` | `string\|null` | `null` | |
| `conversation_summary` | `string\|null` | `null` | |
| `is_voice_mode` | `bool` | `false` | |

---

### `POST /api/rag/query`

Streaming RAG response.

**Request body:** `RAGQueryRequest`

**Response:** `text/plain` — plain-text stream (no SSE envelope, unlike `/api/chat/stream`)

The stream opens with a single transport-metadata frame, then the answer prose:

```
data: {"citations": {"1": {"id": "chunk1", "content": "...", "score": 0.91, "retrieval_method": "semantic", "metadata": {"filename": "doc.pdf", "page_number": 3}}, "2": {...}}}\n\n
According to the document [1], the cat sat on the mat.
```

- The `data: {"citations": {...}}` line is **metadata, not answer text**. Each key is a
  1-based citation index; the value carries the retrieved chunk (`content`), its
  `retrieval_method` (`semantic` / `bm25` / `hybrid`), a normalised `score`, and a
  `metadata` object (typically `filename`, optionally `page_number`).
- The answer prose that follows cites sources with bracketed numbers (`[1]`, `[2]`, …).
  The frontend strips the `data:` frame and renders those markers as clickable
  source chips plus a "Sources (N)" dialog. **Clients must ignore the `data:` frame**
  (do not display it as answer text).

---

### `POST /api/rag/upload`

Upload files for ingestion into the vector store.

**Request:** `multipart/form-data`

| Part | Type | Description |
|---|---|---|
| `files` | file (repeatable) | One or more files to ingest |
| `chunk_size` | int (form field) | Token chunk size, default `500` |

Supported file types: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.xlsx`, `.pptx`, `.html`, `.htm`, `.odt`, `.rtf`, `.yaml`, `.json`

**Response `200`**
```json
{
  "status": "completed",
  "summary": {
    "total_files": 2,
    "successful": 2,
    "duplicates": 0,
    "errors": 0,
    "total_chunks_created": 47
  },
  "results": [
    { "filename": "doc.pdf", "status": "success", "message": "Processed doc.pdf", "chunks_created": 47 }
  ]
}
```

Possible `status` values per result: `"success"` · `"duplicate"` · `"error"`

---

### `GET /api/rag/files`

List all indexed files.

**Response `200`**
```json
{
  "files": [
    { "filename": "doc.pdf", "size": 12345, "chunks": 47 }
  ]
}
```

---

### `DELETE /api/rag/files/{filename}`

Remove a file from the index.

**Response `200`**
```json
{ "message": "Deleted 'doc.pdf'.", "filename": "doc.pdf" }
```

---

### `POST /api/rag/reset`

Wipe the entire vector store and remove all uploaded files.

**Response `200`**
```json
{ "status": "success", "message": "RAG system reset. Removed 3 files.", "files_removed": 3 }
```

---

### `GET /api/rag/status`

Current RAG system status.

**Response `200`**
```json
{
  "status": "ready",
  "uploaded_files": 3,
  "indexed_chunks": 142,
  "bm25_corpus_size": 142
}
```

---

### `POST /api/rag/save_config`

Update the embedding model configuration. Takes effect immediately for subsequent uploads.

**Request body**
```json
{
  "provider": "ollama",
  "embedding_model": "nomic-embed-text",
  "embedding_backend": null,
  "embedding_api_base": null,
  "embedding_api_key": null,
  "chunk_size": 500
}
```

`provider` values: `"ollama"` · `"openrouter"` · `"nvidia_nim"` · `"huggingface"`

---

### `POST /api/rag/check-duplicates`

Check whether a file has already been indexed before uploading.

**Request:** `application/json`
```json
{ "filename": "doc.pdf" }
```

**Response `200`**
```json
{ "is_duplicate": false, "filename": "doc.pdf", "message": "" }
```

---

## 5. Inference backends

The `backend` field routes to a provider via LiteLLM. The `model` value format
differs per provider.

### Ollama (local)

```json
{ "backend": "ollama", "model": "llama3.2:latest" }
```

- `api_base` defaults to `http://localhost:11434` (native) or `http://host.docker.internal:11434` (Docker)
- `api_key` not required

Model names come from `GET /models` or `GET /models/enhanced`.

---

### OpenRouter

```json
{
  "backend": "openrouter",
  "model": "openai/gpt-4o",
  "api_key": "sk-or-...",
  "api_base": "https://openrouter.ai/api/v1"
}
```

`api_base` defaults to `https://openrouter.ai/api/v1` if omitted.

Model name format: `provider/model-name` — see [openrouter.ai/models](https://openrouter.ai/models).

Common models:
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `anthropic/claude-3.5-sonnet`
- `meta-llama/llama-3.3-70b-instruct`
- `google/gemini-2.0-flash-001`

---

### Nvidia NIM

```json
{
  "backend": "nvidia_nim",
  "model": "meta/llama-3.3-70b-instruct",
  "api_key": "nvapi-...",
  "api_base": "https://integrate.api.nvidia.com/v1"
}
```

`api_base` defaults to `https://integrate.api.nvidia.com/v1` if omitted.

> **Note:** The CLI sends `"nim"` as the backend value; the backend normalises
> `"nim"` → `"nvidia_nim"` internally.

---

## 6. Streaming response formats

Two formats are used — **endpoints are consistent**, not mixed:

### SSE (Server-Sent Events) — `/api/chat/stream` only

```
data: {"chunk": "Hello"}\n\n
data: {"chunk": ", how"}\n\n
data: {"chunk": " can I help?"}\n\n
```

Reference parser (Python):
```python
def parse_sse_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("data:"):
        line = line[len("data:"):].strip()
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj.get("chunk", "")
    except json.JSONDecodeError:
        pass
    return line  # plain-text fallback
```

### Plain text stream — all other streaming endpoints

`/api/chat/web-search`, `/api/rag/query`

Raw text chunks, no envelope. Concatenate as received.

---

## 7. Error handling

| HTTP status | Meaning |
|---|---|
| `200` | Success |
| `400` | Bad request (invalid field values, unsupported backend) |
| `503` | Service unavailable (RAG not initialised, Ollama unreachable) |
| `500` | Internal server error |

Error body shape:
```json
{ "detail": "Could not fetch models: connection refused" }
```

Streaming endpoints do not return HTTP error codes mid-stream. On failure they
emit a final plain-text error message in the stream itself.

---

## 8. Common field reference

### Conversation memory

Pass these fields on every turn to maintain context across messages:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "conversation_history": [
    { "role": "user",      "content": "What is RAG?" },
    { "role": "assistant", "content": "RAG stands for..." }
  ],
  "conversation_summary": "User asked about RAG. Assistant explained retrieval-augmented generation."
}
```

- `session_id` — stable UUID per user session; used for server-side memory stats
- `conversation_history` — last N turns to send as context (client manages what to include)
- `conversation_summary` — compressed summary of older turns beyond the context window

Memory stats and management:

| Endpoint | Description |
|---|---|
| `GET /api/chat/memory/stats/{session_id}` | Token counts, summary status |
| `POST /api/chat/memory/clear/{session_id}` | Clear session memory |
| `POST /api/chat/memory/summarize/{session_id}` | Force summarisation |

### Generation parameters

| Parameter | Range | Notes |
|---|---|---|
| `temperature` | `0.0 – 2.0` | Lower = more deterministic |
| `max_tokens` | `1 – model limit` | Hard cap on output length |
| `top_p` | `0.0 – 1.0` | Nucleus sampling |
| `frequency_penalty` | `-2.0 – 2.0` | Penalise token frequency |
| `repetition_penalty` | `0.0 – 2.0` | Penalise repetition (Ollama) |
