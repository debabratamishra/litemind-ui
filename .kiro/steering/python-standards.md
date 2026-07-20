---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# Python Coding Standards

Applies whenever a Python file is in context.

## Toolchain
- **Python 3.13** only — no 3.12 or 3.14 syntax
- **uv** for package management — not pip, poetry, or pipenv
- **ruff** for linting and formatting (line-length 120)
- **ty** for type-checking (zero errors on checked paths)
- Run before finishing any Python change:
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run ty check app/backend app/services app/core app/ingestion app/skills main.py config.py logging_config.py
  uv run pytest -x -q
  ```

## Style rules
- Line length: **120 characters**
- Single quotes for strings (unless the string contains a single quote)
- f-strings preferred over `.format()` or `%`
- No wildcard imports (`from module import *`)
- Imports: stdlib → third-party → local, alphabetically within each group

## Type annotations
- Every function/method must have annotated parameters and return type
- Python 3.13 built-in generics: `list[str]`, `dict[str, int]`, `X | Y` — not `List`, `Dict`, `Optional`
- `Any` requires a justifying comment

## Logging — always use the project logger
```python
from logging_config import get_logger
logger = get_logger(__name__)
# NOT: import logging; logging.getLogger(...)
# NOT: print(...)
```

## Error handling
- Catch specific exceptions — not bare `except Exception` unless at a top-level boundary
- FastAPI routes must raise `HTTPException` or return `JSONResponse` — never let exceptions propagate raw
- Include context in exception messages

## Async rules
- Route handlers and I/O service methods must be `async def`
- Blocking calls inside `async def` must use `asyncio.to_thread()` or a process-pool executor
- Use `httpx.AsyncClient` in async contexts — not `requests`

## Configuration
- All runtime config via `config.py` (`Config`) or `app/backend/core/config.py` (`BackendConfig`)
- New env vars must be added to `.env.example` with a comment

## Security (file upload paths)
```python
from app.backend.api.security_utils import sanitize_filename, validate_file_size
# Always sanitise file names and validate sizes before processing uploads
```

## New code goes here
| What | Where |
|------|-------|
| New LLM provider | `app/services/llm_gateway.py` |
| New chat skill | `app/skills/` + register in `ChatSkillRegistry` |
| New RAG skill | `app/skills/` + register in `RAGSkillRegistry` |
| New document format | `app/ingestion/file_ingest.py` |
| New API route | `app/backend/api/` + register router in `main.py` |
| New Pydantic models | `app/backend/models/` |
