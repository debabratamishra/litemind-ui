---
inclusion: always
---

# Security Rules

These apply to every change in this repository without exception.

## Secrets and credentials
- **Never** store secrets, API keys, or tokens in source code or committed config files
- `.env` is git-ignored — never commit it
- New environment variables go in `.env.example` with a placeholder value and a description comment
- `SECRET_KEY` in `.env.example` is a placeholder — always remind users to rotate it before production
- `NEXT_PUBLIC_*` env vars are baked into the client bundle — **never put secrets here**

## File upload handling
Every upload path must call both helpers from `app/backend/api/security_utils.py`:
```python
from app.backend.api.security_utils import sanitize_filename, validate_file_size

safe_name = sanitize_filename(original_filename)
validate_file_size(file_size_bytes)  # raises HTTPException on violation
```
Never use a raw user-supplied file name for disk paths.

## Input validation
- Validate and sanitise at the Pydantic model boundary — do not re-validate downstream
- Use parameterised queries for all database operations (SQLAlchemy ORM or `?` placeholders)
- Never concatenate user input into SQL strings, shell commands, or file paths

## HTTP responses
- Never return raw Python exceptions or stack traces in HTTP responses
- Use `HTTPException` with appropriate status codes and sanitised detail messages
- Do not expose internal module paths, library versions, or server internals in error responses

## CORS
- CORS origins are configured in `main.py` — do not widen the allowed origins list without explicit approval
- In development, `*` is acceptable; in production it must be a specific origin allowlist

## Logging
- Never log secrets, API keys, or raw user PII
- Log the *type* and *shape* of data, not the values when they may be sensitive

## Dangerous Python patterns — permanently forbidden
```python
eval(user_input)          # forbidden
exec(user_input)          # forbidden
pickle.loads(user_data)   # forbidden — use json.loads
subprocess with shell=True and user input  # forbidden — use list form
```

## Next.js / frontend
- Never expose backend secrets via `NEXT_PUBLIC_*` variables
- Never pass raw server error messages to the UI — map them to user-friendly strings
- Content Security Policy headers are set by Next.js config — do not disable them
