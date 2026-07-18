# Temporal Awareness via a Callable Time Tool

**Date:** 2026-07-18
**Branch:** `feat/nextjs-frontend`
**Status:** Approved design (pending implementation plan)

## 1. Goal & Behavior

Make LiteMindUI temporally aware so the LLM can answer realtime / "current time"
questions correctly. Instead of hard-coding the date into a prompt, we expose a
`get_current_time` **tool** the model calls on demand:

1. The user asks something time-dependent ("What's the date today?", "How many days
   until July 4th?", "Is it morning where I am?").
2. The model decides it needs the current time and emits a tool call.
3. We execute `get_current_time` **locally** (no network) and feed the result back.
4. The model answers using the real, current time.

The time always comes from the system clock. In both standalone (`uv run`) and Docker
the process reads the same kernel clock (Docker does not virtualize time), so the value
is identical; what differs is **timezone**. We resolve timezone from the local system,
with an optional `TZ` environment override so a Docker container can be aligned to the
host's zone.

**Scope (this change):** chat + RAG answer generation only.
**Out of scope:** voice mode (Pipecat LLM leg), web-search temporal grounding, user-
configurable tools, additional tools (date math, etc.). The registry design below makes
adding those later trivial.

## 2. New Module — `app/services/time_service.py`

Single source of truth for time resolution and the tool definition.

### 2.1 Runtime mode detection
```python
def detect_runtime_mode() -> Literal["docker", "standalone"]:
    """Reuse the existing Config container detection."""
    from config import Config
    return "docker" if Config._detect_container_environment() else "standalone"
```
Note: `Config._detect_container_environment()` already checks `/.dockerenv`, the cgroup
docker marker, and `CONTAINER` / `DOCKER_CONTAINER` env vars.

### 2.2 Current time resolver
```python
def get_current_time(timezone: Optional[str] = None) -> dict:
    """Return the current local time, optionally in a named timezone.

    Resolution order: explicit `timezone` arg -> TZ env var -> system local tz.
    Returns a dict with machine-readable + human-readable fields.
    """
```
Return shape:
| key | example | meaning |
|-----|---------|---------|
| `iso` | `2026-07-18T14:30:45+05:30` | ISO 8601 with UTC offset |
| `human` | `Saturday, July 18, 2026, 2:30 PM IST` | human-readable local time |
| `timezone` | `Asia/Kolkata` / `UTC` | resolved IANA zone or `UTC` |
| `utc_offset` | `+05:30` | offset from UTC |
| `runtime_mode` | `docker` / `standalone` | where the process runs |

Resolution rules:
- `tz = timezone or os.getenv("TZ") or None`.
- If `tz` is `None`: use `datetime.now().astimezone()` (system local tz).
- If `tz` is set: `datetime.now(ZoneInfo(tz))`.
- Unknown / invalid `tz` (e.g. `ZoneInfoNotFoundError`): log a warning and fall back to
  system local tz. Never raise to the caller.

`human` is formatted with `strftime("%A, %B %d, %Y, %I:%M %p")` plus the zone abbrev
(`%Z`). `utc_offset` from `utcoffset()` as `+HH:MM`.

### 2.3 Tool schema + registry
```python
TIME_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time. Call this whenever the user's "
                       "question depends on the current date, day, time, or timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone, e.g. 'America/New_York'. "
                                   "Omit to use the system's local timezone.",
                }
            },
            "required": [],
        },
    },
}

TIME_TOOLS = [TIME_TOOL_SPEC]
TOOL_EXECUTORS = {"get_current_time": get_current_time}
```
`timezone` is validated/defaulted inside `get_current_time`, so the executor tolerates
missing or bad args.

## 3. Gateway — `complete_with_tools` in `app/services/llm_gateway.py`

New coroutine handling both backends. The tool-decision turns are non-streaming; only
the final answer streams to the client (standard, and avoids fragile in-stream tool-call
parsing).

```python
async def complete_with_tools(
    messages: Iterable[dict],
    *,
    tools: list[dict],
    tool_executors: dict[str, Callable],
    backend: Optional[str] = "ollama",
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    top_k: int = 40,
    min_p: float = 0.0,
    seed: Optional[int] = None,
    stop: Optional[list[str]] = None,
    max_tool_rounds: int = 3,
) -> AsyncGenerator[str, None]:
    ...
```

Algorithm:
1. `message_list = list(messages)`.
2. For `round in range(max_tool_rounds)`:
   - Get a single, non-streaming assistant message from the model **with tools attached**:
     - **Ollama:** add helper `_ollama_chat_completion(messages, tools, …)` using
       `client.chat(model=…, messages=…, tools=tools, stream=False)`; returns
       `response.message` (has `.content` and `.tool_calls`).
     - **LiteLLM:** `litellm.acompletion(..., tools=tools, stream=False)` →
       `response.choices[0].message`.
   - **Normalize** tool-call shapes: Ollama returns `arguments` as a **dict**; LiteLLM
     returns it as a **JSON string**. Convert both to a parsed dict before execution and
     to a JSON string when re-attaching the assistant message.
   - If the message has no `tool_calls`: break out of the loop (this is the final answer
     turn — its `content` will be streamed in step 3).
   - Else: for each tool call, look up `tool_executors[function.name]`, call it with the
     parsed args, capture the result dict, `json.dumps` it. Append:
     - the assistant message (with normalized `tool_calls`) to `message_list`, and
     - a `{"role": "tool", "tool_call_id": <id>, "name": <name>, "content": <json>}`
       message. Continue to the next round.
3. Stream the final answer with the existing `stream_completion(message_list, …)`.
4. **Fallbacks:**
   - If the non-streaming tool-decision call raises (model/backend without tool support),
     fall back to `stream_completion(messages, …)` **without** tools and return.
   - If `max_tool_rounds` is exhausted, do a final `stream_completion(message_list, …)`
     call **without** `tools` to force a text answer (prevents infinite loops / repeated
     tool calls).

`_ollama_chat_completion` reuses the option-building logic already in
`_stream_ollama_native` (temperature, top_p, num_predict, repeat_penalty, seed, stop) but
with `stream=False` and `tools=tools`.

## 4. Wiring

### 4.1 Chat — `app/backend/api/chat.py`
`_stream_chat_response` (≈ lines 310–339) currently calls `stream_completion(...)`.
Replace that call with `complete_with_tools(..., tools=TIME_TOOLS,
tool_executors=TOOL_EXECUTORS)`, forwarding the same backend/model/temperature/etc.
args. **Non-voice chat only** — voice mode uses the Pipecat LLM leg and is untouched.
Import `complete_with_tools` from `app.services.llm_gateway` alongside the existing
`stream_completion` / `complete_text` import (line 24).

### 4.2 RAG — `app/services/rag_service.py`
The RAG answer generator (≈ line 1650) calls `stream_completion(llm_messages, …)` to
produce the grounded answer. Replace with `complete_with_tools(..., tools=TIME_TOOLS,
tool_executors=TOOL_EXECUTORS)`. The retrieved-context system prompt is unchanged; the
tool is merely *available* if the model needs "now" to interpret the query. The
`citations` JSON line emitted before the stream (line 1645–1648) is unchanged.

### 4.3 Request models
No new fields in `api_models.py` / `main.py`. Tools are server-side and not user-supplied.

## 5. Config & Docker

- `get_current_time` reads `os.getenv("TZ")` directly — no `config.py` change required
  beyond reusing `Config._detect_container_environment()` for `runtime_mode`.
- **docker-compose.yml, docker-compose.dev.yml, docker-compose.prod.yml:** add
  `TZ: ${TZ:-UTC}` to the **backend** service `environment:` block. User sets
  `TZ=Asia/Kolkata` in `.env` to align the container with their host zone.
- **docker-compose.dev.yml (best-effort):** optionally mount
  `/etc/localtime:/etc/localtime:ro` and `/etc/timezone:/etc/timezone:ro` so the dev
  container matches the host clock/timezone exactly (no `TZ` needed). Keep this dev-only.
- **.env.example:** add a commented `TZ=` line with a short note pointing at the IANA
  timezone database.

## 6. Error Handling

| Condition | Behavior |
|-----------|----------|
| Invalid / unknown `TZ` or tool `timezone` arg | Log warning, fall back to system local tz. Never raise. |
| Tool executor raises | Return a short error string as the tool `content` so the model can recover / respond. |
| `max_tool_rounds` exceeded | Final answer streamed with tools omitted (no infinite loop). |
| Model/backend lacks tool support (tool call raises) | Fall back to `stream_completion` without tools. |
| LLM / backend config error | Preserve existing `_build_error_message` behavior. |

## 7. Testing

- **`tests/test_time_service.py`**
  - `detect_runtime_mode()` returns `"docker"` when `/.dockerenv` exists (use tmp path /
    monkeypatch `Config._detect_container_environment`), `"standalone"` otherwise.
  - `get_current_time()` returns all five keys with correct types; `iso` matches
    `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$`; `runtime_mode` set.
  - `TZ` env override produces the requested zone (set `TZ=America/New_York`, assert
    `timezone`/`utc_offset`).
  - Invalid `TZ` (e.g. `TZ=Not/AZone`) → warning logged, falls back to local tz (does
    not raise).
  - Explicit `timezone=` arg overrides env.
- **`tests/test_llm_gateway_tools.py`**
  - Fake LLM returns a `tool_calls` message → executor invoked with parsed args → final
    answer streamed. Assert executor called once with expected args and that streamed
    text equals the fake's final content.
  - Fake LLM returns no tool call → `complete_with_tools` streams directly (no executor
    call).
  - `max_tool_rounds` cap → executor not called more than the cap; final answer still
    produced.
  - Mock `litellm.acompletion` and the Ollama `AsyncClient.chat` to drive both paths
    without a real model.
- Quality gates before "done": `uv run ruff check .`, `uv run ty check app/backend
  app/services app/core app/ingestion app/skills main.py config.py
  logging_config.py`, `uv run pytest`.

## 8. Out of Scope / Follow-ups

- Voice-mode temporal awareness (Pipecat `BaseOpenAILLMService` leg).
- Web-search temporal grounding.
- User-configurable / toggleable tools.
- Additional tools (date math, "day of week", scheduling helpers).
The `TIME_TOOLS` / `TOOL_EXECUTORS` registry makes adding any of these a small,
localized change.
