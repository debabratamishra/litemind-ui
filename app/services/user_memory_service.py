"""
User-level persistent memory: prompt-block formatting, extraction-op parsing,
extraction via the LLM gateway, and the fire-and-forget update orchestration.

All LLM work goes through ``app.services.llm_gateway`` using the request's own
backend/model config, so behavior is identical on Ollama, OpenRouter, and
Nvidia NIM. Extraction failures are logged and skipped — they never affect the
user-visible response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.backend.user_memory_store import get_user_memory_store
from app.services.llm_gateway import complete_text

logger = logging.getLogger(__name__)

MAX_MEMORY_OPS_PER_TURN = 3
MEMORY_EXTRACTION_TIMEOUT_SECONDS = 15
MAX_MEMORY_CONTENT_LENGTH = 500

MEMORY_BLOCK_HEADER = "About the user (persistent memory; use when relevant):"


def build_memory_block(memories: List[Any]) -> str:
    """Format memory records into a system-prompt block ("" when none)."""
    contents = [getattr(m, "content", "") for m in memories if getattr(m, "content", "")]
    if not contents:
        return ""
    lines = [MEMORY_BLOCK_HEADER] + [f"- {c}" for c in contents]
    return "\n".join(lines)


def _valid_op(item: Any) -> Optional[Dict[str, Any]]:
    """Return a normalized op dict, or None if the item is malformed."""
    if not isinstance(item, dict):
        return None
    op = item.get("op")
    if op == "add":
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return {"op": "add", "content": content.strip()[:MAX_MEMORY_CONTENT_LENGTH]}
        return None
    if op in ("update", "delete"):
        memory_id = item.get("id")
        if not isinstance(memory_id, str) or not memory_id:
            return None
        try:
            uuid.UUID(memory_id)
        except ValueError:
            return None  # store would raise on a mangled id; drop just this op
        if op == "delete":
            return {"op": "delete", "id": memory_id}
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return {"op": "update", "id": memory_id, "content": content.strip()[:MAX_MEMORY_CONTENT_LENGTH]}
        return None
    return None


def parse_memory_ops(raw: str) -> List[Dict[str, Any]]:
    """Parse the extraction LLM's output into validated ops (max 3)."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Memory extraction returned non-JSON output, skipping")
        return []
    if not isinstance(data, list):
        return []
    ops = []
    for item in data:
        op = _valid_op(item)
        if op is not None:
            ops.append(op)
        if len(ops) >= MAX_MEMORY_OPS_PER_TURN:
            break
    return ops


_EXTRACTION_SYSTEM_PROMPT = """You maintain long-term memory for a personal assistant. \
Given a conversation exchange and the user's existing memories, decide what durable facts about the \
user should be stored, updated, or removed.

Return ONLY a JSON array. Each element is exactly one of:
{"op": "add", "content": "<new memory, one concise sentence>"}
{"op": "update", "id": "<existing memory id>", "content": "<corrected memory>"}
{"op": "delete", "id": "<existing memory id that is now wrong or obsolete>"}

Rules:
- Store only durable facts: the user's identity, stable preferences, ongoing projects or goals, \
corrections to existing memories, and explicit requests to remember something.
- Never store secrets, credentials, API keys, passwords, or transient task details.
- Prefer updating an existing memory over adding a near-duplicate.
- At most 3 operations. Return [] when nothing durable was shared."""


def _build_extraction_messages(
    user_message: str, assistant_message: str, existing: List[Any]
) -> List[Dict[str, str]]:
    existing_lines = "\n".join(
        f'- id={getattr(m, "id", "?")}: {getattr(m, "content", "")}' for m in existing
    )
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Existing memories:\n{existing_lines or '(none)'}\n\n"
                f"User said:\n{user_message}\n\n"
                f"Assistant replied:\n{assistant_message}"
            ),
        },
    ]


async def extract_memory_ops(
    user_message: str,
    assistant_message: str,
    existing: List[Any],
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask the LLM which memories to add/update/delete; [] on any failure."""
    if not user_message.strip():
        return []
    messages = _build_extraction_messages(user_message, assistant_message, existing)
    try:
        raw = await asyncio.wait_for(
            complete_text(
                messages,
                backend=backend,
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=300,
            ),
            timeout=MEMORY_EXTRACTION_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 — extraction must never raise
        logger.warning("Memory extraction failed/skipped: %s", exc)
        return []
    return parse_memory_ops(raw)


async def apply_memory_ops(store: Any, user_id: str, ops: List[Dict[str, Any]]) -> None:
    """Apply validated ops to the store; unknown ids are ignored by the store.

    Each op is isolated so one failure (transient DB error, race-deleted id)
    cannot abort the remaining ops in the batch.
    """
    for op in ops:
        try:
            if op["op"] == "add":
                await store.add_memory(user_id, op["content"], source="auto")
            elif op["op"] == "update":
                await store.update_memory(user_id, op["id"], op["content"])
            elif op["op"] == "delete":
                await store.delete_memory(user_id, op["id"])
        except Exception as exc:
            logger.warning("Memory op %s failed (skipped): %s", op["op"], exc)


async def run_memory_update(
    user_id: str,
    user_message: str,
    assistant_message: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Extract and apply memory ops for one completed exchange. Never raises."""
    try:
        store = get_user_memory_store()
        existing = await store.list_memories(user_id)
        ops = await extract_memory_ops(
            user_message,
            assistant_message,
            existing,
            backend=backend,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
        await apply_memory_ops(store, user_id, ops)
        if ops:
            logger.info("User memory updated for user %s (%d ops)", user_id, len(ops))
    except Exception as exc:  # noqa: BLE001 — fire-and-forget task must be self-contained
        logger.warning("User memory update failed for user %s: %s", user_id, exc)


async def load_memory_block(user_id: Optional[str]) -> str:
    """Load and format the user's memories for prompt injection ("" on failure)."""
    if not user_id:
        return ""
    try:
        store = get_user_memory_store()
        memories = await store.list_memories(user_id)
        return build_memory_block(memories)
    except Exception as exc:  # noqa: BLE001 — degraded response beats failed response
        logger.warning("Failed to load user memory for user %s: %s", user_id, exc)
        return ""
