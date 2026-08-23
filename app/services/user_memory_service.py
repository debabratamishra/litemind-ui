"""
User-level persistent memory: prompt-block formatting, extraction-op parsing,
extraction via the LLM gateway, and the fire-and-forget update orchestration.

All LLM work goes through ``app.services.llm_gateway`` using the request's own
backend/model config, so behavior is identical on Ollama, OpenRouter, and
Nvidia NIM. Extraction failures are logged and skipped — they never affect the
user-visible response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

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
