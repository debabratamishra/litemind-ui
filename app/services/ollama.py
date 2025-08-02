import httpx
from fastapi import Depends
from config import Config
import json
import re

async def stream_ollama(messages, model="gemma3n:e2b", temperature=0.7):
    """
    Streams RAW Markdown coming from Ollama. Front-end will render it.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{Config.OLLAMA_API_URL}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature}
            }
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk.strip():
                        continue
                    try:
                        data = json.loads(chunk.strip())
                    except json.JSONDecodeError:
                        continue  # skip malformed pieces
                    content = data.get("message", {}).get("content", "")
                    if not content:
                        continue
                    buffer += content
                    # Send partials at natural boundaries or if buffer grows too big
                    if re.search(r"[.!?\n]$", buffer) or data.get("done", False) or len(buffer) > 400:
                        yield buffer
                        buffer = ""
                if buffer:
                    yield buffer
    except (httpx.RequestError, httpx.HTTPStatusError):
        yield (
            "*Ollama service is unavailable. Please ensure it is running at "
            "`http://localhost:11434`.*"
        )
