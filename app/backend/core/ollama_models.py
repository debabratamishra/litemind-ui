"""Shared Ollama model catalog and helpers."""

from typing import Any

import httpx


# Single source of truth for curated cloud models shown in the UI.
OLLAMA_CLOUD_CATALOG: list[dict[str, str]] = [
    {
        "name": "gemma4:31b-cloud",
        "description": "Google Gemma 4 - 31B multimodal (vision + text)",
        "parameter_size": "31B",
        "family": "gemma4",
    },
    {
        "name": "glm-5.1:cloud",
        "description": "GLM-5.1 - Z.AI flagship model for agentic engineering and stronger coding.",
        "parameter_size": "Unknown",
        "family": "glm5.1",
    },
    {
        "name": "nemotron-3-super:cloud",
        "description": "Nemotron-3-Super is a large language model (LLM) trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities",
        "parameter_size": "120B",
        "family": "nemotron-3-super",
    },
    {
        "name": "deepseek-v4-flash:cloud",
        "description": "DeepSeek-V4-Flash is a preview of the DeepSeek-V4 series, a Mixture-of-Experts model with 284B total parameters and 13B activated, built for efficient reasoning across a 1M-token context window.",
        "parameter_size": "284B",
        "family": "deepseek-v4-flash",
    },
    {
        "name": "kimi-k2.6:cloud",
        "description": "Kimi K2.6 is an open-source, native multimodal agentic model that advances practical capabilities in long-horizon coding, coding-driven design, proactive autonomous execution, and swarm-based task orchestration.",
        "parameter_size": "Unknown",
        "family": "kimi-k2.6",
    }
]


async def build_enhanced_model_payload(ollama_url: str) -> dict[str, list[dict[str, Any]]]:
    """Return local models with metadata plus curated cloud models not installed locally."""
    local_models: list[dict[str, Any]] = []
    local_names: set[str] = set()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()

            for model in response.json().get("models", []):
                details = model.get("details", {})
                name = model["name"]
                local_models.append(
                    {
                        "name": name,
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level"),
                        "family": details.get("family"),
                        "is_local": True,
                        "description": None,
                    }
                )
                local_names.add(name)
                local_names.add(name.split(":")[0])
    except Exception:
        # Callers decide how to surface backend/Ollama availability issues.
        pass

    cloud_models: list[dict[str, Any]] = []
    for entry in OLLAMA_CLOUD_CATALOG:
        name = entry["name"]
        base = name.split(":")[0]
        if name not in local_names and base not in local_names:
            cloud_models.append(
                {
                    "name": name,
                    "parameter_size": entry.get("parameter_size"),
                    "family": entry.get("family"),
                    "is_local": False,
                    "description": entry.get("description"),
                }
            )

    return {"local_models": local_models, "cloud_models": cloud_models}