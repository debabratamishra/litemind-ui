"""Kernel provider router: capability-neutral provider resolution.

Owns the resolution logic that used to live in ``llm_gateway`` (moved here
verbatim) and generalizes it per capability. Resolution order everywhere:
explicit request param > key vault > env var.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from backend.app.kernel.keyvault import get_key_vault
from backend.config import Config

logger = logging.getLogger(__name__)

_CAPABILITY_PROVIDERS = {
    "llm": ("openrouter", "nvidia_nim"),
    "vlm": ("openrouter", "nvidia_nim"),
    "stt": ("elevenlabs",),
    "tts": ("elevenlabs",),
    "voice": ("elevenlabs", "sarvam"),
    "embeddings": ("openrouter", "nvidia_nim"),
    "search": ("serpapi",),
}
_KEY_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "serpapi": "SERP_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "sarvam": "SARVAM_API_KEY",
}
_BASE_ENV_VARS = {
    "openrouter": ("OPENROUTER_API_BASE", "OPENROUTER_BASE_URL"),
    "nvidia_nim": ("NVIDIA_NIM_API_BASE",),
}

CAPABILITIES = ("llm", "vlm", "stt", "tts", "voice", "embeddings", "search")

_SUPPORTED_BACKENDS = {"ollama", "openrouter", "nvidia_nim"}
_OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
_NVIDIA_NIM_DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"
_DEFAULT_NVIDIA_NIM_MODEL = "meta/llama3-70b-instruct"


class LLMGatewayConfigurationError(ValueError):
    """Raised when a backend cannot be configured from request/env settings."""


@dataclass(frozen=True)
class ResolvedLLMConfig:
    backend: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None


def normalize_backend(backend: Optional[str]) -> str:
    resolved_backend = (backend or "ollama").strip().lower()
    if resolved_backend not in _SUPPORTED_BACKENDS:
        raise LLMGatewayConfigurationError(
            f"Unsupported backend '{resolved_backend}'. Expected one of: {', '.join(sorted(_SUPPORTED_BACKENDS))}."
        )
    return resolved_backend


def get_ollama_api_base() -> str:
    """Return the runtime Ollama URL for native and containerized execution."""
    try:
        from backend.app.services.host_service_manager import host_service_manager

        url = host_service_manager.environment_config.ollama_url
        logger.debug("Using Ollama URL from host service manager: %s", url)
        return url
    except ImportError:
        logger.warning("Host service manager not available, using fallback config")
        url = Config.OLLAMA_API_URL
        logger.debug("Using fallback Ollama URL: %s", url)
        return url


def resolve_backend_config(
    backend: Optional[str],
    model: Optional[str],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedLLMConfig:
    resolved_backend = normalize_backend(backend)
    resolved_api_base = _resolve_api_base(resolved_backend, api_base)
    resolved_api_key = _resolve_api_key(resolved_backend, api_key)
    resolved_model = _resolve_model_name(resolved_backend, model, purpose="completion")
    return ResolvedLLMConfig(
        backend=resolved_backend,
        model=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
    )


def _resolve_api_base(backend: str, api_base: Optional[str]) -> Optional[str]:
    if api_base:
        return api_base.strip()

    if backend == "ollama":
        return get_ollama_api_base()

    if backend == "openrouter":
        return os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENROUTER_BASE_URL") or _OPENROUTER_DEFAULT_API_BASE

    if backend == "nvidia_nim":
        return os.getenv("NVIDIA_NIM_API_BASE") or _NVIDIA_NIM_DEFAULT_API_BASE

    if backend == "elevenlabs":
        return os.getenv("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io"

    return None


def _resolve_api_key(backend: str, api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key.strip()

    if backend == "openrouter":
        env_api_key = os.getenv("OPENROUTER_API_KEY")
        if env_api_key:
            return env_api_key
        raise LLMGatewayConfigurationError(
            "OpenRouter requires an API key. Set OPENROUTER_API_KEY or pass api_key in the request."
        )

    if backend == "nvidia_nim":
        env_api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if env_api_key:
            return env_api_key
        raise LLMGatewayConfigurationError(
            "Nvidia NIM requires an API key. Set NVIDIA_NIM_API_KEY or pass api_key in the request."
        )

    if backend == "elevenlabs":
        env_api_key = os.getenv("ELEVENLABS_API_KEY")
        if env_api_key:
            return env_api_key
        raise LLMGatewayConfigurationError(
            "ElevenLabs requires an API key. Set ELEVENLABS_API_KEY or pass api_key in the request."
        )

    return None


def _resolve_model_name(
    backend: str,
    model: Optional[str],
    *,
    purpose: str = "completion",
) -> str:
    requested_model = (model or "").strip()
    if not requested_model or requested_model == "default":
        if backend == "ollama":
            requested_model = os.getenv("DEFAULT_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL") or "gemma3:1b"
        elif backend == "openrouter":
            requested_model = os.getenv("DEFAULT_OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL)
        elif backend == "nvidia_nim":
            requested_model = os.getenv("DEFAULT_NVIDIA_NIM_MODEL", _DEFAULT_NVIDIA_NIM_MODEL)
        elif backend == "elevenlabs":
            requested_model = os.getenv("DEFAULT_ELEVENLABS_MODEL", "eleven_turbo_v2_5")

    if backend == "ollama":
        stripped_model = _strip_known_prefix(requested_model, ("ollama_chat/", "ollama/"))
        prefix = "ollama/" if purpose == "embedding" else "ollama_chat/"
        return f"{prefix}{stripped_model}"

    if backend == "openrouter":
        if requested_model.startswith("openrouter/"):
            return requested_model
        return f"openrouter/{requested_model}"

    if backend == "nvidia_nim":
        if requested_model.startswith("nvidia_nim/"):
            return requested_model
        return f"nvidia_nim/{requested_model}"

    return requested_model


def _strip_known_prefix(value: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


# ── Capability layer ─────────────────────────────────────────────────────────


async def _select_provider(capability: str, vault) -> str:
    """First provider with a vault credential, then an env key, else default."""
    candidates = _CAPABILITY_PROVIDERS.get(capability, ())
    if capability in ("llm", "vlm", "embeddings"):
        candidates = ("ollama",) + candidates

    for provider in candidates:
        if await vault.get_credential(capability, provider):
            return provider
        if capability in ("stt", "tts", "voice"):
            for alt_cap in ("stt", "tts", "voice"):
                if alt_cap != capability and await vault.get_credential(alt_cap, provider):
                    return provider

    for provider in candidates:
        env_var = _KEY_ENV_VARS.get(provider)
        if env_var and os.getenv(env_var):
            return provider

    return candidates[0] if candidates else "ollama"


async def resolve_provider_config(
    capability: str,
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> ResolvedLLMConfig:
    """Resolve provider config for a capability: param > vault > env."""
    if capability not in CAPABILITIES:
        raise ValueError(f"Unknown capability '{capability}'. Expected one of: {', '.join(CAPABILITIES)}.")

    vault = get_key_vault()
    provider = await _select_provider(capability, vault)
    vault_key = await vault.get_credential(capability, provider)
    if not vault_key and capability in ("stt", "tts"):
        other_cap = "tts" if capability == "stt" else "stt"
        vault_key = await vault.get_credential(other_cap, provider)
    if not vault_key and capability == "voice":
        for alt_cap in ("stt", "tts"):
            vault_key = await vault.get_credential(alt_cap, provider)
            if vault_key:
                break
    env_key = os.getenv(_KEY_ENV_VARS.get(provider, ""))
    try:
        resolved_key = _resolve_api_key(provider, api_key or vault_key or env_key)
    except LLMGatewayConfigurationError:
        resolved_key = None
    resolved_base = _resolve_api_base(provider, api_base)
    purpose = "embedding" if capability == "embeddings" else "completion"
    resolved_model = _resolve_model_name(provider, model, purpose=purpose)
    return ResolvedLLMConfig(
        backend=provider,
        model=resolved_model,
        api_base=resolved_base,
        api_key=resolved_key,
    )


async def list_provider_status() -> dict[str, dict]:
    """Per-capability provider status for ``GET /api/kernel/providers``.

        ``configured`` means a key/endpoint is present (vault or env), NOT
    live-validated.
    """
    status: dict[str, dict] = {}
    for capability in CAPABILITIES:
        cfg = await resolve_provider_config(capability)
        provider = cfg.backend
        env_present = bool(os.getenv(_KEY_ENV_VARS.get(provider, ""))) or any(
            os.getenv(var) for var in _BASE_ENV_VARS.get(provider, ())
        )
        status[capability] = {
            "provider": provider,
            "configured": bool(cfg.api_key) or env_present,
            "model": cfg.model,
        }
    return status
