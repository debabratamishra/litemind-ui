"""Unified LLM transport backed by the LiteLLM Python SDK.

This module keeps provider-specific details in one place so the rest of the
application can stream chat completions without binding directly to Ollama's
HTTP API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Iterable, Optional


class _LiteLLMOptionalProviderWarningFilter(logging.Filter):
    """Suppress import-time warnings for optional AWS providers we do not use."""

    _SUPPRESSED_MESSAGES = (
        "could not pre-load bedrock-runtime response stream shape",
        "could not pre-load sagemaker-runtime response stream shape",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in self._SUPPRESSED_MESSAGES)


def _install_litellm_log_filter() -> None:
    litellm_logger = logging.getLogger("LiteLLM")
    if any(
        isinstance(existing_filter, _LiteLLMOptionalProviderWarningFilter)
        for existing_filter in litellm_logger.filters
    ):
        return

    litellm_logger.addFilter(_LiteLLMOptionalProviderWarningFilter())


_install_litellm_log_filter()

import litellm

from config import Config

logger = logging.getLogger(__name__)


litellm.drop_params = True
litellm.suppress_debug_info = True

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_SUPPORTED_BACKENDS = {"ollama", "openrouter"}
_OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


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
        from app.services.host_service_manager import host_service_manager

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
    resolved_model = _resolve_model_name(resolved_backend, model)
    return ResolvedLLMConfig(
        backend=resolved_backend,
        model=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
    )


async def stream_completion(
    messages: Iterable[dict[str, Any]],
    *,
    backend: Optional[str] = "ollama",
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> AsyncGenerator[str, None]:
    """Stream completion text via LiteLLM for the requested backend."""
    try:
        config = resolve_backend_config(
            backend=backend,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except LLMGatewayConfigurationError as exc:
        yield f"*{exc}*"
        return

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": _presence_penalty_from_repetition(repetition_penalty),
    }

    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key:
        kwargs["api_key"] = config.api_key

    if config.backend == "openrouter":
        extra_headers = _build_openrouter_headers()
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

    for attempt in range(1, _MAX_RETRIES + 1):
        yielded_any_content = False
        try:
            stream = await litellm.acompletion(**kwargs)
            async for chunk in stream:
                content = _extract_stream_text(chunk)
                if not content:
                    continue

                yielded_any_content = True
                yield content
            return
        except Exception as exc:
            logger.warning(
                "LiteLLM streaming failed for backend=%s model=%s attempt=%s/%s: %s",
                config.backend,
                config.model,
                attempt,
                _MAX_RETRIES,
                exc,
            )
            if not yielded_any_content and attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * attempt)
                continue

            yield _build_error_message(config, interrupted=yielded_any_content)
            return


async def complete_text(
    messages: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> str:
    """Collect a streamed completion into a single response string."""
    parts: list[str] = []
    async for chunk in stream_completion(messages, **kwargs):
        parts.append(chunk)
    return "".join(parts)


def _resolve_api_base(backend: str, api_base: Optional[str]) -> Optional[str]:
    if api_base:
        return api_base.strip()

    if backend == "ollama":
        return get_ollama_api_base()

    if backend == "openrouter":
        return (
            os.getenv("OPENROUTER_API_BASE")
            or os.getenv("OPENROUTER_BASE_URL")
            or _OPENROUTER_DEFAULT_API_BASE
        )

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

    return None


def _resolve_model_name(backend: str, model: Optional[str]) -> str:
    requested_model = (model or "").strip()
    if not requested_model or requested_model == "default":
        if backend == "ollama":
            requested_model = (
                os.getenv("DEFAULT_OLLAMA_MODEL")
                or os.getenv("OLLAMA_MODEL")
                or "gemma3:1b"
            )
        elif backend == "openrouter":
            requested_model = os.getenv("DEFAULT_OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL)

    if backend == "ollama":
        stripped_model = _strip_known_prefix(requested_model, ("ollama_chat/", "ollama/"))
        return f"ollama_chat/{stripped_model}"

    if backend == "openrouter":
        if requested_model.startswith("openrouter/"):
            return requested_model
        return f"openrouter/{requested_model}"

    return requested_model


def _strip_known_prefix(value: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _presence_penalty_from_repetition(repetition_penalty: float) -> float:
    if repetition_penalty <= 1.0:
        return 0.0
    return min(2.0, repetition_penalty - 1.0)


def _extract_stream_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    delta = getattr(first_choice, "delta", None)
    if delta is not None:
        return _coerce_text(getattr(delta, "content", None))

    message = getattr(first_choice, "message", None)
    if message is not None:
        return _coerce_text(getattr(message, "content", None))

    return ""


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    return str(value)


def _build_openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    site_url = os.getenv("OR_SITE_URL") or os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OR_APP_NAME") or os.getenv("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def _build_error_message(config: ResolvedLLMConfig, *, interrupted: bool) -> str:
    if config.backend == "ollama":
        if interrupted:
            return "*The Ollama response stream was interrupted. Please retry.*"
        return "*Could not reach Ollama. Check that the Ollama server is running and reachable.*"

    if config.backend == "openrouter":
        if interrupted:
            return "*The OpenRouter response stream was interrupted. Please retry.*"
        return "*OpenRouter request failed. Check the API key, base URL, and model name, then retry.*"

    return "*The language model request failed. Please retry.*"