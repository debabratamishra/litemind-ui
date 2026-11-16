"""Unified LLM transport backed by the LiteLLM Python SDK.

This module keeps provider-specific details in one place so the rest of the
application can stream chat completions without binding directly to Ollama's
HTTP API. Provider resolution lives in ``backend.app.kernel.router``; the
names below are re-exported from there for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Iterable, Optional

import litellm

from backend.app.kernel.keyvault import get_key_vault
from backend.app.kernel.router import (  # noqa: F401 - back-compat re-exports
    _NVIDIA_NIM_DEFAULT_API_BASE,
    _OPENROUTER_DEFAULT_API_BASE,
    LLMGatewayConfigurationError,
    ResolvedLLMConfig,
    _resolve_api_base,
    _resolve_api_key,
    _resolve_model_name,
    _strip_known_prefix,
    get_ollama_api_base,
    normalize_backend,
    resolve_backend_config,
)
from backend.config import Config  # noqa: F401 - re-exported; tests patch llm_gateway.Config


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
        isinstance(existing_filter, _LiteLLMOptionalProviderWarningFilter) for existing_filter in litellm_logger.filters
    ):
        return

    litellm_logger.addFilter(_LiteLLMOptionalProviderWarningFilter())


_install_litellm_log_filter()

logger = logging.getLogger(__name__)


litellm.drop_params = True
# suppress_debug_info is declared by litellm as a Literal[False]; setattr avoids
# the static-type error while keeping the runtime assignment (the declaration
# lives in the installed package and would be reverted on `uv sync`).
setattr(litellm, "suppress_debug_info", True)
litellm.disable_streaming_logging = True

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def resolve_embedding_config(
    backend: Optional[str],
    model: Optional[str],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedLLMConfig:
    resolved_backend = normalize_backend(backend)
    resolved_api_base = _resolve_api_base(resolved_backend, api_base)
    resolved_api_key = _resolve_api_key(resolved_backend, api_key)
    resolved_model = _resolve_model_name(resolved_backend, model, purpose="embedding")
    return ResolvedLLMConfig(
        backend=resolved_backend,
        model=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
    )


async def _stream_ollama_native(
    messages: list[dict[str, Any]],
    *,
    model: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    repetition_penalty: float,
    top_k: int = 40,
    min_p: float = 0.0,
    seed: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> AsyncGenerator[str, None]:
    """Stream chat completions directly via the ollama Python client.

    This bypasses LiteLLM entirely for the Ollama backend, avoiding a
    ``KeyError('litellm.utils')`` that occurs in LiteLLM's internal stream
    assembler when Ollama returns non-standard streaming chunks (e.g. chunks
    whose first message lacks a ``delta.role`` field).  The ``ollama`` package
    is already a project dependency and handles Ollama's streaming protocol
    natively, so this path is both more reliable and lower overhead.
    """
    import ollama as _ollama

    # Strip the LiteLLM prefix that resolve_backend_config adds (ollama_chat/)
    bare_model = _strip_known_prefix(model, ("ollama_chat/", "ollama/"))

    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        "top_k": top_k,
        "min_p": min_p,
    }
    # Ollama uses repeat_penalty, not frequency_penalty
    if frequency_penalty:
        options["frequency_penalty"] = frequency_penalty
    if repetition_penalty > 1.0:
        options["repeat_penalty"] = repetition_penalty
    if seed is not None:
        options["seed"] = seed
    if stop:
        options["stop"] = stop

    client = _ollama.AsyncClient(host=api_base)
    try:
        stream = await client.chat(
            model=bare_model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            options=options,
        )
        async for chunk in stream:
            content = chunk.message.content if chunk.message else None
            if content:
                yield content
    finally:
        await client._client.aclose()


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
    top_k: int = 40,
    min_p: float = 0.0,
    seed: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> AsyncGenerator[str, None]:
    """Stream completion text via LiteLLM for the requested backend.

    For the Ollama backend the native ``ollama`` Python client is used instead
    of LiteLLM to avoid a ``KeyError('litellm.utils')`` bug in LiteLLM's
    streaming assembler that surfaces in Docker environments.
    """
    try:
        if not api_key:
            # Resolution order: explicit request param > user key vault > env
            # var. The vault row for the requested backend lets a key saved in
            # the UI reach real inference without touching the request params.
            # The lookup is best-effort: if the vault is unreachable (DB down,
            # corrupt key file) the turn must still run on env-var keys as it
            # did before the vault existed, not 500 the chat.
            try:
                vault_key = await get_key_vault().get_credential("llm", backend or "")
            except Exception:
                logger.warning("Key vault read failed; falling back to env keys", exc_info=True)
                vault_key = None
            if vault_key:
                api_key = vault_key
        config = resolve_backend_config(
            backend=backend,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except LLMGatewayConfigurationError:
        logger.warning("LLM configuration error while preparing stream completion", exc_info=True)
        yield "*Unable to process request with current model configuration.*"
        return

    if config.backend == "ollama":
        message_list = list(messages)
        for attempt in range(1, _MAX_RETRIES + 1):
            yielded_any_content = False
            try:
                async for text in _stream_ollama_native(
                    message_list,
                    model=config.model,
                    api_base=config.api_base or "http://localhost:11434",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    min_p=min_p,
                    seed=seed,
                    stop=stop,
                ):
                    yielded_any_content = True
                    yield text
                return
            except Exception as exc:
                logger.warning(
                    "Ollama native streaming failed for model=%s attempt=%s/%s: %s",
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
        return

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": _presence_penalty_from_repetition(repetition_penalty),
        "repetition_penalty": repetition_penalty,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if stop:
        kwargs["stop"] = stop

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

    if config.backend == "nvidia_nim":
        if interrupted:
            return "*The Nvidia NIM response stream was interrupted. Please retry.*"
        return "*Nvidia NIM request failed. Check the API key, base URL, and model name, then retry.*"

    return "*The language model request failed. Please retry.*"
