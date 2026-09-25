"""
LiteMindUI FastAPI Backend
Production-ready API server with chat and RAG capabilities.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.backend.api import auth as auth_api
from backend.app.backend.api import chat as chat_api
from backend.app.backend.api import conversations as conversations_api
from backend.app.backend.api import health as health_api
from backend.app.backend.api import memory as memory_api
from backend.app.backend.api import models as models_api
from backend.app.backend.api import rag as rag_api
from backend.app.backend.api import tts as tts_api
from backend.app.backend.api import voice as voice_api
from backend.app.backend.core.config import DEFAULT_RAG_CONFIG
from backend.app.backend.core.embeddings import create_embedding_function, resolve_embedding_provider
from backend.app.backend.user_memory_store import get_user_memory_store
from backend.app.services.rag_service import RAGService
from backend.app.services.speech_service import preload_stt_model
from backend.app.services.tts_service import preload_tts_model
from backend.config import Config

torch: Any = None
try:
    import torch
except ImportError:
    pass

# Configure logging early so lifespan hooks can use logger
try:
    from backend.logging_config import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Configuration
Config.apply_performance_settings()
dynamic_config = Config.get_dynamic_config()
UPLOAD_FOLDER = Path(dynamic_config["upload_dir"])
storage_dir = dynamic_config.get("storage_dir", Config.get_storage_path())
CONFIG_PATH = Path(storage_dir) / "rag_config.json"
Config.ensure_directories()

# DEFAULT_RAG_CONFIG is now imported from backend.app.backend.core.config

rag_service = None


# Configuration utilities
def load_rag_config() -> Dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return dict(DEFAULT_RAG_CONFIG)


def save_rag_config_local(cfg: Dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        logger.warning(f"Failed to persist RAG config: {e}")


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    app.state.start_time = time.time()
    logger.info("LiteMindUI API starting up...")

    config_info = Config.get_dynamic_config()
    logger.info(f"Environment: {'containerized' if config_info['is_containerized'] else 'native'}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Storage path: {config_info['storage_dir']}")

    try:
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        if config_info["is_containerized"]:
            os.chmod(UPLOAD_FOLDER, 0o755)
        logger.info("Uploads folder cleared")
    except Exception as e:
        logger.warning(f"Failed to clear uploads: {e}")

    # Initialize services
    global rag_service
    try:
        rag_service = RAGService()
        logger.info("RAG service ready")
    except Exception as e:
        logger.warning(f"RAG service initialization failed: {e}")
        rag_service = None

    # Ensure the user-memory table exists (idempotent); degrade gracefully
    try:
        await get_user_memory_store().init_schema()
        logger.info("User memory store ready")
    except Exception as e:
        logger.warning(f"User memory schema init skipped: {e}")

    # Restore configuration
    if rag_service is None:
        logger.warning("RAG service unavailable, skipping config restore")
    else:
        # The embedding function is ALWAYS built through the inference-provider
        # factory. There is no in-process default (no SentenceTransformer /
        # DefaultEmbedding fallback) — only the providers registered in
        # app.backend.core.embeddings.create_embedding_function
        # (ollama / huggingface / openrouter / nvidia_nim) can be used.
        cfg = load_rag_config()
        configured_provider = str(cfg.get("provider", DEFAULT_RAG_CONFIG["provider"]))
        embedding_backend = cfg.get("embedding_backend")
        provider = resolve_embedding_provider(configured_provider, embedding_backend)
        model_name = str(cfg.get("embedding_model", DEFAULT_RAG_CONFIG["embedding_model"]))
        chunk_size = int(cfg.get("chunk_size", DEFAULT_RAG_CONFIG["chunk_size"]))

        rag_service.embedding_function = create_embedding_function(
            provider,
            model_name,
            config_info["ollama_url"],
            embedding_backend=embedding_backend,
            api_base=cfg.get("embedding_api_base"),
            api_key=cfg.get("embedding_api_key"),
        )

        rag_service.default_chunk_size = chunk_size
        logger.info(
            "RAG config restored: provider=%s backend=%s model=%s",
            provider,
            embedding_backend,
            model_name,
        )

    # Performance tuning
    try:
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        if torch:
            torch.set_num_threads(cpu_threads)
        logger.info(f"Thread optimization applied: {cpu_threads} threads")
    except Exception as e:
        logger.warning(f"Thread tuning failed: {e}")

    # Preload speech models for reduced latency
    # This now runs synchronously to ensure models are ready before "startup complete"
    preload_enabled = os.getenv("PRELOAD_SPEECH_MODELS", "1").strip().lower() not in {"0", "false", "no"}
    preload_async_raw = os.getenv("PRELOAD_SPEECH_MODELS_ASYNC")
    preload_async = (
        preload_async_raw.strip().lower() in {"1", "true", "yes"}
        if preload_async_raw is not None
        else bool(config_info["is_containerized"])
    )

    if preload_enabled:
        if preload_async_raw is None and config_info["is_containerized"]:
            logger.info("Containerized environment detected; speech models will preload in background by default")
        logger.info("=" * 60)
        logger.info("LOADING SPEECH MODELS (this may take a moment)...")
        logger.info("=" * 60)

        def preload_models():
            start_time = time.time()
            stt_loaded = False
            tts_loaded = False

            try:
                # Preload STT (Whisper) model
                logger.info("  → Loading STT (Whisper) model...")
                preload_stt_model()
                stt_loaded = True
                logger.info("  ✓ STT model loaded successfully")
            except Exception as e:
                logger.warning(f"  ✗ Failed to preload STT model: {e}")

            try:
                # Preload TTS (Kokoro) model
                logger.info("  → Loading TTS (Kokoro) model...")
                preload_tts_model()
                tts_loaded = True
                logger.info("  ✓ TTS model loaded successfully")
            except Exception as e:
                logger.warning(f"  ✗ Failed to preload TTS model: {e}")

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"SPEECH MODELS READY (took {elapsed:.1f}s)")
            logger.info(f"  STT: {'✓ Ready' if stt_loaded else '✗ Not loaded'}")
            logger.info(f"  TTS: {'✓ Ready' if tts_loaded else '✗ Not loaded'}")
            logger.info("=" * 60)

        if preload_async:
            # Optional: Run in background (set PRELOAD_SPEECH_MODELS_ASYNC=1)
            logger.info("(Running in background mode)")
            preload_thread = threading.Thread(target=preload_models, daemon=True)
            preload_thread.start()
        else:
            # Default: Run synchronously so models are ready before server starts
            preload_models()
    else:
        logger.info("Speech model preloading disabled (PRELOAD_SPEECH_MODELS=0)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("LITEMINDUI API READY")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    try:
        if rag_service:
            await rag_service.reset_system()
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        logger.info("Cleanup completed")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

    # Tear down any open voice (WebRTC) peer connections
    try:
        from backend.app.backend.api.voice import pcs_map

        for conn in list(pcs_map.values()):
            try:
                await conn.disconnect()
            except Exception:
                pass
        pcs_map.clear()
        logger.info("Voice peer connections cleaned up")
    except Exception as e:
        logger.warning(f"Voice peer cleanup failed: {e}")

    logger.info("LiteMindUI API shutting down...")


# FastAPI app
app = FastAPI(
    title="LiteMindUI API",
    description="Production API for LiteMindUI with Chat and RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth_api.router)
app.include_router(conversations_api.router)
app.include_router(chat_api.router)
app.include_router(health_api.router)
app.include_router(models_api.router)
app.include_router(rag_api.router)
app.include_router(tts_api.router)
app.include_router(voice_api.router)
app.include_router(memory_api.router)

# NOTE: Chat endpoints have been moved to backend/app/backend/api/chat.py.
# They are included via the router above to support web search functionality.

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "path": str(request.url.path)})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


# Server runner
def run():
    """Run the server with graceful shutdown"""
    config = uvicorn.Config(
        "backend.main:app", host="localhost", port=8000, reload=bool(int(os.getenv("RELOAD", "0"))), log_level="info"
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def handle_exit(*_args):
        logger.info("Received exit signal")
        stop_event.set()

    # Signal handling
    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)

    for sig in signals:
        try:
            loop.add_signal_handler(sig, handle_exit)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, f: stop_event.set())

    # Windows keyboard handling
    def keyboard_watcher():
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()

    if sys.platform.startswith("win"):
        threading.Thread(target=keyboard_watcher, daemon=True).start()

    async def main():
        server_task = loop.create_task(server.serve())
        await stop_event.wait()
        server.should_exit = True
        await server_task

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        logger.info("Server stopped")


if __name__ == "__main__":
    run()
