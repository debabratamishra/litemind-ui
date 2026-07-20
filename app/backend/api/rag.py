"""
RAG API endpoints
"""

import asyncio
import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.backend.api.security_utils import sanitize_filename, validate_file_size
from app.backend.core.config import backend_config
from app.backend.core.embeddings import create_embedding_function, resolve_embedding_provider
from app.backend.models.api_models import (
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    RAGConfigRequest,
    RagFileInfo,
    RagFilesResponse,
    RAGQueryRequestEnhanced,
    RAGStatusResponse,
    ResetResponse,
    UploadResponse,
)
from app.skills import rag_skill_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/status", response_model=RAGStatusResponse)
async def get_rag_status():
    """Get RAG system status"""
    try:
        from main import rag_service

        if not rag_service:
            return RAGStatusResponse(status="not_initialized", uploaded_files=0, indexed_chunks=0, bm25_corpus_size=0)

        uploaded_files = len([f for f in backend_config.upload_folder.iterdir() if f.is_file()])

        collection_count = 0
        try:
            if getattr(rag_service, "text_collection", None):
                collection_count = rag_service.text_collection.count()
        except Exception:
            pass

        return RAGStatusResponse(
            status="ready",
            uploaded_files=uploaded_files,
            indexed_chunks=collection_count,
            bm25_corpus_size=len(rag_service.bm25_corpus) if rag_service.bm25_corpus else 0,
        )

    except Exception:
        logger.exception("Failed to get RAG status")
        return RAGStatusResponse(status="error", message="Failed to retrieve RAG status")


@router.get("/files", response_model=RagFilesResponse)
async def list_rag_files():
    """List the files currently uploaded to the knowledge base."""
    try:
        files = [
            RagFileInfo(filename=f.name, size=f.stat().st_size)
            for f in backend_config.upload_folder.iterdir()
            if f.is_file()
        ]
        files.sort(key=lambda info: info.filename.lower())
        return RagFilesResponse(files=files)
    except Exception:
        logger.exception("Failed to list RAG files")
        return RagFilesResponse(files=[])


@router.post("/save_config")
async def save_rag_config(request: RAGConfigRequest):
    """Save RAG configuration"""
    try:
        from main import rag_service

        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Save configuration
        config = backend_config.load_rag_config()
        normalized_provider = resolve_embedding_provider(request.provider, request.embedding_backend)
        config.update(
            {
                "provider": normalized_provider,
                "embedding_model": request.embedding_model,
                "embedding_backend": None,
                "embedding_api_base": (
                    request.embedding_api_base if normalized_provider in {"openrouter", "nvidia_nim"} else None
                ),
                "chunk_size": int(request.chunk_size),
            }
        )

        if not backend_config.save_rag_config(config):
            raise HTTPException(status_code=500, detail="Failed to persist configuration")

        # Update embedding function
        rag_service.embedding_function = create_embedding_function(
            normalized_provider,
            request.embedding_model,
            backend_config.get_ollama_url(),
            api_base=request.embedding_api_base,
            api_key=request.embedding_api_key,
        )
        rag_service.default_chunk_size = int(request.chunk_size)

        return {"message": "Configuration saved successfully", "status": "success"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Save config error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def rag_upload(files: List[UploadFile] = File(...), chunk_size: int = Form(500)):
    """Upload and process files for RAG"""
    from main import rag_service

    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    # Validate number of files
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per upload.")

    results = []
    saved_paths = []

    # Save files and check for duplicates
    for upload_file in files:
        try:
            # Validate file size
            await validate_file_size(upload_file)

            # Sanitize filename to prevent path traversal
            safe_filename = sanitize_filename(upload_file.filename)
        except ValueError as e:
            results.append(
                {
                    "filename": upload_file.filename,
                    "status": "error",
                    "message": f"Invalid filename: {str(e)}",
                    "chunks_created": 0,
                }
            )
            continue

        # Save file with sanitized name
        dest_path = backend_config.upload_folder / safe_filename

        # Additional security check: ensure dest_path is within upload_folder
        try:
            dest_resolved = dest_path.resolve()
            upload_resolved = backend_config.upload_folder.resolve()
            if not str(dest_resolved).startswith(str(upload_resolved)):
                raise ValueError("Path traversal attempt detected")
        except (ValueError, OSError, RuntimeError) as e:
            results.append(
                {
                    "filename": upload_file.filename,
                    "status": "error",
                    "message": f"Security error: {str(e)}",
                    "chunks_created": 0,
                }
            )
            continue

        with open(dest_path, "wb") as f:
            f.write(await upload_file.read())

        # Check for duplicates using the saved file path
        is_duplicate, reason = rag_service._is_file_already_processed(str(dest_path), safe_filename)

        if is_duplicate:
            # Remove the saved file since it's a duplicate
            dest_path.unlink(missing_ok=True)
            results.append(
                {"filename": upload_file.filename, "status": "duplicate", "message": reason, "chunks_created": 0}
            )
            continue

        saved_paths.append((dest_path, upload_file.filename))

    # Process files concurrently
    if saved_paths:
        await _process_uploaded_files(saved_paths, chunk_size, results, rag_service)

    # Generate summary
    successful = [r for r in results if r["status"] == "success"]
    duplicates = [r for r in results if r["status"] == "duplicate"]
    errors = [r for r in results if r["status"] == "error"]

    return UploadResponse(
        status="completed",
        summary={
            "total_files": len(files),
            "successful": len(successful),
            "duplicates": len(duplicates),
            "errors": len(errors),
            "total_chunks_created": sum(r.get("chunks_created", 0) for r in successful),
        },
        results=results,
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_rag_system():
    """Reset RAG system"""
    try:
        from main import rag_service

        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Clear uploaded files
        files_removed = 0
        for file_path in backend_config.upload_folder.iterdir():
            if file_path.is_file():
                file_path.unlink()
                files_removed += 1

        # Reset RAG service
        await rag_service.reset_system()

        return ResetResponse(
            status="success", message=f"RAG system reset. Removed {files_removed} files.", files_removed=files_removed
        )

    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@router.post("/duplicate-check", response_model=DuplicateCheckResponse)
async def duplicate_check_rag(request: DuplicateCheckRequest):
    """Preflight check: is a file with this name already in the knowledge base?"""
    try:
        safe_filename = sanitize_filename(request.filename)
    except ValueError as e:
        return DuplicateCheckResponse(is_duplicate=False, reason=f"Invalid filename: {e}")

    dest_path = backend_config.upload_folder / safe_filename
    if dest_path.is_file():
        return DuplicateCheckResponse(is_duplicate=True, reason=f"'{safe_filename}' is already uploaded.")
    return DuplicateCheckResponse(is_duplicate=False, reason="")


@router.delete("/files/{filename}")
async def delete_rag_file(filename: str):
    """Delete a single uploaded knowledge-base file."""
    try:
        safe_filename = sanitize_filename(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")

    dest_path = backend_config.upload_folder / safe_filename

    # Guard against path traversal: ensure the resolved path stays in upload_folder.
    dest_resolved = dest_path.resolve()
    upload_resolved = backend_config.upload_folder.resolve()
    if not str(dest_resolved).startswith(str(upload_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not dest_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    dest_path.unlink()
    return {"status": "deleted", "filename": safe_filename}


@router.post("/query")
async def rag_query(request: RAGQueryRequestEnhanced):
    """Query RAG system"""
    try:
        from main import rag_service

        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        return await _handle_rag_query(request, rag_service)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_uploaded_files(saved_paths, chunk_size, results, rag_service):
    """Process uploaded files with concurrency control"""
    semaphore = asyncio.Semaphore(2)  # Limit concurrent processing

    async def process_single_file(path_info):
        async with semaphore:
            path, filename = path_info
            try:
                result = await rag_service.add_document(str(path), filename, chunk_size=chunk_size)
                results.append(
                    {
                        "filename": filename,
                        **(result or {"status": "success", "message": f"Processed {filename}", "chunks_created": 0}),
                    }
                )
            except Exception as e:
                results.append({"filename": filename, "status": "error", "message": str(e), "chunks_created": 0})

    await asyncio.gather(*(process_single_file(path_info) for path_info in saved_paths))


async def _handle_rag_query(request: RAGQueryRequestEnhanced, rag_service):
    """Handle a RAG query through the registered RAG skill layer."""
    skill = rag_skill_registry.resolve(request)
    if skill is None:
        raise HTTPException(status_code=400, detail="No compatible RAG skill found for request")

    async def event_generator():
        logger.info("Routing RAG query through skill '%s'", skill.name)

        try:
            async for chunk in skill.stream(request, rag_service):
                yield chunk + "\n"
        except Exception as exc:
            logger.error("RAG stream error: %s", exc)
            # Do not expose the exception (which may include a stack trace or
            # internal details) to the client; surface a generic message only.
            yield "\n⚠️ An error occurred while processing your request.\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
