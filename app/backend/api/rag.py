"""
RAG API endpoints
"""
import asyncio
import logging
from typing import List
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.backend.models.api_models import (
    RAGConfigRequest, RAGQueryRequestEnhanced, RAGStatusResponse, 
    UploadResponse, ResetResponse
)
from app.backend.core.config import backend_config
from app.backend.core.embeddings import create_embedding_function
from app.services.vllm_service import vllm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/status", response_model=RAGStatusResponse)
async def get_rag_status():
    """Get RAG system status"""
    try:
        from main import rag_service
        
        if not rag_service:
            return RAGStatusResponse(
                status="not_initialized", 
                uploaded_files=0, 
                indexed_chunks=0,
                bm25_corpus_size=0
            )

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
            bm25_corpus_size=len(rag_service.bm25_corpus) if rag_service.bm25_corpus else 0
        )
        
    except Exception as e:
        logger.error(f"RAG status error: {e}")
        return RAGStatusResponse(status="error", message=str(e))


@router.post("/save_config")
async def save_rag_config(request: RAGConfigRequest):
    """Save RAG configuration"""
    try:
        from main import rag_service
        
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not initialized")

        # Save configuration
        config = backend_config.load_rag_config()
        config.update({
            "provider": request.provider,
            "embedding_model": request.embedding_model,
            "chunk_size": int(request.chunk_size)
        })
        
        if not backend_config.save_rag_config(config):
            raise HTTPException(status_code=500, detail="Failed to persist configuration")

        # Update embedding function
        rag_service.embedding_function = create_embedding_function(
            request.provider, 
            request.embedding_model, 
            backend_config.get_ollama_url()
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

    results = []
    saved_paths = []
    
    # Check for duplicates and save files
    for upload_file in files:
        is_duplicate, reason = rag_service._is_file_already_processed("", upload_file.filename)
        
        if is_duplicate:
            results.append({
                "filename": upload_file.filename,
                "status": "duplicate",
                "message": reason,
                "chunks_created": 0
            })
            continue
        
        # Save file
        dest_path = backend_config.upload_folder / upload_file.filename
        with open(dest_path, "wb") as f:
            f.write(await upload_file.read())
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
            "total_chunks_created": sum(r.get("chunks_created", 0) for r in successful)
        },
        results=results
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
            status="success",
            message=f"RAG system reset. Removed {files_removed} files.",
            files_removed=files_removed
        )
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@router.post("/query")
async def rag_query(request: RAGQueryRequestEnhanced):
    """Query RAG system"""
    try:
        from main import rag_service
        
        if request.backend == "vllm":
            return await _handle_vllm_rag_query(request, rag_service)
        else:
            return await _handle_ollama_rag_query(request, rag_service)

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
                results.append({
                    "filename": filename,
                    **(result or {"status": "success", "message": f"Processed {filename}", "chunks_created": 0})
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "status": "error",
                    "message": str(e),
                    "chunks_created": 0
                })

    await asyncio.gather(*(process_single_file(path_info) for path_info in saved_paths))


async def _handle_vllm_rag_query(request: RAGQueryRequestEnhanced, rag_service):
    """Handle vLLM RAG query"""
    if request.hf_token:
        vllm_service.set_hf_token(request.hf_token)

    # Get context from RAG
    context = ""
    try:
        if request.use_hybrid_search and getattr(rag_service, "bm25_model", None):
            documents = rag_service.hybrid_search(request.query, request.n_results)
            context = " ".join(documents)
        else:
            text_collection = getattr(rag_service, "text_collection", None)
            if text_collection:
                results = text_collection.query(query_texts=[request.query], n_results=request.n_results)
                if results and results.get('documents'):
                    context = " ".join(results['documents'][0])
    except Exception as e:
        logger.warning(f"Vector query failed: {e}")

    # Create messages
    messages = request.messages + [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuery: {request.query}"}
    ]

    # Stream response
    async def event_generator():
        async for chunk in vllm_service.stream_vllm_chat(messages=messages, model=request.model):
            yield chunk + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


async def _handle_ollama_rag_query(request: RAGQueryRequestEnhanced, rag_service):
    """Handle Ollama RAG query"""
    async def event_generator():
        async for chunk in rag_service.query(
            request.query, request.system_prompt, request.messages,
            request.n_results, request.use_hybrid_search, request.model
        ):
            yield chunk + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
