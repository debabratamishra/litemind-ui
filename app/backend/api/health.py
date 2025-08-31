"""
Health check endpoints
"""
import os
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.backend.models.api_models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    return HealthResponse(status="healthy")


@router.get("/ready")
async def readiness_check():
    """Container readiness check with detailed status"""
    try:
        from app.backend.core.config import backend_config
        
        status_data = {
            "status": "ready",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check RAG service
        try:
            from main import rag_service
            if rag_service is None:
                status_data["checks"]["rag_service"] = {"status": "failed", "error": "Not initialized"}
                status_data["status"] = "not_ready"
            else:
                status_data["checks"]["rag_service"] = {"status": "ready"}
        except Exception as e:
            status_data["checks"]["rag_service"] = {"status": "error", "error": str(e)}
            status_data["status"] = "not_ready"
        
        # Check critical directories
        critical_dirs = [backend_config.upload_folder, backend_config.storage_dir]
        for dir_path in critical_dirs:
            if dir_path.exists() and os.access(dir_path, os.R_OK | os.W_OK):
                status_data["checks"][dir_path.name] = {"status": "ready", "path": str(dir_path)}
            else:
                status_data["checks"][dir_path.name] = {"status": "failed", "path": str(dir_path)}
                status_data["status"] = "not_ready"
        
        if status_data["status"] == "ready":
            return status_data
        else:
            return JSONResponse(status_code=503, content=status_data)
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503, 
            content={"status": "error", "error": str(e), "timestamp": time.time()}
        )
