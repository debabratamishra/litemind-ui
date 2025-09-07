"""
n8n Workflow API endpoints
"""
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.n8n_service import n8n_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/n8n", tags=["n8n-workflows"])


# Pydantic models for API requests/responses
class WorkflowExecutionRequest(BaseModel):
    workflow_id: str
    parameters: Optional[Dict] = {}


class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict


class WorkflowCreationRequest(BaseModel):
    workflow_id: str
    name: str
    workflow_definition: Dict


class WorkflowStatusResponse(BaseModel):
    n8n_running: bool
    n8n_url: str
    workflows_count: int
    available_tools: List[str]
    error: Optional[str] = None


class ToolExecutionResponse(BaseModel):
    success: bool
    tool: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


@router.get("/status", response_model=WorkflowStatusResponse)
async def get_n8n_status():
    """Get n8n service status and available workflows"""
    try:
        status = await n8n_service.get_workflow_status()
        return WorkflowStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting n8n status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_n8n_server():
    """Start n8n server"""
    try:
        success = await n8n_service.start_n8n_server()
        if success:
            return {"message": "n8n server started successfully", "status": "running"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start n8n server")
    except Exception as e:
        logger.error(f"Error starting n8n server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_n8n_server():
    """Stop n8n server"""
    try:
        await n8n_service.stop_n8n_server()
        return {"message": "n8n server stopped successfully", "status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping n8n server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows():
    """List all available workflows"""
    try:
        workflows = await n8n_service.list_workflows()
        return {"workflows": workflows, "count": len(workflows)}
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows")
async def create_workflow(request: WorkflowCreationRequest):
    """Create or update a workflow"""
    try:
        success = await n8n_service.create_or_update_workflow(
            workflow_id=request.workflow_id,
            workflow_definition=request.workflow_definition,
            name=request.name
        )
        
        if success:
            return {"message": f"Workflow '{request.workflow_id}' created/updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create/update workflow")
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/execute")
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute a specific workflow with parameters"""
    try:
        result = await n8n_service.execute_workflow(
            workflow_id=request.workflow_id,
            input_data=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def get_available_tools():
    """Get available tools (workflows) with their descriptions"""
    try:
        tools = n8n_service.get_available_tools()
        return {"tools": tools, "count": len(tools)}
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/call", response_model=ToolExecutionResponse)
async def call_tool(request: ToolCallRequest):
    """Call a tool (execute workflow) with parameters"""
    import time
    start_time = time.time()
    
    try:
        result = await n8n_service.call_tool(
            tool_name=request.tool_name,
            parameters=request.parameters
        )
        
        execution_time = time.time() - start_time
        
        return ToolExecutionResponse(
            success=result["success"],
            tool=request.tool_name,
            result=result.get("result"),
            error=result.get("error"),
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        execution_time = time.time() - start_time
        return ToolExecutionResponse(
            success=False,
            tool=request.tool_name,
            error=str(e),
            execution_time=execution_time
        )
