"""
Simple Tools API endpoints for demonstration
"""
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.simple_tools import simple_tool_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["simple-tools"])


# Pydantic models for API requests/responses
class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict


class ToolExecutionResponse(BaseModel):
    success: bool
    tool: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


@router.get("/available")
async def get_available_tools():
    """Get available tools with their descriptions"""
    try:
        tools = simple_tool_service.get_available_tools()
        return {"tools": tools, "count": len(tools)}
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/call", response_model=ToolExecutionResponse)
async def call_tool(request: ToolCallRequest):
    """Call a tool with parameters"""
    import time
    start_time = time.time()
    
    try:
        result = await simple_tool_service.call_tool(
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


@router.get("/test/{tool_name}")
async def test_tool(tool_name: str):
    """Test a tool with default parameters"""
    try:
        # Default test parameters for each tool
        test_params = {
            "web_search": {"query": "artificial intelligence", "num_results": 3},
            "file_reader": {"file_path": "README.md"},
            "system_info": {},
            "calculator": {"expression": "2 + 2 * 3"}
        }
        
        if tool_name not in test_params:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        
        result = await simple_tool_service.call_tool(tool_name, test_params[tool_name])
        return result
    except Exception as e:
        logger.error(f"Error testing tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))
