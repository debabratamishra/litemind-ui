"""
n8n Workflow Integration Service

This service provides integration with n8n workflows to enable tool use and MCP capabilities
for the LiteMind UI application through local LLMs and workflow automation.
"""
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import httpx
import os
import signal
from datetime import datetime

logger = logging.getLogger(__name__)


class N8nWorkflowService:
    """Service for managing n8n workflows and executing them as tools"""
    
    def __init__(self, 
                 n8n_url: str = "http://localhost:5678",
                 n8n_api_key: Optional[str] = None,
                 auto_start: bool = True):
        self.n8n_url = n8n_url
        self.n8n_api_key = n8n_api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        # Only add API key header if provided
        if n8n_api_key:
            self.headers["Authorization"] = f"Bearer {n8n_api_key}"
        
        self.n8n_process = None
        self.workflows_cache = {}
        self.auto_start = auto_start
        
        # Default workflow definitions for common tools
        self.default_workflows = self._get_default_workflows()
        
    def _get_default_workflows(self) -> Dict[str, Dict]:
        """Get default workflow definitions for common tools"""
        return {
            "web_search": {
                "name": "Web Search Tool",
                "description": "Search the web for information",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "workflow_definition": {
                    "nodes": [
                        {
                            "id": "start",
                            "type": "n8n-nodes-base.start",
                            "position": [100, 200],
                            "parameters": {}
                        },
                        {
                            "id": "http_request",
                            "type": "n8n-nodes-base.httpRequest",
                            "position": [300, 200],
                            "parameters": {
                                "url": "https://api.duckduckgo.com/",
                                "options": {
                                    "qs": {
                                        "q": "={{$json.query}}",
                                        "format": "json",
                                        "no_html": "1",
                                        "skip_disambig": "1"
                                    }
                                }
                            }
                        }
                    ],
                    "connections": {
                        "start": {"main": [[{"node": "http_request", "type": "main", "index": 0}]]}
                    }
                }
            },
            "file_operations": {
                "name": "File Operations Tool",
                "description": "Read, write, and manipulate files",
                "parameters": {
                    "operation": {"type": "string", "description": "Operation type: read, write, list"},
                    "file_path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write (for write operation)", "required": False}
                },
                "workflow_definition": {
                    "nodes": [
                        {
                            "id": "start",
                            "type": "n8n-nodes-base.start",
                            "position": [100, 200],
                            "parameters": {}
                        },
                        {
                            "id": "file_operation",
                            "type": "n8n-nodes-base.readWriteFile",
                            "position": [300, 200],
                            "parameters": {
                                "operation": "={{$json.operation}}",
                                "filePath": "={{$json.file_path}}",
                                "fileContent": "={{$json.content || ''}}"
                            }
                        }
                    ],
                    "connections": {
                        "start": {"main": [[{"node": "file_operation", "type": "main", "index": 0}]]}
                    }
                }
            },
            "email_tool": {
                "name": "Email Tool", 
                "description": "Send emails",
                "parameters": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "message": {"type": "string", "description": "Email message"},
                    "smtp_host": {"type": "string", "description": "SMTP host", "default": "smtp.gmail.com"},
                    "smtp_port": {"type": "integer", "description": "SMTP port", "default": 587}
                },
                "workflow_definition": {
                    "nodes": [
                        {
                            "id": "start",
                            "type": "n8n-nodes-base.start",
                            "position": [100, 200],
                            "parameters": {}
                        },
                        {
                            "id": "send_email",
                            "type": "n8n-nodes-base.emailSend",
                            "position": [300, 200],
                            "parameters": {
                                "fromEmail": "your-email@example.com",
                                "toEmail": "={{$json.to}}",
                                "subject": "={{$json.subject}}",
                                "message": "={{$json.message}}",
                                "transport": "smtp",
                                "smtpHost": "={{$json.smtp_host}}",
                                "smtpPort": "={{$json.smtp_port}}",
                                "smtpAuth": True
                            }
                        }
                    ],
                    "connections": {
                        "start": {"main": [[{"node": "send_email", "type": "main", "index": 0}]]}
                    }
                }
            },
            "data_processor": {
                "name": "Data Processing Tool",
                "description": "Process and transform data",
                "parameters": {
                    "data": {"type": "object", "description": "Data to process"},
                    "operation": {"type": "string", "description": "Processing operation: filter, transform, aggregate"}
                },
                "workflow_definition": {
                    "nodes": [
                        {
                            "id": "start",
                            "type": "n8n-nodes-base.start",
                            "position": [100, 200],
                            "parameters": {}
                        },
                        {
                            "id": "code_processor",
                            "type": "n8n-nodes-base.code",
                            "position": [300, 200],
                            "parameters": {
                                "language": "javascript",
                                "jsCode": """
                                const data = $input.all()[0].json.data;
                                const operation = $input.all()[0].json.operation;
                                
                                let result;
                                switch(operation) {
                                    case 'filter':
                                        result = data.filter(item => item !== null && item !== undefined);
                                        break;
                                    case 'transform':
                                        result = data.map(item => typeof item === 'string' ? item.toUpperCase() : item);
                                        break;
                                    case 'aggregate':
                                        result = Array.isArray(data) ? {
                                            count: data.length,
                                            summary: data.slice(0, 5)
                                        } : data;
                                        break;
                                    default:
                                        result = data;
                                }
                                
                                return [{ json: { result, operation, processed_at: new Date().toISOString() } }];
                                """
                            }
                        }
                    ],
                    "connections": {
                        "start": {"main": [[{"node": "code_processor", "type": "main", "index": 0}]]}
                    }
                }
            }
        }
    
    async def start_n8n_server(self) -> bool:
        """Start n8n server if not already running"""
        try:
            # Check if n8n is already running
            if await self.is_n8n_running():
                logger.info("n8n server is already running")
                return True
            
            logger.info("Starting n8n server...")
            
            # Start n8n process
            env = os.environ.copy()
            env["N8N_HOST"] = "localhost"
            env["N8N_PORT"] = "5678"
            env["N8N_PROTOCOL"] = "http"
            
            self.n8n_process = subprocess.Popen(
                ["n8n", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for n8n to start (max 30 seconds)
            for _ in range(30):
                if await self.is_n8n_running():
                    logger.info("n8n server started successfully")
                    await self._setup_default_workflows()
                    return True
                await asyncio.sleep(1)
            
            logger.error("Failed to start n8n server within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting n8n server: {e}")
            return False
    
    async def stop_n8n_server(self):
        """Stop n8n server"""
        if self.n8n_process:
            try:
                if os.name == 'nt':
                    self.n8n_process.terminate()
                else:
                    os.killpg(os.getpgid(self.n8n_process.pid), signal.SIGTERM)
                
                self.n8n_process.wait(timeout=10)
                logger.info("n8n server stopped")
            except Exception as e:
                logger.error(f"Error stopping n8n server: {e}")
                if os.name == 'nt':
                    self.n8n_process.kill()
                else:
                    os.killpg(os.getpgid(self.n8n_process.pid), signal.SIGKILL)
    
    async def is_n8n_running(self) -> bool:
        """Check if n8n server is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.n8n_url}/healthz", timeout=5)
                return response.status_code == 200
        except Exception:
            return False
    
    async def _setup_default_workflows(self):
        """Setup default workflows in n8n"""
        try:
            for workflow_id, workflow_info in self.default_workflows.items():
                await self.create_or_update_workflow(
                    workflow_id=workflow_id,
                    workflow_definition=workflow_info["workflow_definition"],
                    name=workflow_info["name"]
                )
            logger.info("Default workflows setup completed")
        except Exception as e:
            logger.error(f"Error setting up default workflows: {e}")
    
    async def create_or_update_workflow(self, 
                                     workflow_id: str, 
                                     workflow_definition: Dict,
                                     name: str = None) -> bool:
        """Create or update a workflow in n8n"""
        try:
            async with httpx.AsyncClient() as client:
                # Check if workflow exists
                existing_workflows = await self.list_workflows()
                existing_workflow = None
                
                for workflow in existing_workflows:
                    if workflow.get("name") == (name or workflow_id):
                        existing_workflow = workflow
                        break
                
                workflow_data = {
                    "name": name or workflow_id,
                    "nodes": workflow_definition.get("nodes", []),
                    "connections": workflow_definition.get("connections", {}),
                    "active": True,
                    "settings": {
                        "executionOrder": "v1"
                    }
                }
                
                if existing_workflow:
                    # Update existing workflow
                    response = await client.put(
                        f"{self.n8n_url}/api/v1/workflows/{existing_workflow['id']}",
                        headers=self.headers,
                        json=workflow_data,
                        timeout=30
                    )
                else:
                    # Create new workflow
                    response = await client.post(
                        f"{self.n8n_url}/api/v1/workflows",
                        headers=self.headers,
                        json=workflow_data,
                        timeout=30
                    )
                
                if response.status_code in [200, 201]:
                    workflow_result = response.json()
                    self.workflows_cache[workflow_id] = workflow_result
                    logger.info(f"Workflow '{workflow_id}' created/updated successfully")
                    return True
                else:
                    logger.error(f"Failed to create/update workflow: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating/updating workflow: {e}")
            return False
    
    async def list_workflows(self) -> List[Dict]:
        """List all workflows in n8n"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.n8n_url}/api/v1/workflows",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    workflows_data = response.json()
                    return workflows_data.get("data", []) if isinstance(workflows_data, dict) else workflows_data
                else:
                    logger.error(f"Failed to list workflows: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return []
    
    async def execute_workflow(self, 
                             workflow_id: str, 
                             input_data: Dict = None) -> Dict:
        """Execute a workflow with input data"""
        try:
            # Get workflow by name or ID
            workflows = await self.list_workflows()
            target_workflow = None
            
            for workflow in workflows:
                if (workflow.get("id") == workflow_id or 
                    workflow.get("name") == workflow_id):
                    target_workflow = workflow
                    break
            
            if not target_workflow:
                return {
                    "success": False,
                    "error": f"Workflow '{workflow_id}' not found"
                }
            
            async with httpx.AsyncClient() as client:
                execution_data = {
                    "workflowData": target_workflow,
                    "runData": {},
                    "startNodes": [],
                    "destinationNode": None
                }
                
                if input_data:
                    execution_data["runData"] = {
                        "start": [{
                            "json": input_data,
                            "pairedItem": {"item": 0}
                        }]
                    }
                
                response = await client.post(
                    f"{self.n8n_url}/api/v1/workflows/{target_workflow['id']}/run",
                    headers=self.headers,
                    json=execution_data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "execution_id": result.get("data", {}).get("executionId"),
                        "result": result.get("data", {}),
                        "workflow_id": workflow_id
                    }
                else:
                    logger.error(f"Workflow execution failed: {response.status_code} - {response.text}")
                    return {
                        "success": False,
                        "error": f"Execution failed with status {response.status_code}",
                        "details": response.text
                    }
                    
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_tools(self) -> Dict[str, Dict]:
        """Get available tools (workflows) with their descriptions"""
        tools = {}
        for tool_id, tool_info in self.default_workflows.items():
            tools[tool_id] = {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            }
        return tools
    
    async def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Call a tool (execute workflow) with parameters"""
        if not await self.is_n8n_running():
            if self.auto_start:
                await self.start_n8n_server()
            else:
                return {
                    "success": False,
                    "error": "n8n server is not running"
                }
        
        # Execute the workflow
        result = await self.execute_workflow(tool_name, parameters)
        
        if result["success"]:
            # Process the result based on tool type
            return await self._process_tool_result(tool_name, result, parameters)
        else:
            return result
    
    async def _process_tool_result(self, tool_name: str, raw_result: Dict, parameters: Dict) -> Dict:
        """Process tool execution result into a standardized format"""
        try:
            result_data = raw_result.get("result", {})
            
            if tool_name == "web_search":
                # Process web search results
                search_data = result_data.get("data", {})
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": {
                        "query": parameters.get("query"),
                        "results": search_data.get("RelatedTopics", [])[:parameters.get("num_results", 5)],
                        "abstract": search_data.get("Abstract", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            elif tool_name == "file_operations":
                # Process file operation results
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": {
                        "operation": parameters.get("operation"),
                        "file_path": parameters.get("file_path"),
                        "content": result_data.get("data", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            elif tool_name == "data_processor":
                # Process data processing results
                processed_data = result_data.get("data", {})
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": processed_data
                }
            
            else:
                # Generic result processing
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result_data,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing tool result: {e}")
            return {
                "success": False,
                "error": f"Error processing result: {str(e)}",
                "raw_result": raw_result
            }
    
    async def get_workflow_status(self) -> Dict:
        """Get status of n8n service and workflows"""
        status = {
            "n8n_running": await self.is_n8n_running(),
            "n8n_url": self.n8n_url,
            "workflows_count": 0,
            "available_tools": [],
            "error": None
        }
        
        try:
            if status["n8n_running"]:
                workflows = await self.list_workflows()
                status["workflows_count"] = len(workflows)
                status["available_tools"] = list(self.get_available_tools().keys())
            else:
                status["error"] = "n8n server is not running"
        except Exception as e:
            status["error"] = str(e)
        
        return status


# Global n8n service instance
n8n_service = N8nWorkflowService()
