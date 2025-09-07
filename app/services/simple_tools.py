"""
Simple tool demonstration for LiteMind UI
This provides basic tool calling without requiring complex n8n workflows
"""
import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
import httpx
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleToolService:
    """Simple tool service for demonstration purposes"""
    
    def __init__(self):
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, Dict]:
        """Initialize available tools"""
        return {
            "web_search": {
                "name": "Web Search Tool",
                "description": "Search the web using DuckDuckGo",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results", "default": 5}
                }
            },
            "file_reader": {
                "name": "File Reader Tool", 
                "description": "Read contents of a file",
                "parameters": {
                    "file_path": {"type": "string", "description": "Path to the file to read"}
                }
            },
            "system_info": {
                "name": "System Information Tool",
                "description": "Get system information",
                "parameters": {}
            },
            "calculator": {
                "name": "Calculator Tool",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                }
            }
        }
    
    def get_available_tools(self) -> Dict[str, Dict]:
        """Get available tools"""
        return self.tools
    
    async def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Call a specific tool with parameters"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            if tool_name == "web_search":
                return await self._web_search(parameters)
            elif tool_name == "file_reader":
                return await self._read_file(parameters)
            elif tool_name == "system_info":
                return await self._get_system_info(parameters)
            elif tool_name == "calculator":
                return await self._calculate(parameters)
            else:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' implementation not found"
                }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _web_search(self, parameters: Dict) -> Dict:
        """Perform web search using DuckDuckGo"""
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 5)
        
        if not query:
            return {
                "success": False,
                "error": "Query parameter is required"
            }
        
        try:
            # Use DuckDuckGo Instant Answer API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract results
                    results = []
                    if data.get("RelatedTopics"):
                        for topic in data["RelatedTopics"][:num_results]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "title": topic.get("Text", "")[:100] + "...",
                                    "url": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", "")
                                })
                    
                    return {
                        "success": True,
                        "tool": "web_search",
                        "result": {
                            "query": query,
                            "abstract": data.get("Abstract", ""),
                            "results": results,
                            "total_found": len(results),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Search API returned status code {response.status_code}"
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}"
            }
    
    async def _read_file(self, parameters: Dict) -> Dict:
        """Read contents of a file"""
        file_path = parameters.get("file_path", "")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path parameter is required"
            }
        
        try:
            import os
            from pathlib import Path
            
            # Security check - only allow reading files in current directory and subdirectories
            path = Path(file_path).resolve()
            current_dir = Path.cwd().resolve()
            
            try:
                path.relative_to(current_dir)
            except ValueError:
                return {
                    "success": False,
                    "error": "File access denied - path is outside current directory"
                }
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            if not path.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}"
                }
            
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "tool": "file_reader",
                "result": {
                    "file_path": str(path),
                    "content": content,
                    "size": len(content),
                    "lines": len(content.splitlines()),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": "File is not a text file or uses unsupported encoding"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}"
            }
    
    async def _get_system_info(self, parameters: Dict) -> Dict:
        """Get system information"""
        try:
            import platform
            import psutil
            import os
            
            info = {
                "system": platform.system(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('.').total,
                    "free": psutil.disk_usage('.').free,
                    "percent": psutil.disk_usage('.').percent
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "tool": "system_info",
                "result": info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get system info: {str(e)}"
            }
    
    async def _calculate(self, parameters: Dict) -> Dict:
        """Perform mathematical calculations"""
        expression = parameters.get("expression", "")
        
        if not expression:
            return {
                "success": False,
                "error": "expression parameter is required"
            }
        
        try:
            # Simple security check - only allow basic mathematical operations
            allowed_chars = set("0123456789+-*/()., ")
            allowed_words = {"sin", "cos", "tan", "sqrt", "log", "abs", "max", "min", "pow"}
            
            # Basic validation
            clean_expr = expression.lower().strip()
            for word in allowed_words:
                clean_expr = clean_expr.replace(word, "")
            
            if not all(c in allowed_chars for c in clean_expr):
                return {
                    "success": False,
                    "error": "Expression contains disallowed characters"
                }
            
            # Use eval with restricted globals for basic math
            import math
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "pi": math.pi,
                "e": math.e
            }
            
            result = eval(expression, safe_dict, {})
            
            return {
                "success": True,
                "tool": "calculator",
                "result": {
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Calculation failed: {str(e)}"
            }


# Global simple tool service instance
simple_tool_service = SimpleToolService()
