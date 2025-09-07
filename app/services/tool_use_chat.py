"""
Tool-use chat service for integrating tools with LLM conversations
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import httpx

from app.services.simple_tools import simple_tool_service

from app.services.n8n_service import n8n_service

logger = logging.getLogger(__name__)


class ToolUseChatService:
    """Service for handling tool use in chat conversations"""
    
    def __init__(self):
        self.tools = simple_tool_service.get_available_tools()
        
    def get_available_tools(self) -> Dict[str, Dict]:
        """Get available tools"""
        return self.tools
    
    def create_tool_use_conversation(self, user_input: str, chat_history: List[Dict]) -> List[Dict]:
        """Create a conversation with tool use capabilities"""
        # Create system prompt with tool information
        system_prompt = self._create_system_prompt()
        
        # Build conversation with tool context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for message in chat_history[-5:]:  # Keep last 5 messages for context
            messages.append(message)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with tool information"""
        tools_info = []
        for tool_name, tool_data in self.tools.items():
            params_str = []
            for param_name, param_info in tool_data["parameters"].items():
                required = param_info.get("required", True)
                param_type = param_info["type"]
                description = param_info["description"]
                default = param_info.get("default", "")
                
                param_desc = f"{param_name} ({param_type}): {description}"
                if not required:
                    param_desc += " [optional]"
                if default:
                    param_desc += f" [default: {default}]"
                params_str.append(param_desc)
            
            tool_desc = f"""
**{tool_data['name']}** ({tool_name})
Description: {tool_data['description']}
Parameters:
{chr(10).join('  - ' + p for p in params_str)}
"""
            tools_info.append(tool_desc)
        
        system_prompt = f"""You are an AI assistant with access to the following tools:

{chr(10).join(tools_info)}

When you need to use a tool to answer a question, use the following format:

<tool_call>
{{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
</tool_call>

You can call multiple tools if needed. Each tool call should be in its own <tool_call> block.

Guidelines:
1. Only call tools when necessary to answer the user's question
2. Choose the most appropriate tool for the task
3. Provide clear, helpful responses that incorporate tool results
4. If a tool call fails, explain the issue and try an alternative approach if possible
5. Always explain what you're doing and why you're using specific tools

Remember to be helpful, accurate, and explain your reasoning when using tools.
"""
        return system_prompt
    
    async def process_message_with_tools(self, llm_response: str) -> Tuple[str, List[Dict]]:
        """Process LLM response and execute any tool calls"""
        tool_calls = self._extract_tool_calls(llm_response)
        
        if not tool_calls:
            return llm_response, []
        
        # Execute tool calls
        tool_results = await self._execute_tool_calls(tool_calls)
        
        # Generate final response incorporating tool results
        final_response = self._generate_final_response(llm_response, tool_calls, tool_results)
        
        return final_response, tool_results
    
    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        # Find all tool call blocks
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # Parse JSON
                tool_call = json.loads(match.strip())
                if "tool" in tool_call and "parameters" in tool_call:
                    tool_calls.append(tool_call)
                else:
                    logger.warning(f"Invalid tool call format: {tool_call}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}")
        
        return tool_calls
    
    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute a list of tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_name = tool_call["tool"]
                parameters = tool_call["parameters"]
                
                result = await simple_tool_service.call_tool(tool_name, parameters)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                results.append({
                    "success": False,
                    "tool": tool_call.get("tool", "unknown"),
                    "error": str(e)
                })
        
        return results
    
    def _generate_final_response(self, original_response: str, tool_calls: List[Dict], tool_results: List[Dict]) -> str:
        """Generate final response incorporating tool results"""
        # Remove tool call blocks from original response
        response_without_calls = re.sub(r'<tool_call>.*?</tool_call>', '', original_response, flags=re.DOTALL)
        response_without_calls = response_without_calls.strip()
        
        # Add tool results
        if tool_results:
            response_without_calls += "\n\n---\n\n**Tool Execution Results:**\n\n"
            
            for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results), 1):
                tool_name = tool_call.get("tool", "unknown")
                
                if result.get("success"):
                    response_without_calls += f"**{i}. {tool_name}:** ✅ Success\n"
                    
                    # Format result based on tool type
                    if tool_name == "web_search":
                        search_result = result.get("result", {})
                        response_without_calls += f"Query: {search_result.get('query', 'N/A')}\n"
                        if search_result.get("abstract"):
                            response_without_calls += f"Summary: {search_result['abstract'][:200]}...\n"
                        
                        results = search_result.get("results", [])
                        if results:
                            response_without_calls += "Top results:\n"
                            for j, res in enumerate(results[:3], 1):
                                response_without_calls += f"  {j}. {res.get('snippet', 'No description')[:100]}...\n"
                    
                    elif tool_name == "file_reader":
                        file_result = result.get("result", {})
                        response_without_calls += f"File: {file_result.get('file_path', 'N/A')}\n"
                        response_without_calls += f"Size: {file_result.get('size', 0)} characters\n"
                        content = file_result.get("content", "")
                        if len(content) > 300:
                            response_without_calls += f"Content preview: {content[:300]}...\n"
                        else:
                            response_without_calls += f"Content: {content}\n"
                    
                    elif tool_name == "calculator":
                        calc_result = result.get("result", {})
                        response_without_calls += f"Expression: {calc_result.get('expression', 'N/A')}\n"
                        response_without_calls += f"Result: {calc_result.get('result', 'N/A')}\n"
                    
                    elif tool_name == "system_info":
                        sys_result = result.get("result", {})
                        response_without_calls += f"System: {sys_result.get('system', 'N/A')}\n"
                        response_without_calls += f"Platform: {sys_result.get('platform', 'N/A')}\n"
                        memory = sys_result.get("memory", {})
                        if memory:
                            response_without_calls += f"Memory usage: {memory.get('percent', 0):.1f}%\n"
                    
                    else:
                        # Generic result display
                        response_without_calls += f"Result: {json.dumps(result.get('result', {}), indent=2)}\n"
                
                else:
                    response_without_calls += f"**{i}. {tool_name}:** ❌ Failed\n"
                    response_without_calls += f"Error: {result.get('error', 'Unknown error')}\n"
                
                response_without_calls += "\n"
        
        return response_without_calls
    
    async def get_tool_status(self) -> Dict:
        """Get status of tool system"""
        return {
            "tools_available": len(self.tools),
            "tool_names": list(self.tools.keys()),
            "n8n_status": {
                "n8n_running": False,  # Simple tools don't require n8n
                "note": "Using simple tools implementation"
            }
        }


# Global tool use chat instance
tool_use_chat_service = ToolUseChatService()
