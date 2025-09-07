"""
Workflows page for managing n8n workflows and tool use capabilities
"""
import json
import logging
import streamlit as st
from typing import Dict, List, Optional
import asyncio

from app.frontend.services.backend_service import backend_service
from app.frontend.components.text_renderer import render_llm_text
from app.frontend.components.streaming_handler import streaming_handler
from app.services.tool_use_chat import tool_use_chat_service

logger = logging.getLogger(__name__)


class WorkflowsPage:
    """Workflows interface controller"""
    
    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        
    def render(self):
        st.title("🔧 Workflow Automation & Tool Use")
        
        # Initialize session state
        if "workflow_messages" not in st.session_state:
            st.session_state.workflow_messages = []
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Tool Use Chat", "⚙️ Workflow Status", "🛠️ Manage Workflows", "📊 Tool Results"])
        
        with tab1:
            self._render_tool_use_chat()
        
        with tab2:
            self._render_workflow_status()
        
        with tab3:
            self._render_workflow_management()
        
        with tab4:
            self._render_tool_results()
    
    def _render_tool_use_chat(self):
        """Render the tool use chat interface"""
        st.subheader("💬 Chat with Tool Use Capabilities")
        
        # Display chat messages
        for message in st.session_state.workflow_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    render_llm_text(message["content"])
                else:
                    # For assistant messages, check if there are tool results
                    content = message["content"]
                    tool_results = message.get("tool_results", [])
                    
                    render_llm_text(content)
                    
                    # Display tool results if available
                    if tool_results:
                        st.markdown("---")
                        st.markdown("**🔧 Tool Execution Details:**")
                        for i, result in enumerate(tool_results, 1):
                            with st.expander(f"Tool {i}: {result.get('tool', 'Unknown')}"):
                                if result.get('success'):
                                    st.success("✅ Execution successful")
                                    st.json(result.get('result', {}))
                                else:
                                    st.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
        
        # Chat input
        if prompt := st.chat_input("Ask me anything and I'll use tools to help you..."):
            self._process_tool_use_message(prompt)
    
    def _process_tool_use_message(self, user_input: str):
        """Process a message with tool use capabilities"""
        # Add user message
        st.session_state.workflow_messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            render_llm_text(user_input)
        
        # Generate response with tool use
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and preparing tools..."):
                # Get chat configuration
                backend_provider = st.session_state.get("current_backend", "ollama")
                config = self._get_chat_config(backend_provider)
                
                # Create conversation with tool use capabilities
                conversation = tool_use_chat_service.create_tool_use_conversation(
                    user_input, 
                    st.session_state.workflow_messages[:-1]  # Exclude the just-added user message
                )
                
                # Generate initial response
                placeholder = st.empty()
                initial_response = ""
                
                # Stream the LLM response
                for chunk in streaming_handler._stream_response_generator(
                    messages=conversation,
                    model=config["model"], 
                    temperature=config["temperature"],
                    backend=backend_provider,
                    hf_token=config.get("hf_token"),
                    use_fastapi=self.backend_available
                ):
                    initial_response += chunk
                    placeholder.markdown(initial_response + "▌")
                
                placeholder.markdown(initial_response)
                
                # Check if there are tool calls in the response
                if "<tool_call>" in initial_response:
                    with st.spinner("🔧 Executing tools..."):
                        # Process tool calls
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            processed_response, tool_results = loop.run_until_complete(
                                tool_use_chat_service.process_message_with_tools(initial_response)
                            )
                        finally:
                            loop.close()
                        
                        # Update the display with processed response
                        placeholder.markdown(processed_response)
                        
                        # Store message with tool results
                        st.session_state.workflow_messages.append({
                            "role": "assistant", 
                            "content": processed_response,
                            "tool_results": tool_results
                        })
                else:
                    # No tool calls, just store the regular response
                    st.session_state.workflow_messages.append({
                        "role": "assistant", 
                        "content": initial_response
                    })
    
    def _render_workflow_status(self):
        """Render workflow status and health information"""
        st.subheader("⚙️ Workflow System Status")
        
        # Get status
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        try:
            if self.backend_available:
                response = backend_service._make_request("GET", "/api/n8n/status")
                if response and response.get("status_code") == 200:
                    status_data = response["data"]
                    
                    # Display n8n status
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if status_data["n8n_running"]:
                            st.success("✅ n8n Server Running")
                        else:
                            st.error("❌ n8n Server Stopped")
                    
                    with col2:
                        st.metric("Workflows", status_data["workflows_count"])
                    
                    with col3:
                        st.metric("Available Tools", len(status_data["available_tools"]))
                    
                    # Display available tools
                    if status_data["available_tools"]:
                        st.markdown("**Available Tools:**")
                        for tool in status_data["available_tools"]:
                            st.markdown(f"• {tool}")
                    
                    # Display error if any
                    if status_data.get("error"):
                        st.error(f"Error: {status_data['error']}")
                    
                    # n8n Server Controls
                    st.markdown("---")
                    st.subheader("🎛️ Server Controls")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🚀 Start n8n Server"):
                            with st.spinner("Starting n8n server..."):
                                response = backend_service._make_request("POST", "/api/n8n/start")
                                if response and response.get("status_code") == 200:
                                    st.success("✅ n8n server started successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to start n8n server")
                    
                    with col2:
                        if st.button("🛑 Stop n8n Server"):
                            with st.spinner("Stopping n8n server..."):
                                response = backend_service._make_request("POST", "/api/n8n/stop")
                                if response and response.get("status_code") == 200:
                                    st.success("✅ n8n server stopped successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to stop n8n server")
                
                else:
                    st.error("Failed to get workflow status from backend")
            
            else:
                st.warning("⚠️ Backend not available. Using local tool status.")
                
                # Get local tool status
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    status = loop.run_until_complete(tool_use_chat_service.get_tool_status())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tools Loaded", status["tools_available"])
                    with col2:
                        n8n_running = status["n8n_status"]["n8n_running"]
                        if n8n_running:
                            st.success("✅ n8n Connected")
                        else:
                            st.error("❌ n8n Disconnected")
                    
                    st.json(status)
                finally:
                    loop.close()
                
        except Exception as e:
            st.error(f"Error getting status: {e}")
    
    def _render_workflow_management(self):
        """Render workflow management interface"""
        st.subheader("🛠️ Manage Workflows")
        
        # List workflows
        if st.button("📋 List All Workflows"):
            try:
                if self.backend_available:
                    response = backend_service._make_request("GET", "/api/n8n/workflows")
                    if response and response.get("status_code") == 200:
                        workflows = response["data"]["workflows"]
                        
                        if workflows:
                            st.markdown(f"**Found {len(workflows)} workflows:**")
                            for workflow in workflows:
                                with st.expander(f"🔧 {workflow.get('name', 'Unnamed')}"):
                                    st.json(workflow)
                        else:
                            st.info("No workflows found")
                    else:
                        st.error("Failed to fetch workflows")
                else:
                    st.warning("Backend not available")
            except Exception as e:
                st.error(f"Error listing workflows: {e}")
        
        # Get available tools
        st.markdown("---")
        st.subheader("🧰 Available Tools")
        
        try:
            if self.backend_available:
                response = backend_service._make_request("GET", "/api/n8n/tools")
                if response and response.get("status_code") == 200:
                    tools = response["data"]["tools"]
                    
                    for tool_name, tool_info in tools.items():
                        with st.expander(f"🔧 {tool_info['name']}"):
                            st.markdown(f"**Description:** {tool_info['description']}")
                            st.markdown("**Parameters:**")
                            for param_name, param_info in tool_info['parameters'].items():
                                required = "✅" if param_info.get('required', True) else "⚪"
                                default = f" (default: {param_info['default']})" if 'default' in param_info else ""
                                st.markdown(f"- {required} **{param_name}**: {param_info['description']}{default}")
                            
                            # Test tool
                            if st.button(f"Test {tool_name}", key=f"test_{tool_name}"):
                                self._test_tool(tool_name, tool_info)
                else:
                    st.error("Failed to fetch tools")
            else:
                # Show local tools
                tools = tool_use_chat_service.get_available_tools()
                for tool_name, tool_info in tools.items():
                    with st.expander(f"🔧 {tool_info['name']}"):
                        st.markdown(f"**Description:** {tool_info['description']}")
                        st.markdown("**Parameters:**")
                        for param_name, param_info in tool_info['parameters'].items():
                            required = "✅" if param_info.get('required', True) else "⚪"
                            default = f" (default: {param_info['default']})" if 'default' in param_info else ""
                            st.markdown(f"- {required} **{param_name}**: {param_info['description']}{default}")
        except Exception as e:
            st.error(f"Error getting tools: {e}")
    
    def _test_tool(self, tool_name: str, tool_info: Dict):
        """Test a specific tool with user-provided parameters"""
        st.markdown(f"### Test {tool_name}")
        
        # Create input fields for parameters
        params = {}
        for param_name, param_info in tool_info['parameters'].items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info['description']
            param_default = param_info.get('default')
            required = param_info.get('required', True)
            
            if param_type == 'string':
                value = st.text_input(
                    f"{param_name} {'*' if required else ''}",
                    value=param_default or "",
                    help=param_desc,
                    key=f"param_{tool_name}_{param_name}"
                )
                if value:
                    params[param_name] = value
            elif param_type == 'integer':
                value = st.number_input(
                    f"{param_name} {'*' if required else ''}",
                    value=param_default or 0,
                    help=param_desc,
                    key=f"param_{tool_name}_{param_name}"
                )
                params[param_name] = int(value)
            elif param_type == 'object':
                value = st.text_area(
                    f"{param_name} {'*' if required else ''} (JSON)",
                    value=json.dumps(param_default) if param_default else "{}",
                    help=param_desc,
                    key=f"param_{tool_name}_{param_name}"
                )
                try:
                    params[param_name] = json.loads(value)
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON for {param_name}")
                    return
        
        if st.button(f"Execute {tool_name}", key=f"execute_{tool_name}"):
            with st.spinner(f"Executing {tool_name}..."):
                try:
                    if self.backend_available:
                        response = backend_service._make_request(
                            "POST", 
                            "/api/n8n/tools/call",
                            json={"tool_name": tool_name, "parameters": params}
                        )
                        if response and response.get("status_code") == 200:
                            result = response["data"]
                            if result["success"]:
                                st.success("✅ Tool executed successfully!")
                                st.json(result["result"])
                            else:
                                st.error(f"❌ Tool execution failed: {result.get('error')}")
                        else:
                            st.error("Failed to execute tool")
                    else:
                        # Execute locally
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(
                                tool_use_chat_service._execute_tool_calls([{"tool": tool_name, "parameters": params}])
                            )
                            if result and result[0]["success"]:
                                st.success("✅ Tool executed successfully!")
                                st.json(result[0]["result"])
                            else:
                                st.error(f"❌ Tool execution failed: {result[0].get('error') if result else 'Unknown error'}")
                        finally:
                            loop.close()
                except Exception as e:
                    st.error(f"Error executing tool: {e}")
    
    def _render_tool_results(self):
        """Render tool execution results and history"""
        st.subheader("📊 Tool Execution Results")
        
        # Show recent tool executions from chat history
        tool_executions = []
        for message in st.session_state.workflow_messages:
            if message["role"] == "assistant" and "tool_results" in message:
                tool_executions.extend(message["tool_results"])
        
        if tool_executions:
            st.markdown(f"**Recent tool executions ({len(tool_executions)}):**")
            
            for i, result in enumerate(tool_executions, 1):
                with st.expander(f"Execution {i}: {result.get('tool', 'Unknown')}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if result.get('success'):
                            st.success("✅ Success")
                        else:
                            st.error("❌ Failed")
                        
                        tool_name = result.get('tool', 'Unknown')
                        st.markdown(f"**Tool:** {tool_name}")
                        
                        if 'execution_time' in result:
                            st.markdown(f"**Time:** {result['execution_time']:.2f}s")
                    
                    with col2:
                        if result.get('success'):
                            st.markdown("**Result:**")
                            st.json(result.get('result', {}))
                        else:
                            st.markdown("**Error:**")
                            st.error(result.get('error', 'Unknown error'))
        else:
            st.info("No tool executions found. Use the Tool Use Chat to execute tools.")
        
        # Clear history button
        if st.button("🗑️ Clear Tool Results"):
            for message in st.session_state.workflow_messages:
                if "tool_results" in message:
                    del message["tool_results"]
            st.rerun()
    
    def _get_chat_config(self, backend_provider: str) -> Dict:
        """Get chat configuration"""
        config = {
            "temperature": st.session_state.get("chat_temperature", 0.7),
            "hf_token": st.session_state.get("hf_token") if backend_provider == "vllm" else None
        }
        
        if backend_provider == "vllm":
            config["model"] = st.session_state.get("vllm_model", "no-model")
        else:
            config["model"] = st.session_state.get("selected_chat_model", "default")
        
        return config
    
    def render_sidebar_config(self):
        """Render sidebar configuration for workflows"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Workflow Settings")
        
        # Auto-start n8n server
        auto_start = st.sidebar.checkbox(
            "Auto-start n8n server",
            value=st.session_state.get("auto_start_n8n", True),
            help="Automatically start n8n server when needed"
        )
        st.session_state.auto_start_n8n = auto_start
        
        # Tool execution timeout
        timeout = st.sidebar.slider(
            "Tool execution timeout (seconds)",
            min_value=10,
            max_value=120,
            value=st.session_state.get("tool_timeout", 30),
            help="Maximum time to wait for tool execution"
        )
        st.session_state.tool_timeout = timeout
        
        # Clear workflow chat
        if st.sidebar.button("🗑️ Clear Workflow Chat"):
            st.session_state.workflow_messages.clear()
            st.rerun()


def render_workflows_page():
    """Render the workflows page"""
    page = WorkflowsPage()
    page.render()
    page.render_sidebar_config()
