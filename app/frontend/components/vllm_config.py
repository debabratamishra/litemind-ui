"""
vLLM backend configuration component.
"""
import logging
import time
import streamlit as st
from typing import List, Optional, Tuple
from pathlib import Path
import os

from ..services.backend_service import backend_service

logger = logging.getLogger(__name__)


class VLLMConfig:
    """Handles vLLM backend configuration."""
    
    def __init__(self):
        self.backend_service = backend_service
    
    @staticmethod
    def _detect_docker_environment() -> bool:
        """Detect if running inside a Docker container."""
        # Check for common container indicators
        container_indicators = [
            Path('/.dockerenv').exists(),
            Path('/proc/1/cgroup').exists() and 'docker' in Path('/proc/1/cgroup').read_text(errors='ignore'),
            os.getenv('CONTAINER') is not None,
            os.getenv('DOCKER_CONTAINER') is not None
        ]
        return any(container_indicators)
    
    def render_config(self) -> Tuple[bool, Optional[str]]:
        """Render vLLM configuration and return (token_valid, token)."""
        st.sidebar.subheader("🤗 vLLM Configuration")
        
        # Check if running in Docker and show warning
        is_docker = st.session_state.get("is_docker_deployment", self._detect_docker_environment())
        if is_docker:
            st.sidebar.error("❌ vLLM is not supported with Docker installation yet")
            st.sidebar.info("🔄 This feature will be added shortly")
            st.sidebar.info("💡 Please use Ollama backend for now")
            return False, None
        
        # Fetch available models
        models_data = self._fetch_models()
        local_models = models_data.get("local_models", [])
        
        # Model selection (Popular Models removed by request)
        selected_model, hf_token = self._render_model_selection(local_models)
        
        # Token validation
        hf_token_valid = self._render_token_validation(hf_token)
        
        # Model actions
        if selected_model:
            self._render_model_actions(selected_model, local_models)
        
        # Server status
        self._render_server_status()
        
        return hf_token_valid, hf_token
    
    def _fetch_models(self) -> dict:
        """Fetch available models from backend."""
        try:
            return self.backend_service.get_vllm_models()
        except Exception as e:
            st.sidebar.error("❌ Could not fetch models")
            logger.error(f"Error fetching vLLM models: {e}")
            return {"local_models": []}
    
    def _render_model_selection(self, local_models: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Render model selection interface."""
        st.sidebar.subheader("Model Selection")
        
        model_source = st.sidebar.radio(
            "Choose model source:",
            ["Local Models", "Custom Model"],
            key="model_source",
        )

        selected_model = None
        hf_token = None

        if model_source == "Local Models":
            selected_model = self._render_local_models(local_models)
        elif model_source == "Custom Model":
            selected_model, hf_token = self._render_custom_model()

        return selected_model, hf_token
    
    def _render_local_models(self, local_models: List[str]) -> Optional[str]:
        """Render local model selection."""
        if local_models:
            return st.sidebar.selectbox(
                "Select local model:",
                local_models,
                key="local_model_select",
            )
        else:
            st.sidebar.info("No local models found. Choose 'Custom Model' to download and cache one.")
            return None
    
    def _render_custom_model(self) -> Tuple[Optional[str], Optional[str]]:
        """Render custom model input with token."""
        selected_model = st.sidebar.text_input(
            "Enter model repo or local path:",
            placeholder="e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0 or /path/to/local/model",
            key="custom_model_input",
        )
        
        st.sidebar.markdown("---")
        hf_token = st.sidebar.text_input(
            "Huggingface Token",
            type="password",
            help="Enter your Huggingface access token if the model requires authentication.",
            key="hf_token_custom",
        )
        
        return selected_model, hf_token
    
    def _render_token_validation(self, hf_token: Optional[str]) -> bool:
        """Render token validation."""
        if hf_token and st.sidebar.button("Validate Token", key="validate_token"):
            try:
                if self.backend_service.validate_hf_token(hf_token):
                    st.sidebar.success("✅ Token validated successfully!")
                    st.session_state.hf_token_valid = True
                    st.session_state.hf_token = hf_token
                    return True
                else:
                    st.sidebar.error("❌ Invalid token")
                    st.session_state.hf_token_valid = False
                    return False
            except Exception as e:
                st.sidebar.error(f"❌ Error validating token: {e}")
                st.session_state.hf_token_valid = False
                return False
        
        # Store token in session state
        if hf_token:
            st.session_state.hf_token = hf_token
        
        return st.session_state.get("hf_token_valid", False)
    
    def _render_model_actions(self, selected_model: str, local_models: List[str]):
        """Render model download and server control actions."""
        # Download button for non-local models
        if selected_model not in local_models:
            if st.sidebar.button("Download Model", key="download_model"):
                self._download_model(selected_model)
        
        # Server controls
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Start Server", key="start_server"):
                self._start_server(selected_model)
        
        with col2:
            if st.button("Stop Server", key="stop_server"):
                self._stop_server()
    
    def _download_model(self, model_name: str):
        """Download a model."""
        with st.sidebar:
            with st.spinner(f"Downloading {model_name}..."):
                try:
                    success = self.backend_service.download_vllm_model(model_name)
                    if success:
                        st.success(f"✅ Downloaded {model_name}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        hf_token = st.session_state.get("hf_token")
                        if not hf_token:
                            st.error("❌ Download failed. If the repo is gated/private, validate your Hugging Face token above and try again.")
                        else:
                            st.error("❌ Download failed")
                except Exception as e:
                    st.error(f"❌ Download error: {e}")
    
    def _start_server(self, model_name: str):
        """Start vLLM server."""
        with st.spinner("Starting vLLM server..."):
            try:
                success = self.backend_service.start_vllm_server(model_name, "auto")
                if success:
                    st.success("✅ Server started!")
                    st.session_state.vllm_model = model_name
                    time.sleep(1)
                    st.rerun()
                else:
                    hf_token = st.session_state.get("hf_token")
                    if not hf_token:
                        st.error("❌ Failed to start. If the model requires download/auth, validate your Hugging Face token above.")
                    else:
                        st.error("❌ Failed to start server")
            except Exception as e:
                st.error(f"❌ Server start error: {e}")
    
    def _stop_server(self):
        """Stop vLLM server."""
        try:
            success = self.backend_service.stop_vllm_server()
            if success:
                st.success("✅ Server stopped!")
                if "vllm_model" in st.session_state:
                    del st.session_state.vllm_model
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to stop server")
        except Exception as e:
            st.error(f"❌ Stop error: {e}")
    
    def _render_server_status(self):
        """Render current server status."""
        try:
            status = self.backend_service.get_vllm_server_status()
            if status.get("running"):
                st.sidebar.success(f"🟢 vLLM Server: {status.get('current_model', 'Running')}")
            else:
                st.sidebar.info("🔴 vLLM Server: Stopped")
        except Exception:
            st.sidebar.info("🔴 vLLM Server: Unknown")


def setup_vllm_backend() -> Tuple[bool, Optional[str]]:
    """Setup vLLM backend configuration in sidebar."""
    vllm_config = VLLMConfig()
    return vllm_config.render_config()
