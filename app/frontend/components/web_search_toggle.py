"""
Web search toggle component for enabling/disabling web search functionality.
"""
import logging
import streamlit as st
import requests
from typing import Dict, Optional

from app.frontend.config import FASTAPI_URL, FASTAPI_TIMEOUT

logger = logging.getLogger(__name__)


class WebSearchToggle:
    """Component for web search toggle control"""
    
    def __init__(self):
        self.base_url = FASTAPI_URL
        self.timeout = FASTAPI_TIMEOUT
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if "web_search_enabled" not in st.session_state:
            st.session_state.web_search_enabled = False
        
        if "serp_token_status_cache" not in st.session_state:
            st.session_state.serp_token_status_cache = None
    
    def render(self) -> bool:
        """
        Render toggle control and return current state.
        
        Returns:
            bool: Current toggle state (True if enabled, False if disabled)
        """
        # Create a clean toggle with icon
        col1, col2 = st.columns([1, 20])
        
        with col1:
            # Toggle button with web search icon
            toggle_state = st.checkbox(
                "🌐",
                value=st.session_state.web_search_enabled,
                key="web_search_toggle_checkbox",
                help="Enable web search to get real-time information from the internet",
                label_visibility="visible"
            )
        
        # Update session state
        st.session_state.web_search_enabled = toggle_state
        
        return toggle_state
    
    def get_token_status(self) -> Dict[str, str]:
        """
        Check SerpAPI token status from backend.
        Caches the result to minimize API calls.
        
        Returns:
            dict: Status dictionary with 'status' and 'message' keys
                  status can be: 'valid', 'invalid', 'error'
        """
        # Return cached status if available
        if st.session_state.serp_token_status_cache is not None:
            return st.session_state.serp_token_status_cache
        
        try:
            response = requests.get(
                f"{self.base_url}/api/chat/serp-status",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                status_data = response.json()
                result = {
                    "status": status_data.get("status", "error"),
                    "message": status_data.get("message", "Unknown status")
                }
            else:
                result = {
                    "status": "error",
                    "message": f"Failed to check token status (HTTP {response.status_code})"
                }
            
            # Cache the result
            st.session_state.serp_token_status_cache = result
            return result
            
        except requests.RequestException as e:
            logger.error(f"Error checking SerpAPI token status: {e}")
            result = {
                "status": "error",
                "message": f"Connection error: {str(e)}"
            }
            # Cache error result as well
            st.session_state.serp_token_status_cache = result
            return result
    
    def clear_token_status_cache(self):
        """Clear the cached token status to force a refresh"""
        st.session_state.serp_token_status_cache = None


def get_web_search_toggle() -> bool:
    """
    Create and render a web search toggle component.
    
    Returns:
        bool: Current toggle state
    """
    toggle = WebSearchToggle()
    return toggle.render()
