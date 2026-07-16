"""
Generative UI components for dynamic rendering in chat responses.

Inspired by the AG-UI (Agent-User Interaction) protocol's event-driven
component model. Renders rich, interactive UI elements from structured
specifications embedded in LLM output.

Protocol: LLM emits fenced code blocks with a ``ui:<component>`` language
tag. Most components use a JSON props object. ``ui:webapp`` and
``ui:iframe_app`` are the exceptions: their block bodies are raw
HTML/CSS/JS rendered inside a Streamlit iframe for interactive mini-apps.
"""

from .auto_enhance import auto_enhance_content
from .constants import GENERATIVE_UI_SYSTEM_PROMPT
from .dispatch import (
    _COMPONENT_REGISTRY,
    has_ui_blocks,
    render_mixed_content,
    render_ui_component,
)

__all__ = [
    "has_ui_blocks",
    "render_mixed_content",
    "render_ui_component",
    "auto_enhance_content",
    "GENERATIVE_UI_SYSTEM_PROMPT",
    "_COMPONENT_REGISTRY",
]
