# Components package

from .conversation_sidebar import ConversationHistorySidebar, get_chat_sidebar, get_rag_sidebar
from .shared_ui import (
    GenerationConfig,
    create_simple_summary,
    get_generation_config_from_session,
    render_generation_settings,
    render_memory_config,
    render_memory_indicator,
    render_reasoning_config,
    validate_backend_setup,
)

# Legacy imports for backward compatibility
from .tts_playback import InlinePlayButton, TTSPlayback, check_tts_available
from .tts_player import TTSPlayer, get_tts_player, is_tts_available, render_tts_button
from .web_search_toggle import WebSearchToggle, get_web_search_toggle

__all__ = [
    'WebSearchToggle',
    'get_web_search_toggle',
    # New unified TTS player
    'TTSPlayer',
    'render_tts_button',
    'is_tts_available',
    'get_tts_player',
    # Conversation history sidebar
    'ConversationHistorySidebar',
    'get_chat_sidebar',
    'get_rag_sidebar',
    # Shared UI components
    'GenerationConfig',
    'render_memory_indicator',
    'render_generation_settings',
    'render_reasoning_config',
    'render_memory_config',
    'validate_backend_setup',
    'create_simple_summary',
    'get_generation_config_from_session',
    # Legacy (deprecated)
    'TTSPlayback',
    'InlinePlayButton',
    'check_tts_available',
]
