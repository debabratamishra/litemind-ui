# Components package

from .web_search_toggle import WebSearchToggle, get_web_search_toggle
from .tts_player import TTSPlayer, render_tts_button, is_tts_available, get_tts_player
from .conversation_sidebar import (
    ConversationHistorySidebar, 
    get_chat_sidebar, 
    get_rag_sidebar
)
from .shared_ui import (
    GenerationConfig,
    render_memory_indicator,
    render_generation_settings,
    render_reasoning_config,
    render_memory_config,
    validate_backend_setup,
    create_simple_summary,
    get_generation_config_from_session,
)

# Legacy imports for backward compatibility
from .tts_playback import TTSPlayback, InlinePlayButton, check_tts_available

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
