# Components package

from .web_search_toggle import WebSearchToggle, get_web_search_toggle
from .tts_player import TTSPlayer, render_tts_button, is_tts_available, get_tts_player

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
    # Legacy (deprecated)
    'TTSPlayback',
    'InlinePlayButton',
    'check_tts_available',
]
