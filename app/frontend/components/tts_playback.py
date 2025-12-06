"""
Text-to-Speech playback component for Streamlit.

DEPRECATED: This module is kept for backward compatibility.
Please use `tts_player.py` instead for new code:

    from app.frontend.components.tts_player import (
        TTSPlayer,
        render_tts_button,
        is_tts_available
    )

The new unified TTSPlayer provides:
- Single implementation (no code duplication)
- Consistent behavior across Chat and RAG pages
- Better error handling and logging
- Simplified API
"""
import base64
import logging
import warnings
from typing import Optional

import requests
import streamlit as st

from ..config import FASTAPI_URL

logger = logging.getLogger(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "tts_playback module is deprecated. Use tts_player instead.",
    DeprecationWarning,
    stacklevel=2
)


class TTSPlayback:
    """Text-to-Speech playback component."""
    
    def __init__(self):
        self.base_url = FASTAPI_URL
        self._tts_available = None
    
    def _check_tts_available(self) -> bool:
        """Check if TTS service is available."""
        # Use session state for caching to avoid stale results across sessions
        cache_key = "tts_service_available"
        
        # Check session state first
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        try:
            response = requests.get(f"{self.base_url}/api/tts/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                available = status.get("available", False)
                st.session_state[cache_key] = available
                return available
            else:
                logger.warning(f"TTS status check returned {response.status_code}")
                # Don't cache failure - retry next time
                return False
        except Exception as e:
            logger.debug(f"TTS status check failed: {e}")
            # Don't cache failure - retry next time
            return False
    
    def _get_audio(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """Get TTS audio from backend."""
        try:
            logger.info(f"Requesting TTS for text of length {len(text)}")
            response = requests.post(
                f"{self.base_url}/api/tts/synthesize",
                json={
                    "text": text,
                    "voice": voice,
                    "use_cache": True
                },
                timeout=60  # TTS can take a bit for longer texts
            )
            
            logger.info(f"TTS response: status={response.status_code}, size={len(response.content)}, content_type={response.headers.get('content-type')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'audio' in content_type and len(response.content) > 0:
                    return response.content
                else:
                    # Try to get error message
                    try:
                        error_data = response.json()
                        logger.error(f"TTS returned non-audio response: {error_data}")
                    except:
                        logger.error(f"TTS returned unexpected content type: {content_type}")
                    return None
            else:
                try:
                    error_data = response.json()
                    logger.error(f"TTS request failed: {response.status_code} - {error_data}")
                except:
                    logger.error(f"TTS request failed: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.error("TTS request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"TTS connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"TTS error: {type(e).__name__}: {e}")
            return None
    
    def render_play_button(
        self, 
        text: str, 
        message_index: int,
        voice: Optional[str] = None,
        button_text: str = "🗣️",
        help_text: str = "Read aloud"
    ) -> None:
        """
        Render a play button for TTS.
        
        Args:
            text: The text to convert to speech
            message_index: Unique index for this message (for state management)
            voice: Optional voice ID to use
            button_text: Text/emoji to show on the button
            help_text: Tooltip help text
        """
        if not self._check_tts_available():
            return
        
        # Create unique keys for this message
        audio_key = f"tts_audio_{message_index}"
        playing_key = f"tts_playing_{message_index}"
        loading_key = f"tts_loading_{message_index}"
        
        # Initialize state
        if audio_key not in st.session_state:
            st.session_state[audio_key] = None
        if playing_key not in st.session_state:
            st.session_state[playing_key] = False
        if loading_key not in st.session_state:
            st.session_state[loading_key] = False
        
        # Create a small container for the button
        col1, col2 = st.columns([1, 20])
        
        with col1:
            if st.session_state[loading_key]:
                st.spinner("...")
            else:
                if st.button(
                    button_text,
                    key=f"tts_btn_{message_index}",
                    help=help_text,
                    type="secondary"
                ):
                    st.session_state[loading_key] = True
                    st.rerun()
        
        # Handle audio generation and playback
        if st.session_state[loading_key]:
            with st.spinner("Generating audio..."):
                audio_data = self._get_audio(text, voice)
                
                if audio_data:
                    st.session_state[audio_key] = audio_data
                    st.session_state[playing_key] = True
                else:
                    st.toast("Failed to generate audio", icon="❌")
                
                st.session_state[loading_key] = False
                st.rerun()
        
        # Show audio player if we have audio
        if st.session_state[playing_key] and st.session_state[audio_key]:
            import io
            audio_bytes = st.session_state[audio_key]
            if isinstance(audio_bytes, bytes):
                st.audio(io.BytesIO(audio_bytes), format="audio/mp3", autoplay=True)
            else:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Add a close button to hide the player
            if st.button("✕", key=f"tts_close_{message_index}", help="Close audio player"):
                st.session_state[playing_key] = False
                st.rerun()


class InlinePlayButton:
    """Lightweight inline play button that shows audio on click."""
    
    def __init__(self):
        self.base_url = FASTAPI_URL
    
    def render(
        self,
        text: str,
        message_id: str,
        voice: Optional[str] = None
    ) -> None:
        """
        Render a minimal inline play button.
        
        Args:
            text: Text to convert to speech
            message_id: Unique identifier for this message
            voice: Optional voice ID
        """
        audio_key = f"inline_audio_{message_id}"
        show_key = f"inline_show_{message_id}"
        
        # Initialize state
        if audio_key not in st.session_state:
            st.session_state[audio_key] = None
        if show_key not in st.session_state:
            st.session_state[show_key] = False
        
        # If audio is already loaded, show player
        if st.session_state[show_key] and st.session_state[audio_key]:
            import io
            col1, col2 = st.columns([15, 1])
            with col1:
                audio_bytes = st.session_state[audio_key]
                if isinstance(audio_bytes, bytes):
                    st.audio(io.BytesIO(audio_bytes), format="audio/mp3")
                else:
                    st.audio(audio_bytes, format="audio/mp3")
            with col2:
                if st.button("✕", key=f"close_{message_id}", help="Close"):
                    st.session_state[show_key] = False
                    st.rerun()
        else:
            # Show play button
            if st.button("🔊", key=f"play_{message_id}", help="Read aloud"):
                with st.spinner("Generating audio..."):
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/tts/synthesize",
                            json={"text": text, "voice": voice, "use_cache": True},
                            timeout=60
                        )
                        if response.status_code == 200:
                            st.session_state[audio_key] = response.content
                            st.session_state[show_key] = True
                            st.rerun()
                        else:
                            st.toast("Failed to generate audio", icon="❌")
                    except Exception as e:
                        logger.error(f"TTS failed: {e}")
                        st.toast("Audio generation failed", icon="❌")


# Global instances
tts_playback = TTSPlayback()
inline_play_button = InlinePlayButton()


def render_tts_button(text: str, message_index: int, voice: Optional[str] = None) -> None:
    """Convenience function to render TTS play button."""
    tts_playback.render_play_button(text, message_index, voice)


def check_tts_available() -> bool:
    """Check if TTS service is available."""
    # Fresh check each time, don't rely on module-level instance cache
    cache_key = "tts_service_available"
    
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        response = requests.get(f"{FASTAPI_URL}/api/tts/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            available = status.get("available", False)
            st.session_state[cache_key] = available
            return available
        return False
    except Exception:
        return False
