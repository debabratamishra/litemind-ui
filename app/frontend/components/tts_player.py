"""
Unified Text-to-Speech player component for Streamlit.

This module provides a single, reusable TTS player that can be used across
all pages (Chat, RAG, etc.) without code duplication.
"""
import io
import logging
from typing import Optional

import requests
import streamlit as st

from ..config import FASTAPI_URL

logger = logging.getLogger(__name__)


class TTSPlayer:
    """
    Unified TTS player component.
    
    Provides a consistent TTS experience across all pages with:
    - Audio synthesis via Edge TTS (backend)
    - Caching to avoid re-synthesizing same text
    - Error handling with user-friendly messages
    - Clean audio player UI with close button
    """
    
    def __init__(self):
        self.base_url = FASTAPI_URL
        self._status_checked = False
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if TTS service is available.
        
        Uses session state caching to avoid repeated API calls.
        Returns False on connection errors (doesn't cache failures).
        """
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
            # Don't cache failures - retry next time
            return False
    
    @staticmethod
    def clear_cache():
        """Clear TTS availability cache to force re-check."""
        if "tts_service_available" in st.session_state:
            del st.session_state["tts_service_available"]
    
    def _synthesize(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Synthesize text to audio bytes.
        
        Args:
            text: Text to convert to speech
            voice: Optional voice ID (e.g., 'en-US-AriaNeural')
            
        Returns:
            Audio bytes (MP3 format) or None on error
        """
        try:
            logger.debug(f"TTS synthesize request: {len(text)} chars")
            response = requests.post(
                f"{self.base_url}/api/tts/synthesize",
                json={"text": text, "voice": voice, "use_cache": True},
                timeout=60
            )
            
            logger.info(
                f"TTS response: status={response.status_code}, "
                f"size={len(response.content)}, "
                f"content_type={response.headers.get('content-type')}"
            )
            
            if response.status_code == 200 and len(response.content) > 0:
                content_type = response.headers.get('content-type', '')
                if 'audio' in content_type:
                    return response.content
                    
                # Try to parse as error JSON
                try:
                    error_data = response.json()
                    logger.error(f"TTS returned non-audio: {error_data}")
                except:
                    # Not JSON, might still be audio
                    return response.content
            else:
                try:
                    error_data = response.json()
                    logger.error(f"TTS failed: {response.status_code} - {error_data}")
                except:
                    logger.error(f"TTS failed: {response.status_code}")
                    
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
    
    def render(
        self,
        text: str,
        key_prefix: str,
        message_index: int,
        voice: Optional[str] = None,
        button_icon: str = "🗣️",
        show_close: bool = True
    ) -> None:
        """
        Render TTS play button and audio player.
        
        Args:
            text: Text to convert to speech
            key_prefix: Unique prefix for session state keys (e.g., 'chat', 'rag')
            message_index: Index of the message (for unique keys)
            voice: Optional voice ID
            button_icon: Icon/emoji for the play button
            show_close: Whether to show close button for audio player
        """
        # Generate unique keys for this message
        audio_key = f"{key_prefix}_tts_audio_{message_index}"
        show_key = f"{key_prefix}_tts_show_{message_index}"
        
        # Initialize session state
        if audio_key not in st.session_state:
            st.session_state[audio_key] = None
        if show_key not in st.session_state:
            st.session_state[show_key] = False
        
        # If audio is loaded, show the player
        if st.session_state[show_key] and st.session_state[audio_key]:
            self._render_audio_player(
                audio_bytes=st.session_state[audio_key],
                close_key=f"{key_prefix}_close_{message_index}",
                show_key=show_key,
                show_close=show_close
            )
        else:
            # Show play button
            self._render_play_button(
                text=text,
                voice=voice,
                button_key=f"{key_prefix}_play_{message_index}",
                audio_key=audio_key,
                show_key=show_key,
                button_icon=button_icon
            )
    
    def _render_audio_player(
        self,
        audio_bytes: bytes,
        close_key: str,
        show_key: str,
        show_close: bool = True
    ) -> None:
        """Render the audio player with optional close button."""
        try:
            logger.debug(f"Rendering audio player: {len(audio_bytes)} bytes")
            
            if show_close:
                col1, col2 = st.columns([15, 1])
                with col1:
                    if isinstance(audio_bytes, bytes):
                        st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")
                    else:
                        st.audio(audio_bytes, format="audio/mpeg")
                with col2:
                    if st.button("✕", key=close_key, help="Close"):
                        st.session_state[show_key] = False
                        st.rerun()
            else:
                if isinstance(audio_bytes, bytes):
                    st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")
                else:
                    st.audio(audio_bytes, format="audio/mpeg")
                    
        except Exception as e:
            logger.error(f"Audio player error: {type(e).__name__}: {e}")
            st.error(f"Audio playback error: {e}")
    
    def _render_play_button(
        self,
        text: str,
        voice: Optional[str],
        button_key: str,
        audio_key: str,
        show_key: str,
        button_icon: str = "🗣️"
    ) -> None:
        """Render the play button and handle synthesis."""
        if st.button(button_icon, key=button_key, help="Read aloud"):
            with st.spinner("Generating audio..."):
                audio_data = self._synthesize(text, voice)
                
                if audio_data:
                    st.session_state[audio_key] = audio_data
                    st.session_state[show_key] = True
                    logger.info("Audio saved to session state, triggering rerun")
                    st.rerun()
                else:
                    st.error("Failed to generate audio. Please try again.")


# Singleton instance for convenience
_tts_player = None


def get_tts_player() -> TTSPlayer:
    """Get the singleton TTS player instance."""
    global _tts_player
    if _tts_player is None:
        _tts_player = TTSPlayer()
    return _tts_player


def render_tts_button(
    text: str,
    key_prefix: str,
    message_index: int,
    voice: Optional[str] = None
) -> None:
    """
    Convenience function to render a TTS button.
    
    Args:
        text: Text to convert to speech
        key_prefix: Unique prefix for session state keys (e.g., 'chat', 'rag')
        message_index: Index of the message
        voice: Optional voice ID
    """
    get_tts_player().render(text, key_prefix, message_index, voice)


def is_tts_available() -> bool:
    """Check if TTS service is available."""
    return TTSPlayer.is_available()
