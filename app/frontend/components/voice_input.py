"""
Voice input component with speech-to-text transcription.
"""
import logging
import streamlit as st
from typing import Optional

from ..utils.audio_recorder import AUDIO_RECORDER_AVAILABLE, record_audio
from ...services.speech_service import get_speech_service

logger = logging.getLogger(__name__)


class VoiceInput:
    """Voice input with audio recording and transcription"""
    
    def __init__(self, page_key: str):
        self.page_key = page_key
        self.voice_text_key = f"voice_text_{page_key}"
        self.voice_recording_key = f"voice_recording_{page_key}"
        
        # Initialize session state
        if self.voice_text_key not in st.session_state:
            st.session_state[self.voice_text_key] = ""
        if self.voice_recording_key not in st.session_state:
            st.session_state[self.voice_recording_key] = False

    def render_input(self, placeholder_text: str = "Type your message or use voice input...") -> Optional[str]:
        """Render voice input component and return transcribed/typed text."""
        input_container = st.container()
        
        with input_container:
            if not AUDIO_RECORDER_AVAILABLE:
                return self._render_fallback_input(placeholder_text)
            else:
                return self._render_voice_enabled_input(placeholder_text)

    def _render_fallback_input(self, placeholder_text: str) -> Optional[str]:
        """Render fallback input when voice is not available."""
        col_input, col_voice_disabled, col_send = st.columns([10, 1, 1])
        
        with col_input:
            text_input = st.text_input(
                placeholder_text,
                key=f"voice_text_fallback_{self.page_key}",
                label_visibility="collapsed",
                placeholder=placeholder_text
            )
        
        with col_voice_disabled:
            st.button("🎤", key=f"voice_disabled_{self.page_key}", 
                     help="Voice input unavailable", disabled=True)
        
        with col_send:
            submit_clicked = st.button("➤", key=f"voice_submit_fallback_{self.page_key}", 
                                     type="primary", disabled=not text_input.strip(), 
                                     help="Send message")
        
        if submit_clicked and text_input.strip():
            return text_input.strip()
        return None

    def _render_voice_enabled_input(self, placeholder_text: str) -> Optional[str]:
        """Render voice-enabled input interface."""
        col_input, col_voice, col_send = st.columns([10, 1, 1])
        
        with col_input:
            current_text = st.session_state[self.voice_text_key] or ""
            text_input = st.text_input(
                placeholder_text,
                value=current_text,
                key=f"voice_text_input_{self.page_key}",
                label_visibility="collapsed",
                placeholder=placeholder_text
            )
        
        with col_voice:
            if st.session_state[self.voice_recording_key]:
                if st.button("⏹️", key=f"voice_stop_{self.page_key}", 
                           help="Stop recording", type="secondary"):
                    st.session_state[self.voice_recording_key] = False
                    st.rerun()
            else:
                if st.button("🎤", key=f"voice_start_{self.page_key}", 
                           help="Start voice recording"):
                    st.session_state[self.voice_recording_key] = True
                    st.rerun()
        
        with col_send:
            final_text = text_input.strip() if text_input else ""
            submit_clicked = st.button("➤", key=f"voice_submit_{self.page_key}", 
                                     type="primary", disabled=not final_text, 
                                     help="Send message")
        
        # Handle voice recording
        if st.session_state[self.voice_recording_key]:
            self._handle_voice_recording()
        
        # Handle form submission
        if submit_clicked and final_text:
            st.session_state[self.voice_text_key] = ""  # Clear for next input
            return final_text
        
        return None

    def _handle_voice_recording(self) -> None:
        """Handle the voice recording process."""
        st.info("🎤 Recording... Click ⏹️ to stop")
        
        try:
            audio_data = record_audio(
                text="Recording...",
                recording_color="#e74c3c",
                neutral_color="#2ecc71",
                icon_name="microphone"
            )
            
            if audio_data:
                with st.spinner("🎤 Transcribing audio..."):
                    raw_bytes = self._extract_audio_bytes(audio_data)
                    
                    if raw_bytes:
                        try:
                            speech_service = get_speech_service()
                            transcribed_text = speech_service.transcribe_audio(raw_bytes)
                            if transcribed_text:
                                st.session_state[self.voice_text_key] = transcribed_text
                                st.session_state[self.voice_recording_key] = False
                                st.success(f"✅ Transcribed: {transcribed_text}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to transcribe audio")
                                st.session_state[self.voice_recording_key] = False
                        except Exception as e:
                            st.error(f"❌ Transcription error: {e}")
                            st.session_state[self.voice_recording_key] = False
                    else:
                        st.warning("⚠️ No audio captured. Please try again.")
                        st.session_state[self.voice_recording_key] = False
                
        except Exception as rec_err:
            st.error(f"Audio recorder error: {rec_err}")
            st.session_state[self.voice_recording_key] = False

    def _extract_audio_bytes(self, audio_data) -> Optional[bytes]:
        """Extract raw bytes from audio data."""
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            return audio_data['bytes']
        elif isinstance(audio_data, (bytes, bytearray)):
            return bytes(audio_data)
        return None


def get_voice_input(placeholder_text: str = "Type your message or use voice input...", 
                   page_key: str = "chat") -> Optional[str]:
    """Create a voice input component and return user input."""
    voice_input = VoiceInput(page_key)
    return voice_input.render_input(placeholder_text)
