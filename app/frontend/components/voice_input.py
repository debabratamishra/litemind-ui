"""
Voice input component with speech-to-text transcription.
"""
import logging
import streamlit as st
from typing import Optional

from ...services.speech_service import get_speech_service
from .voice_realtime import render_realtime_voice_chat

logger = logging.getLogger(__name__)


class VoiceInput:
    
    def __init__(self, page_key: str):
        self.page_key = page_key
        self.text_key = f"text_{page_key}"
        self.audio_mode_key = f"audio_mode_{page_key}"
        self.realtime_mode_key = f"realtime_voice_mode_{page_key}"
        self.transcription_key = f"transcription_{page_key}"
        self.audio_processed_key = f"audio_processed_{page_key}"
        
        # Initialize session state
        if self.text_key not in st.session_state:
            st.session_state[self.text_key] = ""
        if self.audio_mode_key not in st.session_state:
            st.session_state[self.audio_mode_key] = False
        if self.realtime_mode_key not in st.session_state:
            st.session_state[self.realtime_mode_key] = False
        if self.transcription_key not in st.session_state:
            st.session_state[self.transcription_key] = ""
        if self.audio_processed_key not in st.session_state:
            st.session_state[self.audio_processed_key] = False

    def render_input(self, placeholder_text: str = "Type your message...") -> Optional[str]:
        """Render the input component and return user input."""

        if st.session_state[self.realtime_mode_key]:
            render_realtime_voice_chat(page_key=self.page_key)
            return None

        if st.session_state[self.audio_mode_key]:
            return self._render_audio_mode(placeholder_text)
        else:
            return self._render_text_mode(placeholder_text)

    def _render_text_mode(self, placeholder_text: str) -> Optional[str]:
        """Render clean text input with integrated microphone button."""
        
        if st.session_state[self.transcription_key]:
            st.session_state[f"chat_input_{self.page_key}"] = st.session_state[self.transcription_key]
            st.session_state[self.transcription_key] = ""
        
        # Create clean layout with chat input, realtime voice button, and mic button
        col_input, col_realtime, col_mic = st.columns([20, 1, 1])
        
        with col_input:
            user_input = st.chat_input(
                placeholder=placeholder_text,
                key=f"chat_input_{self.page_key}"
            )

        with col_realtime:
            realtime_clicked = st.button(
                "📞",
                key=f"realtime_voice_toggle_{self.page_key}",
                help="Realtime voice chat (WebRTC)",
                type="secondary",
                use_container_width=True,
            )
        
        with col_mic:
            mic_clicked = st.button(
                "🎤︎︎", 
                key=f"mic_toggle_{self.page_key}",
                help="Voice input",
                type="secondary",
                use_container_width=True
            )

        if realtime_clicked:
            st.session_state[self.realtime_mode_key] = True
            st.session_state[self.audio_mode_key] = False
            st.session_state[self.audio_processed_key] = False
            st.rerun()
        
        # Handle microphone click
        if mic_clicked:
            st.session_state[self.audio_mode_key] = True
            st.session_state[self.realtime_mode_key] = False
            st.session_state[self.audio_processed_key] = False
            st.rerun()
        
        if user_input and user_input.strip():
            st.session_state[self.text_key] = ""
            st.session_state[self.transcription_key] = ""
            return user_input.strip()
        
        return None

    def _render_audio_mode(self, placeholder_text: str) -> Optional[str]:
        """Render minimal audio input mode."""
        
        col_back, col_header = st.columns([1, 10])
        
        with col_back:
            back_clicked = st.button(
                "✕", 
                key=f"close_audio_{self.page_key}",
                help="Close voice input",
                type="secondary"
            )
            if back_clicked:
                st.session_state[self.audio_mode_key] = False
                st.rerun()
        
        with col_header:
            audio_data = st.audio_input(
                "audio",
                key=f"audio_{self.page_key}",
                label_visibility="collapsed"
            )
        
        # Handle transcription
        if audio_data is not None and not st.session_state[self.audio_processed_key]:
            self._handle_transcription(audio_data)
        
        return None

    def _handle_transcription(self, audio_data) -> None:
        """Handle audio transcription with minimal feedback."""
        try:
            st.session_state[self.audio_processed_key] = True
            
            with st.spinner("Transcribing..."):
                audio_bytes = audio_data.read()
                
                if audio_bytes:
                    speech_service = get_speech_service()
                    transcribed_text = speech_service.transcribe_audio(audio_bytes)
                    
                    if transcribed_text and transcribed_text.strip():
                        st.session_state[self.transcription_key] = transcribed_text.strip()
                        st.session_state[f"chat_input_{self.page_key}"] = st.session_state[self.transcription_key]
                        st.session_state[self.audio_mode_key] = False
                        st.session_state[self.audio_processed_key] = False
                        st.rerun()
                    else:
                        st.session_state[self.audio_processed_key] = False
                else:
                    st.session_state[self.audio_processed_key] = False
                        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            st.session_state[self.audio_processed_key] = False


def get_voice_input(placeholder_text: str = "Type your message...", 
                   page_key: str = "chat") -> Optional[str]:
    """Create a voice input component."""
    voice_input = VoiceInput(page_key)
    return voice_input.render_input(placeholder_text)
