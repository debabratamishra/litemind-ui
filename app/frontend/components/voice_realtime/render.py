"""Realtime voice chat component (WebRTC) — main render orchestration.

This wires together the config, ICE configuration, audio processors, streaming
TTS handler, and greeting/playback helpers into the Streamlit UI. It is the
only module that owns the ``render_realtime_voice_chat`` entrypoint that
external callers (e.g. ``voice_input.py``) depend on.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import streamlit as st

from ...services.backend_service import backend_service
from ..streaming_handler import streaming_handler
from .config import _greeting_audio_cache
from .ice_servers import (
    _build_frontend_rtc_configuration,
    _build_server_rtc_configuration,
)
from .processors import (
    PIPECAT_AVAILABLE as _PIPECAT_AVAILABLE,
)
from .processors import (
    WEBRTC_DEPS_AVAILABLE as _WEBRTC_DEPS_AVAILABLE,
)
from .processors import (
    WEBRTCVAD_AVAILABLE as _WEBRTCVAD_AVAILABLE,
)
from .processors import (
    PipecatVADProcessor,
    WebRtcMode,
    WebRTCVADProcessor,
    webrtc_streamer,
)
from .tts import (
    StreamingTTSHandler,
    _get_or_cache_greeting_audio,
    _play_greeting_via_audio_widget,
    _play_greeting_via_webrtc,
    _synthesize_and_play,
)
from .utils import (
    _get_chat_config_from_session,
    _get_memory_manager,
    _get_messages_key,
    _get_realtime_greeting,
    _pcm16_to_wav_bytes,
)

logger = logging.getLogger(__name__)


def render_realtime_voice_chat(page_key: str = "chat") -> None:
    """Render realtime voice chat UI.

    This function manages its own session-state and writes to the shared
    `st.session_state.chat_messages` history for continuity with the Chat page.
    """
    realtime_mode_key = f"realtime_voice_mode_{page_key}"
    initializing_key = f"realtime_initializing_{page_key}"

    # Handle first-time initialization with a loading state
    # This gives the WebRTC component time to load its JavaScript assets
    if st.session_state.get(initializing_key, False):
        st.session_state[initializing_key] = False
        # Show a brief loading message, then auto-refresh to load WebRTC properly
        st.info("🎤 Starting voice chat... Please wait.")
        time.sleep(0.5)  # Brief pause to let the page settle
        st.rerun()

    # Check for required packages
    if not _WEBRTC_DEPS_AVAILABLE:
        st.warning(
            "Realtime voice chat requires additional packages. Install `streamlit-webrtc` and `av` in your environment."
        )
        return

    if not _PIPECAT_AVAILABLE and not _WEBRTCVAD_AVAILABLE:
        st.error("No VAD backend available. Install pipecat-ai[silero] or webrtcvad.")
        return

    # CSS for the modern circle UI
    st.markdown(
        """
        <style>
        .voice-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .voice-circle {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 0 30px rgba(118, 75, 162, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: pulse 3s infinite ease-in-out;
            margin-bottom: 2rem;
            color: white;
            font-size: 4rem;
        }
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(118, 75, 162, 0.7); }
            50% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(118, 75, 162, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(118, 75, 162, 0); }
        }
        .voice-status {
            font-size: 1.5rem;
            font-weight: 600;
            color: #555;
            margin-bottom: 1rem;
        }
        .voice-subtext {
            font-size: 1rem;
            color: #888;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Close button
    col_spacer, col_close = st.columns([10, 1])
    with col_close:
        if st.button("✕", key=f"realtime_voice_close_{page_key}", help="Close realtime voice chat"):
            # Save any ongoing partial response before closing
            active_tts_key = f"active_tts_handler_{page_key}"
            messages_key = _get_messages_key(page_key)

            if active_tts_key in st.session_state and st.session_state[active_tts_key]:
                try:
                    tts_handler = st.session_state[active_tts_key]
                    # Interrupt the TTS handler
                    tts_handler.interrupt()

                    # Get any partial text that was received
                    partial_text = tts_handler.get_full_text()
                    if partial_text and partial_text.strip():
                        # Check if we already have this response in history (avoid duplicates)
                        messages = st.session_state.get(messages_key, [])
                        if (
                            not messages
                            or messages[-1].get("role") != "assistant"
                            or messages[-1].get("content") != partial_text.strip()
                        ):
                            st.session_state[messages_key].append(
                                {"role": "assistant", "content": partial_text.strip() + " [interrupted]"}
                            )
                            logger.info(f"Saved interrupted response: {len(partial_text)} chars")

                    tts_handler.shutdown()
                except Exception as e:
                    logger.debug(f"Error saving partial response on close: {e}")

                st.session_state[active_tts_key] = None

            # Clear live transcript
            live_transcript_key = f"live_transcript_{page_key}"
            if live_transcript_key in st.session_state:
                st.session_state[live_transcript_key] = ""

            st.session_state[realtime_mode_key] = False
            st.rerun()

    # Visual Indicator
    page_label = "RAG Document Assistant" if page_key == "rag" else "Chat Assistant"
    page_subtext = (
        "Ask me anything about your documents." if page_key == "rag" else "Speak naturally. I will listen and respond."
    )

    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    st.markdown('<div class="voice-circle">🎙️</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voice-status">{page_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="voice-subtext">{page_subtext}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # WebRTC Widget Setup
    # ========================================================================

    use_pipecat = _PIPECAT_AVAILABLE
    ProcessorClass = PipecatVADProcessor if use_pipecat else WebRTCVADProcessor
    webrtc_mode = WebRtcMode.SENDRECV if use_pipecat else WebRtcMode.SENDONLY

    # Session state key to track if greeting has been played
    greeting_played_key = f"realtime_greeting_played_{page_key}"

    # Track if this is the first render of the WebRTC component
    # We defer greeting cache to after WebRTC is initialized to avoid blocking component loading
    webrtc_initialized_key = f"webrtc_initialized_{page_key}"

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        # Show loading hint for first-time WebRTC initialization
        if not st.session_state.get(webrtc_initialized_key, False):
            st.info("🎤 Initializing voice chat... Please wait a moment.")

        try:
            frontend_rtc_config = _build_frontend_rtc_configuration()
            server_rtc_config = _build_server_rtc_configuration()
            logger.info(
                "WebRTC ICE servers -> browser: %s | server(in-container): %s",
                [s.get("urls") for s in frontend_rtc_config.get("iceServers", [])],
                [s.get("urls") for s in server_rtc_config.get("iceServers", [])],
            )
            webrtc_ctx = webrtc_streamer(
                key=f"realtime_voice_webrtc_{page_key}",
                mode=webrtc_mode,
                audio_processor_factory=ProcessorClass,
                media_stream_constraints={
                    "audio": {
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": True,
                        "sampleRate": 48000,
                    },
                    "video": False,
                },
                async_processing=False,
                # Browser peer ICE config (reaches TURN via the host LAN IP).
                frontend_rtc_configuration=frontend_rtc_config,
                # In-container aiortc peer ICE config. MUST point at a TURN/STUN
                # server reachable from INSIDE the container (host.docker.internal
                # in Docker). Passing an unreachable server here is what makes
                # aioice crash on teardown ("NoneType has no attribute sendto").
                server_rtc_configuration=server_rtc_config,
            )
        except Exception as e:
            logger.error(f"WebRTC initialization error: {e}")
            st.error(
                "Voice chat component failed to load. This can happen on first access due to network latency. "
                "Please refresh the page to try again."
            )
            if st.button("🔄 Refresh Page"):
                st.rerun()
            return

        with st.expander("Audio Settings", expanded=False):
            backend_info = "Pipecat Silero VAD" if use_pipecat else "webrtcvad"
            st.info(f"Using: {backend_info}")
            st.caption("Audio output is handled by your browser.")

    # Pre-cache greeting audio AFTER WebRTC component is rendered
    # This avoids blocking the component loading on first page access
    # We add a small delay to ensure WebRTC JS assets have time to load
    cache_key = f"greeting_{page_key}"
    if not st.session_state.get(webrtc_initialized_key, False):
        st.session_state[webrtc_initialized_key] = True
        # Start background thread to cache greeting audio only after first render
        # Add delay to let WebRTC component assets load first
        if cache_key not in _greeting_audio_cache:

            def cache_greeting_async():
                time.sleep(2.0)  # Wait for WebRTC component to fully initialize
                _get_or_cache_greeting_audio(page_key)

            threading.Thread(target=cache_greeting_async, daemon=True).start()

    # ========================================================================
    # Play Greeting When Session Starts
    # ========================================================================

    # Check if WebRTC is playing and greeting hasn't been played yet
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        if not st.session_state.get(greeting_played_key, False):
            # Mark greeting as played (do this first to prevent re-triggering)
            st.session_state[greeting_played_key] = True

            # Get the appropriate messages key and greeting for this page
            messages_key = _get_messages_key(page_key)
            greeting_text = _get_realtime_greeting(page_key)

            # Add greeting to appropriate messages list
            st.session_state.setdefault(messages_key, [])
            st.session_state[messages_key].append({"role": "assistant", "content": greeting_text})

            # Get cached or synthesize greeting audio
            greeting_audio = _get_or_cache_greeting_audio(page_key)

            if greeting_audio:
                # Try to play through WebRTC first (lower latency)
                greeting_played = False
                if webrtc_ctx.audio_processor and use_pipecat:
                    if hasattr(webrtc_ctx.audio_processor, "enqueue_assistant_pcm16"):
                        target_sr = webrtc_ctx.audio_processor.output_sample_rate
                        greeting_played = _play_greeting_via_webrtc(
                            webrtc_ctx.audio_processor, greeting_audio, target_sr
                        )

                # Fallback to audio widget if WebRTC playback failed
                if not greeting_played:
                    _play_greeting_via_audio_widget(greeting_audio)
            else:
                # Last resort: synthesize and play directly
                logger.warning("No cached greeting, synthesizing inline...")
                _synthesize_and_play(greeting_text)

            logger.info(f"AI greeting played on realtime voice session start for {page_key}")

    # Reset greeting flag when WebRTC stops
    if webrtc_ctx and not getattr(webrtc_ctx.state, "playing", False):
        if st.session_state.get(greeting_played_key, False):
            st.session_state[greeting_played_key] = False

    # ========================================================================
    # Process Finalized Segments with Streaming TTS
    # ========================================================================

    # Use page-specific messages key
    messages_key = _get_messages_key(page_key)
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    # Session state for live transcription
    live_transcript_key = f"live_transcript_{page_key}"
    if live_transcript_key not in st.session_state:
        st.session_state[live_transcript_key] = ""

    # Placeholder for live transcription display - always visible
    live_transcript_placeholder = st.empty()

    # Display current live transcription state
    current_live_text = st.session_state.get(live_transcript_key, "")
    if current_live_text:
        with live_transcript_placeholder.container():
            st.markdown(
                f"""<div style="
                    padding: 0.75rem 1rem;
                    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
                    border-left: 4px solid #667eea;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    font-style: italic;
                    color: #555;
                    animation: pulse-subtle 1.5s infinite;
                ">
                    🎙️ <strong>Listening:</strong> {current_live_text}
                </div>
                <style>
                    @keyframes pulse-subtle {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.8; }}
                    }}
                </style>""",
                unsafe_allow_html=True,
            )

    if webrtc_ctx and webrtc_ctx.audio_processor:
        # ====================================================================
        # Live Transcription: Show partial text as user speaks
        # ====================================================================
        if use_pipecat and hasattr(webrtc_ctx.audio_processor, "interim_audio"):
            try:
                interim_pcm16, interim_sr = webrtc_ctx.audio_processor.interim_audio.get_nowait()
                if interim_pcm16 and interim_sr:
                    # Transcribe interim audio in a synchronous manner for immediate UI update
                    try:
                        wav_bytes = _pcm16_to_wav_bytes(interim_pcm16, sample_rate=interim_sr, channels=1)
                        partial_text = backend_service.transcribe_audio(wav_bytes, sample_rate=16000)
                        if partial_text and partial_text.strip():
                            st.session_state[live_transcript_key] = partial_text.strip()
                            # Force UI update for live transcript
                            with live_transcript_placeholder.container():
                                st.markdown(
                                    f"""<div style="
                                        padding: 0.75rem 1rem;
                                        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
                                        border-left: 4px solid #667eea;
                                        border-radius: 8px;
                                        margin-bottom: 1rem;
                                        font-style: italic;
                                        color: #555;
                                    ">
                                        🎙️ <strong>Listening:</strong> {partial_text.strip()}
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                    except Exception as e:
                        logger.debug(f"Interim transcription error: {e}")
            except queue.Empty:
                pass

        # ====================================================================
        # Process Completed Segments
        # ====================================================================
        try:
            pcm16, sr = webrtc_ctx.audio_processor.segments.get_nowait()
        except queue.Empty:
            pcm16, sr = None, None

        if pcm16 and sr:
            # Clear live transcript when we have a complete segment
            st.session_state[live_transcript_key] = ""
            live_transcript_placeholder.empty()

            # BARGE-IN: Interrupt any ongoing TTS playback immediately
            # 1. Clear the audio output queue to stop playback instantly
            if hasattr(webrtc_ctx.audio_processor, "interrupt_assistant"):
                webrtc_ctx.audio_processor.interrupt_assistant()
                logger.debug("Cleared audio output queue on barge-in")

            # 2. Interrupt the TTS handler to stop generating more audio
            active_tts_key = f"active_tts_handler_{page_key}"
            if active_tts_key in st.session_state and st.session_state[active_tts_key]:
                try:
                    st.session_state[active_tts_key].interrupt()
                    logger.debug("Interrupted TTS handler on barge-in")
                except Exception as e:
                    logger.debug(f"TTS interrupt error: {e}")
                st.session_state[active_tts_key] = None

            with st.spinner("Transcribing..."):
                wav_bytes = _pcm16_to_wav_bytes(pcm16, sample_rate=sr, channels=1)
                user_text = backend_service.transcribe_audio(wav_bytes, sample_rate=16000)

            if user_text and user_text.strip():
                user_text = user_text.strip()

                cfg = _get_chat_config_from_session(page_key)

                # Get conversation context BEFORE adding the new message
                # This ensures the current question is not included in the history
                memory_manager = _get_memory_manager(page_key)
                conversation_history = memory_manager.get_history_for_api()
                conversation_summary = memory_manager.summary
                session_id = memory_manager.session_id

                # Now add the user message to session state
                st.session_state[messages_key].append({"role": "user", "content": user_text})

                with st.chat_message("user"):
                    st.markdown(user_text)

                # Generate response with streaming TTS
                with st.chat_message("assistant"):
                    out = st.empty()

                    # Set up streaming TTS handler for real-time synthesis
                    tts_handler = None
                    audio_processor = webrtc_ctx.audio_processor
                    tts_audio_queue = None

                    if use_pipecat and hasattr(audio_processor, "enqueue_assistant_pcm16"):
                        # Create streaming TTS handler that synthesizes audio in parallel
                        target_sr = audio_processor.output_sample_rate
                        tts_audio_queue = queue.Queue()
                        tts_handler = StreamingTTSHandler(tts_audio_queue, target_sr)

                        # Store in session state for barge-in interruption
                        st.session_state[active_tts_key] = tts_handler

                        # Set up barge-in callback so VAD interrupt triggers TTS stop
                        if hasattr(audio_processor, "set_tts_interrupt_callback"):
                            audio_processor.set_tts_interrupt_callback(tts_handler.interrupt)

                        # Reset interrupt flag for this new response
                        if hasattr(audio_processor, "reset_interrupt"):
                            audio_processor.reset_interrupt()

                        # Create a callback that feeds text to TTS and queues audio
                        def tts_streaming_callback(text_chunk: str) -> None:
                            """Called for each LLM text chunk to synthesize audio in parallel."""
                            if tts_handler and not tts_handler.is_interrupted():
                                tts_handler.feed(text_chunk)
                                # Immediately drain any ready audio from queue to WebRTC
                                while True:
                                    try:
                                        audio_chunk = tts_audio_queue.get_nowait()
                                        if audio_chunk and not tts_handler.is_interrupted():
                                            audio_processor.enqueue_assistant_pcm16(audio_chunk)
                                    except queue.Empty:
                                        break
                    else:
                        tts_streaming_callback = None

                    # Stream the response based on page type
                    if page_key == "rag":
                        # RAG query with document context
                        config = st.session_state.get("rag_config", {})
                        history = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state[messages_key][:-1]
                        ]

                        reply = streaming_handler.stream_rag_response(
                            query=user_text,
                            messages=history,
                            model=cfg["model"],
                            system_prompt=st.session_state.get("rag_system_prompt", ""),
                            n_results=config.get("n_results", 3),
                            use_multi_agent=config.get("use_multi_agent", False),
                            use_hybrid_search=config.get("use_hybrid_search", False),
                            backend=cfg["backend"],
                            api_base=cfg.get("api_base"),
                            api_key=cfg.get("api_key"),
                            placeholder=out,
                            tts_callback=tts_streaming_callback,
                            conversation_summary=conversation_summary,
                            session_id=session_id,
                            temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"],
                            top_p=cfg["top_p"],
                            frequency_penalty=cfg["frequency_penalty"],
                            repetition_penalty=cfg["repetition_penalty"],
                        )
                    else:
                        # Regular chat response with conversation memory
                        reply = streaming_handler.stream_chat_response(
                            message=user_text,
                            model=cfg["model"],
                            temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"],
                            top_p=cfg["top_p"],
                            frequency_penalty=cfg["frequency_penalty"],
                            repetition_penalty=cfg["repetition_penalty"],
                            backend=cfg["backend"],
                            api_base=cfg.get("api_base"),
                            api_key=cfg.get("api_key"),
                            placeholder=out,
                            use_fastapi=st.session_state.get("backend_available", False),
                            tts_callback=tts_streaming_callback,
                            conversation_history=conversation_history,
                            conversation_summary=conversation_summary,
                            session_id=session_id,
                        )

                    # Finalize any remaining TTS audio (if not interrupted)
                    if tts_handler:
                        if not tts_handler.is_interrupted():
                            tts_handler.finalize()
                            # Drain remaining audio to WebRTC
                            while tts_audio_queue:
                                try:
                                    audio_chunk = tts_audio_queue.get_nowait()
                                    if audio_chunk:
                                        audio_processor.enqueue_assistant_pcm16(audio_chunk)
                                except queue.Empty:
                                    break
                        tts_handler.shutdown()
                        st.session_state[active_tts_key] = None

                # Save response - prefer reply, but use partial text from TTS handler if interrupted
                final_reply = reply
                if tts_handler:
                    # If we have a TTS handler, get the full text it received
                    tts_full_text = tts_handler.get_full_text()
                    if tts_full_text and tts_full_text.strip():
                        # Use TTS handler's text if reply is empty or shorter (interrupted)
                        if not final_reply or len(tts_full_text) > len(final_reply or ""):
                            final_reply = tts_full_text

                if final_reply and final_reply.strip():
                    # Check if response was interrupted (TTS handler has interrupted flag)
                    was_interrupted = tts_handler and tts_handler.is_interrupted() if tts_handler else False

                    # Add message with optional interrupted indicator
                    message_content = final_reply.strip()
                    if was_interrupted:
                        message_content = message_content + " [interrupted]"

                    st.session_state[messages_key].append({"role": "assistant", "content": message_content})

                    # If streaming TTS wasn't available, fall back to full synthesis
                    if not (use_pipecat and hasattr(audio_processor, "enqueue_assistant_pcm16")):
                        _synthesize_and_play(final_reply)
                else:
                    # Even if no reply, log error
                    logger.warning("No response received for realtime voice query")

                st.rerun()

    # Keep UI updating while call is active
    if webrtc_ctx and getattr(webrtc_ctx.state, "playing", False):
        time.sleep(0.2)
        st.rerun()
