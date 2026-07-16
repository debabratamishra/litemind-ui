"""Realtime voice chat component (WebRTC).

This package previously lived in a single 1600+ line module
(``voice_realtime.py``). It has been split into focused submodules:

- ``config``        — dataclasses, constants, greeting cache globals
- ``ice_servers``   — WebRTC ICE/NAT/STUN/TURN configuration
- ``utils``         — PCM/WAV conversion and session-state helpers
- ``audio_output``  — interruptible audio output queue (barge-in)
- ``processors``    — VAD audio processors + availability detection
- ``tts``           — streaming TTS handler and greeting/playback helpers
- ``render``        — the ``render_realtime_voice_chat`` orchestration

Only ``render_realtime_voice_chat`` is part of the public API.
"""

from .render import render_realtime_voice_chat

__all__ = ["render_realtime_voice_chat"]
