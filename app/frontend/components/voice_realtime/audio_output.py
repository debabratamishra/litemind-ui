"""Thread-safe, interruptible audio output queue for barge-in support."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class _InterruptibleAudioOut:
    """Thread-safe audio output queue that supports interruption (barge-in)."""

    def __init__(self) -> None:
        self._q: queue.Queue[bytes] = queue.Queue()
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._bot_speaking = False
        self._interrupted = threading.Event()  # Signal for barge-in
        self._interrupt_callback: Optional[Callable[[], None]] = None

    def set_interrupt_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback to be called when interrupt occurs (for TTS handler)."""
        self._interrupt_callback = callback

    def enqueue(self, pcm16: bytes) -> None:
        """Add audio to the output queue."""
        if not pcm16 or self._interrupted.is_set():
            return
        with self._lock:
            self._bot_speaking = True
            self._q.put(pcm16)

    def interrupt(self) -> None:
        """Clear all queued audio (user started speaking)."""
        self._interrupted.set()  # Signal interruption
        with self._lock:
            self._bot_speaking = False
            self._buf.clear()
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        # Call the interrupt callback if set (to stop TTS synthesis)
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
            except Exception:
                pass

    def reset_interrupt(self) -> None:
        """Reset the interrupt flag for a new response."""
        self._interrupted.clear()

    def is_interrupted(self) -> bool:
        """Check if audio output has been interrupted."""
        return self._interrupted.is_set()

    def is_bot_speaking(self) -> bool:
        """Check if there's audio being output."""
        with self._lock:
            return self._bot_speaking and (len(self._buf) > 0 or not self._q.empty())

    def get_bytes_for_frame(self, nbytes: int) -> bytes:
        """Get bytes for output frame, returns silence if not enough."""
        if nbytes <= 0:
            return b""
        with self._lock:
            while len(self._buf) < nbytes:
                try:
                    chunk = self._q.get_nowait()
                except queue.Empty:
                    break
                self._buf.extend(chunk)

            if len(self._buf) >= nbytes:
                out = bytes(self._buf[:nbytes])
                del self._buf[:nbytes]
            else:
                out = b""

            if len(self._buf) == 0 and self._q.empty():
                self._bot_speaking = False
            return out
