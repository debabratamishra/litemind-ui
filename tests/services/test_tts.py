"""Unit tests for ``app.services.tts_service`` (offline, mocked).

The heavy boundaries (kokoro / soundfile / torch model load + audio synthesis,
and pyttsx3 engine + temp file I/O) are mocked at their first call so that no
real model download, audio synthesis, or filesystem audio write ever occurs.

`kokoro`/`pyttsx3`/`soundfile` are NOT installed in the test environment, so
``_check_backends()`` safely reports both backends unavailable. Tests that need
a specific backend force the corresponding ``_xxx_available`` flag and patch the
synthesis method directly.
"""

from unittest.mock import patch

import pytest

from app.services import tts_service as ts
from app.services.tts_service import TTSService, get_tts_service, preload_tts_model


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the module-level cache dir into tmp_path (keeps /tmp clean)."""
    monkeypatch.setattr(ts, "CACHE_DIR", str(tmp_path / "tts_cache"))
    return tmp_path / "tts_cache"


@pytest.fixture
def service(cache_dir):
    """A fresh TTSService instance (no preload, no real model load)."""
    return TTSService(preload_models=False)


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    """Ensure the module-global singleton does not leak across tests."""
    monkeypatch.setattr(ts, "_tts_service", None)
    yield
    monkeypatch.setattr(ts, "_tts_service", None)


# ── singleton / factory ───────────────────────────────────────────────────────
def test_get_tts_service_singleton(cache_dir):
    a = get_tts_service()
    b = get_tts_service()
    assert a is b


def test_get_tts_service_reset_returns_new(cache_dir, reset_singleton):
    first = get_tts_service()
    ts._tts_service = None
    second = get_tts_service()
    assert first is not second


def test_preload_tts_model_no_raise_when_unavailable(cache_dir):
    # With no kokoro installed, preload short-circuits without downloading.
    preload_tts_model()  # should not raise


def test_preload_method_no_raise_when_unavailable(service):
    service.preload()  # safe; _kokoro_available is False


# ── text cleaning composition ─────────────────────────────────────────────────
def test_clean_text_empty(service):
    assert service._clean_text_for_tts("") == ""


def test_clean_text_removes_markdown_and_urls(service):
    out = service._clean_text_for_tts("**bold** see http://x.com `code` now")
    assert "**" not in out
    assert "http" not in out
    assert "`" not in out
    assert "bold" in out and "see" in out and "now" in out


def test_clean_text_removes_think_tags(service):
    out = service._clean_text_for_tts("answer<think>secret reasoning</think>done")
    assert "secret" not in out
    assert "reasoning" not in out
    assert "answer" in out and "done" in out


def test_clean_text_removes_fenced_code_blocks(service):
    out = service._clean_text_for_tts("text\n```python\nprint(1)\n```\nmore")
    # The fenced code block CONTENT is excluded from the cleaned text.
    assert "print" not in out
    assert "text" in out and "more" in out
    # NOTE (real bug, see report): the "[code block omitted]" placeholder is
    # stripped by the later _remove_brace_blocks('[', ']') pass because it
    # contains square brackets. We therefore assert only that the content is
    # gone, not the placeholder.


def test_clean_text_removes_emojis(service):
    out = service._clean_text_for_tts("hello 😀 world 🚀")
    assert "😀" not in out
    assert "🚀" not in out
    assert "hello" in out and "world" in out


def test_clean_text_strips_special_chars(service):
    out = service._clean_text_for_tts("price #100 @store")
    assert "#" not in out
    assert "@" not in out


def test_clean_text_idempotent(service):
    text = "**Hello** visit https://a.com `code` and _emphasize_ this."
    once = service._clean_text_for_tts(text)
    twice = service._clean_text_for_tts(once)
    assert once == twice


# ── sentence splitting for streaming ──────────────────────────────────────────
def test_split_into_sentences_empty(service):
    assert service._split_into_sentences("") == []


def test_split_into_sentences_combines_short_fragments(service):
    # Below MIN_CHUNK_SIZE fragments are buffered into one chunk.
    out = service._split_into_sentences("Short. Small. Tiny.")
    assert len(out) >= 1
    assert all(isinstance(s, str) and s.strip() for s in out)


def test_split_into_sentences_long_text(service):
    long_text = ("This is a reasonably long sentence that exceeds the minimum "
                 "chunk size threshold. Here is another sufficiently long "
                 "sentence to force a second chunk in the splitter output.")
    out = service._split_into_sentences(long_text)
    assert len(out) >= 1
    assert all(len(s) >= 1 for s in out)


# ── cache key helpers ─────────────────────────────────────────────────────────
def test_get_cache_key_deterministic(service):
    k1 = service._get_cache_key("hello", "af_heart")
    k2 = service._get_cache_key("hello", "af_heart")
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) == 32  # md5 hex digest


def test_get_cache_key_differs_by_voice(service):
    assert service._get_cache_key("hello", "af_heart") != service._get_cache_key("hello", "am_adam")


# ── Kokoro primary synthesis (mocked) ─────────────────────────────────────────
async def test_synthesize_kokoro_primary_success(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    with patch.object(service, "_synthesize_kokoro", return_value=b"wav-bytes") as m:
        audio, ctype = await service.synthesize("hello world")
    assert audio == b"wav-bytes"
    assert ctype == "audio/wav"
    m.assert_called_once()


async def test_synthesize_empty_text_returns_none(service):
    audio, ctype = await service.synthesize("   ")
    assert audio is None
    assert ctype == ""


async def test_synthesize_empty_after_cleaning_returns_none(service):
    # Text that cleans to nothing (only markup) yields no audio.
    audio, ctype = await service.synthesize("** **")
    assert audio is None
    assert ctype == ""


async def test_synthesize_passes_voice_to_kokoro(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    captured = {}

    def fake_kokoro(text, voice=None, speed=None):
        captured["voice"] = voice
        return b"audio"

    with patch.object(service, "_synthesize_kokoro", side_effect=fake_kokoro):
        await service.synthesize("hello", voice="af_bella")
    assert captured["voice"] == "af_bella"


async def test_synthesize_ignores_edge_voice_name(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    captured = {}

    def fake_kokoro(text, voice=None, speed=None):
        captured["voice"] = voice
        return b"audio"

    with patch.object(service, "_synthesize_kokoro", side_effect=fake_kokoro):
        await service.synthesize("hello", voice="en-US-AriaNeural")
    # Edge-style voice names fall back to the default Kokoro voice.
    assert captured["voice"] == ts.TTS_CONFIG["kokoro_voice"]


# ── pyttsx3 fallback path (mocked) ────────────────────────────────────────────
async def test_synthesize_falls_back_to_pyttsx3_when_kokoro_fails(service):
    service._kokoro_available = True
    service._pyttsx3_available = True
    with patch.object(service, "_synthesize_kokoro", return_value=None) as m_k:
        with patch.object(service, "_synthesize_pyttsx3", return_value=b"pyttsx3-audio") as m_p:
            audio, ctype = await service.synthesize("hello")
    assert audio == b"pyttsx3-audio"
    assert ctype == "audio/wav"
    m_k.assert_called_once()
    m_p.assert_called_once()


async def test_synthesize_pyttsx3_only_when_kokoro_unavailable(service):
    service._kokoro_available = False
    service._pyttsx3_available = True
    with patch.object(service, "_synthesize_kokoro", return_value=b"should-not-be-used") as m_k:
        with patch.object(service, "_synthesize_pyttsx3", return_value=b"pyttsx3-audio") as m_p:
            audio, ctype = await service.synthesize("hello")
    assert audio == b"pyttsx3-audio"
    m_k.assert_not_called()
    m_p.assert_called_once()


async def test_synthesize_total_failure_returns_none(service):
    service._kokoro_available = True
    service._pyttsx3_available = True
    with patch.object(service, "_synthesize_kokoro", return_value=None):
        with patch.object(service, "_synthesize_pyttsx3", return_value=None):
            audio, ctype = await service.synthesize("hello")
    assert audio is None
    assert ctype == ""


async def test_synthesize_no_backend_returns_none(service):
    service._kokoro_available = False
    service._pyttsx3_available = False
    audio, ctype = await service.synthesize("hello")
    assert audio is None
    assert ctype == ""


# ── sync convenience methods ──────────────────────────────────────────────────
def test_synthesize_text_chunk_empty(service):
    assert service.synthesize_text_chunk("   ") is None


def test_synthesize_text_chunk_kokoro(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    with patch.object(service, "_synthesize_kokoro", return_value=b"chunk-audio"):
        out = service.synthesize_text_chunk("hello")
    assert out == b"chunk-audio"


def test_synthesize_text_chunk_fallback_when_kokoro_unavailable(service):
    # synthesize_text_chunk has NO fallback: when kokoro is available it returns
    # the (mocked) kokoro result directly, and only uses pyttsx3 when kokoro is
    # unavailable.
    service._kokoro_available = False
    service._pyttsx3_available = True
    with patch.object(service, "_synthesize_kokoro", return_value=b"never-used"):
        with patch.object(service, "_synthesize_pyttsx3", return_value=b"fb"):
            out = service.synthesize_text_chunk("hello")
    assert out == b"fb"


def test_synthesize_text_chunk_returns_kokoro_result_without_fallback(service):
    # No fallback path: a (mocked) None from kokoro is returned as-is.
    service._kokoro_available = True
    service._pyttsx3_available = True
    with patch.object(service, "_synthesize_kokoro", return_value=None):
        with patch.object(service, "_synthesize_pyttsx3", return_value=b"fb"):
            out = service.synthesize_text_chunk("hello")
    assert out is None


def test_synthesize_text_chunk_no_backend(service):
    service._kokoro_available = False
    service._pyttsx3_available = False
    assert service.synthesize_text_chunk("hello") is None


def test_synthesize_sync_wraps_async(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    with patch.object(service, "_synthesize_kokoro", return_value=b"sync-audio"):
        audio, ctype = service.synthesize_sync("hello")
    assert audio == b"sync-audio"
    assert ctype == "audio/wav"


# ── voice / status helpers ────────────────────────────────────────────────────
def test_get_available_voices_returns_list(service):
    voices = service.get_available_voices()
    assert isinstance(voices, list)
    assert any(v["id"] == "af_heart" for v in voices)


def test_is_available_false_when_no_backend(service):
    service._kokoro_available = False
    service._pyttsx3_available = False
    assert service.is_available() is False


def test_is_available_true_when_kokoro(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    assert service.is_available() is True


def test_get_status_reports_flags(service):
    service._kokoro_available = True
    service._pyttsx3_available = False
    service._kokoro_loaded = True
    status = service.get_status()
    assert status["kokoro"] is True
    assert status["kokoro_loaded"] is True
    assert status["pyttsx3_fallback"] is False
    assert status["default_voice"] == ts.TTS_CONFIG["kokoro_voice"]


def test_is_model_loaded(service):
    service._kokoro_loaded = False
    assert service.is_model_loaded() is False
    service._kokoro_loaded = True
    assert service.is_model_loaded() is True


# ── streaming synthesis (kokoro mocked) ───────────────────────────────────────
async def test_synthesize_streaming_kokoro_yields_chunks(service):
    service._kokoro_available = True
    service._pyttsx3_available = False

    def fake_stream(text, voice=None, speed=None):
        yield b"chunk1"
        yield b"chunk2"

    with patch.object(service, "_synthesize_kokoro_streaming", side_effect=fake_stream):
        collected = []
        async for chunk in service.synthesize_streaming(_async_text_gen("Hello world. Second sentence.")):
            collected.append(chunk)
    assert collected == [b"chunk1", b"chunk2"]


async def test_synthesize_streaming_no_backend_yields_nothing(service):
    service._kokoro_available = False
    service._pyttsx3_available = False
    collected = []
    async for chunk in service.synthesize_streaming(_async_text_gen("hello")):
        collected.append(chunk)
    assert collected == []


async def _async_text_gen(text):
    yield text
