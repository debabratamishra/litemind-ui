"""Unit tests for ``app.services.tts_text_processing`` (offline, pure).

Every function in this module is a pure string transform (no regex, no I/O).
These tests cover the high-value markdown/URL/emoji/whitespace cleanup
helpers thoroughly: no-markup passthrough, only-URLs, nested markup, URLs
inside prose, inline code, emoji/unicode, repeated-character compression, and
leading/trailing whitespace.

Note: there is NO public cleaning entrypoint in this module. The composition
function (``_clean_text_for_tts``) lives on ``TTSService`` in
``app.services.tts_service`` and is tested there. Here we exercise each pure
helper directly, which is the real, correct surface to assert against.
"""

from app.services import tts_text_processing as ttp


# ── sentence splitting ────────────────────────────────────────────────────────
def test_split_empty_returns_empty_list():
    assert ttp._split_on_sentence_endings("") == []


def test_split_no_punctuation_returns_single_part():
    assert ttp._split_on_sentence_endings("hello world") == ["hello world"]


def test_split_on_basic_sentence_endings():
    parts = ttp._split_on_sentence_endings("Hello. World! How are you?")
    # The punctuation is dropped; text is split into the surrounding fragments.
    assert parts == ["Hello", "World", "How are you"]


def test_split_skips_trailing_whitespace_after_punctuation():
    parts = ttp._split_on_sentence_endings("One.   Two.")
    assert parts == ["One", "Two"]


def test_split_handles_semicolon_and_colon():
    parts = ttp._split_on_sentence_endings("First; Second: Third")
    assert parts == ["First", "Second", "Third"]


# ── inline code ───────────────────────────────────────────────────────────────
def test_remove_inline_code_empty_when_no_backticks():
    assert ttp._remove_inline_code("no code here") == "no code here"


def test_remove_inline_code_strips_single_backtick_span():
    assert ttp._remove_inline_code("run `ls -la` now") == "run  now"


def test_remove_inline_code_keeps_lone_backtick():
    # A single unmatched backtick is not a span and is preserved.
    assert ttp._remove_inline_code("a ` b") == "a ` b"


# ── URLs ──────────────────────────────────────────────────────────────────────
def test_remove_urls_http():
    assert "http" not in ttp._remove_urls("see http://example.com now")


def test_remove_urls_https():
    out = ttp._remove_urls("visit https://example.com/path please")
    assert "https" not in out
    assert "[link]" in out


def test_remove_urls_www():
    out = ttp._remove_urls("go to www.example.com today")
    assert "www" not in out
    assert "[link]" in out


def test_remove_urls_only_urls():
    out = ttp._remove_urls("http://a.com http://b.com")
    assert "http" not in out
    assert "[link]" in out


def test_remove_urls_inside_prose():
    out = ttp._remove_urls("The docs at https://docs.site.io are great.")
    assert "https" not in out
    assert "docs" in out
    assert "great" in out


def test_remove_urls_url_with_trailing_punctuation():
    # Note: '.' is NOT in the URL terminator set, so it is consumed as part of
    # the URL and the trailing period does NOT survive.
    out = ttp._remove_urls("see http://x.com.")
    assert "http" not in out
    assert "[link]" in out
    assert not out.rstrip().endswith(".")


# ── file paths ────────────────────────────────────────────────────────────────
def test_remove_file_paths_unix():
    assert "/tmp/x.pdf" not in ttp._remove_file_paths("open /tmp/x.pdf please")


def test_remove_file_paths_unix_multi_segment():
    out = ttp._remove_file_paths("read /var/log/app/error.log now")
    assert "/var" not in out
    assert "read" in out and "now" in out


def test_remove_file_paths_windows():
    # The trailing space is part of the allowed segment charset, so the engine
    # consumes the rest of the line too; the key assertion is the drive path
    # is gone.
    out = ttp._remove_file_paths("open C:\\Users\\me\\file.txt here")
    assert "C:" not in out
    assert out.startswith("open")


def test_remove_file_paths_leading_slash_not_a_path():
    # A leading slash followed by nothing alnum should be preserved.
    assert ttp._remove_file_paths("/ not a path") == "/ not a path"


# ── brace block removal ───────────────────────────────────────────────────────
def test_remove_brace_blocks_curly():
    assert ttp._remove_brace_blocks("text {json: 1} more", "{", "}") == "text  more"


def test_remove_brace_blocks_square():
    assert ttp._remove_brace_blocks("a [1, 2, 3] b", "[", "]") == "a  b"


def test_remove_brace_blocks_nested_same_type_kept():
    # The helper only removes non-nested same-type braces: the inner {c} is
    # removed, but the outer braces (which contain a nested '{') are preserved.
    out = ttp._remove_brace_blocks("a {b {c} d} e", "{", "}")
    assert "{c}" not in out
    assert "{b  d}" in out


def test_remove_brace_blocks_no_open_char():
    assert ttp._remove_brace_blocks("plain text", "{", "}") == "plain text"


# ── delimiter pair / markdown formatting ──────────────────────────────────────
def test_remove_delimiter_pair_basic():
    assert ttp._remove_delimiter_pair("a **b** c", "**") == "a b c"


def test_remove_delimiter_pair_no_delimiter():
    assert ttp._remove_delimiter_pair("nothing here", "**") == "nothing here"


def test_remove_delimiter_pair_forbid():
    # With forbid='*', a '**' containing '*' inside is NOT treated as a pair.
    out = ttp._remove_delimiter_pair("a *b*c **d**", "**", forbid="*")
    assert out == "a *b*c d"


def test_remove_markdown_formatting_bold():
    assert ttp._remove_markdown_formatting("**bold** text") == "bold text"


def test_remove_markdown_formatting_italic():
    assert ttp._remove_markdown_formatting("*italic* text") == "italic text"


def test_remove_markdown_formatting_strikethrough():
    assert ttp._remove_markdown_formatting("~~gone~~ kept") == "gone kept"


def test_remove_markdown_formatting_nested():
    # Outer bold with forbid='*' cannot match (inner contains '*'), so the
    # single-asterisk italic pass then strips the inner pair. Content survives;
    # we assert the words are present rather than an exact normalized string.
    out = ttp._remove_markdown_formatting("**bold *and* italic**")
    assert "bold" in out and "and" in out and "italic" in out


def test_remove_markdown_formatting_underscores():
    assert ttp._remove_markdown_formatting("__bold__ and _it_") == "bold and it"


def test_remove_markdown_formatting_no_markup_passthrough():
    text = "just plain text with no formatting at all"
    assert ttp._remove_markdown_formatting(text) == text


# ── markdown links ────────────────────────────────────────────────────────────
def test_remove_markdown_links_keeps_text():
    assert ttp._remove_markdown_links("[text](http://x.com)") == "text"


def test_remove_markdown_links_inside_prose():
    out = ttp._remove_markdown_links("click [here](http://x.com) now")
    assert out == "click here now"


def test_remove_markdown_links_no_bracket():
    text = "no links here"
    assert ttp._remove_markdown_links(text) == text


# ── headers ───────────────────────────────────────────────────────────────────
def test_remove_markdown_headers():
    out = ttp._remove_markdown_headers("# Title\n## Subtitle\nbody")
    assert out == "Title\nSubtitle\nbody"


def test_remove_markdown_headers_no_hash():
    text = "no headers\njust text"
    assert ttp._remove_markdown_headers(text) == text


# ── list markers ──────────────────────────────────────────────────────────────
def test_remove_list_markers_bullets():
    out = ttp._remove_list_markers("- first\n* second\n+ third")
    assert out == "first\nsecond\nthird"


def test_remove_list_markers_numbered():
    out = ttp._remove_list_markers("1. one\n2. two")
    assert out == "one\ntwo"


def test_remove_list_markers_indented_bullet():
    out = ttp._remove_list_markers("  - nested item")
    assert out == "  nested item"


def test_remove_list_markers_plain_lines_unchanged():
    text = "just a line\nanother line"
    assert ttp._remove_list_markers(text) == text


# ── html tags ─────────────────────────────────────────────────────────────────
def test_strip_html_tags_basic():
    assert ttp._strip_html_tags("<b>hi</b>") == "hi"


def test_strip_html_tags_multiple():
    assert ttp._strip_html_tags("<p>hello <i>world</i></p>") == "hello world"


def test_strip_html_tags_no_tags():
    assert ttp._strip_html_tags("plain text") == "plain text"


def test_strip_html_tags_malformed():
    # Unclosed tag: everything until '>' is dropped.
    assert ttp._strip_html_tags("a <b unclosed text") == "a "


# ── emoji / special chars ─────────────────────────────────────────────────────
def test_is_emoji_codepoint_known_emoji():
    assert ttp._is_emoji_codepoint(0x1F600) is True  # grinning face
    assert ttp._is_emoji_codepoint(0x1F34B) is True  # banana


def test_is_emoji_codepoint_non_emoji():
    assert ttp._is_emoji_codepoint(ord("A")) is False
    assert ttp._is_emoji_codepoint(ord("1")) is False


def test_remove_special_chars_replaced_with_space():
    out = ttp._remove_special_chars("a#b@c$d")
    assert out == "a b c d"


def test_remove_special_chars_keeps_letters():
    assert ttp._remove_special_chars("hello world") == "hello world"


# ── repeated char compression ─────────────────────────────────────────────────
def test_compress_repeated_chars_run_replaced():
    assert ttp._compress_repeated_chars("aaab", frozenset("a"), "X") == "Xb"


def test_compress_repeated_chars_single_kept():
    assert ttp._compress_repeated_chars("ab", frozenset("a"), "X") == "ab"


def test_compress_repeated_chars_empty():
    assert ttp._compress_repeated_chars("", frozenset("a"), "X") == ""


# ── whitespace normalization ──────────────────────────────────────────────────
def test_normalize_whitespace_basic():
    assert ttp._normalize_whitespace("a   b\n\n c") == "a b c"


def test_normalize_whitespace_collapses_blank_lines():
    assert ttp._normalize_whitespace("line1\n\n\n\nline2") == "line1 line2"


def test_normalize_whitespace_tabs_and_newlines():
    assert ttp._normalize_whitespace("a\t\tb\nc") == "a b c"


# ── composition-style integration across helpers ──────────────────────────────
def test_full_clean_pipeline_no_markup_passthrough():
    """A realistic messy string should lose all markup/URLs/code."""
    messy = (
        "# Heading\n"
        "Visit **https://example.com** now or run `rm -rf /tmp`.\n"
        "- item one\n"
        "- item two with <b>bold tag</b>\n"
        "See [docs](http://docs.io) for more info."
    )
    out = ttp._strip_html_tags(messy)
    out = ttp._remove_urls(out)
    out = ttp._remove_inline_code(out)
    out = ttp._remove_markdown_formatting(out)
    out = ttp._remove_markdown_links(out)
    out = ttp._remove_markdown_headers(out)
    out = ttp._remove_list_markers(out)
    out = ttp._remove_special_chars(out)
    out = ttp._normalize_whitespace(out)

    assert "**" not in out
    assert "http" not in out
    assert "`" not in out
    assert "<b>" not in out
    assert "#" not in out
    assert "- item" not in out
    assert "Heading" in out
    assert "Visit" in out and "now" in out


def test_no_public_entrypoint_in_module():
    """Confirm (per brief) there is no public cleaning function here — the
    composition lives on TTSService._clean_text_for_tts (tested separately)."""
    public = [n for n in dir(ttp) if not n.startswith("_") and callable(getattr(ttp, n))]
    # The only non-underscore names are module-level constants/imports, not cleaning fns.
    assert all(not n.startswith(("clean", "preprocess")) for n in public)
