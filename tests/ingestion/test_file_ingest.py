"""Unit tests for app/ingestion/file_ingest.py.

These tests cover the pure logic of the ingestion pipeline (format/MIME
detection, text normalisation, chunking strategies) and the offline-safe
text-family extractors (plain text, markdown, HTML, JSON, JSONL, YAML, TOML,
RTF) without invoking heavy document libraries (PyMuPDF/python-pptx/pandas/
LibreOffice/etc.). Heavy-library paths are exercised only via dispatch routing
checks that mock the corresponding extractor.

All tests run fully offline: no network, no model downloads, no real document
libraries are imported on the tested code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion import file_ingest as fi


# ── _guess_mime ────────────────────────────────────────────────────────────
def test_guess_mime_pdf():
    mime = fi._guess_mime(Path("report.pdf"))
    assert mime == "application/pdf"


def test_guess_mime_text():
    mime = fi._guess_mime(Path("note.txt"))
    assert mime == "text/plain"


def test_guess_mime_unmapped_returns_octet_stream():
    # An extension that mimetypes does not map should fall back to octet-stream.
    assert fi._guess_mime(Path("file.zzz")) == "application/octet-stream"


# ── _read_file_bytes ───────────────────────────────────────────────────────
def test_read_file_bytes_roundtrip(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"\x00\x01\x02hello")
    assert fi._read_file_bytes(p) == b"\x00\x01\x02hello"


# ── _normalize_text ────────────────────────────────────────────────────────
def test_normalize_text_collapses_blank_lines_and_indentation():
    # Real behavior: collapses runs of blank lines and strips per-line leading
    # spaces, but does NOT join across paragraphs (the brief's "a b" was wrong).
    assert fi._normalize_text("a\n\n  b") == "a\n\nb"


def test_normalize_text_empty():
    assert fi._normalize_text("") == ""
    assert fi._normalize_text("   \n\n\t  ") == ""


def test_normalize_text_removes_carriage_returns_and_nulls():
    assert fi._normalize_text("x\r\ny\rz") == "x\ny\nz"
    assert "\x00" not in fi._normalize_text("a\x00b")


def test_normalize_text_collapses_inline_whitespace():
    assert fi._normalize_text("a    b\t\tc") == "a b c"


def test_normalize_text_strips_ends():
    assert fi._normalize_text("  spaced  \n") == "spaced"


# ── _make_block ────────────────────────────────────────────────────────────
def test_make_block_basic(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("content")
    block = fi._make_block("  hello world  ", p, "text_content", foo="bar")
    assert block is not None
    assert block["content"] == "hello world"
    assert block["metadata"]["filename"] == "note.txt"
    assert block["metadata"]["content_type"] == "text_content"
    assert block["metadata"]["foo"] == "bar"
    # filetype is derived via _guess_mime
    assert block["metadata"]["filetype"] == "text/plain"


def test_make_block_empty_returns_none(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert fi._make_block("", p, "text_content") is None
    assert fi._make_block("   \n\n  ", p, "text_content") is None


# ── _split_long_text ───────────────────────────────────────────────────────
def test_split_long_text_respects_max_chars():
    chunks = fi._split_long_text("x" * 5000, max_chars=1000)
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)


def test_split_long_text_single_long_sentence():
    # A single very long sentence (no sentence boundary) is hard-split.
    text = "a" * 3500
    chunks = fi._split_long_text(text, max_chars=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text


def test_split_long_text_short_text_returns_single_chunk():
    chunks = fi._split_long_text("Short sentence here.", max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == "Short sentence here."


def test_split_long_text_respects_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence."
    chunks = fi._split_long_text(text, max_chars=1000)
    # All three fit in one chunk since max_chars is large.
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_text_multiple_sentences_split_when_overflow():
    # Build sentences each ~60 chars so several fit per 100-char chunk.
    sentence = "word " * 12  # ~60 chars
    text = " ".join([sentence.strip()] * 10)
    chunks = fi._split_long_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


# ── _split_logical_sections ────────────────────────────────────────────────
def test_split_logical_sections_empty():
    assert fi._split_logical_sections("") == []
    assert fi._split_logical_sections("   \n\n  ") == []


def test_split_logical_sections_short_text_single_section():
    sections = fi._split_logical_sections("Hello world.", max_chars=2800)
    assert sections == ["Hello world."]


def test_split_logical_sections_splits_long_paragraphs():
    # A paragraph longer than max_chars is delegated to _split_long_text.
    para = "word " * 2000  # far exceeds default 2800? ~12000 chars
    sections = fi._split_logical_sections(para, max_chars=2800)
    assert len(sections) > 1
    assert all(len(s) <= 2800 for s in sections)


def test_split_logical_sections_groups_paragraphs():
    text = "\n\n".join(["Paragraph %d with some text." % i for i in range(5)])
    sections = fi._split_logical_sections(text, max_chars=2800)
    assert len(sections) >= 1
    # All original paragraph text is preserved across sections.
    joined = "\n\n".join(sections)
    for i in range(5):
        assert "Paragraph %d with some text." % i in joined


# ── _collect_blocks_from_text ──────────────────────────────────────────────
def test_collect_blocks_from_text_filters_empty_sections(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("x")
    # Short paragraphs that fit under max_chars are joined into a single
    # section; the whitespace-only paragraph produces no block.
    blocks = fi._collect_blocks_from_text("real content\n\n   \n\nmore", p, "text_content")
    assert len(blocks) == 1
    joined = blocks[0]["content"]
    assert "real content" in joined
    assert "more" in joined
    # The blank/whitespace-only paragraph left no trace.
    assert joined.count("\n\n") == 1
    assert all(b["metadata"]["content_type"] == "text_content" for b in blocks)


# ── get_ingestion_capabilities ─────────────────────────────────────────────
def test_get_ingestion_capabilities_returns_dict():
    caps = fi.get_ingestion_capabilities()
    assert isinstance(caps, dict)
    assert "supported_extensions" in caps
    assert "local_pipeline" in caps
    assert isinstance(caps["supported_extensions"], list)
    assert isinstance(caps["local_pipeline"], dict)


def test_get_ingestion_capabilities_lists_expected_extensions():
    caps = fi.get_ingestion_capabilities()
    exts = caps["supported_extensions"]
    # SUPPORTED_EXTENSIONS are dotless ("pdf"), not ".pdf".
    for expected in ["pdf", "txt", "md", "csv", "png", "json"]:
        assert expected in exts


# ── _extract_text_document dispatch routing ────────────────────────────────
def test_extract_text_document_routes_html(tmp_path, monkeypatch):
    p = tmp_path / "page.html"
    p.write_text("<html></html>")
    captured = {}

    def fake(path):
        captured["path"] = path
        return [{"content": "html", "metadata": {}}]

    monkeypatch.setattr(fi, "_extract_html_document", fake)
    result = fi._extract_text_document(p)
    assert captured["path"] == p
    assert result == [{"content": "html", "metadata": {}}]


def test_extract_text_document_routes_json(tmp_path, monkeypatch):
    p = tmp_path / "data.json"
    p.write_text("{}")
    captured = {}

    def fake(path):
        captured["path"] = path
        return [{"content": "json", "metadata": {}}]

    monkeypatch.setattr(fi, "_extract_json_document", fake)
    assert fi._extract_text_document(p) == [{"content": "json", "metadata": {}}]
    assert captured["path"] == p


def test_extract_text_document_routes_jsonl(tmp_path, monkeypatch):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\n')
    captured = {}

    def fake(path):
        captured["path"] = path
        return [{"content": "jsonl", "metadata": {}}]

    monkeypatch.setattr(fi, "_extract_jsonl_document", fake)
    assert fi._extract_text_document(p) == [{"content": "jsonl", "metadata": {}}]
    assert captured["path"] == p


def test_extract_text_document_routes_yaml(tmp_path, monkeypatch):
    p = tmp_path / "data.yaml"
    p.write_text("a: 1\n")
    captured = {}

    def fake(path):
        captured["path"] = path
        return [{"content": "yaml", "metadata": {}}]

    monkeypatch.setattr(fi, "_extract_yaml_document", fake)
    assert fi._extract_text_document(p) == [{"content": "yaml", "metadata": {}}]
    assert captured["path"] == p


def test_extract_text_document_routes_toml(tmp_path, monkeypatch):
    p = tmp_path / "data.toml"
    p.write_text("a = 1\n")
    captured = {}

    def fake(path):
        captured["path"] = path
        return [{"content": "toml", "metadata": {}}]

    monkeypatch.setattr(fi, "_extract_toml_document", fake)
    assert fi._extract_text_document(p) == [{"content": "toml", "metadata": {}}]
    assert captured["path"] == p


def test_extract_text_document_plain_text_reads_file(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello\n\nworld")
    blocks = fi._extract_text_document(p)
    assert blocks
    assert blocks[0]["metadata"]["content_type"] == "text_content"
    contents = [b["content"] for b in blocks]
    assert "hello" in " ".join(contents)
    assert "world" in " ".join(contents)


# ── _extract_html_document ─────────────────────────────────────────────────
def test_extract_html_document_title_and_body(tmp_path, monkeypatch):
    # Force the BeautifulSoup branch (trafilatura is installed in this env and
    # would otherwise pre-empt it), so we deterministically exercise title
    # extraction and script-tag stripping.
    monkeypatch.setattr(fi, "TRAFILATURA_AVAILABLE", False)
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>My Page</title></head>"
        "<body><h1>Heading</h1><p>Some body text.</p>"
        "<script>var x=1;</script></body></html>"
    )
    blocks = fi._extract_html_document(p)
    assert blocks
    # Title block should be present.
    titles = [b for b in blocks if b["metadata"].get("content_type") == "html_title"]
    assert any(b["content"] == "My Page" for b in titles)
    # Script tags are stripped, body text preserved.
    all_text = " ".join(b["content"] for b in blocks)
    assert "Some body text." in all_text
    assert "var x=1" not in all_text


# ── _extract_json_document ─────────────────────────────────────────────────
def test_extract_json_document_valid(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"b":2,"a":1}')
    blocks = fi._extract_json_document(p)
    assert blocks
    block = blocks[0]
    assert block["metadata"]["content_type"] == "structured_text"
    # json.dumps with sort_keys -> "a" before "b". _normalize_text strips the
    # indentation, but the payload remains valid JSON.
    assert json.loads(block["content"]) == {"a": 1, "b": 2}


def test_extract_json_document_invalid_falls_back_to_text(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    blocks = fi._extract_json_document(p)
    assert blocks
    assert blocks[0]["metadata"]["content_type"] == "text_content"
    assert "{not valid json" in blocks[0]["content"]


# ── _extract_jsonl_document ────────────────────────────────────────────────
def test_extract_jsonl_document(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n')
    blocks = fi._extract_jsonl_document(p)
    assert blocks
    content = blocks[0]["content"]
    assert '"a": 1' in content
    assert '"b": 2' in content
    assert blocks[0]["metadata"]["content_type"] == "structured_text"


# ── _extract_toml_document ─────────────────────────────────────────────────
def test_extract_toml_document(tmp_path):
    p = tmp_path / "data.toml"
    p.write_text('title = "hello"\n[section]\nkey = "value"\n')
    blocks = fi._extract_toml_document(p)
    assert blocks
    assert blocks[0]["metadata"]["content_type"] == "structured_text"
    parsed_back = json.loads(blocks[0]["content"])
    assert parsed_back["title"] == "hello"
    assert parsed_back["section"]["key"] == "value"


# ── _extract_yaml_document ─────────────────────────────────────────────────
def test_extract_yaml_document_valid(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text("name: test\nnested:\n  a: 1\n")
    blocks = fi._extract_yaml_document(p)
    assert blocks
    assert blocks[0]["metadata"]["content_type"] == "structured_text"
    parsed_back = json.loads(blocks[0]["content"])
    assert parsed_back["name"] == "test"
    assert parsed_back["nested"]["a"] == 1


# ── _extract_rtf_document ──────────────────────────────────────────────────
def test_extract_rtf_document_plain(tmp_path):
    # RTF without the striprtf library (or with it) should still yield a block
    # whose content_type is rtf_content. A bare "rtf-like" text is treated as
    # raw text when striprtf is unavailable.
    p = tmp_path / "doc.rtf"
    p.write_text("Some rtf body text here.")
    blocks = fi._extract_rtf_document(p)
    assert blocks
    assert blocks[0]["metadata"]["content_type"] == "rtf_content"
    assert "Some rtf body text here." in blocks[0]["content"]


# ── ingest_file end-to-end (light, offline paths) ──────────────────────────
def test_ingest_plain_text_file(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world")
    text_chunks, images, tables = fi.ingest_file(p)
    assert isinstance(text_chunks, list)
    assert isinstance(images, list)
    assert isinstance(tables, list)
    assert text_chunks
    assert images == []
    assert tables == []
    assert "hello world" in text_chunks[0]["content"]


def test_ingest_markdown_file(tmp_path):
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nSome markdown content.")
    text_chunks, images, _ = fi.ingest_file(p)
    assert text_chunks
    joined = " ".join(b["content"] for b in text_chunks)
    assert "Title" in joined
    assert "Some markdown content." in joined


def test_ingest_html_file(tmp_path, monkeypatch):
    # Force the BeautifulSoup branch for deterministic title/body extraction.
    monkeypatch.setattr(fi, "TRAFILATURA_AVAILABLE", False)
    p = tmp_path / "page.html"
    p.write_text("<html><head><title>T</title></head><body><p>Body.</p></body></html>")
    text_chunks, images, _ = fi.ingest_file(p)
    assert text_chunks
    joined = " ".join(b["content"] for b in text_chunks)
    assert "Body." in joined
    assert "T" in joined


def test_ingest_json_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"k": "v"}')
    text_chunks, _, _ = fi.ingest_file(p)
    assert text_chunks
    assert text_chunks[0]["metadata"]["content_type"] == "structured_text"


def test_ingest_unsupported_format_returns_empty(tmp_path):
    p = tmp_path / "file.zzz"
    p.write_text("whatever")
    text_chunks, images, tables = fi.ingest_file(p)
    assert text_chunks == []
    assert images == []
    assert tables == []


@pytest.mark.parametrize(
    "ext,content",
    [
        (".txt", "plain text body"),
        (".md", "# heading\n\nmarkdown body"),
        (".json", '{"a": 1}'),
        (".jsonl", '{"a": 1}\n'),
        (".yaml", "a: 1\n"),
        (".toml", 'a = 1\n'),
        (".html", "<p>html body</p>"),
    ],
)
def test_ingest_text_family_produces_blocks(tmp_path, ext, content):
    p = tmp_path / f"sample{ext}"
    p.write_text(content)
    text_chunks, images, tables = fi.ingest_file(p)
    assert text_chunks, f"expected blocks for {ext}"
    assert images == []
    assert tables == []
