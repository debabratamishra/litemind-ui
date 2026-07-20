"""Unit tests for app/ingestion/enhanced_extractors.py.

Covers the CSV and image extractor paths. External services are mocked at their
boundary so no real OCR / model download / network activity occurs:

  * EasyOCR ``Reader`` is prevented from instantiating (and stubbed with a mock
    returning canned text when the OCR path is exercised).
  * ``pandas`` runs on tiny in-memory CSV files (no network, no model).

All tests run fully offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.ingestion import enhanced_extractors as ee


# ── Helpers ────────────────────────────────────────────────────────────────
def _disable_easyocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the module from building a real EasyOCR Reader at init time."""
    monkeypatch.setattr(ee, "EASYOCR_AVAILABLE", False)
    monkeypatch.setattr(ee, "easyocr", None)


def _make_tiny_png_bytes() -> bytes:
    """Create a small valid PNG image as raw bytes (no file needed)."""
    img = Image.new("RGB", (40, 30), color=(220, 210, 200))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_image_array() -> np.ndarray:
    """Return a small 3-channel numpy array usable as an image variant."""
    return np.full((20, 20, 3), 128, dtype=np.uint8)


def _mock_ocr_reader() -> MagicMock:
    """A mock EasyOCR reader returning canned text results."""
    reader = MagicMock()
    reader.readtext.return_value = [
        ([[0, 0], [10, 0], [10, 5], [0, 5]], "Hello", 0.95),
        ([[0, 6], [10, 6], [10, 11], [0, 11]], "World", 0.55),
        ([[0, 12], [10, 12], [10, 17], [0, 17]], "dup", 0.9),
        ([[0, 12], [10, 12], [10, 17], [0, 17]], "Dup", 0.9),  # duplicate (case) -> deduped
    ]
    return reader


# ── Singleton accessors ──────────────────────────────────────────────────────
def test_get_csv_processor_singleton():
    assert ee.get_csv_processor() is ee.get_csv_processor()


def test_get_image_processor_singleton(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    p1 = ee.get_image_processor()
    p2 = ee.get_image_processor()
    assert p1 is p2
    # No real OCR reader should have been created.
    assert p1.ocr_reader is None


# ── CSV: happy path (real pandas on tiny in-memory CSV) ──────────────────────
def test_extract_csv_enhanced_happy_path(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("name,age,score\nAlice,30,9.5\nBob,25,8.0\nCarol,40,7.2\n")

    blocks = ee.extract_csv_enhanced(csv)
    assert isinstance(blocks, list)
    assert len(blocks) > 0

    content_types = {b["metadata"]["content_type"] for b in blocks}
    assert "dataset_overview" in content_types
    assert "column_analysis" in content_types
    assert "data_chunk" in content_types

    # Numerical column should have been analysed.
    col_names = {b["metadata"].get("column_name") for b in blocks}
    assert "age" in col_names or "score" in col_names


def test_csv_processor_full_extraction_directly(tmp_path: Path):
    csv = tmp_path / "nums.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    processor = ee.EnhancedCSVProcessor()
    blocks = processor.extract_csv_enhanced(csv)

    overview = next(b for b in blocks if b["metadata"]["content_type"] == "dataset_overview")
    assert overview["metadata"]["row_count"] == 3
    assert overview["metadata"]["column_count"] == 2


# ── CSV: empty file (header only) ────────────────────────────────────────────
def test_extract_csv_enhanced_empty_file(tmp_path: Path):
    csv = tmp_path / "empty.csv"
    csv.write_text("a,b,c\n")  # header only -> empty dataframe

    blocks = ee.extract_csv_enhanced(csv)
    assert len(blocks) == 1
    block = blocks[0]
    assert block["metadata"]["content_type"] == "empty_file"
    assert "no data" in block["content"].lower()


def test_csv_processor_empty_dataframe(tmp_path: Path):
    csv = tmp_path / "blank.csv"
    csv.write_text("col1,col2\n")

    processor = ee.EnhancedCSVProcessor()
    block = processor._extract_csv_full(csv)[0]
    assert block["metadata"]["content_type"] == "empty_file"


# ── CSV: missing file -> graceful fallback error block ──────────────────────
def test_extract_csv_enhanced_missing_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    blocks = ee.extract_csv_enhanced(missing)
    assert len(blocks) == 1
    block = blocks[0]
    assert block["metadata"]["content_type"] == "error"
    assert "error" in block["metadata"]


def test_csv_processor_missing_file_returns_fallback(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    processor = ee.EnhancedCSVProcessor()
    blocks = processor._extract_csv_full(missing)
    assert blocks[0]["metadata"]["content_type"] == "error"


# ── CSV: module function uses the singleton processor ───────────────────────
def test_extract_csv_enhanced_module_function(tmp_path: Path):
    csv = tmp_path / "m.csv"
    csv.write_text("id,value\n1,10\n2,20\n")
    out = ee.extract_csv_enhanced(csv)
    assert isinstance(out, list)
    assert any(b["metadata"]["content_type"] == "dataset_overview" for b in out)


# ── CSV: mocked EnhancedCSVProcessor (boundary) ─────────────────────────────
def test_extract_csv_enhanced_mocked(tmp_path: Path):
    # The module function delegates to the get_csv_processor() singleton, so the
    # real boundary to mock is that getter (not the EnhancedCSVProcessor class).
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n")
    fake_processor = MagicMock()
    fake_processor.extract_csv_enhanced.return_value = [{"a": "1", "b": "2"}]
    with patch.object(ee, "get_csv_processor", return_value=fake_processor):
        out = ee.extract_csv_enhanced(csv)
        assert out == [{"a": "1", "b": "2"}]


# ── Image: OCR path mocked -> returns canned text ───────────────────────────
def test_image_ocr_text_extraction_mocked(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    processor = ee.EnhancedImageProcessor()

    # Wire up the mocked OCR reader directly on the instance.
    processor.ocr_reader = _mock_ocr_reader()
    processor.initialized = True

    img_array = _make_image_array()
    metadata = {"filename": "scan.png"}
    result = processor._extract_ocr_text(img_array, metadata)
    assert result is not None

    assert result["metadata"]["content_type"] == "ocr_text"
    content = result["content"]
    assert "Hello" in content
    assert "World" in content
    assert result["metadata"]["text_blocks_count"] == 3  # "dup"/"Dup" deduped to 1
    assert result["metadata"]["high_confidence_blocks"] == 2  # conf > 0.7


def test_image_ocr_dedupes_repeated_text(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    processor = ee.EnhancedImageProcessor()
    processor.ocr_reader = _mock_ocr_reader()
    processor.initialized = True

    result = processor._extract_ocr_text(_make_image_array(), {"filename": "x.png"})
    assert result is not None
    # "dup" and "Dup" map to the same lowercased key -> counted once.
    assert result["metadata"]["text_blocks_count"] == 3


def test_extract_image_content_with_mocked_ocr(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    processor = ee.EnhancedImageProcessor()
    processor.ocr_reader = _mock_ocr_reader()
    processor.initialized = True

    metadata = {"filename": "photo.png", "file_size": 1234}
    blocks = processor.extract_image_content(_make_tiny_png_bytes(), metadata)

    content_types = {b["metadata"]["content_type"] for b in blocks}
    assert "ocr_text" in content_types
    assert "image_analysis" in content_types  # always produced from real image


# ── Image: no OCR available -> analysis + fallback only ──────────────────────
def test_extract_image_content_no_ocr_falls_back(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    processor = ee.EnhancedImageProcessor()
    assert processor.ocr_reader is None
    assert processor.initialized is False

    blocks = processor.extract_image_content(_make_tiny_png_bytes(), {"filename": "img.png"})
    content_types = {b["metadata"]["content_type"] for b in blocks}
    # With no OCR, we still get an image analysis block from the real image.
    assert "image_analysis" in content_types


# ── Image: corrupt bytes -> graceful image_reference fallback ───────────────
def test_extract_image_content_corrupt_bytes_fallback(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    processor = ee.EnhancedImageProcessor()

    blocks = processor.extract_image_content(b"\x89PNG not a real image", {"filename": "bad.png"})
    assert len(blocks) == 1
    block = blocks[0]
    assert block["metadata"]["content_type"] == "image_reference"


# ── Image: module functions ─────────────────────────────────────────────────
def test_process_images_enhanced_mocked(tmp_path: Path):
    # process_images_enhanced delegates to get_image_processor(); mock that.
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    fake_processor = MagicMock()
    fake_processor.extract_image_content.return_value = [{"content": "caption", "metadata": {}}]
    with patch.object(ee, "get_image_processor", return_value=fake_processor):
        out = ee.process_images_enhanced([{"image_bytes": b"\x89PNG", "metadata": {"filename": "pic.png"}}])
        assert out == [{"content": "caption", "metadata": {}}]


def test_process_images_enhanced_real_bytes(monkeypatch: pytest.MonkeyPatch):
    _disable_easyocr(monkeypatch)
    records = [{"image_bytes": _make_tiny_png_bytes(), "metadata": {"filename": "p.png"}}]
    out = ee.process_images_enhanced(records)
    assert isinstance(out, list)
    assert len(out) >= 1
    assert any(b["metadata"]["content_type"] == "image_analysis" for b in out)


def test_process_documents_enhanced_unsupported_ext():
    # Unsupported extensions are skipped (continue) -> empty result.
    records = [{"file_path": "/tmp/notes.txt", "filename": "notes.txt", "metadata": {}}]
    out = ee.process_documents_enhanced(records)
    assert out == []


def test_process_documents_enhanced_missing_pdf_fallback(monkeypatch: pytest.MonkeyPatch):
    # The underlying extractor swallows its own file errors, so to exercise the
    # process_documents_enhanced except-branch we make the extractor raise.
    records = [{"file_path": "/tmp/missing.pdf", "filename": "missing.pdf", "metadata": {}}]
    with patch.object(ee, "extract_pdf_enhanced", side_effect=RuntimeError("boom")):
        out = ee.process_documents_enhanced(records)
    assert len(out) == 1
    assert out[0]["metadata"]["content_type"] == "document_error"
