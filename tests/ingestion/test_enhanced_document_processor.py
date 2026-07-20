"""Unit tests for app/ingestion/enhanced_document_processor.py.

These tests cover the PDF / DOCX / EPUB extraction orchestration and the
pure formatting/classification helpers. Heavy document libraries
(PyMuPDF/`fitz`, `pdfplumber`, `camelot`, `python-docx`, `BeautifulSoup`,
`zipfile`) are mocked at their boundaries so that NO real document parsing,
OCR, model download, or network access occurs. All tests run fully offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion import enhanced_document_processor as edp


@pytest.fixture(autouse=True)
def _disable_easyocr(monkeypatch):
    """Prevent `EnhancedDocumentProcessor.__init__` from loading a real
    EasyOCR model (which would attempt a network download offline).

    The constructor only calls ``easyocr.Reader`` when ``EASYOCR_AVAILABLE``
    is True, so forcing it False keeps every test fully offline.
    """
    monkeypatch.setattr(edp, "EASYOCR_AVAILABLE", False)


# ── Fake PyMuPDF (fitz) objects ────────────────────────────────────────────
class FakeRect:
    def __init__(self, *args):
        if len(args) == 4:
            self.x0, self.y0, self.x1, self.y1 = args
        else:
            self.x0 = self.y0 = self.x1 = self.y1 = 0

    @property
    def width(self):
        return abs(self.x1 - self.x0)

    @property
    def height(self):
        return abs(self.y1 - self.y0)


class FakePixmap:
    def __init__(self, width: int = 200, height: int = 200, n: int = 1):
        self.width = width
        self.height = height
        self.n = n

    def tobytes(self, fmt: str = "png") -> bytes:
        return b"fake-image-bytes"


class FakePage:
    def __init__(self, text_dict: dict, page_text: str = "", images=()):
        self._text_dict = text_dict
        self._page_text = page_text
        self._images = list(images)
        self.rect = FakeRect(0, 0, 612, 792)
        self.parent = "fake-doc-parent"

    def get_text(self, kind: str | None = None):
        if kind == "dict":
            return self._text_dict
        return self._page_text

    def get_images(self, full: bool = True):
        return self._images

    def get_image_rects(self, xref):
        return [FakeRect(10, 10, 110, 110)]

    def get_textbox(self, rect) -> str:
        return "surrounding text context"


class FakeDoc:
    def __init__(self, pages, metadata: dict | None = None, toc=(), name: str = "doc.pdf"):
        self._pages = list(pages)
        self.metadata = metadata or {}
        self._toc = list(toc)
        self.name = name

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def get_toc(self):
        return self._toc

    def close(self):
        return None


class FakeFitz:
    csRGB = "RGB"  # noqa: N815  (mirrors PyMuPDF's mixedCase attribute)

    def __init__(self, pages, metadata=None, toc=()):
        self._pages = pages
        self._metadata = metadata or {}
        self._toc = toc
        self.open = MagicMock(side_effect=lambda p: FakeDoc(pages, self._metadata, self._toc, str(p)))

    def Matrix(self, *args):  # noqa: N802  (mirrors PyMuPDF API)
        return "matrix"

    def Rect(self, *args):  # noqa: N802  (mirrors PyMuPDF API)
        return FakeRect(*args)

    def Pixmap(self, *args):  # noqa: N802  (mirrors PyMuPDF API)
        return FakePixmap()


def _sample_text_dict() -> dict:
    """A simple one-block header + one-block body text dict."""
    return {
        "width": 612,
        "height": 792,
        "blocks": [
            {
                "lines": [{"spans": [
                    {"text": "Chapter One", "size": 18.0, "flags": 16, "font": "Helvetica-Bold"}
                ]}],
                "bbox": [50, 30, 300, 60],
            },
            {
                "lines": [{"spans": [
                    {"text": "This is the body paragraph of the document.", "size": 12.0, "flags": 0, "font": "Helvetica"}
                ]}],
                "bbox": [50, 100, 400, 130],
            },
        ],
    }


def _make_sample_pdf_file(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


class _FakeCamelotModule:
    """Stand-in for the optional `camelot` module (not installed offline)."""

    read_pdf: MagicMock


# ── get_document_processor singleton ───────────────────────────────────────
def test_get_document_processor_singleton():
    assert edp.get_document_processor() is edp.get_document_processor()


# ── PDF integration (mocked fitz + pdfplumber) ─────────────────────────────
def test_extract_pdf_enhanced_mocked(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    page_text = ("Chapter One. " * 30)  # enough words for layout analysis

    fake_fitz = FakeFitz(
        pages=[FakePage(_sample_text_dict(), page_text=page_text)],
        metadata={"title": "My Report", "author": "Jane"},
    )

    # pdfplumber context manager returning one page with one table.
    fake_pdfplumber = MagicMock()
    table_page = MagicMock()
    table_page.extract_tables.return_value = [[["Name", "Value"], ["alpha", "1"], ["beta", "2"]]]
    ctx = fake_pdfplumber.open.return_value
    ctx.__enter__.return_value.pages = [table_page]

    with (
        patch.object(edp, "fitz", fake_fitz),
        patch.object(edp, "pdfplumber", fake_pdfplumber),
        patch.object(edp, "EASYOCR_AVAILABLE", False),
        patch.object(edp.EnhancedDocumentProcessor, "_extract_tables_with_camelot", return_value=[]),
    ):
        out = edp.extract_pdf_enhanced(pdf)

    assert isinstance(out, list)
    assert out  # non-empty

    content_types = {b["metadata"]["content_type"] for b in out}
    # overview, layout analysis, text/header block, structured headers, table, structure analysis
    assert "pdf_document_overview" in content_types
    assert "pdf_table_structured" in content_types
    assert "document_structure_analysis" in content_types

    table_block = next(b for b in out if b["metadata"]["content_type"] == "pdf_table_structured")
    assert "ROW 1" in table_block["content"]
    assert table_block["metadata"]["rows"] == 3


# ── PDF corrupt input -> handled gracefully (no crash) ─────────────────────
def test_extract_pdf_enhanced_corrupt_does_not_raise(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    fake_fitz = MagicMock()
    fake_fitz.open.side_effect = Exception("not a real PDF")

    with patch.object(edp, "fitz", fake_fitz):
        out = edp.extract_pdf_enhanced(pdf)

    # Each sub-extractor swallows its own errors, so a fully corrupt file
    # yields an empty list rather than raising.
    assert isinstance(out, list)
    assert out == []


# ── PDF fallback content block (error contract) ────────────────────────────
def test_create_fallback_pdf_content_error_block(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    processor = edp.EnhancedDocumentProcessor()
    out = processor._create_fallback_pdf_content(pdf, "boom")
    assert len(out) == 1
    assert out[0]["metadata"]["content_type"] == "pdf_processing_error"
    assert "boom" in out[0]["content"]
    assert out[0]["metadata"]["error"] == "boom"


# ── PDF image extraction (mocked fitz Pixmap) ──────────────────────────────
def test_extract_pdf_enhanced_image_extraction_mocked(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    page_text = ("Image page. " * 30)
    # One embedded image: xref=1, width=200, height=200 (passes min_image_size).
    images = [(1, 0, 200, 200, 0, 0)]
    fake_fitz = FakeFitz(
        pages=[FakePage(_sample_text_dict(), page_text=page_text, images=images)],
        metadata={},
    )

    with (
        patch.object(edp, "fitz", fake_fitz),
        patch.object(edp, "pdfplumber", MagicMock()),
        patch.object(edp, "EASYOCR_AVAILABLE", False),
        patch.object(edp.EnhancedDocumentProcessor, "_extract_tables_with_camelot", return_value=[]),
    ):
        out = edp.extract_pdf_enhanced(pdf)

    image_blocks = [b for b in out if b["metadata"].get("content_type") == "pdf_image_enhanced"]
    assert image_blocks, "expected at least one enhanced image block"
    meta = image_blocks[0]["metadata"]
    assert "image_bytes" in meta
    assert meta["image_bytes"]  # base64-encoded fake bytes present


# ── Table formatting (pure helper) ─────────────────────────────────────────
def test_format_table_content_serialization():
    processor = edp.EnhancedDocumentProcessor()
    table = [["Header A", "Header B"], ["r1a", "r1b"], ["r2a", "r2b"]]
    text = processor._format_table_content(table, page_num=2, table_index=0, filename="doc.pdf")

    assert "TABLE 1 (Page 2)" in text
    assert "HEADERS: Header A | Header B" in text
    assert "ROW 1: r1a | r1b" in text
    assert "ROW 2: r2a | r2b" in text


def test_format_table_content_empty_table():
    processor = edp.EnhancedDocumentProcessor()
    text = processor._format_table_content([], page_num=1, table_index=0, filename="doc.pdf")
    assert "[Empty table]" in text


# ── pdfplumber table extraction helper (mocked) ────────────────────────────
def test_extract_tables_with_pdfplumber_mocked(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    processor = edp.EnhancedDocumentProcessor()

    table_page = MagicMock()
    table_page.extract_tables.return_value = [[["A", "B"], ["1", "2"]]]
    fake_pdfplumber = MagicMock()
    ctx = fake_pdfplumber.open.return_value
    ctx.__enter__.return_value.pages = [table_page]

    with patch.object(edp, "pdfplumber", fake_pdfplumber):
        out = processor._extract_tables_with_pdfplumber(Path(str(pdf)))

    assert len(out) == 1
    assert out[0]["metadata"]["content_type"] == "pdf_table_structured"
    assert out[0]["metadata"]["columns"] == 2
    assert out[0]["metadata"]["rows"] == 2


# ── Camelot table extraction (fake camelot module in sys.modules) ──────────
def test_extract_tables_with_camelot_mocked(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)
    processor = edp.EnhancedDocumentProcessor()

    import pandas as pd

    df = pd.DataFrame({"Col": ["x", "y"], "Val": [1, 2]})

    class FakeTable:
        def __init__(self):
            self.df = df
            self.page = 1
            self.accuracy = 0.97

    class FakeTables:
        n = 1

        def __iter__(self):
            return iter([FakeTable()])

    fake_camelot = _FakeCamelotModule()
    fake_camelot.read_pdf = MagicMock(return_value=FakeTables())
    sys.modules["camelot"] = fake_camelot  # type: ignore
    try:
        out = processor._extract_tables_with_camelot(Path(str(pdf)))
    finally:
        sys.modules.pop("camelot", None)

    assert len(out) == 1
    block = out[0]
    assert block["metadata"]["content_type"] == "pdf_table_camelot"
    assert block["metadata"]["accuracy"] == 0.97
    assert "COMPLEX TABLE 1" in block["content"]
    assert "Extraction Accuracy: 0.97" in block["content"]


# ── DOCX integration (mocked python-docx) ──────────────────────────────────
def _make_fake_docx():
    doc = MagicMock()

    heading = MagicMock()
    heading.text = "Section Title"
    heading.style.name = "Heading 1"
    heading.style.font.bold = True

    body1 = MagicMock()
    body1.text = "First paragraph of the document."
    body1.style.name = "Normal"
    body1.style.font.bold = False

    body2 = MagicMock()
    body2.text = "Second paragraph with more content."
    body2.style.name = "Normal"
    body2.style.font.bold = False

    doc.paragraphs = [heading, body1, body2]

    # One table: header row + one data row.
    cell_h1, cell_h2 = MagicMock(), MagicMock()
    cell_h1.text = "Col1"
    cell_h2.text = "Col2"
    cell_d1, cell_d2 = MagicMock(), MagicMock()
    cell_d1.text = "a"
    cell_d2.text = "b"
    row0 = MagicMock()
    row0.cells = [cell_h1, cell_h2]
    row1 = MagicMock()
    row1.cells = [cell_d1, cell_d2]
    table = MagicMock()
    table.rows = [row0, row1]
    doc.tables = [table]

    props = MagicMock()
    props.title = "My Doc"
    props.author = "John"
    props.subject = "Testing"
    props.created = None
    props.modified = None
    doc.core_properties = props

    # One embedded image relationship.
    rel = MagicMock()
    rel.reltype = "image/png"
    target = MagicMock()
    target.blob = b"img-bytes"
    target.content_type = "image/png"
    rel.target_part = target
    doc.part.rels = {"rId1": rel}
    return doc


def test_extract_docx_enhanced_mocked(tmp_path):
    docx = tmp_path / "doc.docx"
    docx.write_bytes(b"PK\x03\x04 fake")

    fake_docx = _make_fake_docx()
    with patch.object(edp, "Document", return_value=fake_docx):
        out = edp.extract_docx_enhanced(docx)

    assert isinstance(out, list)
    content_types = {b["metadata"]["content_type"] for b in out}
    assert "docx_overview" in content_types
    assert "docx_section" in content_types
    assert "docx_table" in content_types
    assert "docx_image" in content_types

    table_block = next(b for b in out if b["metadata"]["content_type"] == "docx_table")
    assert "HEADERS: Col1 | Col2" in table_block["content"]
    assert table_block["metadata"]["rows"] == 2

    image_block = next(b for b in out if b["metadata"]["content_type"] == "docx_image")
    assert image_block["metadata"]["image_bytes"]


# ── DOCX corrupt input -> fallback error block ─────────────────────────────
def test_extract_docx_enhanced_corrupt_returns_fallback(tmp_path):
    docx = tmp_path / "doc.docx"
    docx.write_bytes(b"PK\x03\x04 fake")

    with patch.object(edp, "Document", side_effect=Exception("cannot parse docx")):
        out = edp.extract_docx_enhanced(docx)

    assert isinstance(out, list)
    assert out[-1]["metadata"]["content_type"] == "processing_error"
    assert "cannot parse docx" in out[-1]["content"]


# ── DOCX table formatting (pure helper) ────────────────────────────────────
def test_extract_docx_tables_format(tmp_path):
    processor = edp.EnhancedDocumentProcessor()
    fake_docx = _make_fake_docx()
    out = processor._extract_docx_tables(fake_docx, "doc.docx")
    assert len(out) == 1
    assert out[0]["metadata"]["content_type"] == "docx_table"
    assert "ROW 1: a | b" in out[0]["content"]


# ── EPUB integration (mocked zipfile) ──────────────────────────────────────
CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    '</rootfiles></container>'
)

CONTENT_OPF = (
    '<?xml version="1.0"?>'
    '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<metadata>'
    '<dc:title>Great Book</dc:title>'
    '<dc:creator>Author Name</dc:creator>'
    '<dc:description>A nice book.</dc:description>'
    '<dc:language>en</dc:language>'
    '</metadata>'
    '<manifest>'
    '<item id="c1" href="chap1.html" media-type="application/xhtml+xml"/>'
    '</manifest>'
    '<spine><itemref idref="c1"/></spine>'
    '</package>'
)

CHAPTER_HTML = (
    "<html><head><title>Chap</title></head><body>"
    "<h1>Chapter One</h1>"
    "<p>This is a substantial chapter body with plenty of words to exceed "
    "the one hundred character minimum threshold used by the extractor.</p>"
    "</body></html>"
)


def _make_fake_epub_zip():
    files = {
        "META-INF/container.xml": CONTAINER_XML,
        "OEBPS/content.opf": CONTENT_OPF,
        "OEBPS/chap1.html": CHAPTER_HTML,
    }
    fake_zip = MagicMock()
    fake_zip.read.side_effect = lambda name: files[name].encode("utf-8")
    fake_zip.namelist.return_value = list(files.keys())
    return fake_zip


def test_extract_epub_enhanced_mocked(tmp_path):
    epub = tmp_path / "book.epub"
    epub.write_bytes(b"PK\x03\x04 fake epub")

    fake_zip = _make_fake_epub_zip()
    cm = MagicMock()
    cm.__enter__.return_value = fake_zip
    with patch.object(edp.zipfile, "ZipFile", return_value=cm):
        out = edp.extract_epub_enhanced(epub)

    content_types = {b["metadata"]["content_type"] for b in out}
    assert "epub_metadata" in content_types
    assert "epub_chapter" in content_types

    meta = next(b for b in out if b["metadata"]["content_type"] == "epub_metadata")
    assert meta["metadata"]["title"] == "Great Book"
    assert meta["metadata"]["author"] == "Author Name"

    chapter = next(b for b in out if b["metadata"]["content_type"] == "epub_chapter")
    assert "Chapter One" in chapter["content"]
    assert chapter["metadata"]["word_count"] > 0


# ── EPUB metadata helper (pure) ────────────────────────────────────────────
def test_extract_epub_metadata_unit():
    processor = edp.EnhancedDocumentProcessor()
    fake_zip = _make_fake_epub_zip()
    out = processor._extract_epub_metadata(fake_zip, "OEBPS/content.opf")
    assert out["metadata"]["content_type"] == "epub_metadata"
    assert out["metadata"]["title"] == "Great Book"
    assert "Author Name" in out["content"]


# ── EPUB missing content.opf -> fallback ───────────────────────────────────
def test_extract_epub_missing_content_opf_returns_fallback(tmp_path):
    epub = tmp_path / "book.epub"
    epub.write_bytes(b"PK\x03\x04 fake epub")

    fake_zip = MagicMock()
    fake_zip.read.side_effect = KeyError("META-INF/container.xml")  # no container
    fake_zip.namelist.return_value = []  # no common opf locations

    cm = MagicMock()
    cm.__enter__.return_value = fake_zip
    with patch.object(edp.zipfile, "ZipFile", return_value=cm):
        out = edp.extract_epub_enhanced(epub)

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["metadata"]["content_type"] == "processing_error"
    assert "Could not find EPUB content.opf" in out[0]["content"]


# ── Document type classification (pure helper) ─────────────────────────────
def test_classify_document_type():
    processor = edp.EnhancedDocumentProcessor()

    assert processor._classify_document_type({
        "has_toc": True, "total_pages": 50, "avg_images_per_page": 0, "avg_text_density": 0.1
    }) == "book_or_manual"

    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 3, "avg_images_per_page": 0, "avg_text_density": 0.04
    }) == "article_or_paper"

    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 10, "avg_images_per_page": 0, "avg_text_density": 0.06
    }) == "text_heavy_document"

    # presentation_or_visual_document: many images, very low text density
    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 10, "avg_images_per_page": 5, "avg_text_density": 0.005
    }) == "presentation_or_visual_document"

    # report_with_figures: some images, moderate text density
    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 10, "avg_images_per_page": 2, "avg_text_density": 0.03
    }) == "report_with_figures"

    # academic_paper: pages >= 5, very few images, moderate density (not > 0.05)
    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 10, "avg_images_per_page": 0.05, "avg_text_density": 0.045
    }) == "academic_paper"

    # mixed_content_document: falls through every specific branch
    assert processor._classify_document_type({
        "has_toc": False, "total_pages": 10, "avg_images_per_page": 0.5, "avg_text_density": 0.03
    }) == "mixed_content_document"


def test_no_easyocr_model_load_on_construction(monkeypatch):
    """Constructing the processor must NOT invoke ``easyocr.Reader`` (which
    would download/load a real ML model). We stub ``easyocr`` with a recorder
    and assert it is never called, confirming the offline guarantee.
    """
    recorder = MagicMock()
    fake_easyocr = MagicMock()
    fake_easyocr.Reader = recorder
    monkeypatch.setattr(edp, "EASYOCR_AVAILABLE", False)
    monkeypatch.setattr(edp, "easyocr", fake_easyocr)

    edp.EnhancedDocumentProcessor()

    assert recorder.call_count == 0, "easyocr.Reader was invoked — a model load occurred"


# ── Module-level convenience functions delegate to the processor ───────────
def test_module_level_functions_delegate(tmp_path):
    pdf = _make_sample_pdf_file(tmp_path)

    fake_processor = MagicMock()
    fake_processor.extract_pdf_enhanced.return_value = [{"content": "p", "metadata": {}}]
    fake_processor.extract_docx_enhanced.return_value = [{"content": "d", "metadata": {}}]
    fake_processor.extract_epub_enhanced.return_value = [{"content": "e", "metadata": {}}]

    with patch.object(edp, "get_document_processor", return_value=fake_processor):
        pdf_out = edp.extract_pdf_enhanced(pdf)
        docx_out = edp.extract_docx_enhanced(pdf)
        epub_out = edp.extract_epub_enhanced(pdf)

    assert pdf_out == [{"content": "p", "metadata": {}}]
    assert docx_out == [{"content": "d", "metadata": {}}]
    assert epub_out == [{"content": "e", "metadata": {}}]
    fake_processor.extract_pdf_enhanced.assert_called_once()
    fake_processor.extract_docx_enhanced.assert_called_once()
    fake_processor.extract_epub_enhanced.assert_called_once()
