from app.core.rag_formats import (
    ALLOWED_UPLOAD_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    LEGACY_OFFICE_EXTENSIONS,
    SUPPORTED_EXTENSION_SET,
    SUPPORTED_EXTENSIONS,
    SPREADSHEET_EXTENSIONS,
    TEXT_EXTENSIONS,
    TEXTISH_EXTENSIONS,
)


def test_supported_extensions_is_concatenation():
    assert set(SUPPORTED_EXTENSIONS) == (
        set(DOCUMENT_EXTENSIONS)
        | set(SPREADSHEET_EXTENSIONS)
        | set(TEXT_EXTENSIONS)
        | set(IMAGE_EXTENSIONS)
    )


def test_supported_extension_set_matches_list():
    assert SUPPORTED_EXTENSION_SET == set(SUPPORTED_EXTENSIONS)


def test_allowed_upload_extensions_have_dot_prefix():
    assert all(ext.startswith(".") for ext in ALLOWED_UPLOAD_EXTENSIONS)
    assert ".pdf" in ALLOWED_UPLOAD_EXTENSIONS
    assert ".csv" in ALLOWED_UPLOAD_EXTENSIONS


def test_legacy_office_extensions():
    assert LEGACY_OFFICE_EXTENSIONS == {"doc", "ppt", "xls"}


def test_textish_extensions_includes_code():
    assert {"sql", "py", "js", "ts", "tsx", "jsx"} <= TEXTISH_EXTENSIONS
