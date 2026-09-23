"""Shared RAG upload and extraction format definitions."""

DOCUMENT_EXTENSIONS = (
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "rtf",
    "odt",
    "epub",
    "html",
    "htm",
    "xml",
)

SPREADSHEET_EXTENSIONS = (
    "xls",
    "xlsx",
    "csv",
    "tsv",
)

TEXT_EXTENSIONS = (
    "txt",
    "md",
    "rst",
    "org",
    "json",
    "jsonl",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "log",
)

IMAGE_EXTENSIONS = (
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tiff",
    "webp",
    "gif",
    "heic",
    "svg",
)

SUPPORTED_EXTENSIONS = list(
    DOCUMENT_EXTENSIONS
    + SPREADSHEET_EXTENSIONS
    + TEXT_EXTENSIONS
    + IMAGE_EXTENSIONS
)

SUPPORTED_EXTENSION_SET = set(SUPPORTED_EXTENSIONS)
ALLOWED_UPLOAD_EXTENSIONS = {f".{ext}" for ext in SUPPORTED_EXTENSIONS}

LEGACY_OFFICE_EXTENSIONS = {"doc", "ppt", "xls"}
TEXTISH_EXTENSIONS = set(TEXT_EXTENSIONS) | {"sql", "py", "js", "ts", "tsx", "jsx"}
