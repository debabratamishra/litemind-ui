"""Shared security helpers for file uploads."""
import os
import re
from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".csv", ".md", ".json", ".xml", ".html"}
_CHUNK_SIZE = 1024 * 1024  # 1MB


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to prevent path traversal and reject disallowed extensions."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    filename = os.path.basename(filename)
    filename = filename.replace("/", "").replace("\\", "")
    filename = filename.replace("\0", "")
    filename = filename.lstrip(". ")

    if not filename or filename in (".", ".."):
        raise ValueError("Invalid filename")

    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    filename = name + ext

    dangerous_patterns = [r"\.\./|\\\.\.\\", r"^\.+$", r"\0"]
    for pattern in dangerous_patterns:
        if re.search(pattern, filename):
            raise ValueError(f"Filename contains dangerous pattern: {filename}")

    if ext.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"File extension '{ext}' not allowed. Allowed: {allowed}")

    return filename


def _raise_too_large() -> None:
    raise HTTPException(
        status_code=413,
        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
    )


async def validate_file_size(file: UploadFile) -> None:
    """Ensure file size is within limit without loading full content into memory."""
    try:
        size = await file.seek(0, os.SEEK_END)
    except Exception:
        size = 0
        await file.seek(0)
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                await file.seek(0)
                _raise_too_large()
    finally:
        await file.seek(0)

    if size > MAX_FILE_SIZE:
        _raise_too_large()
