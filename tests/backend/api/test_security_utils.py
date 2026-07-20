"""Unit tests for ``app.backend.api.security_utils`` (offline).

Covers filename sanitisation (path-traversal / extension / character rules)
and streaming file-size validation (HTTP 413 when over the limit).
"""
import io

import pytest
from fastapi import HTTPException, UploadFile

from app.backend.api import security_utils as su


# ── sanitize_filename ──────────────────────────────────────────────
def test_sanitize_filename_preserves_case():
    # Only the extension is lower-cased for the *allowlist check*; the
    # returned filename keeps its original case.
    assert su.sanitize_filename("report.PDF") == "report.PDF"


def test_sanitize_filename_basic_allowed():
    assert su.sanitize_filename("notes.txt") == "notes.txt"


def test_sanitize_filename_strips_path_traversal():
    # os.path.basename removes directory components.
    result = su.sanitize_filename("../../etc/passwd.txt")
    assert ".." not in result
    assert result == "passwd.txt"

    result = su.sanitize_filename("dir/sub/file.pdf")
    assert result == "file.pdf"


def test_sanitize_filename_rejects_disallowed_extension():
    with pytest.raises(ValueError):
        su.sanitize_filename("evil.exe")
    with pytest.raises(ValueError):
        su.sanitize_filename("script.sh")


def test_sanitize_filename_rejects_empty():
    with pytest.raises(ValueError):
        su.sanitize_filename("")
    with pytest.raises(ValueError):
        su.sanitize_filename(None)


def test_sanitize_filename_rejects_all_dot_names():
    # "..." -> lstrip(". ") -> "" -> raises.
    with pytest.raises(ValueError):
        su.sanitize_filename("...")


def test_sanitize_filename_rejects_invalid_characters():
    with pytest.raises(ValueError):
        su.sanitize_filename("my file.txt")  # space not allowed
    with pytest.raises(ValueError):
        su.sanitize_filename("a:b.txt")  # colon not allowed


def test_sanitize_filename_truncates_long_name():
    long_name = "a" * 250 + ".txt"
    result = su.sanitize_filename(long_name)
    name, _ext = __import__("os").path.splitext(result)
    assert len(name) <= 200
    assert result.endswith(".txt")


# ── validate_file_size ─────────────────────────────────────────────
async def test_validate_file_size_ok():
    data = b"small payload"
    uf = UploadFile(filename="x.txt", file=io.BytesIO(data))
    await su.validate_file_size(uf)  # must not raise
    # Stream is rewound to the start for downstream consumers.
    assert uf.file.tell() == 0


async def test_validate_file_size_too_large():
    big = b"x" * (su.MAX_FILE_SIZE + 1)
    uf = UploadFile(filename="x.txt", file=io.BytesIO(big))
    with pytest.raises(HTTPException) as excinfo:
        await su.validate_file_size(uf)
    assert excinfo.value.status_code == 413


async def test_validate_file_size_boundary_allowed():
    # Exactly at the limit is allowed (comparison is strict >).
    data = b"x" * su.MAX_FILE_SIZE
    uf = UploadFile(filename="x.txt", file=io.BytesIO(data))
    await su.validate_file_size(uf)  # must not raise


def test_constants_exposed():
    assert su.MAX_FILE_SIZE == 100 * 1024 * 1024
    assert isinstance(su.ALLOWED_EXTENSIONS, set)
    assert ".pdf" in su.ALLOWED_EXTENSIONS
