"""Repository layout guards.

These tests keep the moved layout (``backend/``, ``infra/docker/``, ``docs/site/``)
the only one referenced by hand-maintained files. They read *tracked* files only,
so gitignored scratch and untracked local files cannot perturb them, and they carry
a self-check so a pattern that is inert cannot pass unnoticed.
"""

import re
import subprocess
from pathlib import Path

# Hand-maintained text files that reference repository paths and must not point at
# the pre-refactor layout. Build output and vendored dependencies are skipped because
# they are not hand-maintained. ``.example``/``.nextjs`` cover ``.env.example`` and
# ``Dockerfile.nextjs``.
_TEXT_SUFFIXES = {".py", ".sh", ".md", ".yml", ".yaml", ".toml", ".ts", ".tsx", ".example", ".nextjs"}
# Known extensionless files that carry repository paths but no text suffix.
_TEXT_FILENAMES = {"Makefile", "makefile", "Dockerfile", "Procfile"}
_SKIP_DIRS = {"node_modules", ".venv", ".git", ".next", "dist", ".astro", "__pycache__", ".ruff_cache"}
# Historical plan/spec records describe the tree as it stood when they were written;
# rewriting them would falsify the record. Live docs are not skipped.
_SKIP_PREFIXES = {Path("docs/superpowers"), Path(".superpowers")}
_SELF = Path(__file__).resolve()

# Each pattern matches only a *root-relative* reference to a moved path. The negative
# lookbehinds keep the canonical forms — ``backend/app/...``, ``backend.main:app``,
# ``docs/site/`` — from matching, so the test cannot pass by accident on a correct file.
_STALE_PATTERNS = {
    "root Dockerfile": r"file: \./Dockerfile",
    "compose build context": r"context: \.\./\.\./\.\.",
    "compose dockerfile path": r"dockerfile: \.\./Dockerfile",
    "bare app/backend path": r"(?<!backend/)app/backend/",
    "bare app/services path": r"(?<!backend/)app/services/",
    "bare app/skills path": r"(?<!backend/)app/skills/",
    "bare app/ingestion path": r"(?<!backend/)app/ingestion/",
    "bare app/core path": r"(?<!backend/)app/core/",
    "bare uvicorn app target": r"(?<!backend\.)main:app",
    "retired frontend port": r"localhost:8501",
    # Backticked path references only: a bare `site/` inside a Markdown link target
    # is a correct *relative* link from docs/, so matching it would false-fire.
    "root site/ directory": r"(?<!docs/)`site/`",
}

# Per pattern: a stale sample it must match, and a canonical sample it must not.
# Without this, a pattern list whose regexes all match nothing looks identical to a
# working guard. Keep the samples in sync with ``_STALE_PATTERNS`` (asserted below).
_STALE_SAMPLES = {
    "root Dockerfile": ("      file: ./Dockerfile", "      file: infra/docker/Dockerfile"),
    "compose build context": ("      context: ../../..", "      context: ."),
    "compose dockerfile path": ("      dockerfile: ../Dockerfile", "      dockerfile: infra/docker/Dockerfile"),
    "bare app/backend path": ("see app/backend/api/rag.py", "see backend/app/backend/api/rag.py"),
    "bare app/services path": ("see app/services/rag_service.py", "see backend/app/services/rag_service.py"),
    "bare app/skills path": ("see app/skills/registry.py", "see backend/app/skills/registry.py"),
    "bare app/ingestion path": ("see app/ingestion/pipeline.py", "see backend/app/ingestion/pipeline.py"),
    "bare app/core path": ("see app/core/rag_formats.py", "see backend/app/core/rag_formats.py"),
    "bare uvicorn app target": ("uv run uvicorn main:app --reload", "uv run uvicorn backend.main:app --reload"),
    "retired frontend port": ("http://localhost:8501", "http://localhost:3000"),
    "root site/ directory": ("the site lives under [`site/`](site/)", "the site lives under [`docs/site/`](site/)"),
}


def _tracked_text_files():
    """Yield the tracked files that are hand-maintained enough to guard.

    ``git ls-files`` is the source of the list: walking the working tree would also
    read gitignored scratch (the ``.superpowers`` ledger quotes old paths as its
    historical record) and untracked local files, so the guard would be neither
    correct nor hermetic.
    """
    listing = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
    for name in listing.split("\0"):
        if not name:
            continue
        path = Path(name)
        if path.suffix not in _TEXT_SUFFIXES and path.name not in _TEXT_FILENAMES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if any(prefix in path.parents for prefix in _SKIP_PREFIXES):
            continue
        if path.resolve() == _SELF:
            continue
        yield path


def test_stale_patterns_match_stale_text_and_ignore_canonical_text():
    assert set(_STALE_SAMPLES) == set(_STALE_PATTERNS)
    for label, pattern in _STALE_PATTERNS.items():
        stale, canonical = _STALE_SAMPLES[label]
        assert re.search(pattern, stale), f"pattern for {label!r} does not match stale text {stale!r}"
        assert not re.search(pattern, canonical), f"pattern for {label!r} matches canonical text {canonical!r}"


def test_no_stale_root_docker_references():
    text = Path(".github/workflows/docker-publish.yml").read_text()
    assert "infra/docker/Dockerfile" in text
    assert "file: ./Dockerfile" not in text


def test_no_stale_path_references_anywhere():
    offenders: list[str] = []
    for path in _tracked_text_files():
        text = path.read_text(errors="ignore")
        for label, pattern in _STALE_PATTERNS.items():
            if re.search(pattern, text):
                offenders.append(f"{path}: {label}")

    assert not offenders, "stale references found:\n" + "\n".join(offenders)


def test_canonical_layout_references_exist():
    root = Path(".")
    assert (root / "backend" / "main.py").exists()
    assert (root / "infra" / "docker" / "Dockerfile").exists()
    assert (root / "infra" / "docker" / "compose" / "docker-compose.yml").exists()
    assert (root / "docs" / "README.md").exists()
    assert (root / "docs" / "docker" / "README.md").exists()


def test_astro_site_is_under_docs_site():
    root = Path(".")
    assert (root / "docs" / "site" / "package.json").exists()
    assert not (root / "site").exists()
