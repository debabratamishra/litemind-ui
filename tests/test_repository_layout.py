import re
from pathlib import Path

# Hand-maintained text files that reference repository paths and must not point
# at the pre-refactor layout. Build output and vendored dependencies are skipped
# because they are not hand-maintained.
_TEXT_SUFFIXES = {".py", ".sh", ".md", ".yml", ".yaml", ".toml", ".ts", ".tsx"}
_SKIP_DIRS = {"node_modules", ".venv", ".git", ".next", "dist", ".astro", "__pycache__", ".ruff_cache"}
# Historical plan/spec records describe the tree as it stood when they were
# written; rewriting them would falsify the record. Live docs are not skipped.
_SKIP_PREFIXES = {Path("docs/superpowers")}
_SELF = Path(__file__).resolve()

# Each pattern matches only a *root-relative* reference to a moved path. The
# negative lookbehind keeps ``backend/app/...`` — the canonical form — from
# matching, so the test cannot pass by accident on a correct file.
_STALE_PATTERNS = {
    "root Dockerfile": r"file: \./Dockerfile",
    "bare app/backend path": r"(?<!backend/)app/backend/",
    "bare app/services path": r"(?<!backend/)app/services/",
    "bare app/skills path": r"(?<!backend/)app/skills/",
    "bare app/ingestion path": r"(?<!backend/)app/ingestion/",
    "bare app/core path": r"(?<!backend/)app/core/",
    "retired frontend port": r"localhost:8501",
    "site/ directory": r"(?<!docs/)site/package\.json",
}


def _repo_text_files():
    for path in Path(".").rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if any(prefix in _SKIP_PREFIXES for prefix in path.parents):
            continue
        if path.resolve() == _SELF:
            continue
        yield path


def test_no_stale_root_docker_references():
    text = Path(".github/workflows/docker-publish.yml").read_text()
    assert "infra/docker/Dockerfile" in text
    assert "file: ./Dockerfile" not in text


def test_no_stale_path_references_anywhere():
    offenders: list[str] = []
    for path in _repo_text_files():
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
