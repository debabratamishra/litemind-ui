from pathlib import Path


def test_canonical_layout_references_exist():
    root = Path(".")
    assert (root / "backend" / "main.py").exists()
    assert (root / "infra" / "docker" / "Dockerfile").exists()
    assert (root / "infra" / "docker" / "compose" / "docker-compose.yml").exists()
    assert (root / "docs" / "README.md").exists()
    assert (root / "docs" / "docker" / "README.md").exists()
