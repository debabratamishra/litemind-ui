"""
Release preparation script for LiteMindUI
Prepares the repository for a new release by updating version files and creating release notes
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_log_since_last_tag():
    """Get git commits since last tag for changelog generation"""
    try:
        # Get the last tag
        last_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Get commits since last tag
        commits = subprocess.check_output([
            "git", "log", f"{last_tag}..HEAD",
            "--pretty=format:- %s (%h)"
        ]).decode().strip()

        return last_tag, commits
    except subprocess.CalledProcessError:
        # No tags found, get all commits
        commits = subprocess.check_output([
            "git", "log", "--pretty=format:- %s (%h)", "--max-count=10"
        ]).decode().strip()

        return None, commits

def create_release_notes(version, last_tag, commits):
    """Generate release notes content"""
    date_str = datetime.now().strftime("%Y-%m-%d")

    release_notes = f"""# Release {version} ({date_str})

## 🚀 What's New

This release includes the following changes:

## 📋 Changes

{commits if commits else "- Initial release"}

## 🐳 Docker Images

- **Backend**: `debabratamishra/litemindui-backend:{version}`
- **Frontend**: `debabratamishra/litemindui-frontend:{version}`

## 📦 Quick Start

### One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/debabratamishra/litemindui/main/install.sh | bash
```

### Docker Compose
```bash
# Download release-specific compose file
curl -O https://github.com/debabratamishra/litemindui/releases/download/v{version}/docker-compose.release.yml

# Start services
docker-compose -f docker-compose.release.yml up -d
```

## 🌐 Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🔧 Prerequisites

- Docker and Docker Compose
- Ollama running locally (optional, for local LLM support)

---

**Full Changelog**: https://github.com/debabratamishra/litemindui/compare/{last_tag if last_tag else "initial"}...v{version}
"""

    return release_notes

def update_install_script_version(version):
    """Update version references in install script"""
    install_script = Path("install.sh")
    if install_script.exists():
        install_script.read_text()
        # Could add version-specific updates here if needed
        print(f"✅ Install script checked for version {version}")
    else:
        print("⚠️  Install script not found")

def main():
    print("🚀 Preparing release for LiteMindUI")

    # Load current version
    version_file = Path("version.json")
    if not version_file.exists():
        print("❌ Version file not found. Run 'python3 scripts/version.py init' first")
        sys.exit(1)

    with open(version_file) as f:
        version_data = json.load(f)

    current_version = version_data["version"]

    print(f"📋 Current version: {current_version}")

    # Get git history
    last_tag, commits = get_git_log_since_last_tag()
    print(f"📝 Found {'changes since ' + last_tag if last_tag else 'commits for initial release'}")

    # Create release notes
    release_notes = create_release_notes(current_version, last_tag, commits)

    # Write release notes to file
    release_notes_file = Path(f"RELEASE-{current_version}.md")
    release_notes_file.write_text(release_notes)
    print(f"📄 Release notes written to {release_notes_file}")

    # Update other files if needed
    update_install_script_version(current_version)

    print("\n🎯 Release preparation complete!")
    print(f"📋 Version: {current_version}")
    print(f"📄 Release notes: {release_notes_file}")

    print("\n📤 Next steps:")
    print("1. Review the release notes")
    print("2. Commit any final changes")
    print(f"3. Create and push the tag: git tag -a v{current_version} -m 'Release {current_version}' && git push origin v{current_version}")
    print("4. The GitHub Actions workflow will automatically build and push Docker images")
    print("5. Create a GitHub release using the generated release notes")

if __name__ == "__main__":
    main()
