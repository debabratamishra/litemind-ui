"""
Version management script for LiteMindUI
Handles semantic versioning and release preparation
"""

import json
import os
import sys
import argparse
from pathlib import Path

class VersionManager:
    def __init__(self, version_file="version.json"):
        self.version_file = Path(version_file)
        self.version_data = self.load_version()
    
    def load_version(self):
        """Load current version from file or initialize with default"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize with version 0.0.1
            return {
                "version": "0.0.1",
                "major": 0,
                "minor": 0,
                "patch": 1,
                "build_date": "",
                "git_commit": ""
            }
    
    def save_version(self):
        """Save current version to file"""
        with open(self.version_file, 'w') as f:
            json.dump(self.version_data, f, indent=2)
        print(f"Version saved to {self.version_file}")
    
    def get_current_version(self):
        """Get current version string"""
        return self.version_data["version"]
    
    def bump_version(self, bump_type="patch"):
        """Bump version according to semantic versioning"""
        if bump_type == "major":
            self.version_data["major"] += 1
            self.version_data["minor"] = 0
            self.version_data["patch"] = 0
        elif bump_type == "minor":
            self.version_data["minor"] += 1
            self.version_data["patch"] = 0
        elif bump_type == "patch":
            self.version_data["patch"] += 1
        else:
            raise ValueError("Invalid bump type. Use 'major', 'minor', or 'patch'")
        
        # Update version string
        self.version_data["version"] = f"{self.version_data['major']}.{self.version_data['minor']}.{self.version_data['patch']}"
        
        # Update metadata
        import datetime
        import subprocess
        
        self.version_data["build_date"] = datetime.datetime.now().isoformat()
        
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            self.version_data["git_commit"] = git_commit
        except:
            self.version_data["git_commit"] = "unknown"
        
        return self.version_data["version"]
    
    def create_tag(self, version=None):
        """Create a git tag for the current version"""
        if version is None:
            version = self.get_current_version()
        
        import subprocess
        
        try:
            # Create annotated tag
            subprocess.run([
                "git", "tag", "-a", f"v{version}", 
                "-m", f"Release version {version}"
            ], check=True)
            print(f"Created git tag: v{version}")
            
            # Push tag to origin
            response = input("Push tag to origin? (y/N): ")
            if response.lower() in ['y', 'yes']:
                subprocess.run(["git", "push", "origin", f"v{version}"], check=True)
                print(f"Pushed tag v{version} to origin")
        except subprocess.CalledProcessError as e:
            print(f"Error creating/pushing tag: {e}")
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Manage LiteMindUI versions")
    parser.add_argument("action", choices=["current", "bump", "tag", "init"], 
                       help="Action to perform")
    parser.add_argument("--type", choices=["major", "minor", "patch"], 
                       default="patch", help="Version bump type")
    parser.add_argument("--version-file", default="version.json", 
                       help="Version file path")
    
    args = parser.parse_args()
    
    vm = VersionManager(args.version_file)
    
    if args.action == "current":
        print(f"Current version: {vm.get_current_version()}")
        
    elif args.action == "bump":
        new_version = vm.bump_version(args.type)
        vm.save_version()
        print(f"Bumped version to: {new_version}")
        
        # Ask if user wants to create a tag
        response = input("Create git tag for this version? (y/N): ")
        if response.lower() in ['y', 'yes']:
            vm.create_tag(new_version)
            
    elif args.action == "tag":
        version = vm.get_current_version()
        if vm.create_tag(version):
            print(f"Successfully tagged version {version}")
        else:
            print("Failed to create tag")
            
    elif args.action == "init":
        vm.save_version()
        print(f"Initialized version file with version {vm.get_current_version()}")

if __name__ == "__main__":
    main()
