#!/usr/bin/env python3
"""
Script to create Docker Hub repositories for the project.
"""

import requests
import json
import os
import sys
import getpass

def create_repository(username: str, password: str, repo_name: str, description: str = "") -> bool:
    """Create a Docker Hub repository using the API."""
    
    # Get JWT token
    login_url = "https://hub.docker.com/v2/users/login/"
    login_data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(login_url, json=login_data)
        if response.status_code != 200:
            print(f"❌ Login failed: {response.text}")
            return False
        
        token = response.json()["token"]
        
        # Create repository
        create_url = f"https://hub.docker.com/v2/repositories/"
        headers = {
            "Authorization": f"JWT {token}",
            "Content-Type": "application/json"
        }
        
        repo_data = {
            "namespace": username,
            "name": repo_name,
            "description": description,
            "is_private": False,
            "full_description": f"Docker image for {repo_name}",
        }
        
        response = requests.post(create_url, json=repo_data, headers=headers)
        
        if response.status_code == 201:
            print(f"✅ Successfully created repository: {username}/{repo_name}")
            return True
        elif response.status_code == 400 and "already exists" in response.text.lower():
            print(f"ℹ️  Repository {username}/{repo_name} already exists")
            return True
        else:
            print(f"❌ Failed to create repository: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False

def main():
    """Main function."""
    print("🐳 Docker Hub Repository Creator")
    print("=" * 40)
    
    # Configuration
    username = "debabratamishra"  # Your Docker Hub username
    repositories = [
        ("litemindui-backend", "LiteMindUI Backend - FastAPI application with LLM integration"),
        ("litemindui-frontend", "LiteMindUI Frontend - Streamlit web interface")
    ]
    
    print(f"Username: {username}")
    print(f"Repositories to create: {len(repositories)}")
    
    # Get password/token
    if len(sys.argv) > 1:
        password = sys.argv[1]  # Can pass token as argument
    else:
        password = getpass.getpass("Enter your Docker Hub password or access token: ")
    
    if not password:
        print("❌ Password/token is required")
        sys.exit(1)
    
    # Create repositories
    success_count = 0
    for repo_name, description in repositories:
        print(f"\n📦 Creating repository: {repo_name}")
        if create_repository(username, password, repo_name, description):
            success_count += 1
    
    print(f"\n🎉 Summary: {success_count}/{len(repositories)} repositories created successfully!")
    
    if success_count == len(repositories):
        print("\n✅ All repositories are ready! You can now run your GitHub Actions workflow.")
    else:
        print("\n⚠️  Some repositories failed to create. You may need to create them manually:")
        print("   1. Go to https://hub.docker.com")
        print("   2. Click 'Create Repository'")
        print("   3. Create the missing repositories")

if __name__ == "__main__":
    main()
