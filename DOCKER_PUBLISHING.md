# Docker Publishing Guide

This guide explains how to set up and troubleshoot Docker Hub publishing for the LiteMindUI project.

## 🚨 Quick Fix for Publishing Issues

If you're getting the error:
```
push access denied, repository does not exist or may require authorization
```

**The solution is simple: Create the repositories on Docker Hub first!**

### Option 1: Manual Creation (Recommended)
1. Go to [Docker Hub](https://hub.docker.com)
2. Log in to your account
3. Click "Create Repository"
4. Create these repositories as **Public**:
   - `litemindui-backend`
   - `litemindui-frontend`

### Option 2: Use Our Creation Script
```bash
make create-docker-repos
# OR
python3 scripts/create-docker-repos.py
```

## 🧪 Local Testing

Before pushing changes that trigger CI/CD, test locally:

```bash
# Test everything (requires repositories to exist)
make test-docker-local

# Test builds only (no push, no repository check)
python3 scripts/test-docker-publish.py --skip-check --skip-push

# Test with repository check but no push
python3 scripts/test-docker-publish.py --skip-push
```

## 📋 Prerequisites Checklist

### Docker Hub Account
- [ ] Have a Docker Hub account
- [ ] Username: `debabratamishra` (or update in scripts)

### GitHub Repository Secrets
Make sure these secrets are set in your GitHub repository:
- [ ] `DOCKER_USERNAME`: Your Docker Hub username
- [ ] `DOCKER_PASSWORD`: Your Docker Hub access token (**not** your password)

#### How to Create Docker Hub Access Token:
1. Go to [Docker Hub Security Settings](https://hub.docker.com/settings/security)
2. Click "New Access Token"
3. Name it (e.g., "GitHub Actions")
4. Set permissions: **Read, Write, Delete**
5. Copy the token and add it as `DOCKER_PASSWORD` secret in GitHub

### Docker Hub Repositories
- [ ] Repository `debabratamishra/litemindui-backend` exists and is public
- [ ] Repository `debabratamishra/litemindui-frontend` exists and is public

### Local Docker Setup
- [ ] Docker is installed and running
- [ ] You can run `docker login` successfully
- [ ] You have push access to your Docker Hub repositories

## 🔧 Troubleshooting

### Error: "repository does not exist"
**Cause:** The Docker Hub repository hasn't been created.
**Solution:** Create the repositories manually or use our script.

### Error: "insufficient_scope: authorization failed"
**Causes:**
1. Wrong Docker Hub credentials in GitHub secrets
2. Access token has insufficient permissions
3. Repository is private and you don't have access

**Solutions:**
1. Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` in GitHub repository secrets
2. Create a new access token with Read, Write, Delete permissions
3. Ensure repositories are public

### Error: "authentication required"
**Cause:** Not logged into Docker Hub.
**Solution:** Run `docker login docker.io`

### Error: Build works locally but fails in CI
**Causes:**
1. Different environment variables
2. Different base images or cache state
3. Repository doesn't exist (most common)

**Solution:** 
1. Use the test script: `make test-docker-local`
2. Check GitHub Actions logs for specific errors
3. Verify all prerequisites above

## 📊 Workflow Explanation

The GitHub Actions workflow (`docker-publish.yml`) does:

1. **Triggers**: On push to `main`/`develop` branches or version tags
2. **Authentication**: Logs into Docker Hub using secrets
3. **Repository Check**: Verifies repositories exist before building
4. **Build**: Creates Docker images for backend and frontend
5. **Push**: Publishes images to Docker Hub
6. **Release**: Creates release artifacts for version tags

## 🚀 Publishing Process

### Automatic Publishing
1. Ensure repositories exist on Docker Hub
2. Verify GitHub secrets are set correctly
3. Push to `main` branch → triggers automatic publishing
4. Create version tag → triggers versioned release

### Manual Publishing
```bash
# Build and test locally first
make test-docker-local

# If successful, push changes
git add .
git commit -m "Your changes"
git push origin main
```

### Creating Releases
```bash
# Bump version and create tag
make tag-release

# This will:
# 1. Increment patch version
# 2. Create git tag
# 3. Push tag (triggers release workflow)
```

## 📁 Repository Structure

```
scripts/
├── test-docker-publish.py     # Local testing script
├── create-docker-repos.py     # Repository creation script
└── ...

.github/workflows/
└── docker-publish.yml         # CI/CD workflow

docker-compose.*.yml           # Various deployment configs
Dockerfile                     # Backend image
Dockerfile.streamlit          # Frontend image
```

## 🎯 Best Practices

1. **Always test locally first** using `make test-docker-local`
2. **Create repositories before first push** to avoid CI failures
3. **Use semantic versioning** for releases
4. **Monitor GitHub Actions** for any issues
5. **Keep access tokens secure** and rotate them periodically

## 🆘 Getting Help

If you're still having issues:

1. **Run the test script**: `make test-docker-local`
2. **Check all prerequisites** in this guide
3. **Look at recent GitHub Actions logs** for specific error messages
4. **Verify Docker Hub repository settings** (public, correct names)
5. **Test Docker login locally**: `docker login docker.io`

The most common issue is simply that Docker Hub repositories don't exist - create them first!
