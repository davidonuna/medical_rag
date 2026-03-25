#!/bin/bash
# Release script - creates a version tag

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh v1.0.0"
    exit 1
fi

VERSION=$1

echo "Creating release $VERSION..."

# Ensure we're on main
git checkout main
git pull origin main

# Create tag
git tag -a "$VERSION" -m "Release $VERSION"

# Push tag
git push origin "$VERSION"

echo "Release $VERSION created and pushed!"
