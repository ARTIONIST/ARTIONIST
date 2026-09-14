#!/usr/bin/env bash
# Usage: ./push_to_github.sh https://github.com/yourname/mmi-psai-x1.git
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <your-github-repo-url>"
  echo "Example: $0 https://github.com/yourname/mmi-psai-x1.git"
  exit 1
fi

REPO_URL="$1"

git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git branch -M main
git push -u origin main

echo "Done. Your code is now on GitHub at: $REPO_URL"
