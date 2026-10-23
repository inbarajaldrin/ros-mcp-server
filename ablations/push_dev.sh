#!/usr/bin/env bash
# Push the cell's branch (dev) to GitHub (operator 10-23: push to dev, never main). This machine has no GitHub key:
# push to the backup repo on dual-a4500 (remote 'origin'), then from there to GitHub (inbarajaldrin/ros-mcp-server).
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(git branch --show-current)" = dev ] || { echo "not on dev - refusing"; exit 1; }
git push origin dev:refs/heads/dev --tags
ssh -o BatchMode=yes aaugus11@100.80.151.46 \
  'cd ~/a4500-backup-20260828/Documents/ros-mcp-server && git push origin dev:refs/heads/dev && git push origin --tags'
git fetch -q github dev && echo "GitHub dev = $(git rev-parse --short github/dev), local = $(git rev-parse --short HEAD)"
