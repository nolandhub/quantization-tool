#!/usr/bin/env bash
# ABOUTME: rsync the qtool repo between this machine and rtx4070:~/qtool (GPU box where images and tests run).
# ABOUTME: Usage: sync.sh push|pull|check   (check = dry-run push listing what would change)
set -euo pipefail
cd "$(dirname "$0")"
REMOTE=${QTOOL_REMOTE:-rtx4070:qtool/}
EXCL=(--exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude /runs --exclude /.lock)
case "${1:-}" in
  push) rsync -a --delete "${EXCL[@]}" ./ "$REMOTE" ;;
  pull) rsync -a "${EXCL[@]}" --exclude .git "$REMOTE" ./ ;;
  check) rsync -a --delete --dry-run --itemize-changes "${EXCL[@]}" ./ "$REMOTE" ;;
  *) echo "usage: $0 push|pull|check" >&2; exit 2 ;;
esac
