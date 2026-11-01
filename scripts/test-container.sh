#!/usr/bin/env bash
# ABOUTME: Runs the container-tier tests (tests/container) inside the qtool-compress image on CPU.
# ABOUTME: Builds the image first if missing; usage: scripts/test-container.sh [test paths, default tests/container]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$(python3 -c 'from qtool import images; print(images.tag("compress", "cu129"))')
docker image inspect "$TAG" >/dev/null 2>&1 || docker build -t "$TAG" --build-arg CUDA=cu129 images/compress
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v "$PWD:/src:ro" -w /src \
  -e PYTHONDONTWRITEBYTECODE=1 "$TAG" python -m pytest -q -m container -p no:cacheprovider "${@:-tests/container}"
