#!/usr/bin/env bash
# Builds the tutorial's MLIR/LLVM toolchain inside a Docker container based
# on ubuntu:22.04, matching Colab's runtime glibc/libstdc++ -- avoids the
# "version `GLIBC_2.3x' not found" failures a build on a newer host OS
# produces when run on Colab. See docker/Dockerfile and docker/build_toolchain.sh.
#
# Usage:
#   scripts/docker_build_toolchain.sh [output-dir]
#
# Output: <output-dir>/mlir-tutorial-llvm.tar.gz (defaults to repo root).
# Re-running reuses a persistent Docker volume for the build tree, so
# iterating on cmake flags doesn't rebuild from scratch. LLVM source itself
# comes from the third_party/llvm-project submodule (see
# scripts/init_submodules.sh, called below), not a build-time clone.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=$(mkdir -p "${1:-$REPO_ROOT}" && cd "${1:-$REPO_ROOT}" && pwd)
IMAGE=sc26-mlir-builder

"$REPO_ROOT/scripts/init_submodules.sh"

echo "==> building builder image"
docker build -t "$IMAGE" "$REPO_ROOT/docker"

echo "==> running build (this rebuilds LLVM+MLIR from scratch the first time -- expect a long build)"
docker run --rm \
  -v "$REPO_ROOT:/work/repo:ro" \
  -v "$OUT_DIR:/out" \
  -v sc26-mlir-build-cache:/work \
  "$IMAGE"

echo "==> toolchain tarball: $OUT_DIR/mlir-tutorial-llvm.tar.gz"
