#!/usr/bin/env bash
# Initializes the third_party/llvm-project submodule with a sparse, partial
# checkout (llvm/ + mlir/ only, ~1.6GB) instead of the full monorepo tree
# (~2.7GB -- clang/lldb/libcxx/flang/etc. are dead weight we never build).
#
# `git submodule update --init` alone can't do this: `.gitmodules`' shallow
# config only limits history depth, and there's no `--sparse` passthrough
# for submodule update the way `git clone --sparse` has. So this clones
# directly with `--filter=blob:none --sparse` (skips fetching blobs outside
# the sparse-checkout patterns entirely, not just hiding them from the
# working tree) and then hands the result to git as the submodule's git dir.
#
# Usage: scripts/init_submodules.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUBMODULE_PATH="$REPO_ROOT/third_party/llvm-project"
URL=$(git -C "$REPO_ROOT" config -f .gitmodules submodule.third_party/llvm-project.url)
TAG=$(git -C "$REPO_ROOT" config -f .gitmodules submodule.third_party/llvm-project.branch)
PINNED_COMMIT=$(git -C "$REPO_ROOT" ls-tree HEAD third_party/llvm-project | awk '{print $3}')

if [ -d "$SUBMODULE_PATH/.git" ] || [ -f "$SUBMODULE_PATH/.git" ]; then
  echo "==> $SUBMODULE_PATH already initialized, skipping clone"
else
  rmdir "$SUBMODULE_PATH" 2>/dev/null || true
  git clone --filter=blob:none --depth 1 --branch "$TAG" --sparse "$URL" "$SUBMODULE_PATH"
  # llvm/CMakeLists.txt includes shared modules from the monorepo-root
  # cmake/ dir (e.g. cmake/Modules/CMakePolicy.cmake) and bundled
  # gtest/gbenchmark sources from third-party/ -- not just llvm/mlir.
  git -C "$SUBMODULE_PATH" sparse-checkout set llvm mlir cmake third-party
  git -C "$REPO_ROOT" submodule absorbgitdirs third_party/llvm-project
fi

ACTUAL_COMMIT=$(git -C "$SUBMODULE_PATH" rev-parse HEAD)
if [ -n "$PINNED_COMMIT" ] && [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then
  echo "warning: checked out $ACTUAL_COMMIT but the superproject pins $PINNED_COMMIT" >&2
  echo "         (run: git -C third_party/llvm-project fetch --depth 1 origin $PINNED_COMMIT && git -C third_party/llvm-project checkout $PINNED_COMMIT)" >&2
fi

echo "==> $SUBMODULE_PATH ready:"
du -sh "$SUBMODULE_PATH"
