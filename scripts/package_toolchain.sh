#!/usr/bin/env bash
# Package a from-source MLIR/LLVM install tree (built per the top-level
# README's cmake flags) into a trimmed, stripped tarball small enough for a
# GitHub Release asset.
#
# The full install tree is ~4.4GB, mostly LLVM's default tool set (llc, opt,
# bugpoint, llvm-mca, mlir-reduce, mlir-lsp-server, ...) that this tutorial
# never uses, plus unstripped binaries and ~550 unused static (.a) libraries
# (we deliberately don't link the plugin against those -- see
# session1/blas-dialect/CMakeLists.txt). None of that is needed to run
# mlir-opt, the Python bindings, or rebuild the blas-dialect plugin.
#
# Usage:
#   scripts/package_toolchain.sh <install-dir> <output.tar.gz>
#
# Re-run session1's smoke tests (see session1/blas-dialect/README or the
# top-level README) against the *extracted* output before publishing it --
# this script is deterministic, but "deterministic" isn't the same as
# "still works."

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <install-dir> <output.tar.gz>" >&2
  exit 1
fi

INSTALL_DIR=$(cd "$1" && pwd)
OUTPUT_TAR=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

# Binaries the notebook / blas-dialect build actually invoke. Nothing
# else in bin/ (llc, opt, bugpoint, llvm-mca, mlir-reduce, mlir-lsp-server,
# the mlir-capi-*-test binaries, Kaleidoscope examples, ...) is used.
KEEP_BIN=(
  mlir-opt
  mlir-tblgen
  llvm-tblgen
)

# Shared libs the notebook actually dlopen()s via ExecutionEngine's
# shared_libs=[...] (see session1_intro_to_mlir.ipynb). Everything else in
# lib/*.so (libLTO.so, libRemarks.so, the arm/amx runner utils, ...) is
# unused -- confirmed via `ldd mlir-opt`, which depends on none of these at
# all (they're only loaded on request).
KEEP_SO_PREFIXES=(
  libmlir_cuda_runtime.so
  libmlir_runner_utils.so
  libmlir_c_runner_utils.so
  libmlir_float16_utils.so   # runtime dep of the two runner-utils libs above
  libmlir_async_runtime.so   # session 2: scf.forall -> async -> CPU threads
)

echo "==> staging copy of $INSTALL_DIR"
rsync -a "$INSTALL_DIR"/ "$STAGE_DIR"/

echo "==> truncating bin/ to empty placeholders, except: ${KEEP_BIN[*]}"
echo "    (same reason as lib/*.a below: LLVMExports.cmake declares"
echo "    imported executable targets for every tool, and find_package()"
echo "    checks each referenced path exists on disk)"
pushd "$STAGE_DIR/bin" >/dev/null
for f in *; do
  keep=0
  for k in "${KEEP_BIN[@]}"; do
    [ "$f" = "$k" ] && keep=1 && break
  done
  if [ "$keep" -eq 0 ]; then
    : > "$f"
    chmod +x "$f"
  fi
done
popd >/dev/null

echo "==> truncating lib/*.a to empty placeholders (static libs; the plugin"
echo "    resolves symbols against the host mlir-opt process at dlopen"
echo "    time, not link time -- but LLVMConfig.cmake's exported targets"
echo "    do a file-existence check on every .a path at find_package()"
echo "    time, even for targets we never link, so the files must still"
echo "    exist on disk)"
find "$STAGE_DIR/lib" -maxdepth 1 -name '*.a' -exec sh -c '> "$1"' _ {} \;

echo "==> truncating lib/*.so to empty placeholders, except: ${KEEP_SO_PREFIXES[*]}"
echo "    (same find_package()-time existence check as bin/ and lib/*.a)"
pushd "$STAGE_DIR/lib" >/dev/null
for f in *.so*; do
  [ -f "$f" ] || continue
  keep=0
  for k in "${KEEP_SO_PREFIXES[@]}"; do
    case "$f" in
      "$k"*) keep=1 ;;
    esac
  done
  [ "$keep" -eq 0 ] && : > "$f"
done
popd >/dev/null

echo "==> stripping remaining binaries/libs (--strip-unneeded: keeps the"
echo "    dynamic symbol table plugins resolve against, drops the rest)"
find "$STAGE_DIR/bin" "$STAGE_DIR/lib" -maxdepth 1 -type f \( -executable -o -name '*.so*' \) \
  -exec strip --strip-unneeded {} + 2>/dev/null || true

echo "==> before/after size"
du -sh "$INSTALL_DIR"
du -sh "$STAGE_DIR"

echo "==> writing $OUTPUT_TAR"
tar czf "$OUTPUT_TAR" -C "$(dirname "$STAGE_DIR")" "$(basename "$STAGE_DIR")" \
  --transform "s/^$(basename "$STAGE_DIR")/mlir-tutorial-llvm/"

ls -lh "$OUTPUT_TAR"
