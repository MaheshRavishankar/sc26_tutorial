#!/usr/bin/env bash
# Runs inside the sc26-mlir-builder image (see Dockerfile). Builds MLIR from
# the third_party/llvm-project submodule (mounted read-only at /work/repo --
# run scripts/init_submodules.sh on the host first if it's not populated yet)
# with this tutorial's flags, installs, and packages the result via
# scripts/package_toolchain.sh. Not meant to be run outside the container --
# use scripts/docker_build_toolchain.sh on the host instead.
set -euo pipefail

SRC_DIR=/work/repo/third_party/llvm-project
BUILD_DIR=/work/build
INSTALL_DIR=/work/install
OUT_DIR=/out

if [ ! -d "$SRC_DIR/llvm" ]; then
  echo "error: $SRC_DIR/llvm not found -- run scripts/init_submodules.sh on the host first" >&2
  exit 1
fi

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  CACHED_HOME=$(grep '^CMAKE_HOME_DIRECTORY:' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2)
  if [ "$CACHED_HOME" != "$SRC_DIR/llvm" ]; then
    echo "==> cached build dir points at a different source dir ($CACHED_HOME) -- wiping it"
    rm -rf "$BUILD_DIR"
  fi
fi

echo "==> configuring"
cmake -S "$SRC_DIR/llvm" -B "$BUILD_DIR" -G Ninja \
  -U 'Python*' \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DPython3_EXECUTABLE="$(command -v python3.13)" \
  -DPython_EXECUTABLE="$(command -v python3.13)" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--export-dynamic"

echo "==> building + installing"
# Force a clean install tree each time -- `ninja install` only adds/overwrites
# files for the current build graph, so a stale $INSTALL_DIR can accumulate
# leftovers from a previous, differently-configured build (e.g. Python
# extension modules built against a Python version no longer in use).
rm -rf "$INSTALL_DIR"
ninja -C "$BUILD_DIR" install

echo "==> packaging"
mkdir -p "$OUT_DIR"
/work/repo/scripts/package_toolchain.sh "$INSTALL_DIR" "$OUT_DIR/mlir-tutorial-llvm.tar.gz"

echo "==> done: $OUT_DIR/mlir-tutorial-llvm.tar.gz"
