# SC26 MLIR Tutorial

A hands-on introduction to MLIR, designed to run entirely on [Google Colab](https://colab.research.google.com)
(free tier, including the free T4/L4 GPU runtime) so attendees need nothing
installed locally.

## Open the notebooks in Colab

| Part | Notebook |
| - | - |
| Session 1, Part 1 (CPU) | [`session1_part1_cpu.ipynb`](https://colab.research.google.com/github/MaheshRavishankar/sc26_tutorial/blob/main/session1/session1_part1_cpu.ipynb) |
| Session 1, Part 2 (GPU) | [`session1_part2_gpu.ipynb`](https://colab.research.google.com/github/MaheshRavishankar/sc26_tutorial/blob/main/session1/session1_part2_gpu.ipynb) |
| Session 2, Part 1 (CPU) | [`session2_part1_cpu.ipynb`](https://colab.research.google.com/github/MaheshRavishankar/sc26_tutorial/blob/main/session2/session2_part1_cpu.ipynb) |
| Session 2, Part 2 (GPU) | [`session2_part2_gpu.ipynb`](https://colab.research.google.com/github/MaheshRavishankar/sc26_tutorial/blob/main/session2/session2_part2_gpu.ipynb) |

For the two GPU parts, switch the runtime to a T4 GPU (Runtime → Change
runtime type) before running any cell.

## Sessions

- **Session 1 — Introduction to MLIR** (`session1/`): attendees build up a
  matrix-matrix product, from a hand-written SCF-loop implementation through
  GPU codegen to a custom `blas` dialect with its own lowering passes. Each
  piece they write has a complete matrix-*vector* version beside it as a
  reference.
- **Session 2 — Linalg and tiling** (`session2/`): the same product one level
  up, as Linalg ops. Part 1 (CPU) lowers them to loops, then tiles them with
  a transform dialect schedule written against the Linalg op interface, which
  attendees extend, and measures matmul single-threaded. Part 2 (GPU)
  distributes the ops across a GPU with a schedule attendees modify, as the
  blocks and threads of a `gpu.launch`.

## How this is put together

Each session's Colab notebook downloads a single prebuilt MLIR toolchain
(`mlir-opt`, `mlir-tblgen`, the MLIR Python bindings, and the CUDA runtime
support library) from this repo's
[Releases](../../releases) page, built from LLVM/MLIR source with:

- `-Wl,--export-dynamic` on every tool (broad dynamic symbol export) — needed
  for a custom MLIR dialect built as an out-of-tree plugin (subclassing
  `mlir::Dialect`) to `dlopen` successfully into `mlir-opt`. The official
  LLVM release binaries already do something equivalent (very broad export),
  but LLVM's own curated `LLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON` mechanism is
  *not* broad enough on its own — see `session1/blas-dialect/CMakeLists.txt`
  for the other half of this fix (`-fno-rtti` on the plugin itself, to match
  how MLIR/LLVM are built — `mlir::Dialect` has no C++ RTTI typeinfo compiled
  in anywhere, since MLIR uses its own `TypeID` system instead).
- `MLIR_ENABLE_BINDINGS_PYTHON=ON` and `MLIR_ENABLE_CUDA_RUNNER=ON` — so the
  same toolchain drives both the interactive Python/numpy steps and the GPU
  steps, with no separate pip install needed (the pip-installable MLIR
  wheels have no CUDA-enabled variant, and no C++ headers/CMake config for
  building a plugin against).

### Building the toolchain (`docker/`)

The toolchain is built inside a Docker container based on `ubuntu:22.04`
with Python 3.13 — matching Colab's runtime OS/glibc *and* Python version —
via `scripts/docker_build_toolchain.sh`. Both matter independently: shared
libraries are backward- but not forward-compatible, so a build done on a
newer host OS (newer glibc/libstdc++) fails on Colab with errors like
`version 'GLIBC_2.38' not found`; separately, MLIR's Python bindings are
compiled native extensions tied to a specific CPython ABI, so building them
against a different Python minor version than Colab actually runs produces
extensions that fail to import there (Ubuntu 22.04 ships Python 3.10 by
default, so the Dockerfile pulls 3.13 from the deadsnakes PPA instead).

LLVM source itself comes from the `third_party/llvm-project` submodule
(pinned to `llvmorg-22.1.8`), sparse-checked-out to just `llvm/`, `mlir/`,
`cmake/`, and `third-party/` (~1.7GB instead of ~2.7GB for the full
monorepo — `clang/`, `lldb/`, `libcxx/`, etc. aren't needed). Initialize it
with:

```
scripts/init_submodules.sh
```

(A plain `git submodule update --init` also works but pulls the full
monorepo tree — the sparse-checkout part isn't something `.gitmodules` can
express on its own, hence the wrapper script.)

Then:

```
scripts/docker_build_toolchain.sh [output-dir]
```

This builds `docker/Dockerfile` (an `ubuntu:22.04` image with the CUDA
toolkit, Python 3.13, and MLIR's Python-binding build deps) and runs
`docker/build_toolchain.sh` inside it, which configures/builds MLIR against
the submodule with the flags above and packages the result via
`scripts/package_toolchain.sh`. A persistent Docker volume caches the build
tree across runs, so iterating on build flags doesn't rebuild from scratch.
Upload the resulting tarball as a GitHub Release asset (`gh release upload
<tag> <tarball> --clobber`).

See `session1/blas-dialect/` for the custom dialect + pass plugin used in
session 1's later steps, and `session1/README.md` for how to build it
yourself.

## License

Apache License v2.0 with LLVM Exceptions, the same license as LLVM; see
[`LICENSE`](LICENSE). The prebuilt toolchain on the Releases page is LLVM/MLIR
itself, under its own copy of that license (`LLVM-LICENSE.TXT`, next to the
tarball).
