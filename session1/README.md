# Session 1 — Introduction to MLIR

Two notebooks, both self-contained (each downloads the prebuilt MLIR
toolchain this tutorial uses — see the top-level README for why it's a
custom build rather than the official LLVM release or a pip package). They
are two independent parts, run start-to-finish in order — not two halves you
switch between: Part 1 needs a CPU runtime only, Part 2 needs a GPU runtime
only, and a GPU runtime is Colab's scarcer free-tier resource, so there's no
reason to hold one during Part 1's CPU-only steps. Part 2 redownloads the
toolchain and rebuilds the plugin from scratch, since a new runtime shares no
state with Part 1.

## The shape of both notebooks

**The attendee writes MLIR and C++ in source files; every notebook cell is
provided.** Nothing has to be typed into Colab — the cells parse, lower, JIT
and check whatever is currently in the repo, so the workflow is: edit a file
from Colab's file browser, save, re-run the cells.

The notebooks work on **matrix-matrix product** throughout. For each
exercise there is a **solved matrix-vector example**, in the same directory,
that attendees can use as a reference.

Both ops **accumulate**: `out[i, j] += sum_k a[i, k] * b[k, j]`. Every
reduction therefore starts from `memref.load %out[...]` rather than from
zero, and `run_and_check` zeroes the output buffer before each call.

| The attendee writes                              | Reference sitting next to it |
| ------------------------------------------------ | ---------------------------- |
| `matmul_scf.mlir` — the loop nest by hand         | `matvec_scf.mlir`            |
| `Blas_MatMulOp` in `BlasOps.td.txt` — the ODS (Part 1) | `Blas_MatVecOp` above it |
| `MatMulOp::verify()` in `BlasOps.cpp` (Part 1)    | `MatVecOp::verify()` above it |
| the loop nest in `MatMulToSCFPattern`, `LowerBlasToSCF.cpp` | `MatVecToSCFPattern` above it |
| the `gpu.launch` in `MatMulToGPUPattern`, `LowerBlasToGPU.cpp` | `MatVecToGPUPattern` above it |

`matmul_gpu.mlir`, the GPU kernel by hand, ships complete: Part 2 walks
through it rather than having attendees write it.

`matmul_custom.mlir` ships complete on purpose: it is the *spec* for the op
the attendee defines. Their ODS is correct when `mlir-opt` round-trips that
file unchanged.

### Editing files from Colab

Colab's file browser refuses to open **`.td`** — and only `.td`; every other
extension the tutorial uses (`.mlir`, `.cpp`, `.h`, `.inc`, `.txt`) opens
fine. Double-clicking one just downloads it. So the one ODS file attendees
have to edit is named **`BlasOps.td.txt`**. `mlir-tblgen` does not care about
the extension, and `CMakeLists.txt` says so at the `LLVM_TARGET_DEFINITIONS`
line. Every other `.td` in the dialect keeps its real name — attendees don't
need to open those.

Colab's file browser is also not how the notebooks open files. Each edit
point has a cell calling `google.colab.files.view()` on the file to change
*and* on its reference, so both open in the editor pane side by side.
Outside Colab those cells print the paths instead, for you to open in your
own editor.

### Where a half-finished notebook stops

Each notebook is a progressive gate — it runs until it reaches work that
hasn't been done, and the failure names what is missing:

| State                        | Stops at                 | With                                                    |
| ---------------------------- | ------------------------ | ------------------------------------------------------- |
| nothing written              | the JIT check in step 1  | all-zero result vs. numpy                               |
| step 1 done                  | the round-trip in step 2 | `custom op 'blas.matmul' is unknown` (Part 1 only)      |
| ODS written, verifier not    | the round-trip in step 2 | `undefined symbol: _ZN4blas8MatMulOp6verifyEv` (Part 1 only) |
| ODS + verifier done          | the pass in step 2       | `convert-blas-to-{scf,gpu} did not convert 'blas.matmul'` |
| everything done              | —                        | runs clean end to end                                   |

Part 2 starts two rows down that table: it clones a branch on which the op is
already defined, so its only gates are the step 1 kernel and the step 2
lowering.

An empty `@matmul` body is valid MLIR, which is why the first gate is a wrong
answer rather than an error.

The third row is the ugly one. `hasVerifier = 1` without a `verify()`
definition still *builds* — the plugin links with `-Wl,-z,undefs`, because
MLIR's own symbols are resolved by `mlir-opt` at `dlopen` time — so the
failure surfaces later, as a mangled name, when the plugin is loaded. Both
notebooks name that symbol in the Step 2 markdown so it is recognisable.

The fourth row exists because a pattern that was never written matches
nothing, and the greedy rewrite driver calls that success. Both passes
therefore check for surviving `blas` ops and name the one they could not
convert — otherwise `blas.matmul` reaches the Python `Context`, which fails
with an unrelated-looking ``Dialect `blas' not found``.

## Part 1 — CPU (`session1_part1_cpu.ipynb`)

No GPU runtime needed. Step numbers are local to this notebook.

- **Step 1 — the loop nest, by hand.** Exercise 1: write `matmul_scf.mlir`
  as nested `scf.for` loops over `memref`s. The cells round-trip it through
  `mlir-opt`, lower it through the LLVM dialect, JIT-compile it, and check it
  against numpy. Reference: `matvec_scf.mlir`.
- **Step 2 — the `blas` dialect.** `blas.matmul` is a single custom op
  standing in for the whole computation, lowered to step 1's SCF form by
  `convert-blas-to-scf`. The step first builds the plugin as shipped and
  round-trips `matvec_custom.mlir` through it. Then three exercises: the ODS
  (2) and the verifier (3) define the op, and `MatMulToSCFPattern` (4)
  lowers it. References: the `matvec` version of each, next to the `TODO`.

## Part 2 — GPU (`session1_part2_gpu.ipynb`)

Needs a GPU runtime: **Runtime -> Change runtime type -> T4 GPU**. Step
numbers are local to this notebook (independent of Part 1's).

- **Step 1 — a GPU kernel, read through.** `matmul_gpu.mlir` ships
  complete and the presenter walks through it: `gpu.launch` with one block,
  one thread per output *element*, each thread reducing over `k`. Lowered to
  NVVM/PTX and run on Colab's GPU. Nothing to write.
The kernel puts `threadIdx.x` on the columns: `memref`s are row-major, so a
warp touches 32 adjacent floats, where putting it on the rows would cost a
transaction per lane. The pattern does the same; session 2 part 2 shows the
same computation coming out the other way round when a pass picks the mapping
instead. No timing — just the argument.

- **Step 2 — lowering `blas` straight to GPU.** `convert-blas-to-gpu` builds
  step 1's `gpu.launch` shape programmatically. The exercise, Exercise 5, is
  `MatMulToGPUPattern`: its setup (operands, host registration, constants,
  sizes) is provided, and attendees write the `gpu::LaunchOp` and its body —
  the reduction loop — then register it. Reference: `MatVecToGPUPattern` in
  the same file.

Part 2 starts from a fresh clone in a fresh runtime, so the attendee's Part 1
edits are gone. Rather than have them re-type the op definition, **Part 2
clones the `session1-part1-solution` branch** (see "Solution branches"
below), so the op is already defined. Part 2's only exercise is then
`MatMulToGPUPattern`.

`main` keeps both TODOs, so Part 1's exercise is untouched. Running Part 2
locally rather than on Colab means checking that branch out yourself: the
notebook only selects it when it clones, and `MLIR_TUTORIAL_REPO_DIR` skips
the clone entirely.

## Solution branches

The answers live on branches, not in the working tree. Each is exactly one
commit on top of its base:

```
main
 +- session1-part1-solution          matmul_scf.mlir, the ODS, the verifier, LowerBlasToSCF.cpp
 |   +- session1-part2-solution      ... plus LowerBlasToGPU.cpp
 +- session2-part1-solution          tile_linalg.mlir
 +- session2-part2-solution          distribute_linalg.mlir
```

`session1-part1-solution` is also what Part 2 clones, which is why session 1's
answers are split in two rather than kept as one "session 1 solved" branch.

After `main` moves, replay them with `scripts/refresh_solution_branches.sh`,
then force-push the ones that changed.

Every notebook clones with `--depth 1 --no-single-branch`, so all of these
branch tips are in an attendee's checkout already: recovering one solved file
is `git checkout origin/<branch> -- <path>`, with no further fetch.

**TODO before the final version:** a "skip this step" cell after each exercise
that does exactly that — 6 cells in total (3 in session 1 part 1, 1 in part 2,
1 in each session 2 notebook) — so an attendee stuck on one exercise can take
the solved file and keep going. It should refuse to overwrite work that has
already been edited, so a reflexive "Run All" cannot wipe someone's answers.

A solution branch is the *end state* of a part, not necessarily a state its
notebook runs from start to finish. `session2-part2-solution` is the clearest
case: its schedule has been widened for matmul, and the notebook applies the
schedule to matvec first, where distributing a reduction loop is a race — so
running that notebook on that branch stops, by design, at exactly the check
that exists to catch it.

## Running locally in VS Code

On Linux x86-64 (glibc 2.35 or newer, e.g. Ubuntu 22.04 or later), you can
run the notebooks in VS Code instead of Colab:

1. Clone the repo: `git clone https://github.com/MaheshRavishankar/sc26_tutorial`.
   For Part 2, also `git checkout session1-part1-solution` in it: on Colab
   the notebook clones that branch itself, but it uses your clone as it is.
2. Download the toolchain,
   [`mlir-tutorial-llvm.tar.gz`](https://github.com/MaheshRavishankar/sc26_tutorial/releases/download/session1-toolchain-v1/mlir-tutorial-llvm.tar.gz),
   and untar it:
   `mkdir mlir-tutorial-llvm && tar xf mlir-tutorial-llvm.tar.gz -C mlir-tutorial-llvm --strip-components=1`
3. Before you start a kernel, create a file `.env` in the repo root with two
   lines, using absolute paths:
   ```
   MLIR_TUTORIAL_INSTALL_DIR=/path/to/mlir-tutorial-llvm
   MLIR_TUTORIAL_REPO_DIR=/path/to/sc26_tutorial
   ```
   VS Code passes `.env` to notebook kernels (the `python.envFile` setting).
   With plain Jupyter, export the two variables before starting it instead.
4. Open the repo folder in VS Code, then a notebook, and pick a Python 3.13
   kernel with `ipykernel`, `numpy` and `requests` installed. It has to be
   3.13, because the toolchain's MLIR Python bindings are built for it. Both
   notebooks build the `blas` plugin, so you also need `cmake`, `ninja` and a
   C++ compiler.
5. Part 2 also needs an NVIDIA GPU and a CUDA 12 toolkit. If the toolkit is
   not in `/usr/local/cuda`, add `CUDA_HOME=/path/to/cuda` to `.env`.

The setup cells then use your toolchain and your clone: they skip the
download when `$MLIR_TUTORIAL_INSTALL_DIR/bin/mlir-opt` exists, and the clone
when `$MLIR_TUTORIAL_REPO_DIR` does. Neither variable is set on Colab, so
nothing changes there.

## `blas-dialect/`

A small out-of-tree MLIR dialect + pass plugin. One dialect, `blas`, with two
ops and four lowering passes, all compiled into one `libBlasPlugin.so` that
`mlir-opt` loads via `--load-dialect-plugin` / `--load-pass-plugin`:

```
include/Blas/BlasDialect.td    the `blas` dialect
include/Blas/BlasOps.td.txt    blas.matvec (given) + blas.matmul (TODO)
include/Blas/Passes.td         both passes declared
lib/BlasOps.cpp                verifyDimsMatch helper; MatVecOp::verify (given),
                               MatMulOp::verify (TODO)
lib/LowerBlasToSCF.cpp         convert-blas-to-scf: MatVecToSCFPattern (given)
                               + MatMulToSCFPattern (TODO), one pass, one
                               pattern per op
lib/LowerBlasToGPU.cpp         convert-blas-to-gpu: MatVecToGPUPattern (given)
                               + MatMulToGPUPattern (TODO), same shape
```

Both lowerings are **one pass with a pattern per op**, and both fail if any
`blas` op survives them — an unwritten pattern matches nothing, and the greedy
rewrite driver calls that success, so without the check `blas.matmul` would
reach the Python `Context` and fail with an unrelated-looking
``Dialect `blas' not found``.

Both matmul patterns ship commented out, with their scaffolding written: the
operands, the index constants, the problem sizes, the `memref.cast` +
`gpu.host_register` calls in the GPU one, and the `eraseOp` at the end. The
attendee uncomments the pattern, writes the loop nest or the `gpu.launch`
where the comment says so, and registers it in `runOnOperation()` — which is
deliberately left undone, since a pattern that is never registered matches
nothing while the greedy driver still reports success. Commenting the whole
pattern out also keeps the plugin building before `blas.matmul` exists, which
matters because Part 1 defines the op in the step before this one.

Patterns build loop bodies by moving the rewriter's insertion point into the
loop's block rather than by passing a body-builder lambda: more verbose, but
it reads top to bottom. The rule worth knowing is that a loop with no
`iter_args` gets its `scf.yield` created automatically, while a loop with
`iter_args` does not — so reduction loops end with an explicit
`scf::YieldOp::create`.

To build it yourself against a local MLIR install:

```
cmake -S blas-dialect -B blas-dialect/build -G Ninja \
  -DMLIR_DIR=<mlir-install>/lib/cmake/mlir \
  -DLLVM_DIR=<mlir-install>/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
ninja -C blas-dialect/build
```

This needs an MLIR build whose tools export a broad set of dynamic symbols
(`-Wl,--export-dynamic`, or the official LLVM release binaries' equivalent —
plain `LLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON` is *not* sufficient, its curated
export list is too narrow). See the top-level README for details.
