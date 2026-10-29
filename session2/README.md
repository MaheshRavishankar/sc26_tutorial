# Session 2 — Linalg and tiling

Two parts, each a self-contained notebook that downloads the same prebuilt MLIR
toolchain as session 1:

- **Part 1 — CPU** (`session2_part1_cpu.ipynb`): Linalg ops lowered to loops,
  then tiled by a transform dialect schedule written against the Linalg op
  interface, with matmul measured single-threaded.
- **Part 2 — GPU** (`session2_part2_gpu.ipynb`): the same Linalg ops,
  distributed across a GPU by a transform dialect schedule that tiles them into
  nested `scf.forall` loops, maps those onto GPU blocks and threads, and
  builds the `gpu.launch` itself.

Session 1 built a matrix-matrix product from the bottom: `scf.for` loops by
hand, then a custom `blas.matmul` op with a C++ pass lowering it to those same
loops. Session 2 starts one level higher, with ops from an existing dialect,
and changes only the *shape* of the loops. Matmul is the theme throughout;
matvec rides along as the smaller example.

## The shape of Part 1

Same contract as session 1: **the attendee writes MLIR in a source file; every
notebook cell is provided.**

| File                 | Role                                                                     |
| -------------------- | ------------------------------------------------------------------------ |
| `linalg_matmul.mlir` | provided: `@matmul` as a Linalg named op                                  |
| `linalg_matvec.mlir` | provided: `@matvec` as a Linalg named op                                  |
| `tile_linalg.mlir`   | provided and working: tiles any Linalg op `[32, 32]`; **the attendee extends it** |

Both ops **accumulate** — every Linalg named op on memrefs reads *and* writes
its `outs` operand, so `linalg.matmul` is `c += a @ b`. That is the same `+=`
convention as session 1, and it is why `run_and_check` zeroes the output
buffer before each call.

**No square shapes, on the presenter's instruction.** Matmul is
`M, N, K = 128, 256, 512` (`a` 128×512, `b` 512×256, `c` 128×256) and matvec
is 128×512, in both parts. All of them are multiples of 32, so every tile is a
full 32×32. They were 512×512 until 2026-10-13.

Steps:

1. **Linalg ops.** Round-trip both files, print their `linalg.generic` form
   (`linalg-generalize-named-ops`) — iterator types, indexing maps, body, and
   the `LinalgOp` interface that reports them — then lower both with
   `convert-linalg-to-loops`, and JIT matmul and check it against numpy.
   `CPU_LOWERING_PIPELINE` here is session 1's seven passes with
   `convert-linalg-to-loops` in front, nothing else.
2. **Tiling.** `tile_linalg.mlir` matches `interface{LinalgOp}` and applies
   `tile_using_for` with `[32, 32]`, so one schedule tiles both ops: both of
   matvec's loops, but only `i` and `j` of matmul. The exercise is to tile
   matmul's `k` loop too.
   The schedule is applied in two cells, matvec and then matmul, and a third
   cell checks that the innermost `linalg.matmul` works on 32×32 slices. After
   editing the schedule, attendees round-trip it (well-formedness), then
   re-run only the matmul cell and the check: the presenter steers them away
   from re-running the matvec cell, which a `[32, 32, 32]` schedule fails. The next cell
   redefines `CPU_LOWERING_PIPELINE` with `expand-strided-metadata` and
   `lower-affine` added, for the `memref.subview`s tiling introduced, and JITs
   the tiled matmul through the same numpy check. Matvec is tiled and shown, never
   run.
3. **Measuring it.** Untiled vs tiled matmul, timed on one thread, then
   *why* it is so much faster: a cell dumps the JIT's object code for both and
   counts scalar vs packed (vectorized) float instructions, and the markdown
   explains the mechanism below.

## Why tiling is worth an order of magnitude here

At the current shapes, tiled matmul ran 10.9× faster than untiled on one
thread on the presenter's machine (9.0 ms vs 0.8 ms, Threadripper 7960X). Not
yet re-measured on Colab: with the earlier square 512×512 shapes it was 11.4×
there (249.7 ms vs 21.9 ms). Cache *capacity* is not the main factor — the
three matrices together are under 1 MB — so the framing is: tiling reshapes
the inner loop so it can be vectorized. Locality improves too (with `k` unrolled, `j` is innermost
and walks `b` and `c` with unit stride), but the vector width is what buys the
order of magnitude.

Untiled, the innermost loop is the `k` reduction,
`c[i,j] += a[i,k] * b[k,j]`. A running float sum cannot be vectorized without
reordering the additions, so LLVM leaves it scalar. Inside a 32-wide tile, `k`
runs a **small constant** number of times, so LLVM fully unrolls it into
straight-line code, and the `j` loop around it — whose iterations are now
independent — gets vectorized.

Both steps are cost-model decisions, which is why the tile size is fixed at 32.
Determined by compiling the tiled kernel (at the earlier square N=512; the
32×32×32 tiles are the same at the current shapes) with LLVM 22.1.8 at `-O3` for
each x86 CPU Colab's free tier hands out, and checking the optimized IR for
vector operations:

| tile size | broadwell | skylake-avx512 | cascadelake | icelake-server | znver2 | znver3 |
| --------- | --------- | -------------- | ----------- | -------------- | ------ | ------ |
| untiled   | scalar    | scalar         | scalar      | scalar         | scalar | scalar |
| 8         | scalar    | scalar         | scalar      | scalar         | scalar | scalar |
| **16**    | vec ×8    | vec ×8         | vec ×8      | vec ×8         | vec ×8 | vec ×8 |
| 24        | scalar    | scalar         | scalar      | scalar         | scalar | scalar |
| **32**    | vec ×8    | vec ×8         | vec ×8      | vec ×8         | vec ×8 | vec ×8 |
| 40        | scalar    | scalar         | scalar      | scalar         | scalar | scalar |
| 64        | scalar    | scalar         | scalar      | scalar         | scalar | scalar |

Identical on every target, so the story does not depend on which machine an
attendee is given. These cliffs belong to LLVM 22's heuristics: moving the
toolchain to a different LLVM version means re-running this table before
trusting the notebook's markdown. The JIT targets the host CPU, so the
object-code cell may report 8 or 16 floats at once depending on the machine.

## The exercise, and the trap in it

The answer is a second, matmul-only step after the shared one:

```mlir
%matmul = transform.structured.match ops{["linalg.matmul"]} in %root
%tiled_k, %k_loop = transform.structured.tile_using_for %matmul tile_sizes [0, 0, 32]
```

A tile size of `0` leaves a loop untiled, and there is one loop handle per
non-zero size. On the matvec file the match is empty and the step does nothing,
so one schedule still serves both.

Changing the shared step to `[32, 32, 32]` instead stops the notebook in one
of two places:

- with `%loops:2` left as it was, the **round-trip** of the schedule rejects it
  (`expected number of loops to tile (3) to match number of loops results (2)`);
- with `%loops:3`, it round-trips and tiles matmul, then fails **applying to
  matvec** (`too many tiles provided, expected at most 2 found 3`).

That is why the round-trip cell's markdown claims only well-formedness, and why
the schedule is applied to both files.

## Where a half-finished notebook stops

| State                        | Stops at                         | With                                                                 |
| ---------------------------- | -------------------------------- | -------------------------------------------------------------------- |
| schedule as provided         | the `k`-loop check, step 2       | `@matmul is not tiled in all three loops yet -- a 512 in these shapes is a loop that still runs untiled` |
| `[32, 32, 32]`, `%loops:2`   | the schedule round-trip, step 2  | the verifier error above                                             |
| `[32, 32, 32]`, `%loops:3`   | the matvec apply cell, step 2    | `too many tiles provided, expected at most 2 found 3`                |
| exercise done                | —                                | runs clean end to end                                                |

The provided schedule is correct, just incomplete, so its gate is a check on
the result rather than an error.

## Things that will cost you an afternoon

- **`expand-strided-metadata` is required** once tiling introduces
  `memref.subview`, which is why the notebook adds it (and `lower-affine`) to
  the pipeline only where it compiles the tiled matmul, and says why there. Without it the lowering fails with a bare
  `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`.
  It emits `affine.apply`, so `lower-affine` has to follow it, or the
  `ExecutionEngine` fails to build with no further detail.
- **Keep the schedule in a separate file from the payload**, loaded with
  `--transform-preload-library`. A payload module carrying
  `transform.with_named_sequence` trips the transform verifier as soon as any
  later pass outlines a function into it (`recursion not allowed in named
  sequences`, reported against the matmul).
- **Check the innermost op, not every subview.** Tiling `k` in a second step
  slices the slices from the first, so the tiled IR still contains 32×512 and
  512×32 subviews even when the exercise is done. Only the operands of the
  innermost `linalg.matmul` show whether all three loops are tiled.

## Part 2 — distributing on the GPU

Needs a GPU runtime (Runtime → Change runtime type → T4 GPU).

| File                     | Role                                                                            |
| ------------------------ | ------------------------------------------------------------------------------- |
| `linalg_matvec_gpu.mlir` | provided: `@matvec`, with its buffers registered for the GPU (`gpu.host_register`) |
| `linalg_matmul_gpu.mlir` | provided: `@matmul`, likewise                                                     |
| `distribute_linalg.mlir` | provided and written for matvec; **the attendee modifies it for matmul**          |

The schedule matches `interface{LinalgOp}` and tiles twice with
`tile_using_forall`, and **states the GPU mapping itself** (on the presenter's
instruction, 2026-10-15: introduce the mapping from the start, then explain
after the exercise why the chosen one is coalesced):

1. `tile_sizes [32]` with `mapping = [#gpu.block<x>]`: an outer `scf.forall`
   whose iterations become the GPU blocks;
2. `tile_sizes [1]` with `mapping = [#gpu.thread<x>]`, on the op inside: a
   nested `scf.forall` whose iterations become the threads of each block;
3. `transform.gpu.map_forall_to_blocks ... generate_gpu_launch` wraps the
   outer loop in a `gpu.launch` (grid = its trip counts) and replaces its
   index with `gpu.block_id`; `transform.gpu.map_nested_forall_to_threads
   ... block_dims = [32, 1, 1]` does the same for the nested loop with
   `gpu.thread_id`. `sync_after_distribute = false` leaves out a
   `gpu.barrier` the kernel does not need.

So the transform interpreter's output is already a `gpu.launch`, with a
Linalg op on one-row slices inside. The notebook then runs
`convert-linalg-to-loops`, `expand-strided-metadata`, `lower-affine` and
`gpu-lower-to-nvvm-pipeline` (for the compute capability `nvidia-smi`
reports) in one pipeline, and JITs with `libmlir_cuda_runtime.so` in
`shared_libs` and `engine.initialize()` to load the compiled GPU module.

| schedule                                        | on     | `gpu.launch` (grid, block)  | one thread computes            |
| ----------------------------------------------- | ------ | --------------------------- | ------------------------------ |
| `[32]`, then `[1]`, x only (provided)           | matvec | (4, 1, 1), (32, 1, 1)       | `out[i]`, reducing over `j`    |
| `[32]`, then `[1]`, x only (provided)           | matmul | (4, 1, 1), (32, 1, 1)       | a row of `c`, over `j` and `k` |
| `[32, 32]`, then `[1, 1]`, `j` on x, `i` on y   | matmul | (8, 4, 1), (32, 32, 1)      | one element `c[i, j]`, over `k` |

**The exercise is the last row**, and its statement says which dimension
goes on x: both steps widen to two sizes, both mappings become
`[#gpu.<block|thread><y>, #gpu.<block|thread><x>]` (the first loop, `i`, on
y), and `block_dims` becomes `[32, 32, 1]`. The check asserts that the
innermost `linalg.matmul` writes a 1x1 slice of `c` **and** that the slice is
cut at `[%thread_id_y, %thread_id_x]`.

The exercise statement sits under the notebook's *Matmul, distributed along
columns too* heading, not in step 2's introduction (on the presenter's
instruction, 2026-10-16), so attendees run matvec and the provided schedule
on matmul before they read it. It tells them to round-trip the schedule
first; below it come four cells to run in order after every edit: the
round-trip, applying the schedule to matmul, the check on the IR, and the
run on the GPU.

Mistakes and what they produce (all verified):

- `block_dims` left at `[32, 1, 1]`: `the number of required parallel
  resources (blocks or threads) 1024 overflows the number of available
  resources 32`.
- A size added without a mapping entry: `scf.forall op requires a mapping
  attribute`.
- x and y swapped (`i` on x): the check stops with `x is not on the columns
  yet`, though the kernel would run and match numpy.
- Part 1's trick, a separate matmul-only step for the columns: the nested
  block loop cannot be mapped, `scf.forall op requires a mapping attribute
  of kind 'thread' or 'warp'`.
- **The modified schedule is for matmul only.** On matvec its second size
  lands on the reduction. `tile_using_forall` only *warns* (`tiling is not
  thread safe at axis #1`) and emits the parallel loop anyway, giving a
  kernel whose threads race on `out[i]`. Every cell that runs the schedule
  keeps `mlir-opt`'s stderr and fails if it mentions `not thread safe`, so
  re-running the matvec cells after the exercise stops with a message
  instead of a wrong answer.

| State                          | Stops at                            | With                                                                 |
| ------------------------------ | ----------------------------------- | -------------------------------------------------------------------- |
| schedule as provided           | the exercise check                  | `matmul is not distributed along columns yet -- each thread still computes a 1x256 slice of c` |
| distributed, x on the rows     | the exercise check                  | `each thread computes one element, but x is not on the columns yet` |
| modified, matvec cells re-run  | applying the schedule to matvec     | `tiling is not thread safe at axis #1` + the race message            |
| exercise done                  | —                                   | runs clean end to end                                                |

**Buffers a kernel has registered must stay alive.** `gpu.host_register`
hands a buffer to the CUDA driver and nothing unregisters it. Free one and a
later numpy array can land on that address, inheriting a registration that
points at memory which is gone: `cuMemHostRegister` reports
`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, the kernel writes somewhere numpy
is not reading, and the result comes back all zeros. The notebook keeps every
buffer it passes to a kernel in a `REGISTERED` list for the session.

## Coalescing, without timing

Part 2 does not time anything, deliberately: Colab GPU timings are too noisy
to settle an argument, and a number that moves for the wrong reason teaches
worse than no number. Coalescing is taught as *reasoning* instead.

The exercise fixes the mapping (`j` on x), and a markdown cell at the end of
the notebook, after the run, explains why: `c` is row-major, a warp is 32 consecutive `threadIdx.x`, so
with x on `j` a warp writes 32 adjacent floats of `c` (one transaction) and
reads 32 adjacent floats of a row of `b`; with x on `i` it would write 32
floats a row (1 KB) apart. That matches Session 1 Part 2's hand-written
kernel, which also puts x on the columns. Nothing to run.

Until 2026-10-15 the schedule left the mapping to the
`scf-forall-to-parallel` / `gpu-map-parallel-loops` /
`convert-parallel-loops-to-gpu` passes, which put the first loop on x and
so x on the rows. That route cannot take a stated mapping: with
`#gpu.block`/`#gpu.thread` attributes on the foralls,
`convert-parallel-loops-to-gpu` stops with `'scf.parallel' op expected
mapping attribute for lowering to GPU`.

## Review decisions (SC'26 reviewer round)

Declined, deliberately:

- **GPU timing, and the device buffers (`gpu.alloc` + `gpu.memcpy`) it would
  require.** Colab GPU numbers are too noisy to be worth presenting, so the
  coalescing lesson is a reasoning prompt instead (see above).
- **A numpy single-thread / peak-FLOPS baseline** next to the GFLOP/s figure.
- **`transform.structured.vectorize`** after tiling, to make the speedup
  independent of LLVM's cost models and to show the `vector` dialect. It would
  also need vector→LLVM lowering added to `CPU_LOWERING_PIPELINE`.

**TODO before the final version:** the controlled experiment that separates
loop *order* from tiling — `transform.structured.generalize` plus
`transform.structured.interchange` to `i, k, j`, with no tiling, then time it.
Measured off-notebook on an RTX 4070 laptop CPU it is the *larger* effect
(~127× vs ~20× for tiling, at the earlier square N=512), so adding it sharpens "transformations
are enablers" but complicates the current narrative; decide how to frame it
before adding a cell.

## Running locally in VS Code

On Linux x86-64 (glibc 2.35 or newer, e.g. Ubuntu 22.04 or later), you can
run the notebooks in VS Code instead of Colab:

1. Clone the repo: `git clone https://github.com/MaheshRavishankar/sc26_tutorial`
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
   3.13, because the toolchain's MLIR Python bindings are built for it.
5. Part 2 also needs an NVIDIA GPU and a CUDA 12 toolkit. If the toolkit is
   not in `/usr/local/cuda`, add `CUDA_HOME=/path/to/cuda` to `.env`.

The setup cells then use your toolchain and your clone: they skip the
download when `$MLIR_TUTORIAL_INSTALL_DIR/bin/mlir-opt` exists, and the clone
when `$MLIR_TUTORIAL_REPO_DIR` does. Neither variable is set on Colab, so
nothing changes there.

Nothing in this session needs compiling: the exercises are `.mlir` files that
`mlir-opt` re-reads on every run. Part 1's object-code cell also uses
`objdump` (from binutils); without it, the cell says so and does nothing.

Timings from a local machine are not a guide to Colab. The cells that open
source files call `google.colab.files.view()` on Colab and fall back to
printing the paths anywhere else, so "Run All" works locally.
