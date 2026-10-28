# Session 1 — Walkthrough Plan

The tutorial's shape: **every step is demonstrated on matrix-vector product,
then repeated by the attendee on matrix-matrix product.** `matvec` is the
worked example and the template; `matmul` is the exercise. Both live in the
same `blas` dialect, so the attendee is extending a dialect that already
works rather than starting from an empty file.

The notebooks work on `matmul` directly. `matvec` is not run as its own
cells — it is referenced from the surrounding markdown as the complete
version to read. **All notebook cells are provided**; the attendee only edits
source files, because copy-pasting code into Colab cells is awkward.

## Part 1 — CPU

1. **Setup.** Walk through the setup cells: downloading/locating the
   toolchain, cloning the repo.

2. **Step 1.** Attendees fill in `matmul_scf.mlir` (ships with an empty
   `@matmul` body), reading `matvec_scf.mlir` as the reference. The cells
   then round-trip it through `mlir-opt` (catches syntax/type errors first),
   parse it into a Python module, run the CPU lowering pipeline, JIT it, and
   check against numpy.

3. **Step 2, round one -- define the op.** Two files: `Blas_MatMulOp` in
   `BlasOps.td.txt` (reading `Blas_MatVecOp` above it), and
   `MatMulOp::verify()` in `BlasOps.cpp` (reading `MatVecOp::verify()`, and
   calling the shared `verifyDimsMatch` helper). Check: `mlir-opt`
   round-trips `matmul_custom.mlir` unchanged.

4. **Step 2, round two -- write the lowering.** Attendees add
   `MatMulToSCFPattern` to `LowerBlasToSCF.cpp` and register it, reading
   `MatVecToSCFPattern` in the same file. Check: the pass output JITs and
   matches numpy. Walk through the generated header here too, to show what
   ODS produced from their `.td.txt`.

## Part 2 — GPU

Targeting the GPU: the presenter walks through `matmul_gpu.mlir`, a matmul
kernel written by hand with `gpu.launch` (blocks, threads, the thread ids, one
thread per output element), then attendees add `MatMulToGPUPattern` to
`convert-blas-to-gpu`. Its setup is provided; they write the launch and the
reduction loop, reading `MatVecToGPUPattern`.

Part 2 runs on a fresh Colab runtime, so the attendee's Part 1 edits are gone.
Rather than have them re-type the op definition, Part 2 clones the
`session1-part1-solution` branch, where `Blas_MatMulOp` and
`MatMulOp::verify()` are already filled in. Its one exercise is
`MatMulToGPUPattern`; `matmul_gpu.mlir` ships complete and is read, not
written.

## TODO before the final version

- "Skip this step" cells, one after each exercise, checking the solved file
  out of the matching solution branch (`git checkout origin/<branch> -- <path>`;
  the clones already fetch every branch tip). 6 cells across the four
  notebooks. Needed because the gates are hard stops: an attendee stuck on the
  ODS or a pattern cannot run the rest of their notebook.

## Notes on what ships

- `matmul_custom.mlir` ships complete: it is the spec for the op the attendee
  defines, not something they write.
- Both lowerings are one pass with one pattern per op: `convert-blas-to-scf`
  and `convert-blas-to-gpu`. The pipeline names are valid before anything is
  implemented.
- Ops accumulate (`out +=`), so every reduction starts from a load of `out`
  rather than from a zero constant, and `run_and_check` zeroes the output
  buffer before each call.
- Patterns build loop bodies by moving the rewriter's insertion point, not
  with body-builder lambdas. Verbose but linear, and it removes a trap: a
  loop with no iter_args gets its terminator for free, so only the reduction
  loop needs an explicit yield.
- Both passes fail if any `blas` op survives them. A pattern that
  was never written matches nothing and the greedy driver reports success,
  so without this the op reaches Python and fails as
  "Dialect `blas' not found".
- `hasVerifier = 1` without a `verify()` definition builds fine (the plugin
  links with -Wl,-z,undefs) and fails at dlopen with a mangled
  "undefined symbol". Both notebooks name that symbol so it is recognisable.
- Each notebook is a progressive gate: it runs until it hits work that has
  not been done. Verified stop points, in order — the step 1 JIT check
  (all-zero result), the step 2 round-trip (`custom op 'blas.matmul' is
  unknown`), the step 2 pass (`did not convert 'blas.matmul'`), then clean.
- `run_and_check` zeroes the output buffer before invoking. The buffers are
  reused across steps, so without that a pass that silently does nothing
  would pass on the previous step's result.

## Notes on the environment

- Colab's editor refuses `.td` and nothing else -- `.mlir`, `.cpp`, `.h`,
  `.inc` and `.txt` all open. Hence `BlasOps.td.txt`: it is the one ODS file
  attendees edit, and mlir-tblgen is indifferent to the extension. The other
  `.td` files keep their real names.
- The notebooks call `google.colab.files.view()` directly at each edit point
  rather than through a helper. An earlier `edit(...)` helper collided with
  IPython's `%edit` magic: automagic substituted it whenever the defining
  cell had not run, giving a confusing TypeError instead of a NameError.
- Setup markdown tells attendees to run cells in order; several of the
  reported problems so far have really been out-of-order execution.
- The lowering passes use the current `OpTy::create(builder, loc, ...)`
  builder API. `builder.create<OpTy>(loc, ...)` is deprecated as of LLVM 22
  and warns on every call site — which matters here because attendees read
  these passes as the template for their own.

## Open items

- The exercise keys live on solution branches (see session1/README.md), one
  commit each on top of `main`, and are public on purpose.
  `scripts/refresh_solution_branches.sh` replays them when `main` moves.
- Timing: Part 2's hand-written kernel was demoted to "read along" on
  2026-10-12, leaving five exercises across the two notebooks. If it still
  runs long, Part 1's hand-written loop nest is the next candidate — the ODS
  and pass exercises are the parts that actually teach MLIR's extension
  points.
