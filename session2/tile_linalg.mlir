// A transform dialect schedule.
//
// This file is not a program to run. It is a *schedule*: a small program whose
// input ("payload") is another MLIR file -- here linalg_matvec.mlir or
// linalg_matmul.mlir -- and whose job is to rewrite it. mlir-opt runs it with
//
//   --transform-preload-library=transform-library-paths=tile_linalg.mlir
//   --transform-interpreter
//
// The interpreter looks for @__transform_main and passes it the payload's
// top-level op as %root.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    // Find the ops to transform. Handles like %ops are not values in the
    // payload -- they are *references* to payload ops, and later transform
    // ops act through them.
    //
    // This matches by interface, not by op name: every op that implements
    // LinalgOp -- linalg.matvec, linalg.matmul, linalg.generic, ... -- so the
    // same schedule works on any Linalg payload.
    %ops = transform.structured.match interface{LinalgOp} in %root
      : (!transform.any_op) -> !transform.any_op

    // Tile the matched ops, one tile size per loop, in the order of the op's
    // iterator_types. [32, 32] tiles the first two loops. Each tiled loop
    // becomes an outer scf.for stepping by 32, and the op is re-created
    // inside on 32-wide slices (memref.subview) of its operands.
    //
    // Returns the tiled op, then one handle per generated loop.
    %tiled, %loops:2 = transform.structured.tile_using_for %ops
      tile_sizes [32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                !transform.any_op)

    // matmul has a third loop, the k reduction, that [32, 32] left untiled.
    // Match it by name -- matvec has no such loop -- and tile only that loop:
    // a tile size of 0 leaves a loop alone, and there is one loop handle per
    // non-zero size.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled_k, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [0, 0, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
