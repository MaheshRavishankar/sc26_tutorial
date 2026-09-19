// A transform dialect schedule that distributes a Linalg op across a GPU.
//
// Written for linalg_matvec_gpu.mlir. Like part 1's schedule it matches Linalg
// ops by interface, so it applies to linalg_matmul_gpu.mlir unchanged. mlir-opt
// runs it with
//
//   --transform-preload-library=transform-library-paths=distribute_linalg.mlir
//   --transform-interpreter
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    // Any op that implements the LinalgOp interface.
    %ops = transform.structured.match interface{LinalgOp} in %root
      : (!transform.any_op) -> !transform.any_op

    // Blocks: tile the first loop by 32 into an scf.forall, a loop whose
    // iterations may run in parallel. 128 rows become 4 iterations, each
    // running the op on a 32-row slice. The mapping gives one entry per tiled
    // loop: this loop becomes the GPU blocks, indexed by the block id in x.
    %block_op, %blocks = transform.structured.tile_using_forall %ops
      tile_sizes [32, 32] (mapping = [#gpu.block<y>, #gpu.block<x>])
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Threads: tile the op inside each block again, by 1, into a nested
    // scf.forall over the block's 32 rows, one row per iteration. It becomes
    // the threads of each block, indexed by the thread id in x.
    %thread_op, %threads = transform.structured.tile_using_forall %block_op
      tile_sizes [1, 1] (mapping = [#gpu.thread<y>, #gpu.thread<x>])
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Build the kernel from those mappings. map_forall_to_blocks wraps the
    // block loop in a gpu.launch, one block per iteration, and replaces its
    // index with gpu.block_id. map_nested_forall_to_threads does the same
    // for the thread loop with gpu.thread_id; block_dims is the size of each
    // block, in x, y and z.
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %launch = transform.gpu.map_forall_to_blocks %func generate_gpu_launch
      : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads %launch block_dims = [32, 32, 1]
      sync_after_distribute = false
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
