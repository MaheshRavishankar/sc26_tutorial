#ifndef BLAS_PASSES
#define BLAS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SCF lowering: one pass for the whole dialect, one pattern per op. The
// matvec pattern ships complete; the matmul pattern is the exercise.
//===----------------------------------------------------------------------===//

def LowerBlasToSCFPass : Pass<"convert-blas-to-scf", "::mlir::ModuleOp"> {
  let summary = "Lower blas ops to scf.for loops";
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
  ];
}

def LowerBlasToGPUPass : Pass<"convert-blas-to-gpu", "::mlir::ModuleOp"> {
  let summary = "Lower blas ops to gpu.launch kernels";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
  ];
}

//===----------------------------------------------------------------------===//
// GPU lowering: same shape as the SCF one -- a single pass over the dialect,
// one pattern per op. The matvec pattern ships complete; the matmul pattern
// is the exercise.
//===----------------------------------------------------------------------===//

#endif // BLAS_PASSES
