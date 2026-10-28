#include "Blas/BlasOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace blas;

#define GET_OP_CLASSES
#include "Blas/BlasOps.cpp.inc"

// Shared helper for the verifiers below: check that two memref dimensions
// agree. A dynamic dimension (`?`) is only known at run time, so it is
// compatible with anything -- that check is the fiddly part, and having it
// here means a verifier is one call per constraint.
static LogicalResult verifyDimsMatch(Operation *op, int64_t lhs, int64_t rhs,
                                     StringRef what) {
  if (ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs)
    return success();
  return op->emitOpError() << what << " (" << lhs << " vs " << rhs << ")";
}

// Hand-written verifier for blas.matvec, pulled in because BlasOps.td.txt
// sets `hasVerifier = 1`. ODS already checked the operand *types* (2D/1D f32
// memrefs); this checks the relationship between their shapes, which ODS
// cannot express.
LogicalResult MatVecOp::verify() {
  auto matType = cast<MemRefType>(getMat().getType());
  auto vecType = cast<MemRefType>(getVec().getType());
  auto outType = cast<MemRefType>(getOut().getType());

  if (failed(verifyDimsMatch(*this, matType.getDimSize(1),
                             vecType.getDimSize(0),
                             "matrix column count must match input vector size")))
    return failure();

  return verifyDimsMatch(*this, matType.getDimSize(0), outType.getDimSize(0),
                         "matrix row count must match output vector size");
}

// TODO(exercise): write `LogicalResult MatMulOp::verify()` here, modelled on
// MatVecOp::verify above. Your blas.matmul definition sets
// `let hasVerifier = 1;`, so this function must exist. If you forget it the
// plugin still builds (we link with -Wl,-z,undefs, because MLIR's own
// symbols are resolved by mlir-opt at dlopen time) and then fails to load:
//   mlir-opt: symbol lookup error: libBlasPlugin.so:
//             undefined symbol: _ZN4blas8MatMulOp6verifyEv
// That mangled name is blas::MatMulOp::verify().
//
// For `out[i, j] += sum_k a[i, k] * b[k, j]` there are three constraints,
// one verifyDimsMatch call each:
//   a.dim(1) == b.dim(0)     the reduction dimension
//   a.dim(0) == out.dim(0)   rows
//   b.dim(1) == out.dim(1)   columns
