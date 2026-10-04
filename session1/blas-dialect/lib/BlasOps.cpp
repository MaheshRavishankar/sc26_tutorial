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

// SOLUTION: verifier for blas.matmul.
LogicalResult MatMulOp::verify() {
  auto aType = cast<MemRefType>(getA().getType());
  auto bType = cast<MemRefType>(getB().getType());
  auto outType = cast<MemRefType>(getOut().getType());

  if (failed(verifyDimsMatch(*this, aType.getDimSize(1), bType.getDimSize(0),
                             "inner dimensions must match")))
    return failure();
  if (failed(verifyDimsMatch(*this, aType.getDimSize(0), outType.getDimSize(0),
                             "row count of 'a' must match row count of 'out'")))
    return failure();
  return verifyDimsMatch(*this, bType.getDimSize(1), outType.getDimSize(1),
                         "column count of 'b' must match column count of 'out'");
}
