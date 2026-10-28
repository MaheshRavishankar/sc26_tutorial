#ifndef BLAS_PASSES_H
#define BLAS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace blas {

#define GEN_PASS_DECL
#include "Blas/Passes.h.inc"

} // namespace blas

#endif // BLAS_PASSES_H
