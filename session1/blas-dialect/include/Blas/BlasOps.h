#ifndef BLAS_BLASOPS_H
#define BLAS_BLASOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Blas/BlasDialect.h"

#define GET_OP_CLASSES
#include "Blas/BlasOps.h.inc"

#endif // BLAS_BLASOPS_H
