#include "Blas/BlasDialect.h"
#include "Blas/BlasOps.h"

using namespace blas;

#include "Blas/BlasOpsDialect.cpp.inc"

void BlasDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Blas/BlasOps.cpp.inc"
      >();
}
