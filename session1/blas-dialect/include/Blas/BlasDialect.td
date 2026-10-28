#ifndef BLAS_DIALECT
#define BLAS_DIALECT

include "mlir/IR/OpBase.td"

def Blas_Dialect : Dialect {
  let name = "blas";
  let summary = "A tiny dense-linear-algebra dialect used in the SC26 "
                "introduction-to-MLIR tutorial.";
  let description = [{
    Holds one worked-through op (`blas.matvec`) and one you write yourself
    (`blas.matmul`). Both name a whole dense-linear-algebra kernel with a
    single operation, so the interesting work happens in the passes that
    lower them.
  }];
  let cppNamespace = "::blas";
}

#endif // BLAS_DIALECT
