// The spec for the blas.matmul op: the definition in
// blas-dialect/include/Blas/BlasOps.td.txt is correct when mlir-opt can parse
// this file and print blas.matmul back in the same syntax.
func.func @matmul(%a: memref<4x3xf32>, %b: memref<3x5xf32>, %out: memref<4x5xf32>) {
  blas.matmul %a, %b, %out : memref<4x3xf32>, memref<3x5xf32>, memref<4x5xf32>
  return
}
