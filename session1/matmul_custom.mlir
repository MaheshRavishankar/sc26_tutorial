// The spec for your blas.matmul op: your ODS definition in
// blas-dialect/include/Blas/BlasOps.td.txt is correct when mlir-opt can parse and
// re-print this file unchanged.
func.func @matmul(%a: memref<4x3xf32>, %b: memref<3x5xf32>, %out: memref<4x5xf32>) {
  blas.matmul %a, %b, %out : memref<4x3xf32>, memref<3x5xf32>, memref<4x5xf32>
  return
}
