func.func @matvec(%mat: memref<4x3xf32>, %vec: memref<3xf32>, %out: memref<4xf32>) {
  blas.matvec %mat, %vec, %out : memref<4x3xf32>, memref<3xf32>, memref<4xf32>
  return
}
