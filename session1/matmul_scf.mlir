// SOLUTION for the matmul_scf.mlir exercise.
module {
  func.func @matmul(%a: memref<4x3xf32>, %b: memref<3x5xf32>, %out: memref<4x5xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = memref.dim %a, %c0 : memref<4x3xf32>
    %K = memref.dim %a, %c1 : memref<4x3xf32>
    %N = memref.dim %b, %c1 : memref<3x5xf32>
    scf.for %i = %c0 to %M step %c1 {
      scf.for %j = %c0 to %N step %c1 {
        %init = memref.load %out[%i, %j] : memref<4x5xf32>
        %acc = scf.for %k = %c0 to %K step %c1
                        iter_args(%s = %init) -> (f32) {
          %av = memref.load %a[%i, %k] : memref<4x3xf32>
          %bv = memref.load %b[%k, %j] : memref<3x5xf32>
          %prod = arith.mulf %av, %bv : f32
          %s2 = arith.addf %s, %prod : f32
          scf.yield %s2 : f32
        }
        memref.store %acc, %out[%i, %j] : memref<4x5xf32>
      }
    }
    return
  }
}
