module {
  func.func @matvec(%mat: memref<4x3xf32>, %vec: memref<3xf32>, %out: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %numRows = memref.dim %mat, %c0 : memref<4x3xf32>
    %numCols = memref.dim %mat, %c1 : memref<4x3xf32>
    scf.for %i = %c0 to %numRows step %c1 {
      // `out[i] +=`, so the running total starts at what is already there.
      %init = memref.load %out[%i] : memref<4xf32>
      %acc = scf.for %j = %c0 to %numCols step %c1
                      iter_args(%a = %init) -> (f32) {
        %m = memref.load %mat[%i, %j] : memref<4x3xf32>
        %v = memref.load %vec[%j] : memref<3xf32>
        %prod = arith.mulf %m, %v : f32
        %a2 = arith.addf %a, %prod : f32
        scf.yield %a2 : f32
      }
      memref.store %acc, %out[%i] : memref<4xf32>
    }
    return
  }
}
