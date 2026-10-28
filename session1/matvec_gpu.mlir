module {
  func.func @matvec(%mat: memref<4x3xf32>, %vec: memref<3xf32>, %out: memref<4xf32>) {
    %mat_u = memref.cast %mat : memref<4x3xf32> to memref<*xf32>
    %vec_u = memref.cast %vec : memref<3xf32> to memref<*xf32>
    %out_u = memref.cast %out : memref<4xf32> to memref<*xf32>
    gpu.host_register %mat_u : memref<*xf32>
    gpu.host_register %vec_u : memref<*xf32>
    gpu.host_register %out_u : memref<*xf32>
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gridX = %c1, %gridY = %c1, %gridZ = %c1)
               threads(%tx, %ty, %tz) in (%blockX = %c4, %blockY = %c1, %blockZ = %c1) {
      // The global row: blockIdx.x * blockDim.x + threadIdx.x. With one
      // block, %bx is 0 and this is just %tx.
      %rowBase = arith.muli %bx, %blockX : index
      %row = arith.addi %rowBase, %tx : index
      %c0 = arith.constant 0 : index
      %step = arith.constant 1 : index
      // `out[row] +=`, so the running total starts at what is already there.
      %init = memref.load %out[%row] : memref<4xf32>
      %acc = scf.for %j = %c0 to %c3 step %step iter_args(%a = %init) -> (f32) {
        %m = memref.load %mat[%row, %j] : memref<4x3xf32>
        %v = memref.load %vec[%j] : memref<3xf32>
        %prod = arith.mulf %m, %v : f32
        %a2 = arith.addf %a, %prod : f32
        scf.yield %a2 : f32
      }
      memref.store %acc, %out[%row] : memref<4xf32>
      gpu.terminator
    }
    return
  }
}
