// Matrix-matrix product as a GPU kernel, written by hand with gpu.launch.
// Complete: step 1 of the Part 2 notebook reads it, and step 2's pattern has
// to build the same thing.
//
//     out[i, j] += sum_k a[i, k] * b[k, j]
//
// One block of (5, 4, 1) threads: x over the 5 columns of `out`, y over its 4
// rows. Thread (%ty, %tx) computes out[%row, %col] by reducing over %k,
// starting from the value already in `out` (the `+=`). %row and %col are the
// thread's global index, block id * block size + thread id in each dimension;
// with one block the block ids are 0, so they are just %ty and %tx.
//
// The memref.cast + gpu.host_register lines make the host buffers reachable
// from the GPU; the cast to an unranked memref is only because
// gpu.host_register takes one.
module {
  func.func @matmul(%a: memref<4x3xf32>, %b: memref<3x5xf32>, %out: memref<4x5xf32>) {
    %a_u = memref.cast %a : memref<4x3xf32> to memref<*xf32>
    %b_u = memref.cast %b : memref<3x5xf32> to memref<*xf32>
    %out_u = memref.cast %out : memref<4x5xf32> to memref<*xf32>
    gpu.host_register %a_u : memref<*xf32>
    gpu.host_register %b_u : memref<*xf32>
    gpu.host_register %out_u : memref<*xf32>
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gridX = %c1, %gridY = %c1, %gridZ = %c1)
               threads(%tx, %ty, %tz) in (%blockX = %c5, %blockY = %c4, %blockZ = %c1) {
      %rowBase = arith.muli %by, %blockY : index
      %row = arith.addi %rowBase, %ty : index
      %colBase = arith.muli %bx, %blockX : index
      %col = arith.addi %colBase, %tx : index
      %c0 = arith.constant 0 : index
      %step = arith.constant 1 : index
      %init = memref.load %out[%row, %col] : memref<4x5xf32>
      %acc = scf.for %k = %c0 to %c3 step %step iter_args(%s = %init) -> (f32) {
        %av = memref.load %a[%row, %k] : memref<4x3xf32>
        %bv = memref.load %b[%k, %col] : memref<3x5xf32>
        %prod = arith.mulf %av, %bv : f32
        %s2 = arith.addf %s, %prod : f32
        scf.yield %s2 : f32
      }
      memref.store %acc, %out[%row, %col] : memref<4x5xf32>
      gpu.terminator
    }
    return
  }
}
