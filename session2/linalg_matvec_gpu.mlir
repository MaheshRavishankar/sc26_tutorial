// Matrix-vector product for the GPU: part 1's linalg_matvec.mlir, plus the
// host registration a GPU kernel needs. It ACCUMULATES into `out`:
//
//     out[i] += sum_j mat[i, j] * vec[j]
//
// The kernel reads and writes these host buffers directly, so each one is
// registered with the CUDA driver (gpu.host_register) before the op runs.
// memref.cast to an unranked memref is only because gpu.host_register
// takes one.
module {
  func.func @matvec(%mat: memref<128x512xf32>, %vec: memref<512xf32>,
                    %out: memref<128xf32>) {
    %mat_u = memref.cast %mat : memref<128x512xf32> to memref<*xf32>
    %vec_u = memref.cast %vec : memref<512xf32> to memref<*xf32>
    %out_u = memref.cast %out : memref<128xf32> to memref<*xf32>
    gpu.host_register %mat_u : memref<*xf32>
    gpu.host_register %vec_u : memref<*xf32>
    gpu.host_register %out_u : memref<*xf32>
    linalg.matvec ins(%mat, %vec : memref<128x512xf32>, memref<512xf32>)
                  outs(%out : memref<128xf32>)
    return
  }
}
