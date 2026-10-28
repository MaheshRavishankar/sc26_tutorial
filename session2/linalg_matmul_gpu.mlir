// Matrix-matrix product for the GPU: part 1's linalg_matmul.mlir, plus the
// host registration a GPU kernel needs. It ACCUMULATES into `c`:
//
//     c[i, j] += sum_k a[i, k] * b[k, j]
//
// The kernel reads and writes these host buffers directly, so each one is
// registered with the CUDA driver (gpu.host_register) before the op runs.
// memref.cast to an unranked memref is only because gpu.host_register
// takes one.
module {
  func.func @matmul(%a: memref<128x512xf32>, %b: memref<512x256xf32>,
                    %c: memref<128x256xf32>) {
    %a_u = memref.cast %a : memref<128x512xf32> to memref<*xf32>
    %b_u = memref.cast %b : memref<512x256xf32> to memref<*xf32>
    %c_u = memref.cast %c : memref<128x256xf32> to memref<*xf32>
    gpu.host_register %a_u : memref<*xf32>
    gpu.host_register %b_u : memref<*xf32>
    gpu.host_register %c_u : memref<*xf32>
    linalg.matmul ins(%a, %b : memref<128x512xf32>, memref<512x256xf32>)
                  outs(%c : memref<128x256xf32>)
    return
  }
}
