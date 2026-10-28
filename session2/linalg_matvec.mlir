// Matrix-vector product as a Linalg named op on memrefs -- the matmul example
// one dimension smaller. It ACCUMULATES into its `outs` operand:
//
//     out[i] += sum_j mat[i, j] * vec[j]
module {
  func.func @matvec(%mat: memref<128x512xf32>, %vec: memref<512xf32>,
                    %out: memref<128xf32>) {
    linalg.matvec ins(%mat, %vec : memref<128x512xf32>, memref<512xf32>)
                  outs(%out : memref<128xf32>)
    return
  }
}
