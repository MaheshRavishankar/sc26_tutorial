// Matrix-matrix product as a Linalg named op on memrefs. Like every Linalg
// named op, it ACCUMULATES into its `outs` operand:
//
//     c[i, j] += sum_k a[i, k] * b[k, j]
//
// which is the same `+=` convention as session 1. `ins` are read, `outs` is
// read *and* written -- that is why there is no separate accumulator here.
module {
  func.func @matmul(%a: memref<128x512xf32>, %b: memref<512x256xf32>,
                    %c: memref<128x256xf32>) {
    linalg.matmul ins(%a, %b : memref<128x512xf32>, memref<512x256xf32>)
                  outs(%c : memref<128x256xf32>)
    return
  }
}
