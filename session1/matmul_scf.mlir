// EXERCISE: matrix-matrix product with scf.for loops.
//
// Template: matvec_scf.mlir, which does the same for out += mat * vec.
//
// Fill in @matmul so it computes out[i, j] += sum_k a[i, k] * b[k, j].
// Structurally this is matvec_scf.mlir with one extra loop: an outer %i loop
// over rows of `a`, then a %j loop over columns of `b`, then the same
// iter_args reduction, now over %k, storing into out[%i, %j]. Note the `+=`:
// the reduction starts from memref.load %out[%i, %j], not from zero.
//
// As shipped this parses and runs, but writes nothing -- so the check in the
// notebook fails with an all-zero `out`. That is what "not done yet" looks
// like here.
module {
  func.func @matmul(%a: memref<4x3xf32>, %b: memref<3x5xf32>, %out: memref<4x5xf32>) {
    // TODO(exercise): your nested scf.for loops go here.
    return
  }
}
