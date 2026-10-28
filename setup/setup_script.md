I want to bringup a tutorial on colab that can be run on any
laptop. The goal of the tutorial is to give an introduction to
MLIR.

1. The first session is a general introduction to MLIR.

- First will be walking through an example of simple function that
  does a matrix vector product using a simple SCF loops and lowers it
  to through LLVM dialect and LLVM to an executable function. The
  function is then invoked on a sample input to get the sample output
  and verified.

- Second they will use the same simple function but this time they
  will start with a function that is using SCF loop and GPU dialect to
  generate a kernel that can be invoked and executed on the GPU.

- Third they will define a custom matrix-vector operation in a custom
  dialect that is pre-defined. They will then use a pre-defined pass
  that will lower the operation to SCF loops, add this to the pass
  pipeline they used earlier to lower a function that contains just
  the matrix vector operation, compiles it down and invokes it just
  like they did in the first step to verify the output.

- Fourth step is that they will now write a lowering to the GPU
  version of the function. The tutorial will create the pass, they
  need to fill in the lowering.