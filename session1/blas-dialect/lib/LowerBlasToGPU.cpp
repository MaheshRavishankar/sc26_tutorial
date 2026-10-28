// Lowers the blas dialect to gpu.launch kernels.
//
// One pass, one pattern per op -- the GPU counterpart of LowerBlasToSCF.cpp.
// MatVecToGPUPattern below is complete and is the template for the exercise:
// write MatMulToGPUPattern, then register it in runOnOperation().
//
// Instead of the nested scf.for loops of the SCF lowering, this
// builds a single GPU kernel launch where each thread computes one row of
// the output: the thread at global row `row` computes
// `out[row] += sum_j mat[row, j] * vec[j]`, looping serially over `j` inside
// the kernel. A thread's global index is blockIdx * blockDim + threadIdx, the
// canonical form; with the single block launched here it is just threadIdx. This is exactly the strategy
// used in Part 2's hand-written gpu.launch example, built programmatically
// instead of by hand.
//
// Loop bodies are built by moving the rewriter's insertion point, not with
// body-builder lambdas -- see LowerBlasToSCF.cpp for the terminator rule.

#include "Blas/BlasOps.h"
#include "Blas/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace blas;

namespace blas {
#define GEN_PASS_DEF_LOWERBLASTOGPUPASS
#include "Blas/Passes.h.inc"
} // namespace blas

namespace {

struct MatVecToGPUPattern : public OpRewritePattern<MatVecOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatVecOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value mat = op.getMat();
    Value vec = op.getVec();
    Value out = op.getOut();

    // Pin the host-allocated memrefs for GPU access -- see this tutorial's
    // Part 2 step 1 for why (some GPU/driver setups silently produce wrong
    // results without this, rather than a crash).
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    Value matU = memref::CastOp::create(rewriter, loc, unrankedF32, mat);
    Value vecU = memref::CastOp::create(rewriter, loc, unrankedF32, vec);
    Value outU = memref::CastOp::create(rewriter, loc, unrankedF32, out);
    gpu::HostRegisterOp::create(rewriter, loc, matU);
    gpu::HostRegisterOp::create(rewriter, loc, vecU);
    gpu::HostRegisterOp::create(rewriter, loc, outU);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value numRows = memref::DimOp::create(rewriter, loc, mat, c0);
    Value numCols = memref::DimOp::create(rewriter, loc, mat, c1);

    // One block, one thread per output row.
    auto launchOp =
        gpu::LaunchOp::create(rewriter, loc, c1, c1, c1, numRows, c1, c1);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      Value bx = launchOp.getBlockIds().x;
      Value blockDimX = launchOp.getBlockSize().x;
      Value tx = launchOp.getThreadIds().x;

      // The global row: blockIdx.x * blockDim.x + threadIdx.x. With one
      // block, bx is 0 and this is just tx, but this is the general form.
      Value rowBase = arith::MulIOp::create(rewriter, loc, bx, blockDimX);
      Value row = arith::AddIOp::create(rewriter, loc, rowBase, tx);

      // `out[row] +=`, so the running total starts at what is already there.
      Value init = memref::LoadOp::create(rewriter, loc, out, ValueRange{row});

      auto redLoop =
          scf::ForOp::create(rewriter, loc, c0, numCols, c1, ValueRange{init});
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(redLoop.getBody());
        Value j = redLoop.getInductionVar();
        Value acc = redLoop.getRegionIterArgs()[0];

        Value matElem =
            memref::LoadOp::create(rewriter, loc, mat, ValueRange{row, j});
        Value vecElem =
            memref::LoadOp::create(rewriter, loc, vec, ValueRange{j});
        Value prod = arith::MulFOp::create(rewriter, loc, matElem, vecElem);
        Value sum = arith::AddFOp::create(rewriter, loc, acc, prod);
        scf::YieldOp::create(rewriter, loc, ValueRange{sum});
      }

      memref::StoreOp::create(rewriter, loc, redLoop.getResult(0), out,
                              ValueRange{row});
      gpu::TerminatorOp::create(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// TODO(exercise): uncomment this pattern, build the gpu.launch where it says
// so, and register it in runOnOperation() below. It is commented out so the
// plugin builds before blas.matmul exists -- Part 1 defines the op, and this
// notebook clones the branch where that is already done.
//
// The scaffolding is done: the operands, the memref.cast + gpu.host_register
// calls that make the buffers reachable from the device, the index constants,
// the problem sizes, and the eraseOp at the end. What is missing is the
// launch. One block, one thread per output *element*: thread (ty, tx)
// computes
//
//   out[row, col] += sum_k a[row, k] * b[k, col]
//
// where row = by * blockDim.y + ty and col = bx * blockDim.x + tx, built the
// way MatVecToGPUPattern above builds its row. So it wants N threads in x and
// M in y -- the 2D grid from step 1 of this notebook. MatVecToGPUPattern is
// the same thing with a 1D grid. Finish the launch body with
// gpu::TerminatorOp.
//
// struct MatMulToGPUPattern : public OpRewritePattern<MatMulOp> {
//   using OpRewritePattern::OpRewritePattern;
//
//   LogicalResult matchAndRewrite(MatMulOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//     Value a = op.getA();
//     Value b = op.getB();
//     Value out = op.getOut();
//
//     auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
//     Value aU = memref::CastOp::create(rewriter, loc, unrankedF32, a);
//     Value bU = memref::CastOp::create(rewriter, loc, unrankedF32, b);
//     Value outU = memref::CastOp::create(rewriter, loc, unrankedF32, out);
//     gpu::HostRegisterOp::create(rewriter, loc, aU);
//     gpu::HostRegisterOp::create(rewriter, loc, bU);
//     gpu::HostRegisterOp::create(rewriter, loc, outU);
//
//     Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
//     Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
//     Value M = memref::DimOp::create(rewriter, loc, a, c0);
//     Value K = memref::DimOp::create(rewriter, loc, a, c1);
//     Value N = memref::DimOp::create(rewriter, loc, b, c1);
//
//     ... your gpu.launch here ...
//
//     rewriter.eraseOp(op);
//     return success();
//   }
// };


struct LowerBlasToGPUPass
    : public blas::impl::LowerBlasToGPUPassBase<LowerBlasToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatVecToGPUPattern>(&getContext());
    // TODO(exercise): register your pattern too:
    //   patterns.add<MatMulToGPUPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // A pattern that hasn't been written matches nothing, and the greedy
    // driver calls that success -- the op simply survives into the output,
    // where it fails much later with a far less obvious error. Catch it here.
    if (getOperation()
            ->walk([&](Operation *nested) {
              if (!isa<BlasDialect>(nested->getDialect()))
                return WalkResult::advance();
              nested->emitError()
                  << "convert-blas-to-gpu did not convert '"
                  << nested->getName()
                  << "' -- is its pattern written and registered in "
                     "blas-dialect/lib/LowerBlasToGPU.cpp?";
              return WalkResult::interrupt();
            })
            .wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
