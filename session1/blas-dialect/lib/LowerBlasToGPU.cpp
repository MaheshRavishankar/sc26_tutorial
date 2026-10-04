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

struct MatMulToGPUPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value out = op.getOut();

    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    Value aU = memref::CastOp::create(rewriter, loc, unrankedF32, a);
    Value bU = memref::CastOp::create(rewriter, loc, unrankedF32, b);
    Value outU = memref::CastOp::create(rewriter, loc, unrankedF32, out);
    gpu::HostRegisterOp::create(rewriter, loc, aU);
    gpu::HostRegisterOp::create(rewriter, loc, bU);
    gpu::HostRegisterOp::create(rewriter, loc, outU);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value M = memref::DimOp::create(rewriter, loc, a, c0);
    Value K = memref::DimOp::create(rewriter, loc, a, c1);
    Value N = memref::DimOp::create(rewriter, loc, b, c1);

    // Threads: x over the columns of `out`, y over its rows.
    auto launchOp = gpu::LaunchOp::create(rewriter, loc, c1, c1, c1, N,
                                          M, c1);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      Value by = launchOp.getBlockIds().y;
      Value bx = launchOp.getBlockIds().x;
      Value blockDimY = launchOp.getBlockSize().y;
      Value blockDimX = launchOp.getBlockSize().x;
      Value ty = launchOp.getThreadIds().y;
      Value tx = launchOp.getThreadIds().x;

      // The global row and column, built as MatVecToGPUPattern builds its
      // row: blockIdx * blockDim + threadIdx, from y and from x.
      Value rowBase = arith::MulIOp::create(rewriter, loc, by, blockDimY);
      Value row = arith::AddIOp::create(rewriter, loc, rowBase, ty);
      Value colBase = arith::MulIOp::create(rewriter, loc, bx, blockDimX);
      Value col = arith::AddIOp::create(rewriter, loc, colBase, tx);

      Value init =
          memref::LoadOp::create(rewriter, loc, out, ValueRange{row, col});

      auto redLoop =
          scf::ForOp::create(rewriter, loc, c0, K, c1, ValueRange{init});
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(redLoop.getBody());
        Value k = redLoop.getInductionVar();
        Value acc = redLoop.getRegionIterArgs()[0];

        Value aElem =
            memref::LoadOp::create(rewriter, loc, a, ValueRange{row, k});
        Value bElem =
            memref::LoadOp::create(rewriter, loc, b, ValueRange{k, col});
        Value prod = arith::MulFOp::create(rewriter, loc, aElem, bElem);
        Value sum = arith::AddFOp::create(rewriter, loc, acc, prod);
        scf::YieldOp::create(rewriter, loc, ValueRange{sum});
      }

      memref::StoreOp::create(rewriter, loc, redLoop.getResult(0), out,
                              ValueRange{row, col});
      gpu::TerminatorOp::create(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};


struct LowerBlasToGPUPass
    : public blas::impl::LowerBlasToGPUPassBase<LowerBlasToGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatVecToGPUPattern, MatMulToGPUPattern>(&getContext());

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
