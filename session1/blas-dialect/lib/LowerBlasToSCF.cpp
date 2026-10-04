// Lowers the blas dialect to scf.for loops.
//
// One pass, one pattern per op. MatVecToSCFPattern below is complete and is
// the template for the exercise: write MatMulToSCFPattern, then register it
// in runOnOperation().

#include "Blas/BlasOps.h"
#include "Blas/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace blas;

namespace blas {
#define GEN_PASS_DEF_LOWERBLASTOSCFPASS
#include "Blas/Passes.h.inc"
} // namespace blas

namespace {

// Rewrites:
//   blas.matvec %mat, %vec, %out : memref<MxNxf32>, memref<Nxf32>, memref<Mxf32>
// into:
//   scf.for %i = 0 to M {
//     %init = memref.load %out[%i]                  // += starts from out
//     %acc = scf.for %j = 0 to N iter_args(%a = %init) {
//       %m = memref.load %mat[%i, %j]
//       %v = memref.load %vec[%j]
//       scf.yield %a + %m * %v
//     }
//     memref.store %acc, %out[%i]
//   }
//
// Loop bodies are built by setting the rewriter's insertion point into the
// loop's block, not by passing a body-builder lambda. Two things to know:
//
//  - A loop with no iter_args gets its `scf.yield` terminator created for
//    you; setInsertionPointToStart puts your ops before it.
//  - A loop *with* iter_args does not, because only you know what it should
//    yield -- so the reduction loop below ends with an explicit scf.yield.
struct MatVecToSCFPattern : public OpRewritePattern<MatVecOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatVecOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value mat = op.getMat();
    Value vec = op.getVec();
    Value out = op.getOut();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value numRows = memref::DimOp::create(rewriter, loc, mat, c0);
    Value numCols = memref::DimOp::create(rewriter, loc, mat, c1);

    auto rowLoop = scf::ForOp::create(rewriter, loc, c0, numRows, c1);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value i = rowLoop.getInductionVar();

      // `out[i] +=`, so the running total starts at what is already there.
      Value init = memref::LoadOp::create(rewriter, loc, out, ValueRange{i});

      auto redLoop =
          scf::ForOp::create(rewriter, loc, c0, numCols, c1, ValueRange{init});
      {
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(redLoop.getBody());
        Value j = redLoop.getInductionVar();
        Value acc = redLoop.getRegionIterArgs()[0];

        Value matElem =
            memref::LoadOp::create(rewriter, loc, mat, ValueRange{i, j});
        Value vecElem =
            memref::LoadOp::create(rewriter, loc, vec, ValueRange{j});
        Value prod = arith::MulFOp::create(rewriter, loc, matElem, vecElem);
        Value sum = arith::AddFOp::create(rewriter, loc, acc, prod);
        scf::YieldOp::create(rewriter, loc, ValueRange{sum});
      }

      memref::StoreOp::create(rewriter, loc, redLoop.getResult(0), out,
                              ValueRange{i});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatMulToSCFPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value out = op.getOut();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value M = memref::DimOp::create(rewriter, loc, a, c0);
    Value K = memref::DimOp::create(rewriter, loc, a, c1);
    Value N = memref::DimOp::create(rewriter, loc, b, c1);

    auto rowLoop = scf::ForOp::create(rewriter, loc, c0, M, c1);
    {
      OpBuilder::InsertionGuard rowGuard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      Value i = rowLoop.getInductionVar();

      auto colLoop = scf::ForOp::create(rewriter, loc, c0, N, c1);
      {
        OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        Value j = colLoop.getInductionVar();

        Value init =
            memref::LoadOp::create(rewriter, loc, out, ValueRange{i, j});

        auto redLoop = scf::ForOp::create(rewriter, loc, c0, K, c1,
                                          ValueRange{init});
        {
          OpBuilder::InsertionGuard redGuard(rewriter);
          rewriter.setInsertionPointToStart(redLoop.getBody());
          Value k = redLoop.getInductionVar();
          Value acc = redLoop.getRegionIterArgs()[0];

          Value aElem =
              memref::LoadOp::create(rewriter, loc, a, ValueRange{i, k});
          Value bElem =
              memref::LoadOp::create(rewriter, loc, b, ValueRange{k, j});
          Value prod = arith::MulFOp::create(rewriter, loc, aElem, bElem);
          Value sum = arith::AddFOp::create(rewriter, loc, acc, prod);
          scf::YieldOp::create(rewriter, loc, ValueRange{sum});
        }

        memref::StoreOp::create(rewriter, loc, redLoop.getResult(0), out,
                                ValueRange{i, j});
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};


struct LowerBlasToSCFPass
    : public blas::impl::LowerBlasToSCFPassBase<LowerBlasToSCFPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatVecToSCFPattern, MatMulToSCFPattern>(&getContext());

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
                  << "convert-blas-to-scf did not convert '"
                  << nested->getName()
                  << "' -- is its pattern written and registered in "
                     "blas-dialect/lib/LowerBlasToSCF.cpp?";
              return WalkResult::interrupt();
            })
            .wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
