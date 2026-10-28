// Combined dialect + pass plugin entry points. One shared library
// (libBlasPlugin.so) exposes both, loaded via:
//   mlir-opt --load-dialect-plugin=libBlasPlugin.so \
//            --load-pass-plugin=libBlasPlugin.so ...

#include "Blas/BlasDialect.h"
#include "Blas/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

extern "C" ::mlir::DialectPluginLibraryInfo mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "BlasDialect", "v1",
          [](DialectRegistry *registry) {
            registry->insert<blas::BlasDialect>();
          }};
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "BlasPasses", "v1", []() {
            mlir::registerPass(blas::createLowerBlasToSCFPass);
            mlir::registerPass(blas::createLowerBlasToGPUPass);
          }};
}
