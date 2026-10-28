cmake_minimum_required(VERSION 3.20)
project(blas_dialect LANGUAGES C CXX)

# Expect MLIR_DIR / LLVM_DIR to point at <extracted-release>/lib/cmake/mlir
# and <extracted-release>/lib/cmake/llvm, e.g.:
#   cmake -S . -B build \
#     -DMLIR_DIR=$LLVM_RELEASE/lib/cmake/mlir \
#     -DLLVM_DIR=$LLVM_RELEASE/lib/cmake/llvm
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
# LLVM_DEFINITIONS is a space-separated *string*, not a CMake list. Passing it
# straight to add_definitions() makes CMake treat it as one argument, so
# everything after the first '=' is swallowed into that macro's value
# (-D_GLIBCXX_USE_CXX11_ABI="1 -D__STDC_CONSTANT_MACROS ...") and the rest of
# the flags are lost. Split it first -- this is what LLVM's own standalone
# example does.
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# BlasOps.td.txt, not BlasOps.td: Colab's editor silently refuses to open
# .td files (it is the only extension it refuses), and this is the one ODS
# file tutorial attendees have to edit. mlir-tblgen itself does not care
# about the extension. See session1/README.md.
set(LLVM_TARGET_DEFINITIONS include/Blas/BlasOps.td.txt)
mlir_tablegen(include/Blas/BlasOps.h.inc -gen-op-decls)
mlir_tablegen(include/Blas/BlasOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/Blas/BlasOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/Blas/BlasOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(BlasOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/Blas/Passes.td)
mlir_tablegen(include/Blas/Passes.h.inc -gen-pass-decls -name Blas)
add_public_tablegen_target(BlasPassesIncGen)

add_library(BlasPlugin SHARED
  lib/BlasDialect.cpp
  lib/BlasOps.cpp
  lib/LowerBlasToSCF.cpp
  lib/LowerBlasToGPU.cpp
  lib/PluginEntry.cpp
)
add_dependencies(BlasPlugin BlasOpsIncGen BlasPassesIncGen)

# Plugins resolve MLIR/LLVM symbols against the host tool (mlir-opt) at
# dlopen time, so we deliberately do NOT link against the MLIR/LLVM static
# libraries here -- only against our own generated code.
# HandleLLVMOptions.cmake sets -Wl,-z,defs globally, which would make the
# linker demand that every MLIR/LLVM symbol we call be resolvable *now*.
# Override it back to -z,undefs for this target: those symbols are meant to
# stay unresolved until the plugin is dlopen()'d by mlir-opt, which resolves
# them against its own (already-loaded) MLIR/LLVM symbols.
set_target_properties(BlasPlugin PROPERTIES
  PREFIX "lib"
  LINK_FLAGS "-Wl,-z,undefs"
)

# MLIR/LLVM are themselves built with -fno-rtti (they use their own TypeID
# system instead of C++ RTTI), so classes like mlir::Dialect never get real
# typeinfo compiled into any MLIR library. If this plugin is compiled WITH
# RTTI (the default), our BlasDialect subclass ends up referencing
# mlir::Dialect's typeinfo symbol expecting someone to define it -- nobody
# ever does, and the plugin fails to dlopen with an unresolved-symbol error.
# Matching MLIR's own -fno-rtti avoids generating that reference at all.
target_compile_options(BlasPlugin PRIVATE -fno-rtti)
