# Builds the from-source MLIR/LLVM toolchain against Ubuntu 22.04 (glibc 2.35),
# matching Colab's runtime. Binaries built on a newer host OS (newer glibc/
# libstdc++) fail to run there with GLIBC_2.3x/GLIBCXX_3.4.3x "version not
# found" errors -- see the top-level README.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Colab's runtime currently ships Python 3.13. MLIR's Python bindings are
# compiled native extensions (via nanobind) tied to a specific CPython ABI,
# so they must be built against the same Python version Colab actually runs
# -- building against Ubuntu 22.04's stock Python 3.10 produces extensions
# that fail to import under 3.13 (a confusing KeyError from importlib, not a
# clean ImportError). Ubuntu 22.04 doesn't ship 3.13, so pull it from
# deadsnakes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    gnupg \
    ca-certificates \
    rsync \
    binutils \
    zlib1g-dev \
    software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.13 python3.13-dev python3.13-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m ensurepip --upgrade

# CUDA toolkit: MLIR_ENABLE_CUDA_RUNNER links libmlir_cuda_runtime.so against
# the CUDA driver-API headers/stub library this provides. nvcc/ptxas from this
# install are NOT what run at tutorial-time -- Colab's own preinstalled CUDA
# toolkit provides ptxas when the notebook's gpu-lower-to-nvvm-pipeline pass
# invokes it.
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-toolkit-12-4 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"

# Build-time-only Python deps for MLIR's Python bindings (mlir/python/requirements.txt).
RUN python3.13 -m pip install --no-cache-dir \
    "nanobind>=2.9,<3.0" \
    "PyYAML>=5.4.0,<=6.0.1" \
    "typing_extensions>=4.12.2" \
    numpy

COPY build_toolchain.sh /usr/local/bin/build_toolchain.sh
RUN chmod +x /usr/local/bin/build_toolchain.sh

ENTRYPOINT ["/usr/local/bin/build_toolchain.sh"]
