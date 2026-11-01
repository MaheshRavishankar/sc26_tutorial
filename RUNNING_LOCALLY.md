# Running the Notebooks Locally

The notebooks are written for Google Colab, but you can also run them in
VS Code on Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04 or later).
The prebuilt toolchain is built on Ubuntu 22.04, so older systems cannot
load it.

## Steps

1. **Clone the repo.**

   ```
   git clone https://github.com/MaheshRavishankar/sc26_tutorial
   ```

   For Session 1 Part 2, also run `git checkout session1-part1-solution` in
   the clone. On Colab that notebook clones the branch itself; locally it
   uses your clone as it is.

2. **Download the toolchain**,
   [`mlir-tutorial-llvm.tar.gz`](https://github.com/MaheshRavishankar/sc26_tutorial/releases/download/session1-toolchain-v1/mlir-tutorial-llvm.tar.gz)
   (about 150 MB), and untar it:

   ```
   mkdir mlir-tutorial-llvm
   tar xf mlir-tutorial-llvm.tar.gz -C mlir-tutorial-llvm --strip-components=1
   ```

3. **Get Python 3.13.** The toolchain's MLIR Python bindings are built for
   Python 3.13, as on Colab, and do not import into any other version.
   `python3.13 --version` tells you whether you have it. If not, on Ubuntu:

   ```
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.13 python3.13-venv
   ```

   Or use [uv](https://docs.astral.sh/uv/), which downloads Python 3.13 for
   you in the next step.

4. **Create a virtual environment** in the repo root, with the packages the
   notebooks use:

   ```
   cd sc26_tutorial
   python3.13 -m venv .venv
   .venv/bin/pip install ipykernel numpy requests
   ```

   With uv, `uv venv --python 3.13` and then
   `uv pip install ipykernel numpy requests`. A `.venv` you already have
   with another Python version will not work; delete it and create a new
   one.

5. **Create a file `.env`** in the repo root, with two lines, using absolute
   paths:

   ```
   MLIR_TUTORIAL_INSTALL_DIR=/path/to/mlir-tutorial-llvm
   MLIR_TUTORIAL_REPO_DIR=/path/to/sc26_tutorial
   ```

   VS Code passes `.env` to notebook kernels (the `python.envFile`
   setting). With plain Jupyter, export the two variables before starting
   it instead.

6. **Install the build tools.** Session 1 builds a plugin, so it needs
   `cmake`, `ninja` and a C++ compiler:
   `sudo apt install cmake ninja-build g++`. Session 2 compiles nothing.

7. **For the GPU parts** (Session 1 Part 2 and Session 2 Part 2), you need
   an NVIDIA GPU and a CUDA 12 toolkit. If the toolkit is not in
   `/usr/local/cuda`, add a third line to `.env`,
   `CUDA_HOME=/path/to/cuda`, naming the directory that has `bin/ptxas`
   and `nvvm/` in it.

8. **Open the repo folder in VS Code**, then a notebook, and pick the
   `.venv` kernel: **Select Kernel** at the top right, then **Python
   Environments**.

The setup cells then use your toolchain and your clone. They skip the
download when `$MLIR_TUTORIAL_INSTALL_DIR/bin/mlir-opt` exists, and the
clone when `$MLIR_TUTORIAL_REPO_DIR` does. Neither variable is set on
Colab, so nothing changes there.

## Differences from Colab

- **`Ctrl+F8`** (run every cell above) is Colab's. In VS Code, use
  **Execute Above Cells**, in the toolbar of the cell you are on (or in the
  menu of its run button, if your VS Code groups them there).
- **The cells that open a file** for you to edit print its path instead.
  Open it in VS Code.
- **Timings** on your machine are not a guide to Colab's.

## Troubleshooting

- **`The MLIR Python bindings are built for Python 3.13; this kernel is
  Python 3.x`**, from the setup cell: the kernel is not the `.venv` from
  step 4. Pick it in the kernel picker. (Without that check, the next cell
  would fail with `cannot import name 'ir' from 'mlir._mlir_libs._mlir'
  (unknown location)`.)
- **`ModuleNotFoundError: No module named 'numpy'`**: the kernel is not the
  environment you installed into. `import sys; print(sys.executable)` shows
  which Python the kernel runs. If you have just installed the packages,
  restart the kernel.
- **The notebook downloads the toolchain or clones the repo anyway**: the
  kernel does not see `.env`. A kernel keeps the environment it started
  with, so restart it after you create or change `.env`, and check that the
  paths are absolute.
