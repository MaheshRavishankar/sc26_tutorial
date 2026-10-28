#!/usr/bin/env python3
"""Run a notebook's code cells in order, outside Colab, stopping on the first
error. Enough to check that a notebook still works end to end after an edit.

    python3 scripts/run_notebook.py presentation/session1_part1_walkthrough.ipynb

Needs ipython, numpy and requests (the notebooks' own imports), and a local
MLIR toolchain:

    MLIR_TUTORIAL_INSTALL_DIR=/path/to/install \
    MLIR_TUTORIAL_REPO_DIR=/path/to/a/checkout \
        python3 scripts/run_notebook.py <notebook>

MLIR_TUTORIAL_REPO_DIR skips the clone, so point it at a checkout of the branch
the notebook expects -- a solution branch for the walkthrough notebooks.
Pass a cell index as a second argument to stop after that cell.
"""

import json
import sys

from IPython.core.interactiveshell import InteractiveShell

notebook = json.load(open(sys.argv[1]))
stop_after = int(sys.argv[2]) if len(sys.argv) > 2 else 10**6

shell = InteractiveShell.instance()
shell.run_cell("import os")  # the notebooks assume os is already imported

for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code" or index > stop_after:
        continue
    print(f"\n{'=' * 30} CELL {index} {'=' * 30}", flush=True)
    result = shell.run_cell("".join(cell["source"]), store_history=True)
    if result.error_before_exec or result.error_in_exec:
        sys.exit(f"*** CELL {index} FAILED ***")

print("\nALL CELLS OK", flush=True)
