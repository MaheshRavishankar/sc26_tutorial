#!/usr/bin/env python3
"""Check the \\colab{...} cell pointers in the decks.

Each deck's tags must name cells of its own notebook (1.* for Session 1 Part 1
up to 4.* for Session 2 Part 2, as scripts/number_cells.py numbers them), that
exist, and that never run backwards: the deck walks the notebook top to
bottom. The Makefile runs this before building a deck.

    python3 scripts/check_colab_tags.py presentation/session1_part1.tex ...
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DECKS = {
    "session1_part1": (1, "session1/session1_part1_cpu.ipynb"),
    "session1_part2": (2, "session1/session1_part2_gpu.ipynb"),
    "session2_part1": (3, "session2/session2_part1_cpu.ipynb"),
    "session2_part2": (4, "session2/session2_part2_gpu.ipynb"),
}

TAG = re.compile(r"\\colab(?:<[^>]*>)?\{([^}]*)\}")
CELL = re.compile(r"(\d+)\.(\d+)")

failed = False
for tex in sys.argv[1:]:
    part, notebook = DECKS[Path(tex).stem]
    cell_count = len(json.loads((REPO / notebook).read_text())["cells"])
    text = Path(tex).read_text()
    last = 0
    for tag in TAG.finditer(text):
        line = text.count("\n", 0, tag.start()) + 1
        cells = CELL.findall(tag.group(1))
        if not cells:
            print(f"{tex}:{line}: no cell number in \\colab{{{tag.group(1)}}}")
            failed = True
        for session_part, number in cells:
            number = int(number)
            problem = None
            if int(session_part) != part:
                problem = f"cell {session_part}.{number} is not in this deck's notebook ({part}.*)"
            elif not 1 <= number <= cell_count:
                problem = f"cell {part}.{number} does not exist ({notebook} has {cell_count})"
            elif number < last:
                problem = f"cell {part}.{number} comes after {part}.{last}: the tags run backwards"
            if problem:
                print(f"{tex}:{line}: {problem}")
                failed = True
            last = max(last, number)

sys.exit(1 if failed else 0)
