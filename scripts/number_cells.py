#!/usr/bin/env python3
"""Number every cell of the four attendee notebooks.

Each cell carries its number, so the decks and READMEs can point at a cell
("Colab 1.12") and the presenter can find it in Colab, which shows no cell
numbers of its own. Session 1 Part 1 is 1.*, Session 1 Part 2 is 2.*,
Session 2 Part 1 is 3.* and Session 2 Part 2 is 4.*, counting from 1:

- a markdown cell that opens with a heading gets it in the heading,
  "## [1.2] Setup"; any other markdown cell starts with "**[1.3]** ";
- a code cell gets a first line "# [1.3]".

Re-run it after adding or removing a cell: it replaces the old numbers. Then
re-run presentation/generate_walkthrough.py, whose notebooks carry the same
numbers (their extra first cell is not numbered), and update any "cell N"
reference in the decks and READMEs.

    python3 scripts/number_cells.py
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    ("session1/session1_part1_cpu.ipynb", 1),
    ("session1/session1_part2_gpu.ipynb", 2),
    ("session2/session2_part1_cpu.ipynb", 3),
    ("session2/session2_part2_gpu.ipynb", 4),
]

TAG = r"\[\d+\.\d+\]"
OLD_HEADING = re.compile(rf"^(#+ ){TAG} ")
OLD_MARKDOWN = re.compile(rf"^\*\*{TAG}\*\* ")
OLD_CODE = re.compile(rf"^# {TAG}\n")


def renumber(source, cell_type, tag):
    if cell_type == "code":
        return f"# {tag}\n" + OLD_CODE.sub("", source, count=1)
    source = OLD_MARKDOWN.sub("", OLD_HEADING.sub(r"\1", source, count=1), count=1)
    heading = re.match(r"#+ ", source)
    if heading:
        return f"{heading.group(0)}{tag} {source[heading.end():]}"
    return f"**{tag}** {source}"


for path, part in NOTEBOOKS:
    notebook_path = REPO / path
    notebook = json.loads(notebook_path.read_text())
    for number, cell in enumerate(notebook["cells"], start=1):
        source = "".join(cell["source"])
        cell["source"] = renumber(source, cell["cell_type"],
                                  f"[{part}.{number}]").splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"{path}: {len(notebook['cells'])} cells, {part}.1 to {part}.{len(notebook['cells'])}")
