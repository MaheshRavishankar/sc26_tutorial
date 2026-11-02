# Presenting the tutorial

What the presenter drives, as opposed to the notebooks the attendees run.
Two things per part:

| File | What it is |
| - | - |
| `session1_part1.tex`, `session1_part2.tex`, `session2_part1.tex`, `session2_part2.tex` | The decks. Beamer source; `make` builds each to a PDF (below). |
| `*_walkthrough.ipynb` | **Generated.** The attendee notebook with its clone cell pointed at the solution branch, so every cell runs green with no editing. |

The deck carries the narration; the walkthrough notebook carries the
execution. Project the notebook when something has to *run*, and the deck for
everything else.

Session 2 Part 2's deck is a first draft (2026-10-15), not yet reviewed.

## Building the decks

Needs a small TeX Live set --- on Ubuntu:

```bash
sudo apt-get install -y --no-install-recommends \
    latexmk texlive-latex-recommended texlive-fonts-recommended texlive-pictures
```

That is beamer, `listings`, TikZ and the fonts, about 400 MB.
`texlive-latex-extra` is deliberately *not* needed: the look comes from
`sc26.sty` and beamer's built-ins rather than a third-party theme.

```bash
cd presentation
make            # every deck's PDF, e.g. session1_part1.pdf -- the only PDFs committed
make handout    # *_handout.pdf -- on demand, 1 page per slide
make clean      # remove .aux/.log/.nav/...; keeps the PDFs
```

The handout PDFs are not committed (`.gitignore` keeps them out).
`make handout` is still the quickest way to review a change: one page per
slide instead of one per overlay step.

## Writing slides

- `sc26.sty` holds the palette, the frame furniture, the `listings` setup
  (MLIR and ODS keyword sets, `escapeinside={(@}{@)}` for inline annotations)
  and these helpers: `\errbox{...}` for a real error message; the `stage`
  environment, a fixed-height area that stops staged code jumping between
  overlays; the `half` listing style, sized for a two-column frame;
  `\df{...}`, which marks what differs from the reference inside a listing's
  `(@ @)` escape; `\hl<2>{...}`, the same highlight on one overlay only, for
  walking a listing a few lines at a time (mark keywords and comments inside
  it with `\lkw{...}` and `\lcm{...}`, since escaped text gets no syntax
  colouring); `halfs`, a size smaller than `half`; and
  `\cmdbox{...}` for a command line or flags worth copying down.
- Any frame containing a listing needs `[fragile]`.
- A verbatim environment **cannot** go inside `\visible<2->{...}`. Use the
  declaration form, `\onslide<2->`, or the `visibleenv` environment instead.
- The decks have no presenter notes (deleted on 2026-11-01); what to do
  while presenting is under "Driving a session" below.
- Walk the **reference** in stages before an exercise, one frame per stage.
  The solution afterwards shows what differs from the reference, marked with
  `\df`, or walks the solution a few highlighted lines at a time with `\hl`.
- A reference listing shows the file **as it is in the repo** -- same names,
  same shapes. Attendees have it open next to the slide. Check against the
  file, and take generated IR and error text from a real `mlir-opt` run.
- Where C++ builds IR, put the C++ on the left and the IR it generates on the
  right (`style=half`), with blank lines keeping each IR line beside the call
  that creates it.
- Do not wrap listing lines in `(@\only<2->{@)` ... `(@}@)` to stage them: the
  lines in between are swallowed as ordinary TeX and come out garbled. Use a
  trailing `(@\only<2>{...}@)` annotation, or separate frames.

## Regenerating the walkthrough notebooks

```bash
python3 presentation/generate_walkthrough.py
```

It rewrites all four from the exercise notebooks, so a change to a notebook
needs this re-run. Never edit a `*_walkthrough.ipynb` by hand.

## Driving a session

Open the walkthrough notebook in Colab from your own account and keep the
attendee notebook in a second tab, unsolved --- that is how you show what a
cell does *before* the exercise is written. Before the room fills, run the
walkthrough's setup cells (3--8; on a T4 runtime for the GPU parts): the
toolchain download is the slowest thing in every part and teaches nothing.
For Session 1, run the first plugin build too (Part 1: cells 20 and 21; Part
2: cells 22 and 23).

Cell numbers below are the attendee notebook's; the walkthrough's are one
higher.

- **Session 1 Part 1**, 90 minutes: 10 setup, 30 for step 1, 28 for defining
  the op, 17 for the lowering, 5 to wrap up. Keyboard time is 12 minutes for
  Exercise 1, 12 for Exercise 2 (both parts) and 10 for Exercise 3. Before
  Exercise 1, run the attendee notebook's JIT check (cell 17) to show the
  all-zero result: an empty function parses and runs, so a clean parse is
  not a passing test. If you are behind at Exercise 3, type it yourself and
  narrate.
- **Session 1 Part 2**: one exercise, the GPU lowering, about 10 minutes.
  After an edit, attendees re-run the rebuild cell, the pass and the run on
  the GPU. If you are behind, type it yourself and narrate.
- **Session 2 Part 1**: one exercise, about 10 minutes. After an edit,
  attendees run the three cells below the exercise, in order: the
  round-trip (25), the apply (27) and the check (29). Not `Ctrl+F8` on any
  of them: it re-runs the matvec cell (23), which stops on a
  `[32, 32, 32]` schedule ("too many tiles provided, expected at most 2
  found 3").
- **Session 2 Part 2**: one exercise, about 10 minutes. After an edit,
  attendees run, in order, the round-trip (26), the apply (28), the check
  (30) and the run (32). Not `Ctrl+F8` on any of them: it re-runs the matvec
  cell (19), which stops on a schedule widened for matmul ("tiling is not
  thread safe at axis #1"). **Known problem:** the walkthrough clones
  `session2-part2-solution`, whose schedule is for matmul only, so its
  matvec cell stops the same way on the first pass.

