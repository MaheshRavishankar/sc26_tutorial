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
make notes      # *_notes.pdf   -- on demand, slides + speaker notes
make clean      # remove .aux/.log/.nav/...; keeps the PDFs
```

The handout and notes PDFs are not committed (`.gitignore` keeps them out).
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
- Speaker notes go in `\note{...}` after the frame. They are typeset text, so
  escape underscores: `iter\_args`, not `iter_args`.
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
cell does *before* the exercise is written (the all-zero gate in Step 1 is
worth 30 seconds). Run the setup cells before the room fills: the toolchain
download and the first cmake configure are the two slowest things in the day
and neither teaches anything.

Timing that Session 1 Part 1 is built around, for a 90-minute slot: 10 minutes
setup, 30 for Step 1 (12 of them exercise time), 28 for Step 2 round one (12
exercise), 17 for round two (10 exercise), 5 to wrap up. If you are behind at
the last exercise, type it yourself and narrate --- the payoff there is
understanding the shape, not producing it.
