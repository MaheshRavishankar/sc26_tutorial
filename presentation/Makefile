# Decks for the SC26 MLIR tutorial. Needs the TeX Live set listed in README.md.
#
#   make            the deck, for every .tex in this directory (the only PDF
#                   that is committed)
#   make deck       the same
#   make handout    on demand: overlays collapsed, one page per slide
#   make notes      on demand: slides interleaved with the presenter notes
#   make clean      remove build droppings (keeps the PDFs)
#   make distclean  remove the PDFs too

DECKS := $(basename $(wildcard session*_part*.tex))
LATEXMK := TEXINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all deck handout notes clean distclean
all: deck
deck:    $(addsuffix .pdf,$(DECKS))
handout: $(addsuffix _handout.pdf,$(DECKS))
notes:   $(addsuffix _notes.pdf,$(DECKS))

%.pdf: %.tex sc26.sty
	$(LATEXMK) $<

# Overlays collapsed: beamer's own handout mode, one page per slide.
%_handout.pdf: %.tex sc26.sty
	$(LATEXMK) -jobname=$*_handout \
	    -pdflatex='pdflatex %O "\PassOptionsToClass{handout}{beamer}\input{%S}"' $<

# Slides plus the \note{} blocks, for printing or a second screen.
%_notes.pdf: %.tex sc26.sty
	$(LATEXMK) -jobname=$*_notes \
	    -pdflatex='pdflatex %O "\def\notesmode{1}\input{%S}"' $<

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk
distclean: clean
	rm -f *.pdf
