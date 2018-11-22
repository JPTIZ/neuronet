# Project configs
BRIEF  := report
BIBREF := refs

TARGETS := $(BRIEF).pdf

# Compiler configs
ifndef LATEX
	LATEX  := xelatex
endif
BIBTEX := bibtex

# Etc.
OBJS   := $(wildcard **/*.tex) $(wildcard *.tex)

ifdef BIBREF
	BIBREF := $(BIBREF).bib
endif

# Rules
.PHONY: clean clean-pdf all

all: $(TARGETS)

$(BRIEF).pdf: $(BRIEF).tex $(BIBREF)

%.pdf: %.tex $(BIBREF)
	$(LATEX) --shell-escape $(basename $@)
ifdef BIBREF
	$(BIBTEX) $(basename $@)
	$(LATEX) --shell-escape $(basename $@)
	$(LATEX) --shell-escape $(basename $@)
endif
	$(LATEX) --shell-escape $(basename $@)

clean:
	rm -f *.{aux,bbl,blg,log,nav,snm,out,toc}
	rm -rf _minted-*

clean-pdf:
	rm $(BRIEF).pdf
