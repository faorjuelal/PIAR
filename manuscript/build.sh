#!/bin/sh
# Build the manuscript in the reprint layout (main.pdf) and check the preprint layout (main_preprint.pdf)
set -e
cd "$(dirname "$0")"
sed 's/^ reprint,%/ preprint,%/' main.tex > main_preprint.tex
for f in main main_preprint; do
  pdflatex -interaction=nonstopmode $f >/dev/null || true
  bibtex $f >/dev/null || true
  pdflatex -interaction=nonstopmode $f >/dev/null || true
  pdflatex -interaction=nonstopmode $f >/dev/null || true
done
