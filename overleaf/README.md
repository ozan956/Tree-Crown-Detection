# Overleaf upload package

Self-contained LaTeX source for the manuscript. Verified to compile standalone
(latexmk → 12-page PDF, 0 undefined references).

## How to use on Overleaf
1. Zip this folder (or use `overleaf.zip` in the parent directory) and
   **New Project → Upload Project** on Overleaf.
2. Set the main document to `main.tex` (Overleaf usually auto-detects it).
3. Menu → Compiler: **pdfLaTeX**. Bibliography is BibTeX (`references.bib`);
   Overleaf runs it automatically. If citations show as `[?]`, hit Recompile
   once more (BibTeX needs a second pass).

## Contents
- `main.tex` — the manuscript (was article-v2.tex).
- `PRIMEarxiv.sty` — the arXiv-style class file it depends on.
- `references.bib` — 30 references; all `\cite` keys resolve.
- `media/` — the 8 figures the manuscript uses (7 result figures + the
  qualitative overlay). `fig_workflow.pdf` is vector; the rest are PNG.
- `main.bbl` / `main.pdf` — a prebuilt bibliography and the reference PDF
  (optional; Overleaf regenerates them).
