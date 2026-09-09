# Manuscript

```bash
cd paper
latexmk -pdf main.tex
```

`main.tex` is self-contained: the bibliography is a `thebibliography` environment, so no
`bibtex` pass is needed. `references.bib` is kept for a future switch to BibTeX.

## Rule

The manuscript reports **no experimental results**. Sections are explicitly separated
into motivation, mathematical method, computational method, validation protocol,
proposed experiments, and expected outcomes; the `Results` section contains only
placeholders naming the artifact each subsection will be populated from.

Do not write a number into `Results` that no run artifact contains. Do not describe an
expected outcome in a way that could be read as an observed one.
