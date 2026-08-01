# Paper v2 draft sections

`v2-sections-draft.tex` contains reframed sections to update `article-v1.tex`
for a revision, based on the Phase-1 re-evaluation, the novelty audit, and the
Tier-B improvements. **Not auto-merged** — review and paste section by section.

## What changes and why

| Section | v1 | v2 | Reason |
|---|---|---|---|
| Abstract | "novel rule-based approach", recall framing | leads with quantified beyond-ceiling recovery; reports precision too | Audit: rule-based/WBF/appearance-filter are established; the measured complementarity is the real novelty |
| Contributions | (none stated) | explicit list, honest about novel vs established | Reviewers expect it; protects against overclaim |
| §5.3 DL results | "12353 / 91.2%", no precision | full P/R/F1 + mAP table under one-to-one matching | Recall-only metric hid false positives |
| §5.4 rule-based | "+4.8%", figure-based claims | quantified beyond-ceiling (+448/3.3pp) + mechanism table + precision gate | Turns anecdote into measured, mechanistic result |
| Limitations | (none) | new subsection | Single-dataset scope, ceiling definition, supervised gate |

## Provenance of every number

All values are produced by the committed code and cross-checked:
- Table `tab:results`: `results/table.md` (+ `IMPROVEMENT_RESULTS.md` for the gate row) — `python -m evaluation.evaluate_all`, `python -m improvement.run_gate`.
- Table `tab:beyond` + the 448/692/94.2%/97.5% figures: `results/BEYOND_CEILING_ANALYSIS.md` — `python -m improvement.beyond_ceiling`.
- The faithful-features basis for the "Gabor bank / illumination-invariant colour" claim: `results/FEATURE_ANALYSIS.md`.

## Not yet reflected in the paper (decide before submission)

- The gate and faithful-features are **post-hoc on saved outputs**; the
  end-to-end pipeline has not been re-run with the faithful features (Phase-3).
  The draft only claims the gate's F1 (measured) and the features' pixel-AUC
  (measured), not a re-run downstream F1 — keep it that way unless the pipeline
  is re-run.
- A second dataset would upgrade the complementarity claim from
  "measured on VHRTrees" to "generalises". Currently scoped honestly as a
  single-dataset result in the Limitations subsection.
- Figures: the PR-curve (`results/pr_curve.png`), the improvement PR
  (`results/improvement_pr.png`), and the beyond-ceiling crop montage
  (`results/beyond_ceiling_crops.png`) are ready to drop in as new figures.
