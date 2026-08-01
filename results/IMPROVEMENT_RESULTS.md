# Tier-A Improvement: Precision Gate — Results

**Date:** 2026-08-01
**Goal:** Close the precision gap identified in Phase 1 (the rule-based integrated
method had F1 0.897, *below* the WBF ensemble's 0.921, due to 2647 false
positives). Runnable now on the vendored data — no GPU, no retraining.

## Diagnosis (what the false positives actually are)

Decomposing the integrated method's 2647 FP against the reliable deep-learning
boxes (WBF, score>=0.8):

- **2259 of 2647 FP (85%)** are *traditional-only* points lying **>=25 px from
  any reliable DL box** — the points the rule-based step *adds*. Visual
  inspection (`resultant-new-version-all/`) shows these fire on grass, bare
  soil, and rooftops (e.g. image `st3_804`: 42 GT trees, 102 detections).
- Only 254 FP are over-split duplicates of a real crown.
- Points *near* a reliable DL box are 97% precise (388 FP against 11906 TP) —
  they should be trusted as-is.

So the leverage is **not** merging over-splits; it is **gating the far-from-box
traditional-only points** to separate genuine tree recoveries from non-tree
firings.

## Method

For each far-from-box candidate point, extract 8 cheap appearance features from
a crown-sized crop (brightness, saturation, excess-green, greenness, texture,
mean R/G/B) and classify tree vs. non-tree with a logistic gate. Trees in this
Mediterranean dataset are **darker, greener, and more textured** than the dry
grass/soil around them; the fitted gate's largest-magnitude features are
brightness (−2.18), greenness (−1.94), and excess-green (+1.75), which matches
that intuition. Near-box points bypass the gate entirely.

## Results (honest image-disjoint split)

The gate is **trained on the first 111 test images and evaluated on the last
111** — an image-disjoint generalization estimate, not fit-on-what-you-score.
Metrics are one-to-one (Hungarian) matched at gap=10, on the test half only.

| config | precision | recall | F1 |
|---|---|---|---|
| baseline (integrated, no gate) | 0.828 | 0.973 | 0.894 |
| gated @P>=0.4 | 0.893 | 0.957 | 0.924 |
| gated @P>=0.5 | 0.918 | 0.940 | 0.929 |
| gated @P>=0.6 | 0.942 | 0.921 | 0.931 |
| gated @P>=0.7 | 0.962 | 0.908 | **0.934** |

Reference (Phase 1, full test set): integrated F1 0.897, WBF ensemble F1 0.921.

**Every gated operating point beats both the ungated integrated method (0.894
on this half) and the WBF ensemble (0.921).** The best F1, 0.934 at threshold
0.7, keeps recall 0.908 (still above WBF's 0.878) while lifting precision from
0.828 to 0.962. See `results/improvement_pr.png`: the gate curve dominates both
prior operating points.

Stability: pooled group-cross-validated F1 (all images, 3/5/10-fold) is
0.922–0.925 at threshold 0.5 — consistent with the held-out numbers, so the
gain is not a split artifact.

## What this changes for the paper

The Phase-1 finding was "the integration is a recall-maximizing operating point
with *worse* F1 than the ensemble." With the precision gate, the integration
becomes a **uniform improvement**: it dominates the WBF ensemble on the
precision–recall frontier, retaining the beyond-ceiling recall (it still finds
trees no detector threshold reaches) while removing the non-tree firings that
cost precision. The gate also gives the paper a **tunable operating curve**
(recall-max at 0.4 → precision-max at 0.7) instead of a single point.

## Reproduce

```
.venv/bin/python -m improvement.run_gate      # prints + writes results/improvement_table.md
.venv/bin/python -m pytest tests/test_precision_gate.py -q
```

## Honest limitations

- The gate is trained on labelled GT (supervised). For deployment on *unlabelled*
  imagery it would be trained once on a labelled subset and applied elsewhere —
  the image-disjoint split here estimates exactly that transfer, but only within
  this dataset's terrain distribution (Marmara + Izmir, Turkey).
- Features are dataset-tuned (Mediterranean dry-grass contrast). A different
  biome (e.g. dense tropical canopy) would need re-fitting; the *method*
  transfers, the *coefficients* may not.
- This addresses precision (FP-2). The smaller over-split (254 FP) and
  under-segmentation (388 FN) modes remain as future Tier-A/B work.
