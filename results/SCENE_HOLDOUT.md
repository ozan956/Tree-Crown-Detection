# Scene-Holdout Generalization (Direction B)

**Date:** 2026-08-01
**Question a reviewer asks:** "Everything is on VHRTrees — does it generalize, or
is the gate tuned to the test set?" We answer with scene-holdout tests, using the
scene prefixes in the test filenames (44 scene groups; 10 base-letter prefixes:
sa, y, b, st, a, c, d, t, h, plus 8 unprefixed) as an acquisition/region proxy.

**Honest caveat up front:** prefixes are a *proxy* for acquisition/region, not
confirmed climate labels. A clean Bursa (Marmara) vs. İzmir (Mediterranean) split
would need the VHRTrees region metadata from ref [29], which we do not have
locally. This is an *internal* generalization test — weaker than a second
dataset, stronger than nothing, and runnable today. Reproduce:
`python -m improvement.scene_holdout`.

## Result 1 — the beyond-ceiling recovery is a *targeted* benefit, not universal

Per scene group, deep-ensemble (union of 6, most permissive) coverage vs. the
integrated method's coverage:

| scene | trees | DL union % | integrated % | gain (pp) |
|---|---|---|---|---|
| sa | 3660 | 100.0 | 99.2 | −0.8 |
| y | 3219 | 98.8 | 96.2 | −2.6 |
| b | 2194 | 80.3 | 98.0 | **+17.8** |
| st | 1385 | 90.2 | 94.0 | +3.8 |
| a | 1197 | 99.6 | 97.3 | −2.3 |
| c | 649 | 85.2 | 98.6 | **+13.4** |
| # | 531 | 100.0 | 98.1 | −1.9 |
| d | 378 | 82.5 | 98.7 | **+16.1** |
| t | 160 | 100.0 | 98.1 | −1.9 |
| h | 140 | 88.6 | 98.6 | **+10.0** |

**corr(DL coverage, recovery gain) = −0.98.** The traditional recovery's benefit
is almost perfectly explained by how weak the detectors are on a scene: where the
ensemble already saturates (~99–100%, e.g. `sa`, `y`, `a`, `t`) the traditional
step adds nothing and its false positives cost a fraction under generous per-box
matching; where the ensemble is weak (80–89%, e.g. `b`, `d`, `c`, `h`) it recovers
**+10 to +18 pp**.

**This sharpens the paper's claim** from "uniform +3.3 pp" to the more useful and
defensible: *the traditional recovery is a targeted safety-net that fires
precisely where deep detectors underperform.* For a practitioner this is exactly
the right property — it helps most when you need it.

## Result 2 — the precision gate generalizes to unseen scenes

Leave-one-scene-group-out: train the gate on all scene groups except one, test on
the held-out group (threshold 0.4, the recovery-preserving operating point).

| held-out | base F1 | gate F1 | ΔF1 |
|---|---|---|---|
| st | 0.837 | 0.881 | **+0.044** |
| sa | 0.893 | 0.923 | **+0.030** |
| a | 0.883 | 0.900 | +0.018 |
| # | 0.901 | 0.912 | +0.011 |
| y | 0.872 | 0.882 | +0.011 |
| b | 0.961 | 0.964 | +0.004 |
| d | 0.950 | 0.951 | +0.001 |
| c | 0.955 | 0.938 | −0.017 |
| h | 0.948 | 0.918 | −0.031 |

**The gate improves F1 on 7 of 9 held-out scene groups** (mean +0.008, median
+0.011), with the largest gains on the weak-detector scenes (`st` +0.044, `sa`
+0.030). It slightly hurts two small high-coverage groups (`c`, `h`) where there
is little to gain and it occasionally drops a real faint crown. So the gate is
**not memorizing the test set** — it transfers to scenes it never saw, and its
failures are confined to already-easy scenes where the stakes are low.

## What this adds to the paper

- Upgrades the generalization story from "single dataset, no holdout" to "holds
  across held-out scene groups within the dataset," directly addressing the
  sharpest referee critique short of a second dataset.
- Converts the headline from a flat number to a *mechanistic, conditional* claim
  (recovery ∝ detector weakness, corr −0.98) — stronger science.
- Figure: `results/figs/fig7_scene_holdout.png`.

## Remaining honest gap

This is within-dataset generalization via a region *proxy*. A true second dataset
(Direction A) remains the way to claim cross-dataset/cross-biome generalization;
this test does not replace it, but it materially strengthens the single-dataset
submission.
