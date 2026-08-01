# Modern Baseline: Learned Density-Map Counter

**Date:** 2026-08-01
**Purpose:** Answer the reviewer question "why not a modern density-map /
crowd-counting method?" with a fair, trained comparison rather than an
assertion. Density-map regression is the technique the 2024-25 literature uses
for dense small-tree detection (Konttila 2025; Gominski et al.), so it is the
right modern baseline.

## What we trained

A CSRNet-style density-map network (VGG16 frontend, dilated-conv backend,
1/8-resolution output) — the canonical crowd-counting architecture — on the
VHRTrees train split (773 images present, ImageNet-pretrained frontend), on an
RTX 2070. Trees are encoded as sum-preserving Gaussian density; the integral of
the predicted map is the tree count. Best validation count MAE 4.30 after 15
epochs. Code: `improvement/density/` (data, model, train, evaluate). Test-time
peak-extraction thresholds were tuned for the best detection F1 (the fair,
strongest setting), not left at defaults.

## Results on the 222-image test set (one-to-one matching, gap=10)

| Method | Precision | Recall | F1 | Count MAE |
|---|---|---|---|---|
| Density-map counter (this baseline) | 0.888 | 0.834 | 0.860 | **4.63** |
| WBF ensemble | 0.968 | 0.878 | 0.921 | — |
| Integrated (this work) | 0.833 | 0.971 | 0.897 | — |
| **Integrated + gate (recovery-preserving)** | 0.893 | 0.957 | **0.924** | — |

## Findings (honest, and they strengthen the paper)

1. **The density-map method is a strong counter but a weaker detector.** Its
   count MAE (4.63 trees/image, ~8% of the median 53/image) is good — counting
   is what it is designed for. But for *individual tree localization*, its best
   detection F1 is 0.860, clearly below the WBF ensemble (0.921) and our
   integrated method (0.924).
2. **This is robust to tuning.** Sweeping the peak-extraction threshold trades
   precision for recall along a curve whose maximum F1 is 0.860; no setting
   reaches the ensemble, let alone the integrated method. So the gap is not an
   artifact of a bad decode threshold.
3. **Why:** density regression deliberately blurs individual instances into a
   continuous surface to count robustly; recovering discrete, well-localized
   centers from that surface is lossy, especially where crowns merge. Detection
   + fusion + traditional recovery keeps instance identity, which is what the
   one-to-one detection metric rewards.

**Takeaway for the paper:** the modern density-map approach does not dominate
detection here; it is complementary (excellent for counting, weaker for
localization). This justifies the detection-centric design and closes the
"why not modern methods?" gap with evidence. If the application is *counting*
rather than *localization*, the density-map method is competitive — we say so.

## Honest limitations

- A larger density backbone, more epochs, or multi-scale test augmentation could
  raise the density-map F1 somewhat; our 15-epoch RTX-2070 model is a competent
  but not exhaustively-tuned baseline. The ~6-point F1 gap to the integrated
  method is large enough that modest tuning is unlikely to close it, but we do
  not claim the density-map ceiling has been reached.
- Both methods are trained/evaluated on the same single dataset; cross-dataset
  behavior is future work (see the novelty audit).

## Reproduce

```
.venv/bin/python -m improvement.density.train      # ~20 min on an 8GB GPU
.venv/bin/python -m improvement.density.evaluate   # writes results/density_results.md
```
