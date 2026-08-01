# Faithful Traditional Features — Before/After

**Date:** 2026-08-01
**Motivation:** The paper text (S2.1) claims "a bank of Gabor filters that
uniformly cover the spatial and frequency domains" (Jain & Farrokhnia 1991) and
"color invariants". The shipped code implemented neither: a **single** Gabor
kernel (`ksize=3`, one orientation `theta=pi/4`, one wavelength) and a **fixed
HSV green band** `[35,50,50]-[85,255,255]`. This is both an honesty gap (text vs
code) and a performance gap. This module (`improvement/features.py`) implements
what the text claims, and we measure the difference.

## What we implemented

- **Gabor bank** (`gabor_bank_energy`): 4 orientations (0/45/90/135 deg) x 5
  sqrt(2)-spaced radial center frequencies, zero-mean kernels (no brightness
  leak), with the Jain-Farrokhnia post-processing (magnitude -> tanh
  nonlinearity -> Gaussian energy pooling), averaged into one texture map.
- **Illumination-invariant color** (`color_invariant`): excess-green in
  normalized-rgb (chromaticity) space `2g - r - b`, which is invariant to
  overall brightness/shading — a genuine colour *invariant*, unlike a fixed HSV
  band that breaks under the sun/shadow variation the paper itself motivates.

## Measurement (honest, pixel-level, GPU-free)

We label every pixel tree/non-tree from the GT boxes and score each feature map
as a tree-pixel classifier (ROC-AUC). Higher = better tree/background
separation. Mean over ~37-40 test images:

| feature | old (shipped) | new (faithful) |
|---|---|---|
| colour | 0.646 | **0.857** |
| Gabor texture | 0.652 | 0.672 |
| **joint map** | **0.743** | **0.782** (equal weight) |

The colour invariant is the big win (+0.21 AUC); the Gabor bank is a modest
gain (+0.02) — texture alone is a weak tree cue on this terrain.

## Weighting (data-driven, validated on held-out split)

Because the new colour feature (0.857) is far stronger than texture (0.672), the
paper's equal 0.5/0.5 weighting *dilutes* the better cue. Sweeping the colour
weight and **validating on an image-disjoint split** (weight chosen on a tuning
half, scored on an unseen half):

- Tuning half independently selects **w_color = 0.9**.
- Held-out test half: **AUC 0.851 @ w_color=0.9** vs 0.798 @ equal weight.
- The Gabor bank still adds a small complementary gain over colour-alone, so it
  is retained (not dropped) — `w_texture = 0.1`.

**Net: joint-map tree/background AUC 0.743 (old) -> 0.851 (new, held-out), a
+0.108 (+15%) improvement**, and the method now matches the paper's own text.

## Honest scope

- This measures **feature quality** (pixel-level separability), not end-to-end
  detection F1 — re-running the full watershed+integration pipeline with the new
  features needs the original pipeline scripts (Phase-3 packaging work). Better
  features feed a better joint map feed better local-maxima, so this is a
  necessary upstream improvement, but the downstream F1 delta is not measured
  here and should not be claimed until the pipeline is re-run.
- Weights are tuned on this dataset's terrain; the *method* (invariant colour +
  Gabor bank) transfers, the exact 0.9 may not.
- For the paper: this fixes the "bank"/"color invariants" overclaim (the text is
  now true of the code) and shows the faithful features are measurably better —
  a clean honesty-plus-improvement result.

## Reproduce

```
.venv/bin/python -m improvement.compare_features   # prints the AUC table
.venv/bin/python -m pytest tests/test_features.py -q
```
