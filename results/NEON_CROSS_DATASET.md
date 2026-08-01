# Cross-Dataset Generalization: NEON (Direction A)

**Date:** 2026-08-01
**Purpose:** Test whether the paper's central finding — a traditional operator
recovers true crowns beyond the deep detector's ceiling, and those crowns are
systematically small — **generalizes to a completely different dataset** than
VHRTrees (Mediterranean satellite). This is the biggest lever on the paper's
acceptance tier.

## Design (fair by construction)

- **Dataset:** NeonTreeEvaluation benchmark — 109 annotated RGB tiles, 3,762
  trees, across US NEON sites (SJER California oak savanna, TEAK Sierra conifer,
  NIWO Colorado subalpine, BART/HARV NE hardwood). Airborne, different biome and
  sensor than VHRTrees satellite. Box annotations (PASCAL-VOC).
- **Deep detector:** **DeepForest** (`weecology/deepforest-tree`), the
  field-standard *NEON-trained* tree detector. Using it on NEON means the deep
  detector is competent on its home data — **no domain-shift confound** that
  would unfairly inflate the traditional method's apparent contribution.
- **Traditional operator:** our faithful features (illumination-invariant colour
  + Gabor bank) → joint map → watershed → local maxima, unchanged from VHRTrees.
- Reproduce: `python -m improvement.neon.download` then
  `python -m improvement.neon.cross_dataset`.

## Result 1 — the finding REPLICATES: recovery beyond the detector, and it's small crowns

| | value |
|---|---|
| DeepForest coverage (center-in-box) | 78.4% |
| + traditional (union) | **94.2%** (+15.7 pp) |
| Beyond-ceiling recovered | 591 trees (15.7%) |
| Median crown area: DeepForest-reached | 1085 px² |
| Median crown area: **beyond-ceiling** | **567 px² (1.91× smaller)** |

**The core mechanism transfers to a different continent, sensor, and detector.**
On VHRTrees the beyond-ceiling crowns were 2.44× smaller than detected ones; on
NEON they are 1.91× smaller. In both datasets, the trees a strong deep detector
misses — and a traditional texture/local-maxima operator recovers — are
systematically the small ones. This is the paper's novel claim, and it is no
longer single-dataset. See `results/figs/fig8_neon_cross.png`.

## Result 2 — the honest boundary: the untuned PIPELINE does not transfer as a detector

Standalone one-to-one detection on NEON:

| method | precision | recall | F1 |
|---|---|---|---|
| DeepForest | 0.880 | 0.671 | 0.762 |
| Traditional (VHRTrees params, unchanged) | **0.250** | 0.798 | 0.381 |

The traditional operator's **precision collapses to 0.25** on NEON — it
over-fires on dense conifer canopy (NIWO/TEAK), producing many spurious peaks.
So while it *recovers* the small crowns (high recall 0.80), as a **standalone
detector** it is poor on NEON without recalibration, and the raw +15.7 pp union
figure is partly inflated by spurious peaks that happen to land on trees.

**We state this plainly**: the *phenomenon* (small crowns lie beyond the deep
detector and carry recoverable texture/intensity signal) generalizes; the
*specific traditional pipeline's parameters*, tuned on Mediterranean satellite
imagery, do **not** transfer unchanged to US airborne conifer forest. A
practitioner applying the method to a new biome must recalibrate the traditional
detector (crown scale, thresholds) and re-fit the precision gate — exactly the
per-biome adaptation our VHRTrees limitations section already anticipated.

## What this adds to the paper

- **Upgrades the central claim from "measured on VHRTrees" to "the mechanism
  replicates cross-dataset"** — on an independent benchmark, different biome and
  sensor, with the field-standard detector (not our own models). This directly
  answers the strongest reviewer critique.
- **Bounds the claim honestly**: it is the *finding* that generalizes, not the
  turnkey pipeline. This is a more credible and useful statement than an
  unqualified "it works everywhere," and it is backed by the precision numbers.

## Honest limitations

- Coverage is the generous center-in-box measure; the traditional detector's low
  precision means the union figure over-states a deployable gain. The
  defensible, conservative claim is the **mechanism** (1.91× smaller recovered
  crowns) + DeepForest's own ceiling (78.4%), not the raw +15.7 pp.
- We did not recalibrate the traditional pipeline or re-fit the gate to NEON;
  doing so (and reporting a fair post-recalibration F1) is natural follow-up.
- NEON tiles vary in size and GSD; matching gap (10 px) and crown-scale
  parameters were kept at VHRTrees values, which is part of why the untuned
  pipeline mismatches.
