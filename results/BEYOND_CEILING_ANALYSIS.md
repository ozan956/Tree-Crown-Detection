# The Novel Core: What Deep Detectors Miss and Traditional Cues Recover

**Date:** 2026-08-01
**This is the paper's defensible novel contribution** (see
`docs/research/2026-08-01-novelty-audit.md` for why the precision gate and the
hybrid pipeline are *not* novel — they are established techniques).

## The claim (conservative, benchmark-backed)

A traditional (non-learned) watershed/local-maxima step recovers true tree
crowns that lie **beyond the detection ceiling of a strong 6-model deep-learning
ensemble** — trees no detector proposes as a box at any confidence threshold —
and we characterize *why* they are missed.

## The ceiling is real (not a threshold artifact)

Generous per-box coverage (a GT box counts as reached if any prediction center
falls in it, gap=10 px), detectors at their **most permissive** thresholds:

| Source | GT trees reachable | % |
|---|---|---|
| Best single detector (DETR @min score) | 11,777 | 86.9% |
| WBF ensemble @min score | 12,532 | 92.5% |
| **Union of all 6 detectors @min score** (theoretical DL ceiling) | 12,762 | **94.2%** |
| Integrated (with traditional recovery) | 13,210 | **97.5%** |

- **Net gain over the full DL ceiling: +448 trees (+3.3 pp).**
- **Beyond-ceiling set: 692 GT trees (5.1%)** are hit by the integrated method
  but by *no* detector in the union; 244 of these are offset by trees the DL
  union catches that the traditional step misses, giving the +448 net.

The skeptic's objection — "just lower the detector threshold" — is answered
directly: every detector is already at its floor and unioned. Those trees are
simply not in any detector's proposal set.

## Why detectors miss them (the mechanism — this is what makes it science)

Median properties of GT crowns, by group:

| feature | DL-reachable (12,762) | beyond-ceiling (692) |
|---|---|---|
| crown area (px²) | 1924 | **788** (2.44× smaller) |
| width (px) | 43 | 28 |
| height (px) | 45 | 30 |
| brightness (HSV V) | 95 | 103 (brighter) |
| saturation | 86 | 60 (paler) |
| excess-green | 22.3 | 11.1 (fainter vegetation signal) |
| texture (grey std) | 26.8 | 32.4 (higher local contrast) |

**82%** of beyond-ceiling trees fall below the 25th percentile of detected-tree
crown area. The pattern is consistent and interpretable:

- **Small + spectrally faint:** the missed trees are small, pale, low-saturation
  crowns against bright dry-soil background. This is exactly where a
  confidence-trained detector is least certain and drops the box below any
  usable threshold.
- **But locally textured:** they retain higher local texture contrast than the
  detected population. The traditional Gabor/watershed/local-maxima step keys on
  *local texture and intensity extrema*, not learned appearance priors, so it
  fires where the detector's global appearance model does not.

**Mechanism statement for the paper:** *Deep detectors systematically miss
small, spectrally-faint tree crowns; a texture/local-maxima traditional operator
recovers a measurable fraction of them precisely because it does not depend on
the learned appearance priors that fail on faint small crowns.*

## Verification: are they real trees?

`results/beyond_ceiling_crops.png` shows 24 randomly-sampled beyond-ceiling
crops. They are unambiguously small tree crowns (dark/grey-green blobs on bright
soil), several faint or partially shadowed — not grass patches or annotation
errors. The recovered set is genuine.

## Honest limitations

- **Single dataset** (Marmara + İzmir, Turkey). The complementarity is
  quantified here; generalization to other biomes is future work and must be
  stated as such. The *mechanism* (small/faint crowns) is plausibly general, but
  the 3.3% magnitude is dataset-specific.
- The traditional step also *adds* false positives (addressed separately by the
  precision gate); the net-recall framing already accounts for the recovery
  trade, but the FP cost is real and reported in `IMPROVEMENT_RESULTS.md`.
- "Beyond-ceiling" is defined against *these six* detectors. A future detector
  trained specifically for tiny crowns could shrink the gap — the claim is about
  the complementarity of the current strong ensemble, not a permanent limit.

## What to claim, precisely

> On a 13,552-tree benchmark, a traditional local-maxima/watershed operator
> recovers 448 true crowns (3.3%) beyond the coverage of a 6-model deep-detector
> ensemble at its most permissive thresholds. These crowns are systematically
> small (2.44× below median detected-crown area; 82% below the detected 25th
> percentile) and spectrally faint, explaining why appearance-trained detectors
> miss them and a texture-based operator recovers them.

That sentence is defensible against the adversarial literature search and
reproducible from the vendored data.
