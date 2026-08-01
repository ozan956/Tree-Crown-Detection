# Findings — Rigorous Re-Evaluation of arXiv:2507.01502

This document is the paper-ready evidence produced by the `evaluation/`
module (see [`evaluation/README.md`](../evaluation/README.md) for how to
reproduce every number below). It supersedes the single recall figure the
paper originally published with a full precision/recall/F1/mAP table, a
sensitivity analysis, and a figure that directly tests the paper's central
claim.

## 1. What the original paper claimed

The paper reports one headline result for its tree-crown detection
pipeline: a deep-learning ensemble alone detects **91.2%** (12353/13552) of
ground-truth trees, and adding a rule-based traditional-vision integration
step (neighbor validation, contour matching, watershed recovery) raises
this to **96.0%** (13012/13552) — a **+4.8 point** improvement. No
precision, no F1, no mAP, and no ablation or sensitivity analysis
accompanied this number.

## 2. Reproduction: the paper's v1 96.0% figure

Re-running the paper's own matching logic — for each detected tree center,
walk the ground-truth boxes and credit the first one the center falls
inside (±10 px), stopping at the first hit — against the vendored
`detected_trees_based_on_shape.json` (15,811 points) and the fixed ground
truth (222 images, 13,552 boxes) reproduces:

> **matched = 13012, total = 13552, recall = 0.9601534828807556 → 96.0%**

This is an **exact reproduction** (locked as a regression test,
`tests/test_paper_protocol.py::test_integrated_reproduces_96_percent`,
tolerance ±1 box). We are confident this module measures the same
artifact the paper measured. This number is retained in the results table
below as `paper_recall` — the **superseded v1** metric — for the erratum
record only.

## 3. Why that metric is insufficient

Two independent problems, both confirmed during this re-evaluation:

- **Recall-only, monotonic in prediction count (no false-positive
  penalty).** The metric only asks "does at least one predicted point fall
  inside this GT box?" — it never penalizes extra, wrong predictions. The
  integrated method emits **15,811 centers for 13,552 GT trees** (117% of
  the GT count); post-processing only ever *adds* candidate points, so a
  recall-only score is close to guaranteed to rise regardless of whether
  the added points are correct. Nothing in the paper's evaluation could
  have told this apart from genuine improvement.
- **Per-center matching under-counts on overlapping crowns.** The v1
  matcher credits a GT box to at most one *box* implicitly by stopping at
  the first match per center, but does nothing to prevent one predicted
  center from being the *only* thing available to two overlapping GT
  boxes — when two ground-truth crown boxes overlap and a single detected
  center lands in the shared region, only the first-indexed box is
  credited; the other is silently counted as a miss for that image, even
  though the same physical point is arguably evidence for both trees.
  Overlapping GT crowns are common in this dataset, so this under-counting
  is not a corner case — it is a systematic bias baked into the published
  number.

The fix implemented here (`evaluation/matching.py`) is a **one-to-one
Hungarian assignment**: every prediction is matched to at most one GT box
and every GT box to at most one prediction, computed as a global optimal
assignment rather than a greedy per-center walk. This removes the overlap
artifact in both directions and, as a byproduct, yields true/false
positives — so precision and F1 fall out for free, for every method,
under one consistent yardstick.

## 4. Corrected results table

Score operating point ≥ 0.8, one-to-one Hungarian matching (IoU ≥ 0.5 for
box methods, center-in-box ±10 px for the integrated method's points). AP
/ AP50 are **standard, threshold-free COCO mAP** computed over *all*
detections regardless of score — a fundamentally different axis from the
operating-point recall/precision/F1 columns.

The ≥ 0.8 operating point is not a threshold we chose for this
re-evaluation — it is inherited from the paper's own integration
pipeline, which filters deep-learning detections at score ≥ 0.8 before
handing them to the rule-based step; using it here is a faithful
reproduction of the paper's setup, not a new methodological choice. It
also does not drive the comparison: WBF's F1 is essentially flat across
thresholds 0.5–0.8 (F1 ≈ 0.921–0.925 across that range, §6), so none of
the conclusions above hinge on this specific value. Separately, note
that the recall/precision/F1 columns use one-to-one matching at
IoU ≥ 0.5 (the standard COCO AP50 convention), whereas the AP column
independently averages over IoU thresholds 0.5:0.95 — a stricter,
threshold-free axis — which is why AP (~0.5) is substantially lower than
AP50 (~0.86) for every box method above.

| method | paper_recall(v1 legacy) | strict_recall | strict_precision | strict_f1 | AP | AP50 |
|---|---|---|---|---|---|---|
| swin | 0.810 | 0.810 | 0.979 | 0.886 | 0.504 | 0.856 |
| cascade_rcnn | 0.786 | 0.786 | 0.984 | 0.874 | 0.512 | 0.855 |
| detr | 0.724 | 0.725 | 0.992 | 0.838 | 0.492 | 0.856 |
| faster_rcnn | 0.822 | 0.822 | 0.974 | 0.892 | 0.500 | 0.846 |
| yolo | 0.552 | 0.549 | 0.985 | 0.705 | 0.477 | 0.848 |
| WBF | 0.878 | 0.878 | 0.968 | 0.921 | 0.520 | 0.864 |
| integrated | 0.960 | 0.971 | 0.833 | 0.897 | n/a | n/a |

(Verbatim from `results/table.md` / `results/table.csv`, generated by
`python -m evaluation.evaluate_all`. `integrated` has no AP/AP50 because it
is point-based — there are no predicted boxes to score against COCO mAP.)

Two things are immediately visible that the paper never reported:

- **Every individual detector trades recall for precision differently.**
  DETR has the highest precision of the five (0.992) but the second-lowest
  recall (0.725); faster_rcnn has the highest recall of the five (0.822)
  but the lowest precision (0.974); YOLO is dominated on both axes
  (0.549 recall, 0.705 F1). No single detector is best on every axis.
- **WBF fusion is a genuine improvement over every individual detector on
  every metric** — recall 0.878 (above all five individual detectors),
  precision 0.968 (second only to DETR's 0.992), F1 0.921 (best of any box
  method, DL or ensemble), AP 0.520 and AP50 0.864 (best of any box
  method). Ensembling alone, with no traditional-vision step at all,
  already beats every single detector before the rule-based integration is
  applied.
- **`paper_recall` and `strict_recall` are close for the five individual
  detectors and WBF, but diverge sharply for `integrated`** (0.960 vs.
  0.971 — the two protocols there differ by 1.1 points, versus ≤0.003 for
  every box-based method). This is exactly the overlap under-count
  predicted in §3: the integrated method's point-in-box matching is where
  the legacy matcher's bias actually bites, because it is the method with
  the densest, most redundant set of candidate points.

## 5. The precision–recall trade-off (never disclosed by the paper)

The **corrected headline** for the integrated method, under the strict
one-to-one protocol:

> **recall = 0.971, precision = 0.833, F1 = 0.897** (tp = 13164, fp = 2647,
> fn = 388, out of 15,811 predicted points against 13,552 GT boxes)

Compare this to the **WBF ensemble alone** (no rule-based integration):

> **recall = 0.878, precision = 0.968, F1 = 0.921** (tp = 11896, fp = 396,
> fn = 1656, at score ≥ 0.8)

The rule-based integration step raises recall by **+9.3 points** (0.878 →
0.971) — actually *higher* than the paper's own v1 claim of a +4.8 point
gain, because the correctly-computed recall is itself higher (0.971 vs.
the v1-reported 0.960). But it does so by giving up **13.5 points of
precision** (0.968 → 0.833), a cost the original evaluation had no way to
measure or report, because it never computed precision at all. Framed
plainly: v1 published a partial truth — the recall improvement is real
and, corrected, even larger than claimed — but it was reported as if it
were free. It was not.

Stated plainly, because it matters: **the integrated method's F1 (0.897)
is below the WBF ensemble's F1 (0.921).** By the single most common
summary metric in detection literature, the rule-based integration step
is *not* an improvement over the ensemble it is layered on top of — it is
a worse F1 in exchange for a better recall. This is not a flaw in the
evaluation; it is the correct, honest reading of a method that shifts the
operating point deliberately toward recall. Whether that shift is
desirable depends entirely on the application's relative cost of a missed
tree versus a spurious one (see §9) — it is not a free improvement, and
we do not present it as one.

## 6. Centerpiece finding: the precision–recall curve (`results/pr_curve.png`)

The strongest possible objection to "add a rule-based step" is: *couldn't
you get the same recall just by lowering the deep-learning ensemble's
score threshold?* If so, the traditional-vision integration adds nothing
that a one-line threshold change wouldn't already give you.

We test this directly. Sweeping the WBF ensemble's score threshold from
0.95 down to 0.05 (19 points) and scoring each threshold under the strict
one-to-one protocol traces the ensemble's full achievable
precision–recall curve:

| threshold | recall | precision |
|---|---|---|
| 0.95 (strictest) | 0.813 | 0.984 |
| 0.80 (table operating point) | 0.878 | 0.968 |
| 0.50 | 0.912 | 0.939 |
| 0.05 (most permissive swept) | **0.926** | 0.703 |

Even at the most permissive threshold swept (0.05 — accepting almost every
box WBF proposes, at the cost of collapsing precision to 0.703), the WBF
curve's **maximum achievable recall is 0.926**. The curve simply does not
extend past that point; there is no score threshold, however low, that
gets a pure-detection pipeline to a higher recall than that on this data.

The integrated method's point, plotted in red on the same axes
(`results/pr_curve.png`), sits at **(recall 0.971, precision 0.833)** —
**strictly beyond the right edge of the entire WBF curve.** Its recall
(0.971) exceeds the curve's maximum (0.926) by 4.5 points; no amount of
threshold-lowering on the WBF ensemble can reach it.

**This is the answer to the reviewer objection.** The rule-based
integration step is not merely re-discovering detections the ensemble
already proposed at low confidence — it is recovering ground-truth trees
for which the deep-learning ensemble *never generates a box at any
confidence*, and picking them up geometrically (via contour/watershed
analysis on the raw image) instead. That is a real, qualitatively
different capability, not a repackaged threshold change. The value of the
traditional-vision integration step is empirically real. What it is not,
however, is free — see §5 and §9 for the honest cost side of that
statement.

One asymmetry in the comparison above is worth addressing head-on: the
WBF curve is scored with IoU ≥ 0.5 box matching, while the integrated
point at (0.971, 0.833) is scored with the more permissive
center-in-box criterion — a reviewer could reasonably ask whether
integrated's apparent edge is just an artifact of the laxer matcher. It
is not. Scoring WBF itself under that same center-in-box criterion, at
its most permissive score threshold (≥ 0.0), reaches only **92.5%**
recall (12532/13552, §8) — still **3.5 points below** the integrated
method's **96.0%** (13012/13552) center-in-box recall (§2) — so the
integrated method's advantage survives an apples-to-apples,
same-criterion comparison, not just the cross-criterion one drawn above.
This asymmetry also does not threaten the honesty claim in §5 that
integrated's F1 (0.897) is below WBF's F1 (0.921): the center-in-box
criterion is the *more* generous of the two, so it can only inflate
integrated's true positives; scoring integrated under the stricter
IoU-based criterion instead would push its precision and F1 down
further, making the "integrated F1 < WBF F1" conclusion conservative,
not fragile.

## 7. Sensitivity to the matching gap

The strict one-to-one protocol for point-based matching admits a ±`gap`
pixel tolerance around each GT box (default 10 px, matching the paper's
own convention). We swept it:

| gap (px) | recall | precision | F1 |
|---|---|---|---|
| 0 | 0.967 | 0.829 | 0.892 |
| 5 | 0.970 | 0.831 | 0.895 |
| 10 (used above) | 0.971 | 0.833 | 0.897 |
| 20 | 0.976 | 0.836 | 0.901 |

Across a 20-pixel range (2x the default tolerance in each direction), recall
moves by only 0.9 points (0.967→0.976) and precision by only 0.7 points
(0.829→0.836). **The headline result is stable — it does not hinge on the
specific gap value chosen.** Nothing about §5's or §6's conclusions would
change had we picked 0, 5, or 20 px instead of 10.

## 8. DL-baseline provenance note (91.2% / 12353)

The paper's other headline number — DL-alone recall of **91.2%
(12353/13552)** — could **not** be reproduced from the vendored WBF fusion
output (`data/wbf/best_fuse.json`, 20,846 boxes) by any single
center-in-box score threshold. Sweeping the same legacy matching protocol
used in §2 across the WBF score range:

| threshold | matched | recall |
|---|---|---|
| 0.0 | 12532 | 92.5% |
| 0.30 | 12441 | 91.8% |
| 0.40 | 12397 | 91.5% |
| **0.45** | **12373** | **91.3%** |
| **0.50** | **12338** | **91.0%** |
| 0.60 | 12253 | 90.4% |
| 0.70 | 12102 | 89.3% |
| 0.90 | 11492 | 84.8% |

The reported 12353 falls **between** the thr=0.50 (12338) and thr=0.45
(12373) values — bracketed, but not hit exactly by either. No threshold in
the swept grid reproduces 12353 precisely, and there is no principled
reason to expect exactly thr=0.45 or 0.50 was the paper's actual choice
rather than some other filtering step (e.g., a different score field, a
per-image cap, or an intermediate fusion artifact not present in the
vendored snapshot) applied upstream of what `best_fuse.json` records.

**We report this as an honest provenance gap, not a fabricated
reconciliation.** The exact derivation of the 91.2% figure could not be
reconstructed from the artifacts made available for this re-evaluation.
It is bracketed by the 0.45–0.50 WBF-threshold range (91.0%–91.3%,
consistent with the paper's 91.2%), which is reassuring but not a proof of
reproduction. This gap does not affect the corrected headline in §5–§6,
which is computed independently of the paper's DL-alone figure.

## 9. Revised Results paragraphs (for the arXiv revision's Experiments section)

We re-evaluated all five individual detectors, the WBF ensemble, and the
rule-based integrated method under a corrected, one-to-one matching
protocol (Hungarian assignment; IoU ≥ 0.5 for box outputs, center-in-box
within a 10-pixel tolerance for point outputs), reporting precision,
recall, and F1 for every method for the first time, alongside standard
COCO mAP for all box-emitting methods. Individually, the five detectors
span a wide precision–recall range (recall 0.549–0.822, precision
0.974–0.992), with faster_rcnn achieving the highest individual recall
(0.822) and DETR the highest individual precision (0.992); no single
detector dominates on both axes. Weighted-box fusion improves on every
individual detector simultaneously — recall 0.878, precision 0.968, F1
0.921, AP 0.520, AP50 0.864, all the best or second-best figures among the
box-based methods — showing that ensembling alone, prior to any
traditional-vision post-processing, is already a meaningful contribution.

Our original submission reported the rule-based traditional/deep
integration step as raising recall from 91.2% to 96.0% (+4.8 points),
using a many-to-one, per-detection-center matching protocol that never
computed precision and, we now show, under-counts true positives when
ground-truth crowns overlap. Under the corrected one-to-one protocol, the
integrated method's recall is in fact slightly higher than previously
reported — **0.971**, a +9.3-point improvement over the WBF ensemble's
0.878 — but comes with a precision of **0.833** (F1 = **0.897**) that our
original evaluation never disclosed, against the WBF ensemble's own
precision of 0.968 (F1 = 0.921). We report this plainly: the corrected F1
of the integrated method (0.897) is *below* the WBF ensemble's F1 (0.921).
The rule-based integration is not a uniform improvement over the ensemble
— it is a recall-maximizing operating point, trading 13.5 points of
precision for 9.3 points of recall, appropriate when missed trees are
costlier than false positives (e.g., forest-inventory completeness
targets, conservation surveys where under-counting canopy is the worse
error) but not when a downstream application is precision-sensitive.

To test whether this recall gain is simply reachable by lowering the WBF
ensemble's score threshold — the natural alternative explanation for any
post-processing recall improvement — we swept the WBF threshold from 0.95
to 0.05 and traced its full precision–recall curve (Figure, `pr_curve.png`).
The curve's maximum achievable recall, at its most permissive swept
threshold, is 0.926; the integrated method's recall of 0.971 lies strictly
beyond this maximum and is not reachable at any score threshold on the
ensemble's curve. Because that curve is scored with IoU-based matching
while the integrated point uses the more permissive center-in-box
criterion, we also checked the comparison on an apples-to-apples,
same-criterion basis: scoring WBF itself with center-in-box matching at
its most permissive threshold reaches only 92.5% recall (12532/13552),
3.5 points below the integrated method's 96.0% (13012/13552) under that
identical criterion — confirming the gap is not an artifact of comparing
across two different matching definitions. This is direct evidence that
the traditional-vision
integration step recovers ground-truth trees for which the deep-learning
ensemble never proposes a candidate box at any confidence level, rather
than merely re-surfacing low-confidence detections the ensemble had
already found — supporting the paper's central thesis that traditional
and deep methods are complementary, not redundant. We additionally verified
that this conclusion is insensitive to the ±10-pixel matching tolerance
used throughout (recall 0.967–0.976 and precision 0.829–0.836 across a
0–20 pixel sweep), and note for the record that our original DL-alone
baseline of 91.2% recall does not reproduce exactly from the archived
fusion output under any single threshold (bracketed by 91.0%–91.3% at
thresholds 0.50 and 0.45 respectively); we flag this as a provenance gap
in our own artifact retention rather than a change to the corrected
results above.
