# `evaluation/` — rigorous re-evaluation module

This package is a GPU-free, from-scratch re-evaluation of the results in
*"Integrating Traditional and Deep Learning Methods to Detect Tree Crowns in
Satellite Images"* (arXiv:2507.01502). The paper's original evaluation
reported a single recall-style number (91.2% DL-alone → 96.0% after
rule-based integration) with no precision, no F1, no mAP, and no sensitivity
analysis. This module recomputes every method's results under a corrected,
one-to-one matching protocol, adds precision/F1/COCO mAP, reproduces the
paper's original number exactly (for the erratum record), and produces the
precision–recall figure that tests whether the rule-based integration step
does something a deep-learning ensemble cannot do by simply lowering its
score threshold.

It consumes a vendored, read-only snapshot of the paper's own detector
outputs and ground truth (see **Data provenance** below) — no detector is
re-run and no GPU is required.

For the results themselves and the honest interpretation, see
[`results/FINDINGS.md`](../results/FINDINGS.md). This README covers only
how to install, run, and navigate the code.

## Install

```bash
.venv/bin/pip install -r requirements-eval.txt
```

Dependencies: `numpy`, `scipy` (Hungarian/linear-sum-assignment matching),
`pycocotools` (COCO mAP), `matplotlib` (figure), `pytest` (tests). All
CPU-only, Python 3.12.

## Run

Generate the full results table (`results/table.md` + `results/table.csv`)
for all 5 detectors, the WBF ensemble, and the integrated method:

```bash
.venv/bin/python -m evaluation.evaluate_all
```

This takes roughly **80 seconds** — most of it is COCO mAP evaluation,
which runs independently for each of the 6 box-based methods (swin,
cascade_rcnn, detr, faster_rcnn, yolo, WBF).

Generate the centerpiece precision–recall figure
(`results/pr_curve.png`, WBF score-threshold sweep vs. the integrated
method's single operating point):

```bash
.venv/bin/python -c "from evaluation.sensitivity import wbf_pr_curve, integrated_point; from evaluation.figures import plot_pr_curve; plot_pr_curve(wbf_pr_curve(), integrated_point())"
```

Run the test suite:

```bash
.venv/bin/python -m pytest
```

This works as a bare `pytest` invocation now (`pytest.ini` scopes
collection to `tests/` and excludes the vendored third-party
`wbf_test-*/` upload folders, which contain an unrelated mmdetection test
suite that requires `torch`/`mmcv`/`mmdet`). The suite takes **about 6
minutes** — `tests/test_evaluate_all.py` calls `build_table()`, which runs
full COCO mAP for all 6 box methods, multiple times.

## Module map

| Module | Responsibility |
|---|---|
| `evaluation/io_utils.py` | Load COCO ground truth, detector JSONs, and the integrated-method's point output; convert `[x,y,w,h]` boxes to corner form; group detections/points by image; filter detections by score threshold (`dets_by_image`). |
| `evaluation/matching.py` | Geometry and matching primitives: `iou` (box IoU); `match_center_in_box` (the **legacy** per-center matcher that reproduces the paper's v1 protocol); `match_one_to_one_points` / `match_one_to_one_iou` (the **canonical** one-to-one matchers, Hungarian assignment via `scipy.optimize.linear_sum_assignment`). |
| `evaluation/metrics.py` | `pr_f1` (precision/recall/F1 from TP/FP/FN, zero-safe); `coco_map` (standard, threshold-free COCO AP/AP50/AP75 via `pycocotools`, over *all* detections regardless of score). |
| `evaluation/paper_protocol.py` | Reproduces the paper's **original, superseded v1 recall metric** (`paper_recall_points`, `paper_recall_boxes`) using the legacy matcher, for erratum/audit purposes only. `investigate_dl_baseline()` sweeps WBF score thresholds to investigate the paper's DL-alone 91.2% figure. |
| `evaluation/evaluate_all.py` | Main entry point. Builds the results table for every method under both protocols plus COCO mAP, and writes `results/table.md` / `results/table.csv`. |
| `evaluation/sensitivity.py` + `evaluation/figures.py` | `wbf_pr_curve` (WBF precision/recall at a sweep of score thresholds), `integrated_point` (integrated method's single precision/recall point), `gap_sensitivity` (integrated precision/recall/F1 as the matching gap varies); `plot_pr_curve` renders both onto `results/pr_curve.png`. |

## The two protocols

Every method is scored two ways:

1. **Legacy v1 (`match_center_in_box`, reproduction only).** Many-to-one:
   for each detection/tree center, walk the GT boxes in order and credit
   the *first* box the center falls inside (±`gap` pixels), then stop.
   This is exactly what the paper's original code did. It is
   order-dependent and **under-counts** distinct matched GT boxes when
   crowns overlap, because a center sitting in the overlap of two boxes
   can only ever credit one of them. Kept solely to reproduce the
   published 96.0% figure — do not use it for anything else. Surfaced in
   the results table as the `paper_recall` column.

2. **Strict one-to-one (`match_one_to_one_points` / `match_one_to_one_iou`,
   the headline protocol).** Each prediction is matched to at most one GT
   box and each GT box to at most one prediction, found by a global
   Hungarian assignment over a feasibility matrix (center-in-box for
   points, IoU ≥ 0.5 for boxes). This removes the overlap artifact in
   *either* direction and yields TP/FP/FN, from which precision, recall,
   and F1 are computed uniformly for every method — DL detectors, the WBF
   ensemble, and the integrated method alike. Surfaced in the results
   table as `strict_recall` / `strict_precision` / `strict_f1`.

COCO `AP`/`AP50` are a third, independent, threshold-free measure (all
detections, no score cutoff) computed only for the five box-emitting
detectors and WBF — the integrated method emits points, not boxes, so it
has no mAP and is marked `n/a`.

## Data provenance

All inputs are a vendored, read-only snapshot under `data/`, copied
verbatim from the project's original upload folders with record counts
verified against the fixed ground truth (222 images, 13552 boxes). Every
module in this package reads exclusively from `data/` — never from the
original `wbf_test-*` folders. See [`data/MANIFEST.md`](../data/MANIFEST.md)
for the exact source-path-to-`data/`-path mapping and verification method.
