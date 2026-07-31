# Design: Strengthening "Integrating Traditional and Deep Learning Methods to Detect Tree Crowns in Satellite Images"

**Date:** 2026-07-31
**Paper:** arXiv:2507.01502 (Durgut, Kallfelz-Sirmacek, Ünsalan)
**Author of this work:** Ozan Durgut (repo owner)
**Goal:** Act as a rigorous reviewer, then fix the paper's weaknesses with runnable code and paper-ready results.

---

## 1. Problem statement (reviewer findings)

The paper reports a single headline result: deep learning alone detects **91.2%** (12353/13552) of test trees, and the
rule-based integration of traditional + deep methods detects **96.0%** (13012/13552) — a **+4.8%** improvement. During
review the integrated **96.0%** was **reproduced exactly** by re-running the paper's own matching logic on the provided
`detected_trees_based_on_shape.json` (13012 unique GT boxes matched), and the DL-alone **91.2%** is arithmetically
consistent (12353/13552 = 91.15%) though not yet independently recomputed from the detector outputs — Phase 1's
`paper_protocol.py` will close that gap. The artifact is sound; the *evaluation methodology and presentation* are the
problem.

Ranked weaknesses:

- **W1 (Major): Recall-only metric, monotonic in prediction count.** A GT tree counts as "detected" if ≥1 predicted
  center falls within its box (±10px). False positives are never penalized. The integrated method emits **15,811 centers
  for 13,552 GT trees**; only ~86% of centers land in a box. Because post-processing *adds* candidate points, the metric
  can essentially only increase — so +4.8% is close to guaranteed by construction regardless of correctness.
- **W2 (Major): No precision / F1 / mAP.** The MMDetection configs already compute `CocoMetric` (mAP) and all detector
  outputs + GT exist in COCO format, yet only the custom recall number is reported.
- **W3 (Major): No ablations, no sensitivity, single run.** Individual rule contributions (neighbor validation, contour
  matching, watershed recovery) are never isolated. Magic constants (score≥0.8, gap=10, required_nearby∈{3,5},
  threshold∈{40,180}, tolerance∈{0.02,0.2}) have no sensitivity analysis.
- **W4 (Moderate): Text overclaims vs. code.** "Color invariants" = fixed HSV green threshold; "Gabor bank with
  systematic [Jain–Farrokhnia] selection" = a single Gabor kernel (one θ, ksize 3). Loose matching double-counts (raw
  inside-count 13,587 > unique matched 13,012).
- **W5 (Moderate): Reproducibility.** Public repo has 3 scripts and no evaluation. Real pipeline is hardcoded to
  `C:/wbf_test/...`, duplicated across `beril_work.py` / `beril_work_multip.py` / `refactored-try.py` / `NEW_VERSION2.py`,
  no requirements/entry point.
- **W6 (Minor): Thin external baselines** (only own DL models; no DeepForest / local-maxima / template-matching).

## 2. Scope and priority

Chosen: **all three phases, in priority order**; deliverables are **both reproducible code in this repo and paper-ready
tables/text**.

- **Phase 1 — Rigorous re-evaluation** (this spec's focus; runnable now, no GPU). Fixes W1, W2, W3.
- **Phase 2 — Method improvements** (separate spec later). Fixes W4.
- **Phase 3 — Reproducibility & packaging** (separate spec later). Fixes W5.

W6 is acknowledged as future work unless an off-the-shelf baseline (DeepForest) proves cheap to add in Phase 1.

## 3. Data inventory (verified present)

| Artifact | Canonical path (chosen) | Content |
|---|---|---|
| GT annotations (COCO) | `wbf_test-…-002/wbf_test/beril-ozan-cem-work/new_idea/_annotations.coco.json` | 222 images, 13552 boxes, 1 class |
| GT (flat list form) | `wbf_test-…-001/wbf_test/test_notation.json` | 13552 boxes, used by original scripts |
| Swin dets | `…-001/wbf_test/swin/bbox_swin.json` | 13387 |
| Cascade-RCNN dets | `…-001/wbf_test/cascade-rcnn/bbox_rcnn.json` | 12894 |
| DETR dets | `…-001/wbf_test/detr/bbox_detr.json` | 22200 |
| Faster-RCNN dets | `…-001/wbf_test/faster_rcnn/bbox_fasterrcnn.json` | 12610 |
| YOLO dets | `…-002/wbf_test/yolo/bbox_yolo.json` | 14084 |
| WBF fusion (consumed by integration) | `…-001/wbf_test/beril-ozan-cem-work/new_idea/best_fuse.json` | 20846 boxes, score 0.005–1.0 |
| Integrated output (points) | `…-001/wbf_test/beril-ozan-cem-work/new_idea/detected_trees_based_on_shape.json` | 222 imgs, 15811 centers |
| Test images | present across zips | all 222 GT images physically found |

All coordinates are in the 640×640 image frame. Detector JSONs use COCO `[x,y,w,h]` bbox + `score` + `category_id`.
Integrated output uses `{image_id, tree_x, tree_y}` points.

**Path robustness:** the three `wbf_test-*` folders overlap with near-duplicate files. Phase 1 code will resolve inputs
through a single `paths.py` resolver that (a) prefers an explicit path, (b) falls back to a glob across the zips, and
(c) asserts the file hash/shape (e.g. detector count) so we never silently read the wrong copy.

## 4. Phase 1 architecture

New top-level `evaluation/` package in the repo. Each module has one clear responsibility, is independently testable, and
depends only on `numpy`, `scipy` (Hungarian), `pycocotools`, and `matplotlib`.

```
evaluation/
  paths.py            # resolve + verify canonical input files across the wbf_test-* folders
  io_utils.py         # load COCO GT, load detector JSONs, reduce boxes->centers, group by image
  matching.py         # box–box IoU matching; point-in-box one-to-one (Hungarian) matching
  metrics.py          # precision/recall/F1 from TP/FP/FN; COCO mAP wrapper (pycocotools)
  paper_protocol.py   # reproduces 91.2% / 96.0% EXACTLY (regression lock on the original metric)
  evaluate_all.py     # main entry: produces results table (Markdown + CSV) for all methods
  sensitivity.py      # score-threshold sweep -> PR curve + integrated point; tolerance/gap sweeps
  figures.py          # PR-curve-with-integrated-point plot; saved to results/
  requirements-eval.txt
  README.md
tests/
  test_matching.py    # synthetic cases: perfect match, double-count, off-by-one, empty
  test_paper_protocol.py  # asserts reproduction == 91.2% / 96.0% within tolerance
results/                # generated CSV/MD/PNG (git-tracked so paper can cite them)
```

### 4.1 Data flow

```
GT (COCO) ─┐
           ├─► io_utils.load ─► matching ─► metrics ─► evaluate_all ─► results/table.{md,csv}
detectors ─┘                        │
WBF ───────────────────────────────┤
integrated points ─────────────────┘
                                    └─► sensitivity ─► figures ─► results/pr_curve.png
```

### 4.2 The two-protocol reconciliation (core intellectual fix for W1)

1. **Original protocol** (`paper_protocol.py`): many-to-one, center-in-box ±gap, recall = unique GT boxes matched / total
   GT. Must reproduce 91.2% (DL/WBF) and 96.0% (integrated) — a regression test locks this so we prove we measure the
   *same system* the paper did.
2. **Strict protocol** (`matching.py` + `metrics.py`): one-to-one greedy/Hungarian matching (each prediction and each GT
   used at most once), yielding TP/FP/FN → **precision, recall, F1**. Applied uniformly to every method by reducing
   boxes to their centers, so DL-alone, WBF, and integrated share one yardstick.
3. **COCO mAP** (`metrics.py`): AP / AP50 / AP75 via `pycocotools` for the box methods (5 detectors + WBF). The integrated
   method is point-only, so it appears in the strict-protocol table but is marked "n/a" for mAP.

### 4.3 The centerpiece figure

Sweep the WBF ensemble's score threshold to trace its full **precision–recall curve** under the strict protocol. Overlay
the **integrated method's single (recall, precision) point**.
- Point **above** the curve → rule-based integration genuinely recovers trees a threshold-drop cannot; **paper thesis
  proven**.
- Point **below** the curve → "just lower the WBF threshold" dominates; paper must reframe honestly.

The outcome is unknown at design time; either way the paper becomes more credible. This figure + table replaces the
single 96% number as the paper's evidentiary core.

### 4.4 Ablations (W3)

`evaluate_all.py` reports the integrated result with each rule turned off, reusing the saved intermediate JSONs where
possible (WBF-only points; +neighbor validation; +contour matching; +watershed recovery). Where an ablation requires
re-running the rule logic (not just re-scoring), Phase 1 re-implements that rule cleanly in `evaluation/` driven by the
saved inputs — no GPU, no detector re-run. If a specific ablation cannot be produced from saved artifacts alone, it is
listed explicitly as "requires pipeline re-run (Phase 3)" rather than silently omitted.

### 4.5 Sensitivity (W3)

`sensitivity.py` sweeps the key constants (score threshold τ_a, matching gap, contour tolerance) and outputs how
recall/precision/F1 move — turning magic numbers into justified choices or flagged fragilities.

## 5. Error handling & correctness guards

- Every input load asserts expected shape (detector counts above; GT=13552/222) and fails loud on mismatch.
- Matching is unit-tested on synthetic fixtures (perfect, double-count, boundary ±gap, empty image) before trusting it on
  real data.
- `paper_protocol.py` reproduction is a hard regression test (≡91.2% / 96.0% within ±0.1%).
- Coordinate frame (640×640) is validated; any detection out of frame is reported, not clipped silently.
- All generated numbers are written to `results/` with the exact input paths + a config header, so every table cell is
  traceable.

## 6. Testing strategy

- `tests/test_matching.py` — synthetic unit tests for IoU and one-to-one point matching.
- `tests/test_paper_protocol.py` — reproduction regression lock.
- Smoke run of `evaluate_all.py` on real data producing a complete table with no NaNs (except intended mAP n/a).
- Spot-check: strict-protocol recall ≤ original-protocol recall for every method (sanity — stricter can't score higher).

## 7. Deliverables (Phase 1)

1. `evaluation/` package + `tests/`, committed and runnable via `python -m evaluation.evaluate_all`.
2. `results/table.md` + `table.csv`: per-model (×5) + WBF + integrated, columns = original-recall, strict
   recall/precision/F1, AP/AP50.
3. `results/pr_curve.png`: WBF PR curve + integrated point.
4. Paper-ready Markdown: a revised "Experiments and Results" results table and 2–3 paragraphs reporting precision/F1 and
   interpreting the PR-curve figure, honestly stating whichever outcome we get.

## 8. Out of scope for Phase 1

- Re-running detectors / re-training (GPU) — outputs are reused from saved JSONs.
- Phase 2 method changes (color invariants, Gabor bank) — separate spec.
- Phase 3 packaging of the `C:/`-hardcoded pipeline — separate spec.
- New external baselines (W6) — attempted only if DeepForest is cheap; otherwise future work.

## 9. Open questions

- None blocking. DeepForest baseline (W6) is optional/opportunistic in Phase 1.
