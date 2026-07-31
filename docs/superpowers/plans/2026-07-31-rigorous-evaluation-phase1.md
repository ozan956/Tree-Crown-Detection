# Rigorous Re-Evaluation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, GPU-free `evaluation/` module that re-scores every tree-crown detection method (5 detectors, WBF ensemble, integrated) under both the paper's original protocol and a strict one-to-one protocol, producing precision/recall/F1 + COCO mAP tables and a PR-curve-with-integrated-point figure — turning the paper's single unverified 96% recall number into defensible, reproducible evidence.

**Architecture:** A small Python package reads a vendored 20 MB snapshot of the real GT + detector + fusion + integrated JSONs, matches predictions to ground truth two ways (many-to-one center-in-box = paper protocol; one-to-one Hungarian = strict protocol), computes metrics, and emits Markdown/CSV tables + a PNG figure into `results/`. Every module has one responsibility and is unit-tested on synthetic fixtures before touching real data.

**Tech Stack:** Python 3.12, numpy, scipy (Hungarian assignment), pycocotools (mAP), matplotlib (figure), pytest (tests). All CPU-only. Installed into the existing empty `.venv`.

## Global Constraints

- Python 3.12 via the repo's `.venv` (`.venv/bin/python`, `.venv/bin/pip`).
- CPU-only. No detector re-run, no training, no GPU. All results derive from saved JSON outputs.
- Ground truth is fixed: **222 images, 13552 boxes, 1 class ("Tree")**, image_id range 0–221, 640×640 frame.
- COCO bbox format is `[x, y, w, h]` (top-left + size); detector JSON fields are exactly `image_id`, `bbox`, `score`, `category_id`.
- Integrated-output points file is a **list of 222 per-image lists**, each element `{"image_id", "tree_x", "tree_y"}`.
- Reproduction lock: the integrated method's **96.0% = 13012 unique GT boxes** matched (center-in-box, gap=10) MUST reproduce within ±1 box. (The DL-alone 91.2%/12353 does NOT reproduce trivially from `best_fuse.json` — it is an investigation output, NOT a hard assertion; see Task 5.)
- **RULING (2026-07-31, supersedes narrative below):** the published 96.0% is a *superseded v1* number produced by a per-center-`break` matcher that drops a GT box whenever a detection point falls in the overlap of two crowns (217/222 images have overlapping capture zones). It remains a **reproduction lock** — we must still reproduce 13012 to document what v1 published — but it is NOT the corrected headline. The **corrected headline is one-to-one (Hungarian) matching: recall 97.14% (tp=13164, fn=388), precision 83.26% (fp=2647)** for the integrated method at gap=10. `match_center_in_box` is retained as the *legacy reproduction* matcher; `match_one_to_one_points` is the *canonical* matcher for all headline recall/precision/F1. Every place the plan text below says "reproduce/lock 96.0% as the result", read it as "reproduce 96.0% as the superseded v1 figure, and report one-to-one 97.14%/83.26% as the corrected result." FINDINGS.md frames this as an erratum: v1 96.0% → corrected 97.14% recall with 83.26% precision (which v1 never reported).
- All generated artifacts go under `results/` and are git-tracked so the paper can cite exact numbers.
- Input data is vendored under `data/` (see Task 1); code reads only from `data/`, never from `wbf_test-*` or `C:/` paths.

---

## File Structure

```
data/                         # vendored 20MB snapshot (Task 1)
  annotations.coco.json       # GT: 222 imgs / 13552 boxes
  dets/swin.json cascade_rcnn.json detr.json faster_rcnn.json yolo.json
  wbf/best_fuse.json          # fused boxes consumed by integration
  integrated/points.json      # detected_trees_based_on_shape.json
  MANIFEST.md                 # provenance: which wbf_test-* path each file came from + count
evaluation/
  __init__.py
  io_utils.py                 # load GT, load detector json, load integrated points, box<->center
  matching.py                 # iou(); match_center_in_box() many-to-one; match_one_to_one() Hungarian
  metrics.py                  # pr_f1_from_counts(); coco_map()
  paper_protocol.py           # reproduce original recall metric (locks 96.0% integrated)
  evaluate_all.py             # main: build the full results table (md + csv)
  sensitivity.py              # score-threshold sweep + tolerance/gap sweeps
  figures.py                  # PR curve + integrated point -> png
requirements-eval.txt
tests/
  __init__.py
  conftest.py                 # synthetic fixtures
  test_matching.py
  test_metrics.py
  test_paper_protocol.py      # regression lock on 13012
results/                      # generated (tables, figure)
```

---

### Task 1: Vendored data snapshot + dependency setup

**Files:**
- Create: `data/` tree (copied JSONs), `data/MANIFEST.md`
- Create: `requirements-eval.txt`
- Create: `.gitignore` (ignore `.venv/`, `wbf_test-*/`, `__pycache__/`, `*.pyc`)

**Interfaces:**
- Produces: canonical on-disk paths every later task reads:
  - `data/annotations.coco.json`
  - `data/dets/{swin,cascade_rcnn,detr,faster_rcnn,yolo}.json`
  - `data/wbf/best_fuse.json`
  - `data/integrated/points.json`

- [ ] **Step 1: Copy the verified input files into `data/`**

```bash
cd "/home/odurgut/academy/Tree-Crown-Detection"
Z1="wbf_test-20260731T192532Z-1-001/wbf_test"
Z2="wbf_test-20260731T192532Z-1-002/wbf_test"
mkdir -p data/dets data/wbf data/integrated
cp "$Z2/beril-ozan-cem-work/new_idea/_annotations.coco.json" data/annotations.coco.json
cp "$Z1/swin/bbox_swin.json"                 data/dets/swin.json
cp "$Z1/cascade-rcnn/bbox_rcnn.json"         data/dets/cascade_rcnn.json
cp "$Z1/detr/bbox_detr.json"                 data/dets/detr.json
cp "$Z1/faster_rcnn/bbox_fasterrcnn.json"    data/dets/faster_rcnn.json
cp "$Z2/yolo/bbox_yolo.json"                 data/dets/yolo.json
cp "$Z1/beril-ozan-cem-work/new_idea/best_fuse.json"                     data/wbf/best_fuse.json
cp "$Z1/beril-ozan-cem-work/new_idea/detected_trees_based_on_shape.json" data/integrated/points.json
```

- [ ] **Step 2: Verify counts (fail loud if wrong copy)**

```bash
.venv/bin/python - <<'PY'
import json
def n(p): 
    d=json.load(open(p)); return len(d)
assert n('data/dets/swin.json')==13387, n('data/dets/swin.json')
assert n('data/dets/cascade_rcnn.json')==12894
assert n('data/dets/detr.json')==22200
assert n('data/dets/faster_rcnn.json')==12610
assert n('data/dets/yolo.json')==14084
assert n('data/wbf/best_fuse.json')==20846
ann=json.load(open('data/annotations.coco.json'))
assert len(ann['images'])==222 and len(ann['annotations'])==13552
pts=json.load(open('data/integrated/points.json'))
assert len(pts)==222 and sum(len(g) for g in pts)==15811, sum(len(g) for g in pts)
print("data snapshot OK")
PY
```
Expected: `data snapshot OK`

- [ ] **Step 3: Write `data/MANIFEST.md`** documenting each file's source path in the `wbf_test-*` folders and its record count (the six detector/GT/wbf/integrated counts from Step 2), so provenance is traceable.

- [ ] **Step 4: Write `requirements-eval.txt`**

```
numpy>=1.26
scipy>=1.11
pycocotools>=2.0.7
matplotlib>=3.7
pytest>=7.4
```

- [ ] **Step 5: Write `.gitignore`**

```
.venv/
wbf_test-*/
__pycache__/
*.pyc
```

- [ ] **Step 6: Install dependencies into `.venv`**

```bash
.venv/bin/pip install -r requirements-eval.txt
```
Expected: installs succeed; `.venv/bin/python -c "import numpy,scipy,pycocotools,matplotlib,pytest"` prints nothing (no error).

- [ ] **Step 7: Commit**

```bash
git add data/ requirements-eval.txt .gitignore
git commit -m "chore: vendor evaluation data snapshot + eval deps"
```

---

### Task 2: `io_utils.py` — data loading & geometry helpers

**Files:**
- Create: `evaluation/__init__.py` (empty), `evaluation/io_utils.py`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_metrics.py` (metrics tests land in Task 4; create file in Task 4)
- Test: `tests/test_io_utils.py`

**Interfaces:**
- Produces:
  - `load_gt(path="data/annotations.coco.json") -> dict[int, list[tuple[float,float,float,float]]]` — image_id → list of GT boxes as `(x0,y0,x1,y1)` corner form.
  - `gt_total(gt) -> int` — total box count (13552).
  - `load_dets(path) -> list[dict]` — raw detector records (each has `image_id,bbox,score,category_id`).
  - `dets_by_image(dets, score_thr=0.0) -> dict[int, list[tuple[tuple,float]]]` — image_id → list of `((x0,y0,x1,y1), score)` corner boxes with score ≥ thr.
  - `load_points(path="data/integrated/points.json") -> dict[int, list[tuple[float,float]]]` — image_id → list of `(x,y)` centers (flattens the 222 groups).
  - `box_center(box_xywh) -> tuple[float,float]` and `xywh_to_corners(b) -> tuple`.

- [ ] **Step 1: Write failing tests** `tests/test_io_utils.py`

```python
from evaluation.io_utils import (load_gt, gt_total, load_dets, dets_by_image,
                                  load_points, box_center, xywh_to_corners)

def test_xywh_to_corners():
    assert xywh_to_corners([10, 20, 30, 40]) == (10, 20, 40, 60)

def test_box_center():
    assert box_center([10, 20, 30, 40]) == (25.0, 40.0)

def test_load_gt_shape():
    gt = load_gt()
    assert gt_total(gt) == 13552
    assert len(gt) == 222
    # every box is a 4-tuple of corners with x1>x0, y1>y0
    x0, y0, x1, y1 = gt[0][0]
    assert x1 > x0 and y1 > y0

def test_dets_by_image_threshold():
    dets = load_dets("data/dets/swin.json")
    all_n = sum(len(v) for v in dets_by_image(dets, 0.0).values())
    hi_n  = sum(len(v) for v in dets_by_image(dets, 0.8).values())
    assert all_n == 13387
    assert hi_n < all_n  # thresholding removes some

def test_load_points_shape():
    pts = load_points()
    assert len(pts) == 222
    assert sum(len(v) for v in pts.values()) == 15811
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module not written yet).

- [ ] **Step 3: Implement `evaluation/io_utils.py`**

```python
import json

def xywh_to_corners(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)

def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def load_gt(path="data/annotations.coco.json"):
    d = json.load(open(path))
    gt = {}
    for a in d["annotations"]:
        gt.setdefault(a["image_id"], []).append(xywh_to_corners(a["bbox"]))
    return gt

def gt_total(gt):
    return sum(len(v) for v in gt.values())

def load_dets(path):
    return json.load(open(path))

def dets_by_image(dets, score_thr=0.0):
    out = {}
    for d in dets:
        if d["score"] < score_thr:
            continue
        out.setdefault(d["image_id"], []).append((xywh_to_corners(d["bbox"]), d["score"]))
    return out

def load_points(path="data/integrated/points.json"):
    groups = json.load(open(path))
    out = {}
    for g in groups:
        for t in g:
            out.setdefault(t["image_id"], []).append((t["tree_x"], t["tree_y"]))
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/__init__.py evaluation/io_utils.py tests/__init__.py tests/test_io_utils.py
git commit -m "feat(eval): data loading + geometry helpers"
```

---

### Task 3: `matching.py` — IoU, center-in-box, one-to-one matching

**Files:**
- Create: `evaluation/matching.py`
- Create: `tests/conftest.py` (synthetic fixtures)
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: corner boxes `(x0,y0,x1,y1)` and centers `(x,y)` from `io_utils`.
- Produces:
  - `iou(a, b) -> float` — IoU of two corner boxes.
  - `match_center_in_box(centers, gt_boxes, gap=10.0) -> set[int]` — indices of GT boxes that contain ≥1 center (±gap). Many-to-one (paper protocol); returns the set of matched GT indices.
  - `match_one_to_one_points(centers, gt_boxes, gap=10.0) -> tuple[int,int,int]` — `(tp, fp, fn)` under one-to-one assignment: a center matches a GT box if the center is inside (±gap); each center and each GT used at most once (Hungarian on a 0/1 feasibility cost).
  - `match_one_to_one_iou(pred_boxes, gt_boxes, iou_thr=0.5) -> tuple[int,int,int]` — `(tp, fp, fn)` one-to-one by IoU≥thr (Hungarian on 1−IoU cost, infeasible pairs excluded).

- [ ] **Step 1: Write synthetic fixtures** `tests/conftest.py`

```python
import pytest

@pytest.fixture
def two_gt_boxes():
    # two 10x10 boxes, corner form
    return [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
```

- [ ] **Step 2: Write failing tests** `tests/test_matching.py`

```python
from evaluation.matching import (iou, match_center_in_box,
                                  match_one_to_one_points, match_one_to_one_iou)

def test_iou_identical():
    assert iou((0,0,10,10), (0,0,10,10)) == 1.0

def test_iou_disjoint():
    assert iou((0,0,10,10), (100,100,110,110)) == 0.0

def test_iou_half():
    # overlap area 50, union 150 -> 1/3
    assert abs(iou((0,0,10,10), (5,0,15,10)) - (50/150)) < 1e-9

def test_center_in_box_matches_both(two_gt_boxes):
    centers = [(5,5), (105,105)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == {0, 1}

def test_center_in_box_double_count_is_one_box(two_gt_boxes):
    # three centers all in box 0 -> only box 0 matched (set semantics)
    centers = [(1,1), (2,2), (3,3)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == {0}

def test_center_gap_boundary(two_gt_boxes):
    # center just outside box 0 by 5px; gap=0 misses, gap=10 hits
    centers = [(15, 5)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == set()
    assert match_center_in_box(centers, two_gt_boxes, gap=10) == {0}

def test_one_to_one_points_penalizes_extra(two_gt_boxes):
    # 3 centers in box0, 0 in box1: one-to-one -> tp=1, fp=2, fn=1
    centers = [(1,1), (2,2), (3,3)]
    tp, fp, fn = match_one_to_one_points(centers, two_gt_boxes, gap=0)
    assert (tp, fp, fn) == (1, 2, 1)

def test_one_to_one_points_perfect(two_gt_boxes):
    centers = [(5,5), (105,105)]
    assert match_one_to_one_points(centers, two_gt_boxes, gap=0) == (2, 0, 0)

def test_one_to_one_iou_perfect(two_gt_boxes):
    preds = [(0,0,10,10), (100,100,110,110)]
    assert match_one_to_one_iou(preds, two_gt_boxes, iou_thr=0.5) == (2, 0, 0)

def test_one_to_one_iou_no_match(two_gt_boxes):
    preds = [(0,0,3,3)]  # iou with box0 = 9/100 < 0.5
    assert match_one_to_one_iou(preds, two_gt_boxes, iou_thr=0.5) == (0, 1, 2)
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: FAIL (module not written).

- [ ] **Step 4: Implement `evaluation/matching.py`**

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0

def _center_in(c, box, gap):
    x, y = c
    x0, y0, x1, y1 = box
    return (x0 - gap) <= x <= (x1 + gap) and (y0 - gap) <= y <= (y1 + gap)

def match_center_in_box(centers, gt_boxes, gap=10.0):
    matched = set()
    for c in centers:
        for bi, box in enumerate(gt_boxes):
            if _center_in(c, box, gap):
                matched.add(bi)
                break
    return matched

def _one_to_one(feasible):
    # feasible: bool matrix [n_pred, n_gt]; maximize matched pairs.
    if feasible.size == 0 or not feasible.any():
        return 0
    cost = np.where(feasible, 0.0, 1e6)
    r, c = linear_sum_assignment(cost)
    return int(sum(feasible[i, j] for i, j in zip(r, c)))

def match_one_to_one_points(centers, gt_boxes, gap=10.0):
    n_pred, n_gt = len(centers), len(gt_boxes)
    feasible = np.zeros((n_pred, n_gt), dtype=bool)
    for i, c in enumerate(centers):
        for j, box in enumerate(gt_boxes):
            feasible[i, j] = _center_in(c, box, gap)
    tp = _one_to_one(feasible)
    return tp, n_pred - tp, n_gt - tp

def match_one_to_one_iou(pred_boxes, gt_boxes, iou_thr=0.5):
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    feasible = np.zeros((n_pred, n_gt), dtype=bool)
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            feasible[i, j] = iou(p, g) >= iou_thr
    tp = _one_to_one(feasible)
    return tp, n_pred - tp, n_gt - tp
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluation/matching.py tests/conftest.py tests/test_matching.py
git commit -m "feat(eval): IoU + center-in-box + one-to-one matching"
```

---

### Task 4: `metrics.py` — precision/recall/F1 + COCO mAP

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `(tp, fp, fn)` tuples from `matching`; raw detector list + GT COCO dict for mAP.
- Produces:
  - `pr_f1(tp, fp, fn) -> dict` — `{"precision","recall","f1","tp","fp","fn"}` (0.0 when denominators are 0).
  - `coco_map(gt_coco_path, dets, cat_id=1) -> dict` — `{"AP","AP50","AP75"}` via pycocotools `COCOeval` (bbox). `dets` is the raw detector list (COCO result format).

- [ ] **Step 1: Write failing tests** `tests/test_metrics.py`

```python
from evaluation.metrics import pr_f1, coco_map
from evaluation.io_utils import load_dets

def test_pr_f1_basic():
    m = pr_f1(tp=8, fp=2, fn=4)
    assert abs(m["precision"] - 0.8) < 1e-9
    assert abs(m["recall"] - (8/12)) < 1e-9
    assert abs(m["f1"] - (2*0.8*(8/12))/(0.8+8/12)) < 1e-9

def test_pr_f1_zero_safe():
    assert pr_f1(0, 0, 0) == {"precision":0.0,"recall":0.0,"f1":0.0,"tp":0,"fp":0,"fn":0}

def test_coco_map_swin_runs():
    dets = load_dets("data/dets/swin.json")
    m = coco_map("data/annotations.coco.json", dets)
    # AP50 for a trained tree detector should be a sane, high-ish number in (0,1]
    assert 0.0 < m["AP50"] <= 1.0
    assert 0.0 <= m["AP"] <= m["AP50"]  # AP (averaged over IoUs) <= AP50
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL (module not written).

- [ ] **Step 3: Implement `evaluation/metrics.py`**

```python
import contextlib, io, json, tempfile, os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def pr_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}

def coco_map(gt_coco_path, dets, cat_id=1):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_coco_path)
        # pycocotools needs results as a file or list of dicts with int category_id
        results = [{"image_id": d["image_id"], "category_id": int(d["category_id"]),
                    "bbox": d["bbox"], "score": float(d["score"])} for d in dets]
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(results, tf); tf.close()
        coco_dt = coco_gt.loadRes(tf.name)
        os.unlink(tf.name)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 3 passed. (First real signal: Swin AP50 prints via the test.)

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(eval): precision/recall/f1 + COCO mAP"
```

---

### Task 5: `paper_protocol.py` — reproduce the original metric (lock 96.0%)

**Files:**
- Create: `evaluation/paper_protocol.py`
- Test: `tests/test_paper_protocol.py`

**Interfaces:**
- Consumes: `load_gt`, `load_points`, `dets_by_image`, `box_center` from `io_utils`; `match_center_in_box` from `matching`.
- Produces:
  - `paper_recall_points(points_by_image, gt, gap=10.0) -> dict` — `{"matched","total","recall"}` unique-GT-box recall for a point set (used for the integrated method).
  - `paper_recall_boxes(dets_by_img, gt, gap=10.0) -> dict` — same, using box centers (used for DL/WBF box sets).
  - `investigate_dl_baseline() -> dict` — sweeps `best_fuse.json` score thresholds and reports the recall curve + the closest threshold to the paper's 12353, documenting that 91.2% is not reproduced by a single obvious rule.

- [ ] **Step 1: Write failing test** `tests/test_paper_protocol.py`

```python
from evaluation.paper_protocol import paper_recall_points
from evaluation.io_utils import load_gt, load_points

def test_integrated_reproduces_96_percent():
    gt = load_gt()
    pts = load_points()
    r = paper_recall_points(pts, gt, gap=10.0)
    assert r["total"] == 13552
    # Paper reports 13012 / 13552 = 96.0%. Lock within +/- 1 box.
    assert abs(r["matched"] - 13012) <= 1, r["matched"]
    assert abs(r["recall"] - 0.960) < 0.002
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_paper_protocol.py -v`
Expected: FAIL (module not written).

- [ ] **Step 3: Implement `evaluation/paper_protocol.py`**

```python
from evaluation.io_utils import load_gt, load_dets, dets_by_image, box_center
from evaluation.matching import match_center_in_box

def paper_recall_points(points_by_image, gt, gap=10.0):
    total = sum(len(v) for v in gt.values())
    matched = 0
    for iid, boxes in gt.items():
        centers = points_by_image.get(iid, [])
        matched += len(match_center_in_box(centers, boxes, gap))
    return {"matched": matched, "total": total, "recall": matched / total}

def paper_recall_boxes(dets_by_img, gt, gap=10.0):
    total = sum(len(v) for v in gt.values())
    matched = 0
    for iid, boxes in gt.items():
        centers = [ (( (b[0]+b[2])/2.0, (b[1]+b[3])/2.0 )) for (b, s) in dets_by_img.get(iid, []) ]
        matched += len(match_center_in_box(centers, boxes, gap))
    return {"matched": matched, "total": total, "recall": matched / total}

def investigate_dl_baseline():
    """The paper's 91.2% (12353) does not reproduce from best_fuse.json with a
    single center-in-box rule. Sweep thresholds and report the curve so the
    discrepancy is documented rather than hidden."""
    gt = load_gt()
    wbf = load_dets("data/wbf/best_fuse.json")
    rows = []
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = paper_recall_boxes(dets_by_image(wbf, thr), gt, gap=10.0)
        rows.append((thr, r["matched"], r["recall"]))
    closest = min(rows, key=lambda t: abs(t[1] - 12353))
    return {"paper_reported": 12353, "curve": rows, "closest_threshold": closest}
```

- [ ] **Step 4: Run test, verify pass**

Run: `.venv/bin/python -m pytest tests/test_paper_protocol.py -v`
Expected: 1 passed (13012 reproduced).

- [ ] **Step 5: Print the DL-baseline investigation for the record**

Run:
```bash
.venv/bin/python -c "from evaluation.paper_protocol import investigate_dl_baseline; import json; print(json.dumps(investigate_dl_baseline(), indent=2))"
```
Expected: a threshold→recall table; closest-to-12353 near thr≈0.45. This output is captured in the results write-up (Task 8) as the documented provenance gap for the 91.2% figure.

- [ ] **Step 6: Commit**

```bash
git add evaluation/paper_protocol.py tests/test_paper_protocol.py
git commit -m "feat(eval): reproduce paper protocol; lock integrated 96.0%; document DL-baseline gap"
```

---

### Task 6: `evaluate_all.py` — the full results table

**Files:**
- Create: `evaluation/evaluate_all.py`
- Test: `tests/test_evaluate_all.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build_table(score_thr=0.8, gap=10.0) -> list[dict]` — one row per method: keys `method, paper_recall, strict_recall, strict_precision, strict_f1, AP, AP50`. Methods: `swin, cascade_rcnn, detr, faster_rcnn, yolo, WBF, integrated`. `AP/AP50` are `None` for `integrated` (point-only). Strict metrics for box methods use `match_one_to_one_iou`; for `integrated` use `match_one_to_one_points`.
  - `write_table(rows, md_path="results/table.md", csv_path="results/table.csv") -> None`.
  - `main()` — build + write + print.
- Run: `.venv/bin/python -m evaluation.evaluate_all`

- [ ] **Step 1: Write failing test** `tests/test_evaluate_all.py`

```python
from evaluation.evaluate_all import build_table

def test_table_has_all_methods():
    rows = build_table(score_thr=0.8, gap=10.0)
    methods = {r["method"] for r in rows}
    assert {"swin","cascade_rcnn","detr","faster_rcnn","yolo","WBF","integrated"} <= methods

def test_integrated_row_is_point_only_and_reproduces():
    rows = {r["method"]: r for r in build_table()}
    integ = rows["integrated"]
    assert integ["AP"] is None and integ["AP50"] is None  # point-only
    assert abs(integ["paper_recall"] - 0.960) < 0.002       # reproduction holds

def test_strict_precision_present_for_box_methods():
    rows = {r["method"]: r for r in build_table()}
    for m in ["swin","WBF"]:
        assert 0.0 <= rows[m]["strict_precision"] <= 1.0
        assert 0.0 <= rows[m]["strict_recall"] <= 1.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate_all.py -v`
Expected: FAIL (module not written).

- [ ] **Step 3: Implement `evaluation/evaluate_all.py`**

```python
import os, csv
from evaluation.io_utils import load_gt, load_dets, dets_by_image, load_points
from evaluation.matching import match_one_to_one_iou, match_one_to_one_points
from evaluation.metrics import pr_f1, coco_map
from evaluation.paper_protocol import paper_recall_boxes, paper_recall_points

DET_FILES = {
    "swin": "data/dets/swin.json",
    "cascade_rcnn": "data/dets/cascade_rcnn.json",
    "detr": "data/dets/detr.json",
    "faster_rcnn": "data/dets/faster_rcnn.json",
    "yolo": "data/dets/yolo.json",
    "WBF": "data/wbf/best_fuse.json",
}

def _strict_box(dets_by_img, gt, iou_thr=0.5):
    tp = fp = fn = 0
    keys = set(dets_by_img) | set(gt)
    for iid in keys:
        preds = [b for (b, s) in dets_by_img.get(iid, [])]
        t, f, n = match_one_to_one_iou(preds, gt.get(iid, []), iou_thr)
        tp += t; fp += f; fn += n
    return pr_f1(tp, fp, fn)

def _strict_points(points_by_image, gt, gap=10.0):
    tp = fp = fn = 0
    keys = set(points_by_image) | set(gt)
    for iid in keys:
        t, f, n = match_one_to_one_points(points_by_image.get(iid, []), gt.get(iid, []), gap)
        tp += t; fp += f; fn += n
    return pr_f1(tp, fp, fn)

def build_table(score_thr=0.8, gap=10.0):
    gt = load_gt()
    rows = []
    for method, path in DET_FILES.items():
        dets = load_dets(path)
        dbi = dets_by_image(dets, score_thr)
        paper = paper_recall_boxes(dbi, gt, gap)
        strict = _strict_box(dbi, gt, 0.5)
        cm = coco_map("data/annotations.coco.json", [d for d in dets if d["score"] >= score_thr])
        rows.append({"method": method, "paper_recall": paper["recall"],
                     "strict_recall": strict["recall"], "strict_precision": strict["precision"],
                     "strict_f1": strict["f1"], "AP": cm["AP"], "AP50": cm["AP50"]})
    # integrated (points)
    pts = load_points()
    paper = paper_recall_points(pts, gt, gap)
    strict = _strict_points(pts, gt, gap)
    rows.append({"method": "integrated", "paper_recall": paper["recall"],
                 "strict_recall": strict["recall"], "strict_precision": strict["precision"],
                 "strict_f1": strict["f1"], "AP": None, "AP50": None})
    return rows

def write_table(rows, md_path="results/table.md", csv_path="results/table.csv"):
    os.makedirs("results", exist_ok=True)
    cols = ["method","paper_recall","strict_recall","strict_precision","strict_f1","AP","AP50"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    def fmt(v): return "n/a" if v is None else f"{v:.3f}"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r["method"] if c=="method" else fmt(r[c]) for c in cols) + " |\n")

def main():
    rows = build_table()
    write_table(rows)
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate_all.py -v`
Expected: 3 passed.

- [ ] **Step 5: Generate the real table**

Run: `.venv/bin/python -m evaluation.evaluate_all`
Expected: prints 7 rows; `results/table.md` and `results/table.csv` created. **Sanity checks to eyeball:** integrated `paper_recall≈0.960`; every box method's `strict_precision` and `strict_recall` in (0,1); DETR (22200 dets) should show notably lower precision than Swin — the false-positive cost the paper never reported.

- [ ] **Step 6: Commit**

```bash
git add evaluation/evaluate_all.py tests/test_evaluate_all.py results/table.md results/table.csv
git commit -m "feat(eval): full results table (paper + strict + mAP) for all methods"
```

---

### Task 7: `sensitivity.py` + `figures.py` — PR curve with integrated point

**Files:**
- Create: `evaluation/sensitivity.py`, `evaluation/figures.py`
- Test: `tests/test_sensitivity.py`

**Interfaces:**
- Consumes: `load_gt`, `load_dets`, `dets_by_image`, `load_points`; `_strict_box`/`_strict_points` logic (import from `evaluate_all`).
- Produces:
  - `wbf_pr_curve(thresholds=None) -> list[dict]` — for each WBF score threshold: `{"threshold","precision","recall"}` under the strict IoU protocol.
  - `integrated_point(gap=10.0) -> dict` — `{"precision","recall"}` for the integrated method (strict points).
  - `gap_sensitivity(gaps=(0,5,10,20)) -> list[dict]` and `iou_sensitivity(...)` — how integrated P/R move with the matching gap.
  - `figures.plot_pr_curve(curve, point, out="results/pr_curve.png") -> None` — plots the WBF PR curve and overlays the integrated point, annotated above/below.

- [ ] **Step 1: Write failing test** `tests/test_sensitivity.py`

```python
from evaluation.sensitivity import wbf_pr_curve, integrated_point

def test_pr_curve_monotone_recall():
    curve = wbf_pr_curve([0.9, 0.7, 0.5, 0.3, 0.1])
    recalls = [c["recall"] for c in curve]
    # lower threshold -> more predictions -> recall non-decreasing
    assert recalls == sorted(recalls)

def test_integrated_point_in_range():
    p = integrated_point()
    assert 0.0 < p["recall"] <= 1.0 and 0.0 < p["precision"] <= 1.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sensitivity.py -v`
Expected: FAIL (module not written).

- [ ] **Step 3: Implement `evaluation/sensitivity.py`**

```python
from evaluation.io_utils import load_gt, load_dets, dets_by_image, load_points
from evaluation.evaluate_all import _strict_box, _strict_points

def wbf_pr_curve(thresholds=None):
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95
    gt = load_gt()
    wbf = load_dets("data/wbf/best_fuse.json")
    out = []
    for thr in sorted(thresholds, reverse=True):
        m = _strict_box(dets_by_image(wbf, thr), gt, 0.5)
        out.append({"threshold": thr, "precision": m["precision"], "recall": m["recall"]})
    return out

def integrated_point(gap=10.0):
    gt = load_gt()
    m = _strict_points(load_points(), gt, gap)
    return {"precision": m["precision"], "recall": m["recall"]}

def gap_sensitivity(gaps=(0, 5, 10, 20)):
    gt = load_gt(); pts = load_points()
    return [{"gap": g, **{k: _strict_points(pts, gt, g)[k] for k in ("precision","recall","f1")}} for g in gaps]
```

- [ ] **Step 4: Implement `evaluation/figures.py`**

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_pr_curve(curve, point, out="results/pr_curve.png"):
    os.makedirs("results", exist_ok=True)
    rec = [c["recall"] for c in curve]
    prec = [c["precision"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, "-o", ms=3, label="WBF ensemble (score sweep)")
    ax.scatter([point["recall"]], [point["precision"]], c="red", zorder=5,
               label="Rule-based integrated")
    ax.annotate(f'({point["recall"]:.3f}, {point["precision"]:.3f})',
                (point["recall"], point["precision"]),
                textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall: WBF ensemble vs. rule-based integration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
```

- [ ] **Step 5: Run test, verify pass**

Run: `.venv/bin/python -m pytest tests/test_sensitivity.py -v`
Expected: 2 passed.

- [ ] **Step 6: Generate the figure**

Run:
```bash
.venv/bin/python -c "from evaluation.sensitivity import wbf_pr_curve, integrated_point; from evaluation.figures import plot_pr_curve; plot_pr_curve(wbf_pr_curve(), integrated_point()); print('wrote results/pr_curve.png')"
```
Expected: `results/pr_curve.png` created. **This is the paper's new evidentiary centerpiece** — whether the red integrated point sits above or below the WBF curve is the finding.

- [ ] **Step 7: Commit**

```bash
git add evaluation/sensitivity.py evaluation/figures.py tests/test_sensitivity.py results/pr_curve.png
git commit -m "feat(eval): WBF PR curve + integrated point figure + sensitivity"
```

---

### Task 8: Results write-up + module README

**Files:**
- Create: `evaluation/README.md`, `results/FINDINGS.md`

**Interfaces:** none (documentation). Consumes the generated `results/table.md`, `results/pr_curve.png`, and the Task-5 DL-baseline investigation output.

- [ ] **Step 1: Write `evaluation/README.md`** — how to install (`.venv/bin/pip install -r requirements-eval.txt`), how to run (`python -m evaluation.evaluate_all`, the figure one-liner), what each module does, and the two protocols explained.

- [ ] **Step 2: Generate final numbers and write `results/FINDINGS.md`**

Run (capture output to paste in):
```bash
.venv/bin/python -m evaluation.evaluate_all
.venv/bin/python -c "from evaluation.paper_protocol import investigate_dl_baseline; import json; print(json.dumps(investigate_dl_baseline(), indent=2))"
.venv/bin/python -c "from evaluation.sensitivity import gap_sensitivity; import json; print(json.dumps(gap_sensitivity(), indent=2))"
```

Then write `results/FINDINGS.md` with: (a) the results table; (b) the reproduction statement (integrated 96.0% reproduced exactly = 13012; DL-alone 91.2% not reproduced by a single rule, closest at thr≈0.45 — documented provenance gap); (c) precision/F1 numbers the paper never reported; (d) the PR-curve verdict — does rule-based integration beat threshold-lowering? (e) sensitivity of the result to `gap`. **This file is the paper-ready evidence.** Write the honest interpretation of whichever way the PR-curve point falls.

- [ ] **Step 3: Draft paper-ready text** — append to `results/FINDINGS.md` a "Revised Results paragraphs" section with 2–3 paragraphs suitable for pasting into the arXiv revision's Experiments section, reporting precision/recall/F1/mAP and interpreting the figure.

- [ ] **Step 4: Commit**

```bash
git add evaluation/README.md results/FINDINGS.md
git commit -m "docs(eval): findings, reproduction notes, paper-ready results text"
```

---

## Self-Review

**1. Spec coverage:**
- W1 (recall-only, no FP penalty) → Tasks 3 (`match_one_to_one_*`), 6 (strict precision/recall/F1 columns), 7 (PR curve). ✓
- W2 (no precision/F1/mAP) → Tasks 4 (`coco_map`, `pr_f1`), 6 (table). ✓
- W3 (no ablation/sensitivity) → Task 7 (sensitivity sweeps); *note:* rule-by-rule ablation (neighbor/contour/watershed) from the spec §4.4 is **partially** covered — this plan does the score-threshold/gap/IoU sensitivity and the WBF-vs-integrated comparison, but does NOT re-implement each rule toggle, because that requires re-running the integration logic (deferred to Phase 3 packaging where the pipeline is consolidated). This is a deliberate scope cut, flagged in FINDINGS.md. ✓ (with documented deferral)
- W4 (text vs code) → out of scope for Phase 1 per spec (Phase 2). ✓
- W5 (reproducibility) → Task 1 (vendored data + requirements + entry point) partially; full packaging is Phase 3. ✓
- Two-protocol reconciliation (spec §4.2) → Tasks 5 + 6. ✓
- Centerpiece figure (spec §4.3) → Task 7. ✓
- Reproduction lock (spec §5) → Task 5 test (13012 within ±1). ✓ Note the spec's 91.2% is correctly downgraded to an investigation (verified during planning: it does not reproduce from best_fuse.json).

**2. Placeholder scan:** No TBD/TODO. Every code step has real code. Documentation tasks (8) describe exact content and the commands that produce the numbers to write. ✓

**3. Type consistency:** `dets_by_image` returns `{iid: [((corners), score)]}` — consumed consistently in `paper_recall_boxes`, `_strict_box`, `wbf_pr_curve`. `load_points`/`load_gt` return `{iid: [...]}` — consumed consistently. `pr_f1` dict keys (`precision/recall/f1/tp/fp/fn`) used consistently in Tasks 4/6/7. `match_one_to_one_*` returns `(tp,fp,fn)` everywhere. `build_table` row keys match `write_table` cols and Task 6 tests. ✓

**Fixed inline:** downgraded 91.2% from assertion to investigation (Task 5) based on planning-time verification; flagged rule-ablation deferral (Task 7/W3).
