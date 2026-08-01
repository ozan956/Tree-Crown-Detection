"""Qualitative result overlays on real images from both datasets.

Draws, per image:
  - grey box  = ground-truth tree crown
  - blue dot  = GT tree the DEEP detector reaches (WBF on VHRTrees, DeepForest on NEON)
  - red dot   = BEYOND-CEILING recovery: GT tree the deep detector MISSED but the
                traditional/integrated step found (the paper's contribution, made visible)

So red dots are literally "trees the deep detector missed that we recovered."

Run: .venv/bin/python -m improvement.qualitative
Writes results/figs/qual_vhrtrees.png and results/figs/qual_neon.png.
"""

import os
import json
import numpy as np
import cv2

GREY = (170, 170, 170)
BLUE = (200, 130, 0)      # BGR ~ Okabe-Ito blue
RED = (0, 40, 220)        # BGR red
WHITE = (255, 255, 255)


def _draw(img, gt_boxes, reached_flags, thickness=2):
    """gt_boxes: list of (x0,y0,x1,y1); reached_flags: list of bool (deep reached)."""
    out = img.copy()
    for (x0, y0, x1, y1), reached in zip(gt_boxes, reached_flags):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), GREY, 1)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if reached:
            cv2.circle(out, (cx, cy), 4, BLUE, -1)
        else:
            # beyond-ceiling recovery — bigger red dot + ring for visibility
            cv2.circle(out, (cx, cy), 6, RED, -1)
            cv2.circle(out, (cx, cy), 6, WHITE, 1)
    return out


def _legend(w=520, h=90):
    leg = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(leg, (12, 18), (32, 34), GREY, 1); cv2.putText(leg, "GT crown", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.circle(leg, (22, 58), 5, BLUE, -1); cv2.putText(leg, "reached by deep detector", (42, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.circle(leg, (300, 58), 6, RED, -1); cv2.circle(leg, (300, 58), 6, WHITE, 1)
    cv2.putText(leg, "beyond-ceiling recovery", (315, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return leg


def _grid(panels, cols=3, pad=6):
    """Stack labelled panels into a grid (all resized to a common size)."""
    H = min(p.shape[0] for p in panels)
    W = min(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (W, H)) for p in panels]
    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        while len(row) < cols:
            row.append(np.full((H, W, 3), 255, np.uint8))
        rows.append(np.hstack([np.pad(p, ((pad, pad), (pad, pad), (0, 0)), constant_values=255) for p in row]))
    return np.vstack(rows)


def vhrtrees_panels(ids=(88, 83, 72, 184, 154, 100)):  # size-varied: large->mixed->small crowns
    from evaluation.io_utils import load_gt, load_points
    from improvement.beyond_ceiling import dl_union_centers, _box_hit
    IMGDIR = "wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs"
    ann = json.load(open("data/annotations.coco.json")); names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = load_gt(); union = dl_union_centers(); integ = load_points()
    panels = []
    for iid in ids:
        img = cv2.imread(os.path.join(IMGDIR, names[iid]))
        if img is None:
            continue
        boxes = gt[iid]; uc = union.get(iid, []); ic = integ.get(iid, [])
        # a crown is "reached" if DL union hits it; "recovered" if integrated hits but DL doesn't
        flags = [_box_hit(uc, b) for b in boxes]
        show = [b for b in boxes if _box_hit(uc, b) or _box_hit(ic, b)]  # only show detected-by-something
        show_flags = [_box_hit(uc, b) for b in show]
        n_rec = sum(1 for f in show_flags if not f)
        import numpy as _np
        med_area = int(_np.median([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])) if boxes else 0
        p = _draw(img, show, show_flags)
        cv2.putText(p, f"+{n_rec} recovered", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        cv2.putText(p, f"median crown {med_area}px^2", (8, img.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        panels.append(p)
    return panels


def neon_panels(n=6):
    from deepforest import main as dfm
    from improvement.neon.cross_dataset import load_voc_boxes, _box_hit
    from improvement.neon.traditional import traditional_centers
    m = dfm.deepforest(); m.load_model("weecology/deepforest-tree")
    index = json.load(open("data_neon/index.json"))
    # rank tiles by #recoveries to pick illustrative ones
    scored = []
    for it in index:
        img = cv2.imread(os.path.join("data_neon/rgb", it["tif"]))
        if img is None:
            continue
        gt = load_voc_boxes(os.path.join("data_neon/ann", it["xml"]))
        if len(gt) < 8:
            continue
        pred = m.predict_image(path=os.path.join("data_neon/rgb", it["tif"]))
        dfc = [((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2) for r in pred.itertuples()] if pred is not None and len(pred) else []
        tr = traditional_centers(img)
        rec = sum(1 for b in gt if (_box_hit(dfc, b) or _box_hit(tr, b)) and not _box_hit(dfc, b))
        scored.append((rec, it, img, gt, dfc, tr))
    scored.sort(key=lambda x: -x[0])
    panels = []
    for rec, it, img, gt, dfc, tr in scored[:n]:
        show = [b for b in gt if _box_hit(dfc, b) or _box_hit(tr, b)]
        flags = [_box_hit(dfc, b) for b in show]
        p = _draw(img, show, flags)
        cv2.putText(p, f"+{sum(1 for f in flags if not f)} recovered", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        panels.append(p)
    return panels


def main():
    os.makedirs("results/figs", exist_ok=True)
    print("VHRTrees panels...")
    vp = vhrtrees_panels()
    grid = _grid(vp, cols=3)
    grid = np.vstack([grid, np.pad(_legend(grid.shape[1]), ((0, 0), (0, 0), (0, 0)))]) if _legend().shape[1] <= grid.shape[1] else grid
    cv2.imwrite("results/figs/qual_vhrtrees.png", grid)
    print("  wrote results/figs/qual_vhrtrees.png")
    print("NEON panels (runs DeepForest)...")
    npnl = neon_panels()
    ng = _grid(npnl, cols=3)
    cv2.imwrite("results/figs/qual_neon.png", ng)
    print("  wrote results/figs/qual_neon.png")
    cv2.imwrite("results/figs/qual_legend.png", _legend())


if __name__ == "__main__":
    main()
