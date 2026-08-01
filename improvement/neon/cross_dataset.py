"""Cross-dataset beyond-ceiling test on NEON (Direction A).

Fair design: the deep detector is DeepForest (the field-standard, NEON-trained
tree detector), so it is competent on its home data — no domain-shift confound.
The question: does the traditional operator recover true crowns beyond
DeepForest's ceiling on a completely different biome/sensor (US airborne forest)
than VHRTrees (Mediterranean satellite), and are those crowns small/faint —
replicating the VHRTrees mechanism?

Run: .venv/bin/python -m improvement.neon.cross_dataset
Writes results/NEON_CROSS_DATASET.md and results/figs/fig8_neon_cross.png.
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from improvement.neon.traditional import traditional_centers

NEON = "data_neon"


def load_voc_boxes(xml_path):
    """PASCAL-VOC XML -> list of (x0,y0,x1,y1) corner boxes."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append((float(b.find("xmin").text), float(b.find("ymin").text),
                      float(b.find("xmax").text), float(b.find("ymax").text)))
    return boxes


def _box_hit(centers, box, gap=10.0):
    x0, y0, x1, y1 = box
    return any((x0 - gap) <= cx <= (x1 + gap) and (y0 - gap) <= cy <= (y1 + gap)
               for cx, cy in centers)


def _one_to_one_points(centers, boxes, gap=10.0):
    n, m = len(centers), len(boxes)
    feas = np.zeros((n, m), bool)
    for i, (x, y) in enumerate(centers):
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            feas[i, j] = (x0 - gap) <= x <= (x1 + gap) and (y0 - gap) <= y <= (y1 + gap)
    tp = 0
    if n and m and feas.any():
        r, c = linear_sum_assignment(np.where(feas, 0.0, 1e6))
        tp = int(sum(feas[i, j] for i, j in zip(r, c)))
    return tp, n - tp, m - tp


def run(limit=None, score_thr=0.3):
    from deepforest import main as dfm
    model = dfm.deepforest()
    model.load_model("weecology/deepforest-tree")

    index = json.load(open(f"{NEON}/index.json"))
    if limit:
        index = index[:limit]

    # coverage counters
    gt_total = 0
    df_cov = 0        # GT boxes DeepForest reaches
    union_cov = 0     # GT boxes reached by DeepForest OR traditional
    beyond = 0        # reached by traditional but NOT DeepForest
    # size of recovered vs DeepForest-reached crowns
    df_areas, beyond_areas = [], []
    # standalone one-to-one for both detectors
    df_tp = df_fp = df_fn = 0
    tr_tp = tr_fp = tr_fn = 0

    n_used = 0
    for it in index:
        tif = os.path.join(NEON, "rgb", it["tif"])
        xml = os.path.join(NEON, "ann", it["xml"])
        img = cv2.imread(tif)
        if img is None or not os.path.exists(xml):
            continue
        gt = load_voc_boxes(xml)
        if not gt:
            continue
        n_used += 1
        gt_total += len(gt)

        # DeepForest boxes -> centers (score-thresholded)
        pred = model.predict_image(path=tif)
        if pred is not None and len(pred):
            pred = pred[pred["score"] >= score_thr]
            df_centers = [((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2)
                          for r in pred.itertuples()]
        else:
            df_centers = []

        # traditional centers
        tr_centers = traditional_centers(img)

        union_centers = df_centers + tr_centers
        for box in gt:
            d = _box_hit(df_centers, box)
            u = _box_hit(union_centers, box)
            df_cov += d; union_cov += u
            area = (box[2] - box[0]) * (box[3] - box[1])
            if d:
                df_areas.append(area)
            elif u:  # beyond-ceiling: traditional got it, DeepForest didn't
                beyond += 1; beyond_areas.append(area)

        t, f, n = _one_to_one_points(df_centers, gt); df_tp += t; df_fp += f; df_fn += n
        t, f, n = _one_to_one_points(tr_centers, gt); tr_tp += t; tr_fp += f; tr_fn += n

    def prf(tp, fp, fn):
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        return P, R, (2 * P * R / (P + R) if P + R else 0.0)

    res = {
        "images": n_used, "gt_total": gt_total,
        "deepforest_cov": df_cov, "deepforest_cov_pct": 100 * df_cov / gt_total,
        "union_cov": union_cov, "union_cov_pct": 100 * union_cov / gt_total,
        "beyond_ceiling": beyond, "beyond_pct": 100 * beyond / gt_total,
        "df_area_median": float(np.median(df_areas)) if df_areas else 0.0,
        "beyond_area_median": float(np.median(beyond_areas)) if beyond_areas else 0.0,
        "deepforest_prf": prf(df_tp, df_fp, df_fn),
        "traditional_prf": prf(tr_tp, tr_fp, tr_fn),
    }
    return res, df_areas, beyond_areas


if __name__ == "__main__":
    res, _, _ = run()
    print(json.dumps(res, indent=2))
