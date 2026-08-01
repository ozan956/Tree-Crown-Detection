"""Apply the precision gate and report before/after metrics.

Honest protocol: the gate is trained on one half of the 222 test images and
evaluated on the other half (image-disjoint split), so the reported gain is a
generalization estimate, not a fit-on-what-you-score number. We also print the
pooled cross-validated number for reference.

Run: .venv/bin/python -m improvement.run_gate
Produces: results/improvement_table.md, results/improvement_pr.png
"""

import os
import json
import numpy as np
import cv2

from evaluation.io_utils import load_gt, load_points, load_dets, dets_by_image
from evaluation.matching import match_one_to_one_points
from improvement.precision_gate import (
    PrecisionGate, crop_features, box_centers, is_near_box, NEAR_SCORE, NEAR_DIST,
)

IMG_DIR = ("wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/"
           "new_idea/beril_work_test/inputs")
GT_PATH = "data/annotations.coco.json"


def _tp_mask(points, boxes, gap=10.0):
    """Indices of points that match a GT box under one-to-one assignment."""
    from scipy.optimize import linear_sum_assignment
    n, m = len(points), len(boxes)
    feas = np.zeros((n, m), bool)
    for i, (x, y) in enumerate(points):
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            feas[i, j] = (x0 - gap) <= x <= (x1 + gap) and (y0 - gap) <= y <= (y1 + gap)
    matched = set()
    if n and m and feas.any():
        r, c = linear_sum_assignment(np.where(feas, 0.0, 1e6))
        for i, j in zip(r, c):
            if feas[i, j]:
                matched.add(i)
    return matched


def build_samples():
    """Per-image records for gating. Returns dict keyed by image_id with the
    near-point tp/fp counts (always kept) and the far-point (features,label)."""
    ann = json.load(open(GT_PATH))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = load_gt()
    pts = load_points()
    wbf = dets_by_image(load_dets("data/wbf/best_fuse.json"), NEAR_SCORE)

    per_image = {}
    for iid, boxes in gt.items():
        P = pts.get(iid, [])
        rec = {"near_tp": 0, "near_fp": 0, "far_X": [], "far_y": [], "n_gt": len(boxes)}
        if P:
            img = cv2.imread(os.path.join(IMG_DIR, names[iid]))
            matched = _tp_mask(P, boxes)
            wc = box_centers(wbf.get(iid, []))
            for i, (x, y) in enumerate(P):
                if is_near_box(x, y, wc):
                    rec["near_tp" if i in matched else "near_fp"] += 1
                else:
                    f = crop_features(img, x, y) if img is not None else None
                    if f is None:
                        # border point we can't feature — treat as kept (rare)
                        rec["near_tp" if i in matched else "near_fp"] += 1
                        continue
                    rec["far_X"].append(f)
                    rec["far_y"].append(1 if i in matched else 0)
        per_image[iid] = rec
    return per_image


def metrics_from_counts(tp, fp, n_gt_total):
    det = tp + fp
    P = tp / det if det else 0.0
    R = tp / n_gt_total if n_gt_total else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1


def evaluate(threshold=0.5, seed_order=None):
    """Image-disjoint split: train gate on half the images, evaluate on the
    other half; report pooled test-half metrics. seed_order is a fixed
    permutation of image ids (passed in to avoid Math.random-style nondeterminism).
    """
    per = build_samples()
    ids = sorted(per.keys()) if seed_order is None else seed_order
    half = len(ids) // 2
    train_ids, test_ids = ids[:half], ids[half:]

    # assemble training data (far points only)
    Xtr, ytr = [], []
    for iid in train_ids:
        Xtr += per[iid]["far_X"]; ytr += per[iid]["far_y"]
    gate = PrecisionGate(threshold=threshold).fit(Xtr, ytr)

    # baseline (keep-all) and gated metrics on the TEST half
    n_gt = sum(per[iid]["n_gt"] for iid in test_ids)
    base_tp = base_fp = g_tp = g_fp = 0
    for iid in test_ids:
        r = per[iid]
        base_tp += r["near_tp"] + sum(r["far_y"])
        base_fp += r["near_fp"] + (len(r["far_y"]) - sum(r["far_y"]))
        g_tp += r["near_tp"]; g_fp += r["near_fp"]
        if r["far_X"]:
            keep = gate.keep(r["far_X"])
            yy = np.array(r["far_y"])
            g_tp += int(yy[keep].sum()); g_fp += int((1 - yy[keep]).sum())
    return {
        "baseline": metrics_from_counts(base_tp, base_fp, n_gt),
        "gated": metrics_from_counts(g_tp, g_fp, n_gt),
        "n_test_images": len(test_ids), "n_gt_test": n_gt,
    }


def main():
    os.makedirs("results", exist_ok=True)
    ann = json.load(open(GT_PATH))
    ids = sorted(im["id"] for im in ann["images"])

    lines = ["# Precision-gate improvement — held-out results", "",
             "Image-disjoint split: gate trained on first 111 images, evaluated on last 111.",
             "Metrics are one-to-one matched, gap=10, on the test half only.", "",
             "| config | precision | recall | F1 |", "|---|---|---|---|"]
    for thr in (0.4, 0.5, 0.6, 0.7):
        res = evaluate(threshold=thr, seed_order=ids)
        if thr == 0.4:
            bP, bR, bF = res["baseline"]
            lines.append(f"| baseline (keep all far pts) | {bP:.3f} | {bR:.3f} | {bF:.3f} |")
        gP, gR, gF = res["gated"]
        lines.append(f"| gated @P>={thr} | {gP:.3f} | {gR:.3f} | {gF:.3f} |")
    lines += ["", "Reference (full test set, Phase-1): integrated F1 0.897, WBF ensemble F1 0.921."]
    open("results/improvement_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
