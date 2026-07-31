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
