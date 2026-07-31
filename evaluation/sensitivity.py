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
