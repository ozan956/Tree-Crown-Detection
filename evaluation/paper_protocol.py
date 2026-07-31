"""Reproduction of the paper's ORIGINAL v1 recall metric.

This module exists for documentation/erratum purposes: it reproduces the
legacy per-center matching protocol (``match_center_in_box``) that the
published paper actually used, so that the paper's headline numbers
(96.0% integrated recall, 91.2% DL-alone recall) can be locked down and
audited against the vendored data snapshot.

IMPORTANT: this is the SUPERSEDED v1 metric, not the corrected headline.
``match_center_in_box`` is a per-center greedy matcher that under-counts
true positives when GT crowns overlap (a center in the overlap region can
only ever credit one box). The corrected, one-to-one metric (97.14%) is
computed elsewhere (see Task 6 / ``match_one_to_one_points``) and is the
number that should be reported going forward. Nothing in this module
should be treated as the current headline result — it is kept solely to
reproduce and document what the paper originally published.
"""

from evaluation.io_utils import load_gt, load_dets, dets_by_image
from evaluation.matching import match_center_in_box


def paper_recall_points(points_by_image, gt, gap=10.0):
    """Reproduce the paper's v1 recall for a point set (legacy per-center matching).

    Uses the legacy ``match_center_in_box`` matcher (order-dependent,
    under-counts on overlapping GT crowns) to match this exactly as the
    paper originally computed it for the integrated method. This is the
    ORIGINAL v1 metric, not the corrected one-to-one headline.

    Returns {"matched", "total", "recall"} counting unique matched GT boxes.
    """
    total = sum(len(v) for v in gt.values())
    matched = 0
    for iid, boxes in gt.items():
        centers = points_by_image.get(iid, [])
        matched += len(match_center_in_box(centers, boxes, gap))
    return {"matched": matched, "total": total, "recall": matched / total}


def paper_recall_boxes(dets_by_img, gt, gap=10.0):
    """Reproduce the paper's v1 recall for a box/detection set (legacy per-center matching).

    Same legacy protocol as ``paper_recall_points``, but derives centers
    from detection box centers — used for DL-alone / WBF box sets. This is
    the ORIGINAL v1 metric, not the corrected one-to-one headline.

    Returns {"matched", "total", "recall"} counting unique matched GT boxes.
    """
    total = sum(len(v) for v in gt.values())
    matched = 0
    for iid, boxes in gt.items():
        centers = [ (( (b[0]+b[2])/2.0, (b[1]+b[3])/2.0 )) for (b, s) in dets_by_img.get(iid, []) ]
        matched += len(match_center_in_box(centers, boxes, gap))
    return {"matched": matched, "total": total, "recall": matched / total}


def investigate_dl_baseline():
    """Investigate the paper's reported DL-alone v1 recall of 91.2% (12353 boxes).

    The paper's 91.2% (12353) does not reproduce from best_fuse.json with a
    single center-in-box rule. Sweep thresholds and report the curve so the
    discrepancy is documented rather than hidden. This is provenance
    investigation for the SUPERSEDED v1 figure only; it does not affect the
    corrected one-to-one headline computed elsewhere.
    """
    gt = load_gt()
    wbf = load_dets("data/wbf/best_fuse.json")
    rows = []
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = paper_recall_boxes(dets_by_image(wbf, thr), gt, gap=10.0)
        rows.append((thr, r["matched"], r["recall"]))
    closest = min(rows, key=lambda t: abs(t[1] - 12353))
    return {"paper_reported": 12353, "curve": rows, "closest_threshold": closest}
