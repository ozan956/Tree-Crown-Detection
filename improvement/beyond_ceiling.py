"""Quantify and characterize the beyond-ceiling recovery — the paper's novel core.

Measures how many GT tree crowns the integrated (traditional-recovery) method
reaches that lie beyond the coverage of the full deep-detector ensemble at its
most permissive thresholds, and characterizes those crowns (size, spectral,
texture) against the detected population.

Run: .venv/bin/python -m improvement.beyond_ceiling
Fully reproducible from the vendored data; CPU-only.
"""

import json
import os
import numpy as np
import cv2

from evaluation.io_utils import load_gt, load_points, load_dets, dets_by_image

IMG_DIR = ("wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/"
           "new_idea/beril_work_test/inputs")
GT_PATH = "data/annotations.coco.json"
DET_FILES = [
    "data/dets/swin.json", "data/dets/cascade_rcnn.json", "data/dets/detr.json",
    "data/dets/faster_rcnn.json", "data/dets/yolo.json", "data/wbf/best_fuse.json",
]


def dl_union_centers():
    """Center points of every detector's boxes at that detector's MIN score —
    the most permissive union, i.e. the deep-learning detection ceiling."""
    union = {}
    for p in DET_FILES:
        d = load_dets(p)
        dbi = dets_by_image(d, min(x["score"] for x in d))
        for iid, bs in dbi.items():
            union.setdefault(iid, []).extend(
                ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, s in bs)
    return union


def _box_hit(centers, box, gap=10.0):
    x0, y0, x1, y1 = box
    return any((x0 - gap) <= cx <= (x1 + gap) and (y0 - gap) <= cy <= (y1 + gap)
               for cx, cy in centers)


def coverage_counts():
    """Return (dl_union_cov, integrated_cov, beyond_ceiling_set, net_gain, total)."""
    gt = load_gt()
    total = sum(len(b) for b in gt.values())
    union = dl_union_centers()
    integ = load_points()
    dl_cov = integ_cov = beyond_set = 0
    for iid, boxes in gt.items():
        uc = union.get(iid, [])
        ic = integ.get(iid, [])
        for box in boxes:
            din = _box_hit(uc, box)
            iin = _box_hit(ic, box)
            dl_cov += din
            integ_cov += iin
            if iin and not din:
                beyond_set += 1
    return dl_cov, integ_cov, beyond_set, integ_cov - dl_cov, total


def characterize():
    """Median crown features for DL-reachable vs beyond-ceiling GT crowns."""
    ann = json.load(open(GT_PATH))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = load_gt()
    union = dl_union_centers()
    integ = load_points()

    def feats(img, box):
        x0, y0, x1, y1 = (int(v) for v in box)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            return None
        rgb = crop.astype(float)
        B, G, R = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(float)
        return {
            "area": (x1 - x0) * (y1 - y0), "val": hsv[:, :, 2].mean(),
            "sat": hsv[:, :, 1].mean(), "exg": 2 * G - R - B,
            "tex": cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std(),
        }

    grp = {"dl_reachable": [], "beyond_ceiling": []}
    for iid, boxes in gt.items():
        img = cv2.imread(os.path.join(IMG_DIR, names[iid]))
        if img is None:
            continue
        uc, ic = union.get(iid, []), integ.get(iid, [])
        for box in boxes:
            f = feats(img, box)
            if f is None:
                continue
            if _box_hit(uc, box):
                grp["dl_reachable"].append(f)
            elif _box_hit(ic, box):
                grp["beyond_ceiling"].append(f)
    med = {k: {ft: float(np.median([g[ft] for g in v])) for ft in ("area", "val", "sat", "exg", "tex")}
           for k, v in grp.items() if v}
    sizes = {k: len(v) for k, v in grp.items()}
    return med, sizes


def main():
    dl, integ, beyond, net, total = coverage_counts()
    print(f"DL-union coverage : {dl}/{total} = {dl/total*100:.1f}%")
    print(f"Integrated coverage: {integ}/{total} = {integ/total*100:.1f}%")
    print(f"Beyond-ceiling set : {beyond} trees ({beyond/total*100:.1f}%)")
    print(f"NET gain over ceiling: +{net} trees (+{net/total*100:.1f} pp)")
    med, sizes = characterize()
    print(f"\nCharacterization ({sizes}):")
    print(f"  {'feat':6s} {'DL-reachable':>13s} {'beyond-ceiling':>15s}")
    for ft in ("area", "val", "sat", "exg", "tex"):
        print(f"  {ft:6s} {med['dl_reachable'][ft]:>13.1f} {med['beyond_ceiling'][ft]:>15.1f}")
    ratio = med["dl_reachable"]["area"] / med["beyond_ceiling"]["area"]
    print(f"\n  beyond-ceiling crowns are {ratio:.2f}x smaller by median area")


if __name__ == "__main__":
    main()
