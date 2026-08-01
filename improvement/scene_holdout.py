"""Scene-holdout generalization tests (Direction B).

Answers "does the method generalize to unseen scenes, or is it tuned to the test
set?" using scene prefixes in the VHRTrees test filenames as an
acquisition/region proxy (44 scene groups; 10 base-letter prefixes).

Two tests:
  1. Beyond-ceiling recovery per scene group + its correlation with detector
     strength — is the traditional recovery a targeted or universal benefit?
  2. Leave-one-scene-group-out for the precision gate — does the gate improve
     F1 on scene groups it never trained on?

Caveat: prefixes are a PROXY for acquisition/region, not confirmed climate
labels; a clean Bursa-vs-Izmir split needs VHRTrees region metadata (ref [29]).

Run: .venv/bin/python -m improvement.scene_holdout
"""

import json
import os
import re
import collections
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from evaluation.io_utils import load_gt, load_points, load_dets, dets_by_image
from improvement.beyond_ceiling import dl_union_centers, _box_hit
from improvement.precision_gate import crop_features, box_centers, is_near_box, PrecisionGate

IMG_DIR = ("wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/"
           "new_idea/beril_work_test/inputs")
GT_PATH = "data/annotations.coco.json"


def base_prefix(fn):
    m = re.match(r"([a-zA-Z]+)", fn)
    return m.group(1) if m else "#"


def _tp_mask(P, boxes, gap=10.0):
    n, m = len(P), len(boxes)
    feas = np.zeros((n, m), bool)
    for i, (x, y) in enumerate(P):
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            feas[i, j] = (x0 - gap) <= x <= (x1 + gap) and (y0 - gap) <= y <= (y1 + gap)
    mm = set()
    if n and m and feas.any():
        r, c = linear_sum_assignment(np.where(feas, 0.0, 1e6))
        for i, j in zip(r, c):
            if feas[i, j]:
                mm.add(i)
    return mm


def _prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return (2 * P * R / (P + R)) if (P + R) else 0.0


def beyond_ceiling_per_scene(names, gt):
    union = dl_union_centers()
    integ = load_points()
    grp = collections.defaultdict(lambda: {"gt": 0, "dl": 0, "int": 0})
    for iid, boxes in gt.items():
        b = base_prefix(names[iid]); uc = union.get(iid, []); ic = integ.get(iid, [])
        for box in boxes:
            grp[b]["gt"] += 1
            grp[b]["dl"] += _box_hit(uc, box)
            grp[b]["int"] += _box_hit(ic, box)
    rows = []
    for b, g in grp.items():
        if g["gt"] < 50:
            continue
        dlp = g["dl"] / g["gt"]; ip = g["int"] / g["gt"]
        rows.append((b, g["gt"], dlp, ip, ip - dlp))
    dl = np.array([r[2] for r in rows]); gain = np.array([r[4] for r in rows])
    corr = float(np.corrcoef(dl, gain)[0, 1]) if len(rows) > 2 else float("nan")
    return sorted(rows, key=lambda r: -r[1]), corr


def gate_leave_one_scene_out(names, gt, threshold=0.4):
    pts = load_points()
    wbf08 = dets_by_image(load_dets("data/wbf/best_fuse.json"), 0.8)
    rec = {}
    for iid, boxes in gt.items():
        P = pts.get(iid, [])
        r = {"near_tp": 0, "near_fp": 0, "X": [], "y": [], "ngt": len(boxes),
             "scene": base_prefix(names[iid])}
        if P:
            img = cv2.imread(os.path.join(IMG_DIR, names[iid]))
            matched = _tp_mask(P, boxes); wc = box_centers(wbf08.get(iid, []))
            for i, (x, y) in enumerate(P):
                if is_near_box(x, y, wc):
                    r["near_tp" if i in matched else "near_fp"] += 1
                else:
                    f = crop_features(img, x, y) if img is not None else None
                    if f is None:
                        r["near_tp" if i in matched else "near_fp"] += 1; continue
                    r["X"].append(f); r["y"].append(1 if i in matched else 0)
        rec[iid] = r
    scenes = sorted(set(r["scene"] for r in rec.values()))
    big = [s for s in scenes
           if sum(len(rec[i]["X"]) for i in rec if rec[i]["scene"] == s) >= 30]
    out = []
    for held in big:
        Xtr = [f for i in rec for f in rec[i]["X"] if rec[i]["scene"] != held]
        ytr = [l for i in rec for l in rec[i]["y"] if rec[i]["scene"] != held]
        if sum(ytr) < 5 or (len(ytr) - sum(ytr)) < 5:
            continue
        gate = PrecisionGate(threshold=threshold).fit(Xtr, ytr)
        btp = bfp = gtp = gfp = ngt = 0
        for i in rec:
            if rec[i]["scene"] != held:
                continue
            r = rec[i]; ngt += r["ngt"]
            btp += r["near_tp"] + sum(r["y"]); bfp += r["near_fp"] + (len(r["y"]) - sum(r["y"]))
            gtp += r["near_tp"]; gfp += r["near_fp"]
            if r["X"]:
                keep = gate.keep(r["X"]); yy = np.array(r["y"])
                gtp += int(yy[keep].sum()); gfp += int((1 - yy[keep]).sum())
        bF = _prf(btp, bfp, ngt - btp); gF = _prf(gtp, gfp, ngt - gtp)
        out.append((held, bF, gF, gF - bF))
    return out


def main():
    ann = json.load(open(GT_PATH))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = load_gt()

    rows, corr = beyond_ceiling_per_scene(names, gt)
    print("== Beyond-ceiling recovery per scene group ==")
    print(f"  {'scene':6s} {'trees':>6s} {'DL%':>6s} {'int%':>6s} {'gain_pp':>7s}")
    for b, n, dlp, ip, gain in rows:
        print(f"  {b:6s} {n:>6d} {dlp*100:>6.1f} {ip*100:>6.1f} {gain*100:>+7.1f}")
    print(f"  corr(DL-coverage, gain) = {corr:+.2f}  (negative => helps where detectors weak)")

    print("\n== Gate leave-one-scene-out (thr=0.4) ==")
    print(f"  {'held-out':9s} {'base_F1':>7s} {'gate_F1':>7s} {'dF1':>6s}")
    res = gate_leave_one_scene_out(names, gt)
    dfs = [d for _, _, _, d in res]
    for held, bF, gF, d in res:
        print(f"  {held:9s} {bF:>7.3f} {gF:>7.3f} {d:>+6.3f}")
    print(f"  gate improves F1 on {sum(1 for x in dfs if x>0)}/{len(dfs)} unseen scene groups; "
          f"mean dF1={np.mean(dfs):+.3f} median={np.median(dfs):+.3f}")


if __name__ == "__main__":
    main()
