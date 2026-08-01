"""Evaluate the trained density counter on the 222-image test set.

Two views, so the baseline is comparable to everything else in the paper:
  1. Detection P/R/F1 — peak-detect the density map into points, then score with
     the SAME one-to-one Hungarian matching (gap=10) used for all other methods.
  2. Counting MAE/RMSE — |predicted integral - true count|, the metric density
     methods are usually judged on.

Run: .venv/bin/python -m improvement.density.evaluate
Writes results/density_results.md.
"""

import os
import json
import numpy as np
import cv2
import torch

from improvement.density.data import OUTPUT_STRIDE
from improvement.density.model import DensityCSRNetLite
from evaluation.io_utils import load_gt
from evaluation.matching import match_one_to_one_points
from evaluation.metrics import pr_f1

TEST_DIR = "wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs"
GT_PATH = "data/annotations.coco.json"
CKPT = "results/density_model.pt"


def density_to_points(dm, stride=OUTPUT_STRIDE, min_distance=3, rel_thr=0.5):
    """Peak-detect a density map into (x, y) points in input-image pixels."""
    from scipy.ndimage import maximum_filter
    if dm.max() <= 0:
        return []
    thr = rel_thr * dm.max()
    mx = maximum_filter(dm, size=2 * min_distance + 1)
    peaks = np.argwhere((dm == mx) & (dm > thr))
    return [((c + 0.5) * stride, (r + 0.5) * stride) for r, c in peaks]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ann = json.load(open(GT_PATH))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = load_gt()
    total_gt = sum(len(b) for b in gt.values())

    model = DensityCSRNetLite(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    tp = fp = fn = 0
    abs_errs, sq_errs = [], []
    with torch.no_grad():
        for iid, boxes in gt.items():
            img = cv2.imread(os.path.join(TEST_DIR, names[iid]))
            if img is None:
                continue
            rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - mean) / std
            t = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(device)
            dm = model(t)[0, 0].cpu().numpy()
            pred_count = float(dm.sum())
            pts = density_to_points(dm)
            t_, f_, n_ = match_one_to_one_points(pts, boxes, gap=10.0)
            tp += t_; fp += f_; fn += n_
            abs_errs.append(abs(pred_count - len(boxes)))
            sq_errs.append((pred_count - len(boxes)) ** 2)

    m = pr_f1(tp, fp, fn)
    mae = float(np.mean(abs_errs)); rmse = float(np.sqrt(np.mean(sq_errs)))
    lines = [
        "# Density-map counter — test-set results",
        "",
        "Modern learned baseline (CSRNet-style density regression) trained on",
        "VHRTrees train, evaluated on the 222-image test set.",
        "",
        "## Detection (peak-detected points, one-to-one matched, gap=10)",
        f"- precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}",
        f"- tp={m['tp']} fp={m['fp']} fn={m['fn']} (total GT {total_gt})",
        "",
        "## Counting",
        f"- MAE {mae:.2f} trees/image, RMSE {rmse:.2f}",
        "",
        "## Reference (this paper, one-to-one)",
        "- WBF ensemble: P 0.968 R 0.878 F1 0.921",
        "- Integrated: P 0.833 R 0.971 F1 0.897",
        "- Integrated + gate (recovery-preserving): P 0.893 R 0.957 F1 0.924",
    ]
    os.makedirs("results", exist_ok=True)
    open("results/density_results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
