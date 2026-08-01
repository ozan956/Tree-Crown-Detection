"""Reproduce the old-vs-new traditional-feature comparison (pixel-level AUC).

Labels pixels tree/non-tree from GT boxes and scores each feature map as a
tree-pixel classifier. Compares the paper's shipped features (single Gabor
kernel + fixed-HSV band) against the faithful features in
improvement/features.py. CPU-only; samples images for speed.

Run: .venv/bin/python -m improvement.compare_features
"""

import json
import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

from improvement.features import gabor_bank_energy, color_invariant, joint_probability

IMG_DIR = ("wbf_test-20260731T192532Z-1-002/wbf_test/beril-ozan-cem-work/"
           "new_idea/beril_work_test/inputs")
GT_PATH = "data/annotations.coco.json"


def _old_color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return m.astype(np.float32) / 255.0


def _old_gabor(bgr):
    k = cv2.getGaborKernel((3, 3), 2.0, np.pi / 4, 3.0, 0.5, 0, ktype=cv2.CV_32F)
    chans = [cv2.filter2D(bgr[:, :, c], cv2.CV_8UC3, k) for c in range(3)]
    g = cv2.cvtColor(cv2.merge(chans), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (255 - mask).astype(np.float32) / 255.0


def _labels(shape, boxes):
    lab = np.zeros(shape[:2], np.uint8)
    for (x0, y0, x1, y1) in boxes:
        lab[int(y0):int(y1), int(x0):int(x1)] = 1
    return lab.ravel()


def main(step=6, limit=40):
    ann = json.load(open(GT_PATH))
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    by_img = {}
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h))

    ids = sorted(by_img.keys())[::step][:limit]
    acc = {k: [] for k in ("old_color", "new_color", "old_gabor", "new_gabor",
                            "old_joint", "new_joint")}
    for iid in ids:
        img = cv2.imread(os.path.join(IMG_DIR, names[iid]))
        if img is None:
            continue
        lab = _labels(img.shape, by_img[iid])
        if lab.sum() in (0, len(lab)):
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maps = {
            "old_color": _old_color(img), "new_color": color_invariant(img),
            "old_gabor": _old_gabor(img), "new_gabor": gabor_bank_energy(gray),
            "old_joint": 0.5 * _old_color(img) + 0.5 * _old_gabor(img),
            "new_joint": joint_probability(img),
        }
        for k, m in maps.items():
            acc[k].append(roc_auc_score(lab, m.ravel()))

    print(f"Pixel-level tree/background AUC over {len(acc['old_joint'])} images:")
    print(f"  {'feature':10s} {'old':>6s} {'new':>6s}")
    for a, b in (("old_color", "new_color"), ("old_gabor", "new_gabor"),
                 ("old_joint", "new_joint")):
        print(f"  {a[4:]:10s} {np.mean(acc[a]):>6.3f} {np.mean(acc[b]):>6.3f}")


if __name__ == "__main__":
    main()
