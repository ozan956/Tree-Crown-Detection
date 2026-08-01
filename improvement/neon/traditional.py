"""Standalone traditional tree-center detector for the NEON cross-dataset test.

Reuses the faithful features (illumination-invariant colour + Gabor bank) from
improvement/features.py, then the classic joint-map -> Otsu -> distance-transform
-> local-maxima pipeline the paper describes, to produce candidate tree centers.
This is the "traditional" side of the beyond-ceiling test on NEON, independent of
any deep detector. CPU-only.
"""

import numpy as np
import cv2
from scipy.ndimage import maximum_filter, label

from improvement.features import joint_probability


def traditional_centers(bgr, min_distance=6, rel_thr=0.3, min_area=8):
    """Detect candidate tree centers via the traditional joint-map + watershed
    + local-maxima pipeline. Returns list of (x, y) in image pixels.

    min_distance: peak neighbourhood (px) — set near expected crown radius.
    rel_thr: fraction of the distance-transform max a peak must exceed.
    """
    J = joint_probability(bgr)                         # [0,1] tree-crown prob
    prob = (J * 255).astype(np.uint8)
    # Otsu threshold + opening to clean small vegetation
    _, binm = cv2.threshold(prob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # distance transform -> local maxima = crown centers
    dt = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
    if dt.max() <= 0:
        return []
    thr = rel_thr * dt.max()
    mx = maximum_filter(dt, size=2 * min_distance + 1)
    peaks_mask = (dt == mx) & (dt > thr)
    # collapse connected peak plateaus to their centroid, drop tiny specks
    lab, n = label(peaks_mask)
    centers = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) == 0:
            continue
        # require the supporting segment to be crown-sized (min_area)
        if binm[int(ys.mean()), int(xs.mean())] == 0 and len(xs) < min_area:
            continue
        centers.append((float(xs.mean()), float(ys.mean())))
    return centers
