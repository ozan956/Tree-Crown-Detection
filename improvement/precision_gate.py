"""Tier-A precision gate for the rule-based integrated tree-crown detector.

Diagnosis (see docs/research/2026-08-01-novelty-and-improvement-roadmap.md and
results/IMPROVEMENT_FINDINGS.md): of the integrated method's 2647 false
positives, ~2259 are *traditional-only* points that lie far (>=25 px) from any
reliable deep-learning box (score>=0.8). These fire on grass, bare soil, and
rooftops. Points *near* a reliable DL box are trusted as-is (they are 97%
precise). This module gates only the far-from-box points, using cheap
appearance features of a crown-sized crop, so that genuine traditional-only
tree recoveries are kept while non-tree firings are dropped.

The gate is a small logistic classifier over 8 interpretable features
(brightness, saturation, excess-green, greenness, texture, mean R/G/B). Trees
in this Mediterranean dataset are darker, greener, and more textured than the
dry grass / soil around them, so the features separate the two populations.

Nothing here re-runs detectors or the traditional pipeline; it post-processes
the saved integrated points against the saved WBF boxes and the source images,
so it is fully CPU-only and reproducible from the vendored data.
"""

import numpy as np
import cv2

# Radius (px) of the crop analysed around each candidate point. ~0.37x the
# median GT crown diameter (43.5 px) — a patch centred on the crown interior.
CROP_RADIUS = 16
# A candidate point is "near" a reliable DL box if a WBF box centre (score>=
# NEAR_SCORE) lies within NEAR_DIST px. Near points bypass the gate.
NEAR_DIST = 25.0
NEAR_SCORE = 0.8

FEATURE_NAMES = ["val", "sat", "exg", "green", "texture", "R", "G", "B",
                 "dist_wbf", "compactness"]


def box_centers(dets_by_img_entry):
    """Centres of (corner_box, score) tuples as returned by dets_by_image."""
    return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, s in dets_by_img_entry]


def crop_features(img_bgr, x, y, r=CROP_RADIUS, wbf_centers=None):
    """Appearance feature vector for the crop centred at (x, y).

    Returns None if the crop is empty (point on the image border). img_bgr is a
    BGR uint8 image as read by cv2.imread. When `wbf_centers` is given, two
    context features are appended (distance to the nearest reliable DL box, and
    the circularity of the crop's dark blob) — a tree crown is compact and its
    recoveries cluster near canopy, while grass/soil firings are irregular and
    scattered; these lift precision (permutation-importance: dist_wbf is the top
    non-texture feature). Without `wbf_centers` the original 8-D vector is
    returned, so old callers/tests still work.
    """
    h, w = img_bgr.shape[:2]
    x0, y0 = max(0, int(x - r)), max(0, int(y - r))
    x1, y1 = min(w, int(x + r)), min(h, int(y + r))
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    rgbf = crop.astype(float)
    B, G, R = rgbf[:, :, 0].mean(), rgbf[:, :, 1].mean(), rgbf[:, :, 2].mean()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(float)
    val = hsv[:, :, 2].mean()
    sat = hsv[:, :, 1].mean()
    texture = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()
    exg = 2 * G - R - B          # excess-green vegetation index
    green = G - max(R, B)
    feats = [val, sat, exg, green, texture, R, G, B]
    if wbf_centers is None:
        return feats
    if len(wbf_centers):
        dist_wbf = float(np.min(np.linalg.norm(np.asarray(wbf_centers) - [x, y], axis=1)))
    else:
        dist_wbf = 999.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compact = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        compact = 4 * np.pi * cv2.contourArea(c) / (per * per) if per > 0 else 0.0
    return feats + [dist_wbf, compact]


def is_near_box(x, y, wbf_centers, dist=NEAR_DIST):
    """True if a reliable DL box centre lies within `dist` of (x, y)."""
    if len(wbf_centers) == 0:
        return False
    d = np.linalg.norm(np.asarray(wbf_centers) - np.array([x, y]), axis=1)
    return bool(d.min() < dist)


def split_points(points, wbf_centers, dist=NEAR_DIST):
    """Partition points into (near_idx, far_idx) by proximity to a DL box."""
    near, far = [], []
    for i, (x, y) in enumerate(points):
        (near if is_near_box(x, y, wbf_centers, dist) else far).append(i)
    return near, far


def merge_close_points(points, min_dist=26.0):
    """Collapse points closer than min_dist to their first occurrence (greedy).
    Kills over-split duplicates (one crown -> several local maxima). Returns the
    kept subset in input order. min_dist default ~0.6x the VHRTrees median crown
    diameter (43px).
    ponytail: O(n^2) greedy; fine for <~250 pts/image, swap to KDTree if it grows.
    """
    kept = []
    for p in points:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_dist ** 2 for q in kept):
            kept.append(p)
    return kept


class PrecisionGate:
    """Gate deciding whether a far-from-box point is a real tree.

    Fit on (features, label) pairs where label=1 means the point matched a GT
    tree. `keep(X)` compares the predicted probability to `threshold`. Uses a
    gradient-boosted tree (nonlinear boundary on the crop features); it beat
    logistic regression by ~0.4 F1 in the image-disjoint comparison at no extra
    cost. Pass `kind="logistic"` for the linear model.
    """

    def __init__(self, threshold=0.5, kind="hgb"):
        # Imported lazily so the module imports without scikit-learn present
        # (e.g. for the geometry-only tests).
        if kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.clf = HistGradientBoostingClassifier()
        self.threshold = threshold
        self._fitted = False

    def fit(self, X, y):
        self.clf.fit(np.asarray(X), np.asarray(y))
        self._fitted = True
        return self

    def proba(self, X):
        if not self._fitted:
            raise RuntimeError("PrecisionGate must be fit before proba().")
        return self.clf.predict_proba(np.asarray(X))[:, 1]

    def keep(self, X):
        """Boolean mask: keep far-from-box points whose P(tree) >= threshold."""
        return self.proba(X) >= self.threshold
