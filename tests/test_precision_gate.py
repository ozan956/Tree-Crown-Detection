import numpy as np
from improvement.precision_gate import (
    crop_features, box_centers, is_near_box, split_points, PrecisionGate,
    merge_close_points,
)


def test_box_centers():
    # (corner_box, score) tuples -> centers
    entry = [((0.0, 0.0, 10.0, 10.0), 0.9), ((20.0, 20.0, 40.0, 40.0), 0.8)]
    assert box_centers(entry) == [(5.0, 5.0), (30.0, 30.0)]


def test_is_near_box():
    centers = [(100.0, 100.0)]
    assert is_near_box(110.0, 100.0, centers, dist=25.0) is True   # 10px away
    assert is_near_box(200.0, 200.0, centers, dist=25.0) is False  # far
    assert is_near_box(5.0, 5.0, [], dist=25.0) is False           # no boxes


def test_split_points():
    pts = [(10, 10), (500, 500)]          # first near, second far
    near, far = split_points(pts, [(12.0, 10.0)], dist=25.0)
    assert near == [0] and far == [1]


def test_crop_features_shape_and_border():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 120  # green-ish
    f = crop_features(img, 32, 32, r=16)
    assert f is not None and len(f) == 8          # 8-D without context (back-compat)
    f2 = crop_features(img, 32, 32, r=16, wbf_centers=[(0.0, 0.0)])
    assert len(f2) == 10                          # +dist_wbf +compactness
    assert f2[8] > 0                              # dist to (0,0) box is positive
    # a point whose crop is fully outside the image yields None
    assert crop_features(img, -100, -100, r=4) is None


def test_merge_close_points():
    pts = [(10, 10), (12, 11), (100, 100)]        # first two within 26px -> merge
    kept = merge_close_points(pts, min_dist=26.0)
    assert kept == [(10, 10), (100, 100)]
    # nothing merges when all far apart
    assert merge_close_points([(0, 0), (50, 50)], 26.0) == [(0, 0), (50, 50)]


def test_gate_learns_separable_features():
    # class 1 clusters high, class 0 low on feature 0 — gate must separate them
    rng = np.random.RandomState(0)
    X1 = rng.normal(10, 1, (50, 8)); y1 = np.ones(50)
    X0 = rng.normal(0, 1, (50, 8)); y0 = np.zeros(50)
    X = np.vstack([X1, X0]); y = np.concatenate([y1, y0])
    gate = PrecisionGate(threshold=0.5).fit(X, y)
    # a clearly-class-1 sample is kept, a clearly-class-0 sample is dropped
    assert gate.keep([[10] * 8])[0] == True
    assert gate.keep([[0] * 8])[0] == False
