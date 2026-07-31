from evaluation.io_utils import (load_gt, gt_total, load_dets, dets_by_image,
                                  load_points, box_center, xywh_to_corners)

def test_xywh_to_corners():
    assert xywh_to_corners([10, 20, 30, 40]) == (10, 20, 40, 60)

def test_box_center():
    assert box_center([10, 20, 30, 40]) == (25.0, 40.0)

def test_load_gt_shape():
    gt = load_gt()
    assert gt_total(gt) == 13552
    assert len(gt) == 222
    # every box is a 4-tuple of corners with x1>x0, y1>y0
    x0, y0, x1, y1 = gt[0][0]
    assert x1 > x0 and y1 > y0

def test_dets_by_image_threshold():
    dets = load_dets("data/dets/swin.json")
    all_n = sum(len(v) for v in dets_by_image(dets, 0.0).values())
    hi_n  = sum(len(v) for v in dets_by_image(dets, 0.8).values())
    assert all_n == 13387
    assert hi_n < all_n  # thresholding removes some

def test_load_points_shape():
    pts = load_points()
    assert len(pts) == 222
    assert sum(len(v) for v in pts.values()) == 15811
