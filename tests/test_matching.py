from evaluation.matching import (iou, match_center_in_box,
                                  match_one_to_one_points, match_one_to_one_iou)

def test_iou_identical():
    assert iou((0,0,10,10), (0,0,10,10)) == 1.0

def test_iou_disjoint():
    assert iou((0,0,10,10), (100,100,110,110)) == 0.0

def test_iou_half():
    # overlap area 50, union 150 -> 1/3
    assert abs(iou((0,0,10,10), (5,0,15,10)) - (50/150)) < 1e-9

def test_center_in_box_matches_both(two_gt_boxes):
    centers = [(5,5), (105,105)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == {0, 1}

def test_center_in_box_double_count_is_one_box(two_gt_boxes):
    # three centers all in box 0 -> only box 0 matched (set semantics)
    centers = [(1,1), (2,2), (3,3)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == {0}

def test_center_gap_boundary(two_gt_boxes):
    # center just outside box 0 by 5px; gap=0 misses, gap=10 hits
    centers = [(15, 5)]
    assert match_center_in_box(centers, two_gt_boxes, gap=0) == set()
    assert match_center_in_box(centers, two_gt_boxes, gap=10) == {0}

def test_one_to_one_points_penalizes_extra(two_gt_boxes):
    # 3 centers in box0, 0 in box1: one-to-one -> tp=1, fp=2, fn=1
    centers = [(1,1), (2,2), (3,3)]
    tp, fp, fn = match_one_to_one_points(centers, two_gt_boxes, gap=0)
    assert (tp, fp, fn) == (1, 2, 1)

def test_one_to_one_points_perfect(two_gt_boxes):
    centers = [(5,5), (105,105)]
    assert match_one_to_one_points(centers, two_gt_boxes, gap=0) == (2, 0, 0)

def test_one_to_one_iou_perfect(two_gt_boxes):
    preds = [(0,0,10,10), (100,100,110,110)]
    assert match_one_to_one_iou(preds, two_gt_boxes, iou_thr=0.5) == (2, 0, 0)

def test_one_to_one_iou_no_match(two_gt_boxes):
    preds = [(0,0,3,3)]  # iou with box0 = 9/100 < 0.5
    assert match_one_to_one_iou(preds, two_gt_boxes, iou_thr=0.5) == (0, 1, 2)

def test_center_in_box_legacy_is_order_dependent_on_overlap(two_overlapping_boxes):
    # a single center inside BOTH boxes credits only the first (lowest index) box:
    # documented legacy (per-center break) behavior.
    centers = [(7.0, 7.0)]
    assert match_center_in_box(centers, two_overlapping_boxes, gap=0) == {0}
    # reversing box order flips which single box is credited -> proves order dependence
    reversed_boxes = list(reversed(two_overlapping_boxes))
    assert match_center_in_box(centers, reversed_boxes, gap=0) == {0}

def test_one_to_one_points_no_double_credit_on_overlap(two_overlapping_boxes):
    # ONE center in the overlap of two boxes: one-to-one credits exactly ONE tree
    # (tp=1), never both. The other box is a false negative. No double-credit.
    centers = [(7.0, 7.0)]
    tp, fp, fn = match_one_to_one_points(centers, two_overlapping_boxes, gap=0)
    assert (tp, fp, fn) == (1, 0, 1)

def test_one_to_one_points_two_centers_two_overlapping_boxes(two_overlapping_boxes):
    # two distinct centers, each cleanly in one box's exclusive region -> both matched
    centers = [(2.0, 2.0), (13.0, 13.0)]
    tp, fp, fn = match_one_to_one_points(centers, two_overlapping_boxes, gap=0)
    assert (tp, fp, fn) == (2, 0, 0)
