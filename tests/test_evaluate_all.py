from evaluation.evaluate_all import build_table

def test_table_has_all_methods():
    rows = build_table(score_thr=0.8, gap=10.0)
    methods = {r["method"] for r in rows}
    assert {"swin","cascade_rcnn","detr","faster_rcnn","yolo","WBF","integrated"} <= methods

def test_integrated_row_is_point_only_and_reproduces():
    rows = {r["method"]: r for r in build_table()}
    integ = rows["integrated"]
    assert integ["AP"] is None and integ["AP50"] is None  # point-only
    assert abs(integ["paper_recall"] - 0.960) < 0.002       # reproduction holds

def test_strict_precision_present_for_box_methods():
    rows = {r["method"]: r for r in build_table()}
    for m in ["swin","WBF"]:
        assert 0.0 <= rows[m]["strict_precision"] <= 1.0
        assert 0.0 <= rows[m]["strict_recall"] <= 1.0

def test_wbf_ap50_standard():
    # AP/AP50 must be standard COCO mAP over ALL detections (threshold-free),
    # not score-filtered. Pins the standard-mAP behavior so a future
    # prefilter regression is caught.
    rows = {r["method"]: r for r in build_table()}
    assert 0.80 < rows["WBF"]["AP50"] < 0.90
    assert 0.80 < rows["swin"]["AP50"] < 0.90
