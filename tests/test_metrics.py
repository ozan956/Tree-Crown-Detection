from evaluation.metrics import pr_f1, coco_map
from evaluation.io_utils import load_dets

def test_pr_f1_basic():
    m = pr_f1(tp=8, fp=2, fn=4)
    assert abs(m["precision"] - 0.8) < 1e-9
    assert abs(m["recall"] - (8/12)) < 1e-9
    assert abs(m["f1"] - (2*0.8*(8/12))/(0.8+8/12)) < 1e-9

def test_pr_f1_zero_safe():
    assert pr_f1(0, 0, 0) == {"precision":0.0,"recall":0.0,"f1":0.0,"tp":0,"fp":0,"fn":0}

def test_coco_map_swin_runs():
    dets = load_dets("data/dets/swin.json")
    m = coco_map("data/annotations.coco.json", dets)
    # AP50 for a trained tree detector should be a sane, high-ish number in (0,1]
    assert 0.0 < m["AP50"] <= 1.0
    assert 0.0 <= m["AP"] <= m["AP50"]  # AP (averaged over IoUs) <= AP50
