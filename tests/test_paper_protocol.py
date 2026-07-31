from evaluation.paper_protocol import paper_recall_points
from evaluation.io_utils import load_gt, load_points

def test_integrated_reproduces_96_percent():
    gt = load_gt()
    pts = load_points()
    r = paper_recall_points(pts, gt, gap=10.0)
    assert r["total"] == 13552
    # Paper reports 13012 / 13552 = 96.0%. Lock within +/- 1 box.
    assert abs(r["matched"] - 13012) <= 1, r["matched"]
    assert abs(r["recall"] - 0.960) < 0.002
