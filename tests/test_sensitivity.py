from evaluation.sensitivity import wbf_pr_curve, integrated_point

def test_pr_curve_monotone_recall():
    curve = wbf_pr_curve([0.9, 0.7, 0.5, 0.3, 0.1])
    recalls = [c["recall"] for c in curve]
    # lower threshold -> more predictions -> recall non-decreasing
    assert recalls == sorted(recalls)

def test_integrated_point_in_range():
    p = integrated_point()
    assert 0.0 < p["recall"] <= 1.0 and 0.0 < p["precision"] <= 1.0
