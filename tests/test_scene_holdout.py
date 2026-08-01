"""Locks the scene-holdout generalization findings against regression."""
import pytest
pytest.importorskip("cv2")
pytest.importorskip("sklearn")

import json
from improvement.scene_holdout import (
    base_prefix, beyond_ceiling_per_scene, gate_leave_one_scene_out,
)
from evaluation.io_utils import load_gt


def test_base_prefix():
    assert base_prefix("sa19_92_image_3_2.jpg") == "sa"
    assert base_prefix("y22_1080.jpg") == "y"
    assert base_prefix("12345.jpg") == "#"


def _names_gt():
    ann = json.load(open("data/annotations.coco.json"))
    return {im["id"]: im["file_name"] for im in ann["images"]}, load_gt()


def test_recovery_correlates_negatively_with_detector_strength():
    names, gt = _names_gt()
    rows, corr = beyond_ceiling_per_scene(names, gt)
    # the mechanistic claim: recovery helps where detectors are weak
    assert corr < -0.8, corr
    # weak-detector scenes (DL<90%) show a large positive gain
    weak = [g for _, _, dl, _, g in rows if dl < 0.90]
    assert max(weak) > 0.08  # at least one scene with >8pp gain


def test_gate_generalizes_to_unseen_scenes():
    names, gt = _names_gt()
    res = gate_leave_one_scene_out(names, gt, threshold=0.4)
    dfs = [d for _, _, _, d in res]
    # gate improves F1 on the majority of held-out scene groups
    assert sum(1 for x in dfs if x > 0) >= 0.6 * len(dfs)
    assert sum(dfs) / len(dfs) > 0  # mean improvement positive
