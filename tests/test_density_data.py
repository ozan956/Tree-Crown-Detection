"""Tests for density-target generation — the sum-preserving property is the
one that must hold (integral of density == tree count) or the counting metric
is meaningless. These do not require torch/GPU."""
import numpy as np
import pytest

# _density_target imports torch at module load via data.py; guard so the test
# suite still collects if torch is absent, but the maths is torch-free.
density = pytest.importorskip("improvement.density.data", reason="torch not installed")


def test_density_sum_equals_count():
    centers = [(100, 100), (300, 250), (500, 400), (123, 456)]
    dm = density._density_target(centers, (640, 640))
    # Gaussian splats preserve mass; integral should equal the number of points
    assert abs(dm.sum() - len(centers)) < 0.05


def test_density_empty_is_zero():
    dm = density._density_target([], (640, 640))
    assert dm.sum() == 0.0
    assert dm.shape == (80, 80)  # 640 / stride(8)


def test_density_points_out_of_frame_dropped():
    # points outside the image contribute nothing
    dm = density._density_target([(-50, -50), (10000, 10000)], (640, 640))
    assert dm.sum() == 0.0


def test_density_resolution_is_input_over_stride():
    dm = density._density_target([(320, 320)], (640, 640), stride=8)
    assert dm.shape == (80, 80)
    # the single splat should sit near the center cell
    r, c = np.unravel_index(dm.argmax(), dm.shape)
    assert abs(r - 40) <= 1 and abs(c - 40) <= 1
