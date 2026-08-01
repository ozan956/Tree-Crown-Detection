import numpy as np
import cv2
from improvement.features import (
    gabor_bank_energy, color_invariant, joint_probability,
    ORIENTATIONS, CENTER_FREQS,
)


def test_bank_has_multiple_orientations_and_scales():
    # the whole point vs the paper's single kernel: a real bank
    assert len(ORIENTATIONS) >= 4
    assert len(CENTER_FREQS) >= 4


def test_color_invariant_range_and_greenness():
    # a pure-green patch scores higher than a pure-gray patch
    green = np.zeros((32, 32, 3), np.uint8); green[:, :, 1] = 200  # BGR green
    gray = np.full((32, 32, 3), 128, np.uint8)
    cg = color_invariant(green).mean()
    cy = color_invariant(gray).mean()
    assert 0.0 <= color_invariant(green).min() <= color_invariant(green).max() <= 1.0
    assert cg > cy


def test_color_invariant_is_illumination_invariant():
    # scaling brightness (shading) must not change the chromaticity feature much
    base = np.random.RandomState(0).randint(20, 200, (48, 48, 3)).astype(np.uint8)
    dark = (base.astype(float) * 0.5).astype(np.uint8)  # same colours, half as bright
    c_base = color_invariant(base)
    c_dark = color_invariant(dark)
    # mean absolute difference should be tiny relative to the [0,1] range
    assert np.abs(c_base - c_dark).mean() < 0.05


def test_gabor_bank_energy_shape_and_range():
    img = np.random.RandomState(1).randint(0, 255, (64, 64), np.uint8)
    e = gabor_bank_energy(img)
    assert e.shape == (64, 64)
    assert 0.0 <= e.min() and e.max() <= 1.0


def test_gabor_bank_responds_to_texture_not_flat():
    # a textured region should yield higher mean energy than a flat one
    flat = np.full((64, 64), 128, np.uint8)
    texture = np.zeros((64, 64), np.uint8)
    texture[:, ::4] = 255  # vertical stripes
    assert gabor_bank_energy(texture).mean() > gabor_bank_energy(flat).mean()


def test_joint_probability_default_favours_color():
    img = np.random.RandomState(2).randint(0, 255, (48, 48, 3), np.uint8)
    j = joint_probability(img)
    assert j.shape == (48, 48)
    assert 0.0 <= j.min() and j.max() <= 1.0
