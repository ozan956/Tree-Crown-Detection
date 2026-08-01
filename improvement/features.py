"""Faithful traditional features: a real Gabor bank + illumination-invariant color.

The paper text (arXiv:2507.01502, S2.1) claims "a bank of Gabor filters that
uniformly cover the spatial and frequency domains" following Jain & Farrokhnia
(1991), and "color invariants". The shipped code instead used a SINGLE Gabor
kernel (ksize=3, one theta=pi/4, one lambda) and a fixed HSV green threshold.
This module implements what the text actually claims, so the method matches the
paper and (measurably) improves tree-vs-background separation.

- gabor_bank_energy: multi-orientation, multi-scale Gabor bank with the
  Jain-Farrokhnia post-processing (magnitude -> tanh nonlinearity -> Gaussian
  energy in a local window), averaged into one texture response.
- color_invariant: an illumination-invariant vegetation feature. We use the
  normalized excess-green on normalized-rgb (chromaticity) coordinates, which
  is invariant to overall intensity/shading — a genuine colour *invariant*,
  unlike a fixed HSV band.

CPU-only; depends on numpy + opencv.
"""

import numpy as np
import cv2

# Jain-Farrokhnia canonical bank: 4 orientations, sqrt(2)-spaced radial
# center frequencies (cycles/image-width). Kept modest for 640px crops.
ORIENTATIONS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
# center frequencies in cycles/image -> wavelengths lambda = W / f (px)
CENTER_FREQS = [4.0, 5.66, 8.0, 11.31, 16.0]  # ~ 4 * sqrt(2)^k


def _gabor_wavelengths(image_width):
    """Convert center frequencies (cycles/image) to Gabor wavelengths (px)."""
    return [image_width / f for f in CENTER_FREQS]


def gabor_bank_energy(gray, image_width=None, gamma=0.5, bandwidth_sigma=0.56):
    """Average Gabor energy response over a full orientation x scale bank.

    gray: single-channel float/uint8 image. Returns a float32 map in [0,1],
    the mean over all (orientation, scale) filters of the Jain-Farrokhnia
    texture energy (tanh of magnitude, Gaussian-smoothed).
    """
    g = gray.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    h, w = g.shape[:2]
    if image_width is None:
        image_width = w
    lambdas = _gabor_wavelengths(image_width)
    responses = []
    for lam in lambdas:
        # kernel size ~ 3x the wavelength (captures the full oscillation), odd
        ks = int(max(7, 2 * round(1.5 * lam) + 1))
        sigma = bandwidth_sigma * lam
        for theta in ORIENTATIONS:
            kern = cv2.getGaborKernel((ks, ks), sigma, theta, lam, gamma, 0,
                                      ktype=cv2.CV_32F)
            # zero-mean kernel -> pure texture response, no DC/brightness leak
            kern -= kern.mean()
            filt = cv2.filter2D(g, cv2.CV_32F, kern)
            mag = np.abs(filt)
            # Jain-Farrokhnia nonlinearity + local energy pooling
            energy = np.tanh(0.25 * mag)
            energy = cv2.GaussianBlur(energy, (0, 0), sigmaX=max(1.0, 0.5 * lam))
            responses.append(energy)
    resp = np.mean(responses, axis=0)
    mn, mx = resp.min(), resp.max()
    return ((resp - mn) / (mx - mn + 1e-8)).astype(np.float32)


def color_invariant(bgr):
    """Illumination-invariant vegetation feature in [0,1].

    Works in normalized-rgb (chromaticity) space r=R/(R+G+B) etc., which is
    invariant to overall intensity/shading. The vegetation score is the
    normalized excess-green chromaticity 2g - r - b, mapped to [0,1]. Unlike a
    fixed HSV band this does not depend on absolute brightness, so it survives
    the sun/shadow variation the paper's text specifically motivates.
    """
    x = bgr.astype(np.float32) + 1e-6
    s = x.sum(axis=2, keepdims=True)
    chroma = x / s                      # normalized rgb, sums to 1 per pixel
    b, g, r = chroma[:, :, 0], chroma[:, :, 1], chroma[:, :, 2]
    exg = 2 * g - r - b                 # excess-green chromaticity, ~[-1,1]
    return np.clip((exg + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def joint_probability(bgr, w_color=0.9, w_texture=0.1):
    """Combine the two invariant features into a tree-crown probability map,
    matching the paper's J = w1*C + w2*G formulation but with the faithful
    features. Returns float32 in [0,1].

    Default weights favour colour (w_color=0.9): the illumination-invariant
    colour feature separates tree pixels far better than texture on this data
    (pixel-AUC 0.86 vs 0.67), so an equal 0.5/0.5 split dilutes the stronger
    cue. The weighting was selected on a tuning split and confirmed on a
    held-out split (see results/FEATURE_ANALYSIS.md); the Gabor bank still adds
    a small complementary gain over colour alone, so it is retained, not
    dropped. The paper's original equal-weight formulation used fixed-HSV +
    single-Gabor features (pixel-AUC 0.74)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    C = color_invariant(bgr)
    G = gabor_bank_energy(gray)
    return (w_color * C + w_texture * G).astype(np.float32)
