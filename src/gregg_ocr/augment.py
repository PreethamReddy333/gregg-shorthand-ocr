"""Augmentation of line crops for fine-tuning Topaz. Crops are uint8, ink high, height 128,
as Topaz's training crops are stored.

``grid_distort`` is the random grid perturbation of Wigington et al. (ICDAR 2017): control points
on a regular grid move by Gaussian offsets and the image is warped smoothly between them, which
varies local proportions, slant and baseline the way different hands do. ``basic`` is the usual
geometric and photometric jitter (stretch, slant, stroke width, blur, contrast, noise).
"""

import random

import cv2
import numpy as np


def grid_distort(img: np.ndarray, rng: random.Random, spacing: int = 36, sigma: tuple = (1.5, 3.5)) -> np.ndarray:
    """Move grid points every ``spacing`` px by N(0, σ) px (σ drawn from ``sigma``), then warp the
    image smoothly (bicubic interpolation of the offsets) between them. Larger σ starts closing
    open curves into loops, which changes the stroke, not just the hand."""
    h, w = img.shape
    s = rng.uniform(*sigma)
    gh, gw = h // spacing + 2, w // spacing + 2
    nrng = np.random.default_rng(rng.randrange(1 << 31))
    dx = cv2.resize(nrng.normal(0, s, (gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(nrng.normal(0, s, (gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=int(np.median(img)))  # the paper, which isn't always black


def basic(img: np.ndarray, rng: random.Random, max_width: int = 1600) -> np.ndarray:
    h, w = img.shape
    # Width stretch, then slant (shear about the middle of the line).
    nw = int(np.clip(round(w * rng.uniform(0.85, 1.15)), 16, max_width))
    img = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR)
    sh = rng.uniform(-0.2, 0.2)
    pad = int(abs(sh) * h / 2) + 1
    M = np.float32([[1, sh, pad - sh * h / 2], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (min(nw + 2 * pad, max_width), h), flags=cv2.INTER_LINEAR,
                         borderValue=int(np.median(img)))
    # Stroke width.
    r = rng.random()
    if r < 0.25:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    elif r < 0.35:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.1))
    # Contrast and noise.
    out = img.astype(np.float32) * rng.uniform(0.65, 1.0)
    if rng.random() < 0.5:
        out += np.random.default_rng(rng.randrange(1 << 31)).normal(0, rng.uniform(2, 12), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
