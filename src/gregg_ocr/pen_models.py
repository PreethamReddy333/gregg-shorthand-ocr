"""Motor models that redraw the renderer's pen paths the way a moving hand would, as variation for
synthetic training lines. Two models, each replacing the renderer's ad-hoc pen lag (Style.lag):

``sigma_lognormal``: the Kinematic Theory of rapid movements (Plamondon 1995; O'Reilly &
Plamondon 2009). A path is a sum of overlapping strokes, each a circular arc travelled with a
lognormal speed profile. The path is cut into strokes at its RDP corners, each stroke is fitted as
the arc between its ends (so the unperturbed strokes end exactly where the path's pieces do), and
then every stroke's parameters (size D, start direction θs, turn Δθ, onset t0, log-time μ,
log-response σ) are jittered, as in sigma-lognormal duplication of handwriting and signatures
(Galbally et al. 2012; Fischer et al. 2014; Diaz et al. 2017). Overlap between strokes rounds the
corners, as in fast writing.

``pen_dynamics``: the second-order pen of the DSR module, r'' + 2ζωr' + ω²(r - g) = 0, following
the path g at two-thirds-power-law speed (dynamical_stroke_representation.vary_path). ω is given
per unit of path length, so long and short paths lag alike.

Both work on Drawing paths in units (y up), after layout, and leave dots and text alone. The
parameters are estimates of plausible hands, not measurements: they were set so the redrawn lines
read back about as well as the renderer's own writers.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .dynamical_stroke_representation import _arc_length, rdp, resample, vary_path


# --- sigma-lognormal --------------------------------------------------------------------------

@dataclass
class LognormalStrokes:
    start: np.ndarray       # (2,) pen-down point
    D: np.ndarray           # (k,) arc length of each stroke
    theta: np.ndarray       # (k,) start direction, radians
    turn: np.ndarray        # (k,) change of direction along the stroke, radians
    t0: np.ndarray          # (k,) onset times
    mu: np.ndarray          # (k,) log-time delay
    sigma: np.ndarray       # (k,) log-response time


def lognormal_strokes(path, eps: float = 0.03, step: float = 0.02, mu: float = -1.6, sigma: float = 0.28,
                      spacing: float = 0.9) -> LognormalStrokes:
    """A pen path -> one sigma-lognormal stroke per RDP piece. Onsets are ``spacing`` · e^μ apart:
    smaller spacing, more overlap, rounder corners. Each piece's turn is the change of the path's
    direction along it, and D and θs are set so the arc ends exactly at the piece's end."""
    _, p = resample(np.asarray(path, float), step)
    a = np.unwrap(np.arctan2(*np.diff(p, axis=0)[:, ::-1].T))   # direction of each step
    D, theta, turn = [], [], []
    idx = rdp(p, eps)
    for i, j in zip(idx[:-1], idx[1:]):
        c = p[j] - p[i]
        chord = float(np.hypot(*c))
        if chord < 1e-9:
            continue
        dth = float(np.clip(a[j - 1] - a[i], -1.9 * math.pi, 1.9 * math.pi))
        half = dth / 2
        D.append(chord * (half / math.sin(half)) if abs(half) > 1e-6 else chord)
        theta.append(math.atan2(c[1], c[0]) - half)
        turn.append(dth)
    k = len(D)
    return LognormalStrokes(p[0], np.array(D), np.array(theta), np.array(turn),
                            np.arange(k) * spacing * math.exp(mu), np.full(k, mu), np.full(k, sigma))


def lognormal_path(s: LognormalStrokes, dt: float = 0.003, step: float = 0.03) -> np.ndarray:
    """Strokes -> the pen path: the sum of each stroke's arc, travelled to the fraction its
    lognormal speed profile has covered by time t (the CDF), resampled every ``step`` units."""
    if not len(s.D):
        return s.start[None]
    t = np.arange(0, (s.t0 + np.exp(s.mu + 4 * s.sigma)).max() + dt, dt)
    pos = np.repeat(s.start[None], len(t), 0).astype(float)
    for D, th, turn, t0, mu, sg in zip(s.D, s.theta, s.turn, s.t0, s.mu, s.sigma):
        F = ndtr((np.log(np.maximum(t - t0, 1e-12)) - mu) / sg) * (t > t0)
        if abs(turn) > 1e-6:
            R, ang = D / turn, th + turn * F
            pos[:, 0] += R * (np.sin(ang) - math.sin(th))
            pos[:, 1] += -R * (np.cos(ang) - math.cos(th))
        else:
            pos[:, 0] += D * F * math.cos(th)
            pos[:, 1] += D * F * math.sin(th)
    keep = np.r_[True, np.linalg.norm(np.diff(pos, axis=0), axis=1) > 1e-9]
    pos = pos[keep]
    if len(pos) < 2:
        return pos
    return resample(pos, step)[1]


def perturb(s: LognormalStrokes, rng: random.Random, amount: float = 1.0) -> LognormalStrokes:
    """Independent jitter of every stroke's parameters; ``amount`` scales all of it."""
    k = len(s.D)
    g = np.random.default_rng(rng.randrange(1 << 31))
    e = math.exp(float(np.mean(s.mu))) if k else 1.0
    return LognormalStrokes(
        s.start,
        s.D * np.exp(g.normal(0, 0.08 * amount, k)),
        s.theta + g.normal(0, 0.07 * amount, k),
        s.turn + g.normal(0, 0.07 * amount, k),
        s.t0 + g.normal(0, 0.08 * amount * e, k),
        s.mu + g.normal(0, 0.08 * amount, k),
        s.sigma * np.exp(g.normal(0, 0.08 * amount, k)))


def sigma_lognormal(path, rng: random.Random, sigma: float = 0.28, spacing: float = 0.9,
                    amount: float = 1.0) -> np.ndarray:
    path = np.asarray(path, float)
    if len(path) < 2 or _arc_length(path) < 0.05:
        return path
    s = lognormal_strokes(path, sigma=sigma, spacing=spacing)
    return lognormal_path(perturb(s, rng, amount) if amount else s)


# --- second-order pen -----------------------------------------------------------------------

def pen_dynamics(path, zeta: float, omega: float, timing: str = "power_law") -> np.ndarray:
    """``omega``: natural frequency per unit of path length (the pen covers the path in unit time)."""
    path = np.asarray(path, float)
    L = _arc_length(path)
    if len(path) < 2 or L < 0.05:
        return path
    return vary_path(path, zeta, omega * L, timing, n=max(60, int(L / 0.03)))


# --- writers ---------------------------------------------------------------------------------

@dataclass
class Hand:
    """One writer's motor parameters for a model ("sln" or "dsr")."""
    kind: str
    sigma: float = 0.28
    spacing: float = 0.9
    amount: float = 1.0
    zeta: float = 0.7
    omega: float = 20.0

    @classmethod
    def sample(cls, kind: str, rng: random.Random) -> "Hand":
        if kind == "sln":
            return cls(kind, sigma=rng.uniform(*SLN_SIGMA), spacing=rng.uniform(*SLN_SPACING),
                       amount=rng.uniform(*SLN_AMOUNT))
        if kind == "dsr":
            return cls(kind, zeta=rng.uniform(*DSR_ZETA), omega=rng.uniform(*DSR_OMEGA))
        raise ValueError(kind)

    def redraw(self, path, rng: random.Random) -> list:
        if self.kind == "sln":
            out = sigma_lognormal(path, rng, self.sigma, self.spacing, self.amount)
        else:
            out = pen_dynamics(path, self.zeta, self.omega)
        return [tuple(map(float, q)) for q in out]


# Ranges of plausible hands, set by read-back (see make_synth_lines.py --vary).
SLN_SIGMA = (0.2, 0.36)
SLN_SPACING = (0.6, 1.1)
SLN_AMOUNT = (0.5, 1.5)
DSR_ZETA = (0.45, 0.95)
DSR_OMEGA = (6.0, 14.0)


def redraw_lines(lines: list, hand: Hand, rng: random.Random) -> list:
    """Every pen-down path of laid-out lines (render.layout) redrawn by ``hand``, in place."""
    for ln in lines:
        for d in ln.drawings:
            d.paths = [hand.redraw(p, rng) for p in d.paths]
    return lines
