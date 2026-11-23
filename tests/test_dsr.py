"""Checks of the numerical parts of gregg_ocr.dynamical_stroke_representation against closed forms."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gregg_ocr import dynamical_stroke_representation as dsr  # noqa: E402


def step_response(zeta, omega, t):
    """Closed-form unit step response of ω² / (p² + 2ζωp + ω²) from rest."""
    if zeta < 1:
        wd = omega * math.sqrt(1 - zeta ** 2)
        phi = math.acos(zeta)
        return 1 - np.exp(-zeta * omega * t) / math.sqrt(1 - zeta ** 2) * np.sin(wd * t + phi)
    if zeta == 1:
        return 1 - (1 + omega * t) * np.exp(-omega * t)
    a = omega * math.sqrt(zeta ** 2 - 1)
    p1, p2 = -zeta * omega + a, -zeta * omega - a
    return 1 + (p2 * np.exp(p1 * t) - p1 * np.exp(p2 * t)) / (p1 - p2)


@pytest.mark.parametrize("zeta", [0.1, 0.3, 0.707, 1.0, 1.5])
@pytest.mark.parametrize("method", ["exact", "rk4"])
def test_step_response(zeta, method):
    omega, dt = 5.0, 0.002
    t = np.arange(0, 3, dt)
    g = np.ones_like(t)
    r = dsr.simulate(g, dt, zeta, omega, r0=0.0, v0=0.0, method=method)
    assert np.max(np.abs(r - step_response(zeta, omega, t))) < 1e-4


def test_exact_is_stable_at_coarse_steps():
    t = np.arange(0, 2, 0.1)
    r = dsr.simulate(np.ones_like(t), 0.1, 0.5, 50.0, r0=0.0)  # ω dt = 5: RK4 would blow up
    assert np.all(np.isfinite(r)) and abs(r[-1] - 1) < 1e-3


@pytest.mark.parametrize("zeta", [0.1, 0.3, 0.5])
def test_overshoot(zeta):
    t = np.arange(0, 10, 0.001)
    r = dsr.simulate(np.ones_like(t), 0.001, zeta, 3.0, r0=0.0)
    assert r.max() - 1 == pytest.approx(dsr.resonance(zeta, 3.0)["overshoot"], abs=1e-3)


def test_tracks_a_ramp_with_lag_2zeta_over_omega():
    dt, zeta, omega = 0.001, 0.7, 10.0
    t = np.arange(0, 5, dt)
    r = dsr.simulate(t, dt, zeta, omega)
    assert t[-1] - r[-1] == pytest.approx(2 * zeta / omega, rel=1e-3)


@pytest.mark.parametrize("zeta,omega", [(0.2, 3.0), (0.707, 10.0), (1.5, 1.0)])
def test_frequency_response_matches_scipy(zeta, omega):
    W = np.logspace(-2, 2, 200)
    _, H = signal.freqs([omega ** 2], [1, 2 * zeta * omega, omega ** 2], worN=W)
    assert np.allclose(dsr.transfer(zeta, omega, W), H)
    mag, ph = dsr.bode(zeta, omega, np.array([1e-4, omega]))
    assert mag[0] == pytest.approx(0, abs=1e-6)
    assert ph[1] == pytest.approx(-90, abs=1e-6)
    assert 10 ** (mag[1] / 20) == pytest.approx(1 / (2 * zeta))


@pytest.mark.parametrize("zeta", [0.1, 0.3, 0.6, 0.8, 1.2])
def test_resonance_and_bandwidth(zeta):
    omega = 4.0
    W = np.linspace(1e-3, 3 * omega, 200001)
    m = np.abs(dsr.transfer(zeta, omega, W))
    res = dsr.resonance(zeta, omega)
    if zeta < 1 / math.sqrt(2):
        assert W[np.argmax(m)] == pytest.approx(res["peak_freq"], rel=1e-3)
        assert m.max() == pytest.approx(res["peak_gain"], rel=1e-6)
    else:
        assert np.argmax(m) == 0
    assert W[np.argmin(np.abs(m - 1 / math.sqrt(2)) + (W < res["peak_freq"]))] == pytest.approx(res["bandwidth"], rel=1e-3)


def test_inversion_recovers_reference():
    dt = 0.001
    t = np.arange(0, 2, dt)
    g = np.c_[np.sin(2 * np.pi * t), np.cos(3 * t)]
    r = dsr.simulate(g, dt, 0.5, 20.0, r0=g[0], v0=[0, 0])
    # Skip the start-up transient: the pen starts at rest while g is already moving.
    est = dsr.invert(r, dt, 0.5, 20.0, derivative="savgol", window=31)
    assert np.max(np.abs(est - g)[500:-50]) < 2e-3


def test_smoothed_inversion_beats_finite_differences_on_noise():
    rng = np.random.default_rng(0)
    dt = 0.002
    t = np.arange(0, 2, dt)
    g = np.c_[np.sin(2 * np.pi * t), 0.5 * np.sin(np.pi * t)]
    r = dsr.simulate(g, dt, 0.7, 15.0, r0=g[0], v0=[2 * np.pi, 0.5 * np.pi]) + rng.normal(0, 1e-3, (len(t), 2))
    err = {m: np.sqrt(np.mean((dsr.invert(r, dt, 0.7, 15.0, derivative=m, window=41) - g)[50:-50] ** 2))
           for m in ("finite", "savgol", "spline")}
    assert err["savgol"] < err["finite"] / 10
    assert err["spline"] < err["finite"] / 10


def test_fit_dynamics_recovers_parameters_with_the_true_reference():
    dt = 1 / 128
    corners = np.array([[0, 0], [1, 0], [1, 1], [2, 1]], float)
    k = np.linspace(0, 3, 129)
    g = np.c_[np.interp(k, range(4), corners[:, 0]), np.interp(k, range(4), corners[:, 1])]
    r = dsr.simulate(g, dt, 0.3, 20.0)
    fit = dsr.fit_dynamics(r, dt, g)
    assert (fit["zeta"], fit["omega"]) == (0.3, 20.0)
    assert fit["error"] < 1e-9 and fit["identifiability"] > 0.5


def test_timings():
    s = np.linspace(0, 2, 401)
    k = np.r_[np.zeros(200), np.full(201, 10.0)]
    for model in dsr.TIMINGS:
        t = dsr.time_parameterize(s, k, model)
        assert t[0] == 0 and t[-1] == pytest.approx(1) and np.all(np.diff(t) >= 0)
    # Two-thirds power law: the curved half is slower, so it takes longer.
    t = dsr.time_parameterize(s, k, "power_law")
    assert t[200] < 0.5
    # Minimum jerk: peak speed 1.875 × mean, at mid-time.
    t = dsr.time_parameterize(s, np.zeros_like(s), "min_jerk")
    v = np.diff(s) / np.diff(t)
    assert v.max() / 2 == pytest.approx(1.875, rel=1e-2)


def draw(shape, size=(400, 200), width=5):
    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    shape(d, width)
    return np.array(im)


def test_circle_curvature_and_loop():
    R = 60
    gray = draw(lambda d, w: d.ellipse([200 - R, 100 - R, 200 + R, 100 + R], outline=0, width=w))
    sk = dsr.extract_strokes(gray)
    loops = [s for s in sk.segments if s.kind == "loop"]
    assert len(loops) == 1 and len(sk.segments) == 1
    g = dsr.geometry(loops[0], unit=1.0)
    assert np.median(np.abs(g.curvature)) == pytest.approx(1 / (R - 2.5), rel=0.05)  # centreline of a 5 px ring drawn inward
    f = dsr.geometric_features(loops[0], g, 1.0, 200)
    assert f["abs_turning"] == pytest.approx(2 * math.pi, rel=0.05)


def test_straight_line():
    gray = draw(lambda d, w: d.line([50, 150, 350, 50], fill=0, width=w))
    sk = dsr.extract_strokes(gray)
    (seg,) = sk.segments
    g = dsr.geometry(seg, unit=1.0)
    f = dsr.geometric_features(seg, g, 1.0, 200)
    assert f["straightness"] > 0.99 and f["mean_abs_curv"] < 0.01
    assert math.degrees(math.atan2(f["chord_sin"], f["chord_cos"])) == pytest.approx(math.degrees(math.atan2(100, 300)), abs=1)
    assert f["length"] == pytest.approx(math.hypot(300, 100), rel=0.03)


def test_junction_graph():
    def tee(d, w):
        d.line([50, 50, 350, 50], fill=0, width=w)
        d.line([200, 50, 200, 180], fill=0, width=w)
    sk = dsr.extract_strokes(draw(tee))
    paths = [s for s in sk.segments if s.kind == "path"]
    assert len(paths) == 3
    assert np.sum(sk.node_degree >= 3) == 1 and np.sum(sk.node_degree == 1) == 3
    tr = dsr.transition_features(sk, unit=200.0)
    # The two halves of the bar continue each other; the stem meets them at a right angle.
    bar = [i for i, s in enumerate(sk.segments) if s.kind == "path" and abs(s.points[0, 1] - s.points[-1, 1]) < 5]
    assert all(tr[i]["best_continuation"] < 0.2 for i in bar)


def test_light_ink_on_dark():
    gray = draw(lambda d, w: d.line([50, 150, 350, 50], fill=0, width=w))
    (a,), (b,) = dsr.extract_strokes(gray).segments, dsr.extract_strokes(255 - gray).segments
    assert np.allclose(a.points, b.points)


def test_dot():
    sk = dsr.extract_strokes(draw(lambda d, w: d.ellipse([195, 95, 203, 103], fill=0)))
    assert [s.kind for s in sk.segments] == ["dot"]


def test_fourier_descriptors_ignore_rotation():
    s = np.linspace(0, 1, 200)
    xy = np.c_[s, 0.2 * np.sin(3 * s)]
    a = 0.7
    rot = xy @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    g1 = dsr.Geometry(s, xy, None, None, False)
    g2 = dsr.Geometry(s, rot, None, None, False)
    assert np.allclose(dsr.fourier_descriptors(g1), dsr.fourier_descriptors(g2))


def test_features_and_columns():
    def outline(d, w):
        d.arc([40, 40, 160, 160], 180, 360, fill=0, width=w)
        d.line([160, 100, 300, 60], fill=0, width=w)
        d.ellipse([320, 90, 330, 100], fill=0)
    gray = draw(outline)
    res = dsr.extract_features(gray)
    assert res.features.shape == (len(res.skeleton.segments), len(res.names))
    assert np.all(np.isfinite(res.features))
    assert set(res.groups) == set(dsr.GROUPS)
    cols = dsr.column_features(res, gray.shape[1])
    assert cols.shape == ((gray.shape[1] + 3) // 4, len(res.names) + 1)
    assert cols[:5, -1].max() == 0 and cols[20, -1] > 0  # no ink at the far left; ink on the arc
    scaler = dsr.FeatureScaler(res.names).fit(np.r_[res.features, res.features + 1])
    assert np.all(np.abs(scaler.transform(res.features)) <= scaler.clip)


def test_vary_path_keeps_shape_for_fast_pen():
    path = np.c_[np.linspace(0, 1, 50), np.zeros(50)]
    path = np.r_[path, np.c_[np.ones(50), np.linspace(0, 1, 50)]]
    tight = dsr.vary_path(path, 1.0, 200.0)
    loose = dsr.vary_path(path, 0.3, 10.0)
    def deviation(r):  # farthest the pen gets from the reference path
        return np.max(np.min(np.linalg.norm(r[:, None] - path[None], axis=2), axis=1))
    # A fast, damped pen stays on the path; a slow, springy one cuts the corner and overshoots.
    assert deviation(tight) < 0.02
    assert deviation(loose) > 0.05
