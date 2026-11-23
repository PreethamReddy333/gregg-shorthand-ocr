"""Dynamical Stroke Representation (DSR): geometric, pseudo-kinematic, dynamical, frequency-domain
and transition features of Gregg strokes, measured from still images.

What is and isn't measured. A scan has no pen timing, so every "velocity", "acceleration" and "jerk"
here is pseudo-kinematic: geometry plus an assumed timing model (``TIMINGS``), and it changes with
that assumption. The damping ratio ζ and natural frequency ω of the pen model

    r''(t) + 2ζω r'(t) + ω² (r(t) - g(t)) = 0

can't be recovered from a still image either: for any ζ, ω there is a reference path g that the pen
would have traced exactly as observed (``invert``). ``fit_dynamics`` therefore fits them against an
explicit prior for g (straight pieces between corners), and its ζ, ω, tracking error and
identifiability score describe the shape under that prior, not the writer's motor system.

Skeleton segments are not Gregg strokes. The skeleton is split at junctions and ends, so one outline
is usually one connected component of several segments, and a circle vowel is often a closed loop.
The skeleton also has no writing direction; ``canonical_direction`` picks one by rule.

Lines come in as Topaz's rectified line crops (``topaz.find_lines``), which already handles skew,
line finding and size normalization, so features line up with Topaz's output frames (one per 4 px).
Lengths are in units of ``DSRConfig.unit`` pixels (default: the crop height, which Topaz scales to
the writer's line spacing), so size differences such as T vs D survive.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import interpolate, linalg, ndimage, signal
from skimage import filters, morphology

# --------------------------------------------------------------------------------------------
# 1. Stroke extraction
# --------------------------------------------------------------------------------------------


@dataclass
class ExtractConfig:
    threshold: str = "otsu"     # "otsu", or "sauvola" for uneven lighting (photos)
    sauvola_window: int = 25
    min_component: int = 10     # pixels; smaller ink blobs are noise (a Gregg dot is larger)
    skeleton: str = "zhang"     # "zhang", "lee" or "medial_axis"
    prune: float = 1.5          # spurs shorter than this many pen widths are dropped


@dataclass
class Segment:
    """One skeleton path between nodes (or a closed loop, or a dot). Coordinates are image
    pixels (x right, y down) along the skeleton; ``width`` is the ink width at each point."""
    points: np.ndarray
    width: np.ndarray
    component: int
    start: int                  # node ids; -1 for loops and dots
    end: int
    kind: str                   # "path", "loop" or "dot"


@dataclass
class Skeleton:
    image: np.ndarray           # the input, untouched
    ink: np.ndarray             # binarized, True = ink
    skeleton: np.ndarray
    segments: list[Segment]
    nodes: np.ndarray           # (n, 2) node positions, x, y
    node_degree: np.ndarray     # segment ends meeting at each node
    n_components: int
    config: ExtractConfig


def binarize(gray: np.ndarray, cfg: ExtractConfig) -> np.ndarray:
    """Grayscale -> True where ink. Ink is taken to be the minority tone, so both dark-on-light
    scans and light-on-dark images (Topaz's own training crops, screenshots) work."""
    g = gray.astype(float)
    if np.median(g) < 0.5 * (g.min() + g.max()):
        g = g.max() + g.min() - g
    if g.max() - g.min() < 1:
        return np.zeros(g.shape, bool)
    if cfg.threshold == "sauvola":
        ink = g < filters.threshold_sauvola(g, window_size=cfg.sauvola_window)
        ink &= g < filters.threshold_otsu(g) + 0.25 * (g.max() - g.min())  # paper texture isn't ink
    else:
        ink = g < filters.threshold_otsu(g)
    return morphology.remove_small_objects(ink, max_size=cfg.min_component - 1)


def skeletonize(ink: np.ndarray, method: str = "zhang") -> np.ndarray:
    if method == "medial_axis":
        skel = morphology.medial_axis(ink)
    else:
        skel = morphology.skeletonize(ink, method=method)
    return morphology.thin(skel)  # removes the staircase pixels that look like junctions


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = _N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _graph(skel: np.ndarray):
    """Skeleton -> nodes (ends and junction clusters) and the pixel paths between them."""
    H, W = skel.shape
    deg = ndimage.convolve(skel.astype(np.int32), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                           mode="constant") * skel
    node_mask = skel & (deg != 2)
    labels, n_nodes = ndimage.label(node_mask, structure=np.ones((3, 3)))
    nodes = np.array(ndimage.center_of_mass(node_mask, labels, range(1, n_nodes + 1))).reshape(-1, 2)[:, ::-1]

    def nbrs(y, x):
        for dy, dx in _N8:
            v, u = y + dy, x + dx
            if 0 <= v < H and 0 <= u < W and skel[v, u]:
                yield v, u

    visited = np.zeros_like(skel)
    edges = []  # (start node, end node, [(y, x)...])
    for y, x in zip(*np.nonzero(node_mask)):
        a = labels[y, x] - 1
        for v, u in nbrs(y, x):
            if node_mask[v, u] or visited[v, u]:
                continue
            path, prev, cur = [(y, x)], (y, x), (v, u)
            while True:
                path.append(cur)
                if node_mask[cur]:
                    edges.append((a, labels[cur] - 1, path))
                    break
                visited[cur] = True
                nxt = [q for q in nbrs(*cur) if q != prev and (node_mask[q] or not visited[q])]
                if not nxt:  # ran into its own start (a node of the same cluster)
                    edges.append((a, a, path))
                    break
                prev, cur = cur, nxt[0]
    # Loops with no node on them: circle vowels, closed O hooks.
    loops = []
    for y, x in zip(*np.nonzero(skel & ~node_mask & ~visited)):
        if visited[y, x]:
            continue
        path, prev, cur = [], None, (y, x)
        while cur is not None and not visited[cur]:
            visited[cur] = True
            path.append(cur)
            nxt = [q for q in nbrs(*cur) if q != prev and not visited[q]]
            prev, cur = cur, (nxt[0] if nxt else None)
        if len(path) > 2:
            loops.append(path + [path[0]])
    return nodes, edges, loops, labels


def _merge_and_prune(nodes, edges, prune_px):
    """Drop short spurs (end node -> junction), then join paths through nodes they now pass
    straight through (degree 2), until nothing changes."""
    edges = [list(e) for e in edges]
    changed = True
    while changed:
        changed = False
        degree = np.zeros(len(nodes), int)
        for a, b, _ in edges:
            degree[a] += 1
            degree[b] += 1
        keep = []
        for a, b, p in edges:
            spur = (degree[a] == 1) != (degree[b] == 1) and max(degree[a], degree[b]) >= 3
            if spur and len(p) < prune_px(p):
                changed = True
                continue
            keep.append([a, b, p])
        edges = keep
        degree = np.zeros(len(nodes), int)
        for a, b, _ in edges:
            degree[a] += 1
            degree[b] += 1
        for n in np.nonzero(degree == 2)[0]:
            ends = [(i, e) for i, e in enumerate(edges) if e[0] == n or e[1] == n]
            if len(ends) != 2:
                continue  # a loop through the node: both ends on one edge
            (i, e1), (j, e2) = ends
            p1 = e1[2] if e1[1] == n else e1[2][::-1]
            p2 = e2[2] if e2[0] == n else e2[2][::-1]
            a = e1[0] if e1[1] == n else e1[1]
            b = e2[1] if e2[0] == n else e2[0]
            edges = [e for k, e in enumerate(edges) if k not in (i, j)] + [[a, b, p1 + p2[1:]]]
            changed = True
            break
    return edges


def extract_strokes(gray: np.ndarray, cfg: ExtractConfig | None = None) -> Skeleton:
    """Grayscale image (dark ink on light) -> skeleton segments, nodes and component metadata."""
    cfg = cfg or ExtractConfig()
    ink = binarize(gray, cfg)
    skel = skeletonize(ink, cfg.skeleton)
    width_map = 2.0 * ndimage.distance_transform_edt(ink)
    comp, n_comp = ndimage.label(ink, structure=np.ones((3, 3)))
    nodes, edges, loops, _ = _graph(skel)
    median_w = float(np.median(width_map[skel])) if skel.any() else 1.0
    edges = _merge_and_prune(nodes, edges, lambda p: cfg.prune * median_w)

    segments = []

    def add(path, start, end, kind):
        yx = np.array(path)
        pts = yx[:, ::-1].astype(float)
        if start >= 0:
            pts[0] = nodes[start]
        if end >= 0:
            pts[-1] = nodes[end]
        segments.append(Segment(pts, width_map[yx[:, 0], yx[:, 1]], int(comp[yx[len(yx) // 2, 0], yx[len(yx) // 2, 1]]),
                                start, end, kind))

    for a, b, p in edges:
        add(p, a, b, "loop" if a == b and len(p) > 4 else "path")
    for p in loops:
        add(p, -1, -1, "loop")
    # Dots: ink components whose skeleton has no path (or a tiny one).
    covered = {s.component for s in segments if _arc_length(s.points) > cfg.prune * median_w}
    for c in range(1, n_comp + 1):
        if c not in covered:
            yx = np.argwhere(comp == c)
            centre = yx.mean(0)[::-1]
            segments = [s for s in segments if s.component != c]
            segments.append(Segment(centre[None], np.array([width_map[comp == c].max()]), c, -1, -1, "dot"))
    segments.sort(key=lambda s: s.points[:, 0].min())
    degree = np.zeros(len(nodes), int)
    for s in segments:
        for n in (s.start, s.end):
            if n >= 0:
                degree[n] += 1
    return Skeleton(gray, ink, skel, segments, nodes, degree, n_comp, cfg)


# --------------------------------------------------------------------------------------------
# 2. Geometry
# --------------------------------------------------------------------------------------------


def _arc_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def resample(pts: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points at even arc-length spacing ``step`` -> (s, xy)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    n = max(int(round(d[-1] / step)), 1) + 1
    s = np.linspace(0, d[-1], n)
    return s, np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def canonical_direction(pts: np.ndarray) -> np.ndarray:
    """The skeleton doesn't say which way the pen went. Gregg runs left to right, and its steep
    strokes (P, B, F, V, SH, CH, J) are written downward, so: left to right, unless the segment
    is steep, then top to bottom. An assumption, not an observation."""
    dx, dy = pts[-1] - pts[0]
    steep = abs(dx) < 0.3 * abs(dy)
    return pts[::-1] if (dy < 0 if steep else dx < 0) else pts


def derivatives(y: np.ndarray, dt: float, method: str = "savgol", window: int = 9, closed: bool = False,
                lam: float | None = None, order: int = 3) -> list[np.ndarray]:
    """Smoothed signal and its first ``order`` derivatives, per column of ``y`` (n, d).
    "finite": plain differences (the noisy baseline); "savgol": Savitzky-Golay, cubic, in a
    ``window``-sample window; "spline": a smoothing spline, ``lam`` chosen by GCV when None."""
    y = np.asarray(y, float)
    one_d = y.ndim == 1
    y = y[:, None] if one_d else y
    n = len(y)
    out = []
    if method == "finite" or n < 5:
        d = y
        for _ in range(order + 1):
            out.append(d)
            d = np.gradient(d, dt, axis=0) if len(d) > 1 else np.zeros_like(d)
    elif method == "savgol":
        w = min(window | 1, n if n % 2 else n - 1)
        p = min(3, w - 1)
        mode = "wrap" if closed else "interp"
        out = [signal.savgol_filter(y, w, p, deriv=k, delta=dt, axis=0, mode=mode) if k <= p
               else np.zeros_like(y) for k in range(order + 1)]
    elif method == "spline":
        t = np.arange(n) * dt
        cols = [interpolate.make_smoothing_spline(t, y[:, j], lam=lam) for j in range(y.shape[1])]
        out = [np.stack([c.derivative(k)(t) if k else c(t) for c in cols], 1) for k in range(order + 1)]
    else:
        raise ValueError(method)
    return [o[:, 0] for o in out] if one_d else out


@dataclass
class Geometry:
    s: np.ndarray               # arc length, units
    xy: np.ndarray              # smoothed path, units, y up
    tangent: np.ndarray         # tangent angle, radians, unwrapped
    curvature: np.ndarray       # signed, 1/units (+ = counterclockwise)
    closed: bool


def geometry(seg: Segment, unit: float, smoothing: str = "savgol", window_units: float = 0.12) -> Geometry | None:
    """Smoothed arc-length geometry of one segment, in writing units with y pointing up.

    Skeleton pixels zigzag, so the path is smoothed over ``window_units`` (at least 2.5 pen
    widths) first, then its arc length is measured on the smooth curve: measured on the pixel
    staircase, a slanted line comes out up to 8% long."""
    if seg.kind == "dot" or len(seg.points) < 3:
        return None
    pts = seg.points if seg.kind == "loop" else canonical_direction(seg.points)
    pts = np.c_[pts[:, 0], -pts[:, 1]] / unit
    step = 0.5 / unit  # half a pixel
    closed = seg.kind == "loop"
    s, xy = resample(pts, step)
    if len(s) < 5:
        return None
    if closed:
        xy = xy[:-1]
    window_px = max(window_units * unit, 2.5 * float(np.median(seg.width)))
    window = max(5, int(round(window_px / 0.5)))
    if not closed:
        window = min(window, len(xy) - 1 if len(xy) % 2 == 0 else len(xy))
    smooth = derivatives(xy, step, smoothing, window, closed, order=0)[0]
    s, p = resample(np.r_[smooth, smooth[:1]] if closed else smooth, step)
    if closed:
        s, p = s[:-1], p[:-1]
    if len(s) < 5:
        return None
    _, d1, d2, _ = derivatives(p, step, "finite" if smoothing == "finite" else "savgol", window, closed)
    speed = np.linalg.norm(d1, axis=1) + 1e-9
    curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    tangent = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    return Geometry(s, p, tangent, curvature, closed)


def geometric_features(seg: Segment, g: Geometry | None, unit: float, height: float) -> dict:
    f = {}
    width = float(np.median(seg.width)) / unit
    cy = float(seg.points[:, 1].mean()) / height  # vertical position in the line, 0 = top
    if g is None:
        return dict(is_dot=1.0, length=0.0, log_length=math.log(1e-3), chord=0.0, straightness=1.0,
                    chord_cos=0.0, chord_sin=0.0, axis_cos2=0.0, axis_sin2=0.0, aspect=1.0,
                    mean_curv=0.0, mean_abs_curv=0.0, max_abs_curv=0.0, turning=0.0, abs_turning=0.0,
                    inflections=0.0, curv_slope=0.0, curv_std=0.0, width=width, width_cv=0.0,
                    y_centre=cy, is_loop=0.0)
    length = float(g.s[-1] - g.s[0]) + (float(np.linalg.norm(g.xy[-1] - g.xy[0])) if g.closed else 0.0)
    chord_v = g.xy[-1] - g.xy[0]
    chord = float(np.linalg.norm(chord_v))
    ang = math.atan2(chord_v[1], chord_v[0])
    c = g.xy - g.xy.mean(0)
    ev, evec = np.linalg.eigh(c.T @ c / len(c))
    axis = math.atan2(evec[1, 1], evec[0, 1])
    k = g.curvature
    ds = np.gradient(g.s)
    big = np.abs(k) > 2.0  # ignore sign flips of a nearly straight stroke
    signs = np.sign(k[big])
    u = (g.s - g.s[0]) / max(g.s[-1] - g.s[0], 1e-9)
    f.update(is_dot=0.0, length=length, log_length=math.log(max(length, 1e-3)), chord=chord,
             straightness=chord / max(length, 1e-9),
             chord_cos=math.cos(ang) if not g.closed else 0.0, chord_sin=math.sin(ang) if not g.closed else 0.0,
             axis_cos2=math.cos(2 * axis), axis_sin2=math.sin(2 * axis),
             aspect=float(math.sqrt(max(ev[0], 0) / max(ev[1], 1e-12))),
             mean_curv=float(np.sum(k * ds) / max(length, 1e-9)) if not g.closed else 0.0,
             mean_abs_curv=float(np.sum(np.abs(k) * ds) / max(length, 1e-9)),
             max_abs_curv=float(np.percentile(np.abs(k), 95)),
             # The sign of a loop's turning depends on a direction the skeleton doesn't have.
             turning=float(np.sum(k * ds)) if not g.closed else 0.0,
             abs_turning=float(np.sum(np.abs(k) * ds)),
             inflections=float(np.count_nonzero(np.diff(signs) != 0)) if len(signs) > 1 else 0.0,
             curv_slope=float(np.polyfit(u, k, 1)[0]) if len(u) > 2 and not g.closed else 0.0,
             curv_std=float(np.std(k)), width=width,
             width_cv=float(np.std(seg.width) / max(np.mean(seg.width), 1e-9)),
             y_centre=cy, is_loop=float(g.closed))
    return f


# --------------------------------------------------------------------------------------------
# 3. Pseudo-kinematics
# --------------------------------------------------------------------------------------------

TIMINGS = ("uniform", "power_law", "min_jerk")


def time_parameterize(s: np.ndarray, curvature: np.ndarray, model: str = "power_law", beta: float = 1 / 3,
                      kappa0: float = 1.0) -> np.ndarray:
    """Assumed pen time at each arc length, normalized to [0, 1].

    "uniform": constant speed. "power_law": speed ∝ (|κ| + κ0)^(-β), the two-thirds power law of
    drawing movements (Lacquaniti et al. 1983; β = 1/3), with κ0 (1/units) keeping straight
    stretches finite. "min_jerk": the bell-shaped speed of a minimum-jerk point-to-point movement
    (Flash & Hogan 1985) over the whole segment."""
    u = (s - s[0]) / max(s[-1] - s[0], 1e-12)
    if model == "uniform":
        return u
    if model == "power_law":
        dt = (np.abs(curvature) + kappa0) ** beta
        t = np.r_[0, np.cumsum(0.5 * (dt[1:] + dt[:-1]) * np.diff(s))]
        return t / max(t[-1], 1e-12)
    if model == "min_jerk":
        tau = np.linspace(0, 1, 2001)
        pos = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        return np.interp(u, pos, tau)
    raise ValueError(model)


@dataclass
class Kinematics:
    t: np.ndarray
    xy: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    speed: np.ndarray
    curvature: np.ndarray
    timing: str


def kinematics(g: Geometry, timing: str = "power_law", n: int = 128, derivative: str = "savgol") -> Kinematics:
    """Resample the path at even (assumed) time steps and differentiate. Duration is 1."""
    t_s = time_parameterize(g.s, g.curvature, timing)
    t = np.linspace(0, 1, n)
    xy = np.c_[np.interp(t, t_s, g.xy[:, 0]), np.interp(t, t_s, g.xy[:, 1])]
    k = np.interp(t, t_s, g.curvature)
    p, v, a, j = derivatives(xy, t[1] - t[0], derivative, window=max(9, n // 10), closed=g.closed)
    return Kinematics(t, p, v, a, j, np.linalg.norm(v, axis=1), k, timing)


def kinematic_features(kin: Kinematics | None) -> dict:
    names = ["speed_peak_ratio", "speed_minima", "mean_acc", "mean_tan_acc", "mean_norm_acc", "mean_jerk",
             "ldlj", "direction_changes", "curv_rate", "curv_acc"]
    if kin is None:
        return dict.fromkeys(names, 0.0)
    dt = kin.t[1] - kin.t[0]
    sp = kin.speed
    tang = kin.velocity / (sp[:, None] + 1e-9)
    a_tan = np.sum(kin.acceleration * tang, 1)
    a_norm = kin.velocity[:, 0] * kin.acceleration[:, 1] - kin.velocity[:, 1] * kin.acceleration[:, 0]
    a_norm = a_norm / (sp + 1e-9)
    inner = sp[2:-2]
    minima = np.count_nonzero((inner[1:-1] < inner[:-2]) & (inner[1:-1] < inner[2:])) if len(inner) > 2 else 0
    peak = max(sp.max(), 1e-9)
    # Log dimensionless jerk (Hogan & Sternad 2009): duration 1, so -ln(∫|j|² dt / v_peak²).
    ldlj = -math.log(max(float(np.sum(np.sum(kin.jerk ** 2, 1)) * dt) / peak ** 2, 1e-12))
    heading = np.unwrap(np.arctan2(kin.velocity[:, 1], kin.velocity[:, 0]))
    turn_rate = np.gradient(heading, dt)
    big = np.abs(turn_rate) > 2 * math.pi * 0.25
    flips = np.count_nonzero(np.diff(np.sign(turn_rate[big])) != 0) if big.sum() > 1 else 0
    dk = np.gradient(kin.curvature, dt)
    return dict(speed_peak_ratio=float(peak / max(sp.mean(), 1e-9)), speed_minima=float(minima),
                mean_acc=float(np.linalg.norm(kin.acceleration, axis=1).mean()),
                mean_tan_acc=float(np.abs(a_tan).mean()), mean_norm_acc=float(np.abs(a_norm).mean()),
                mean_jerk=float(np.linalg.norm(kin.jerk, axis=1).mean()), ldlj=ldlj,
                direction_changes=float(flips), curv_rate=float(np.abs(dk).mean()),
                curv_acc=float(np.abs(np.gradient(dk, dt)).mean()))


# --------------------------------------------------------------------------------------------
# 4. Dynamics: the second-order pen model
# --------------------------------------------------------------------------------------------


def _discretize(zeta: float, omega: float, dt: float):
    """Exact discretization of x' = A x + B g with g linear between samples (first-order hold).
    Returns Φ, Γ0, Γ1 with x[k+1] = Φ x[k] + Γ0 g[k] + Γ1 (g[k+1] - g[k]) / dt."""
    A = np.array([[0.0, 1.0], [-omega ** 2, -2 * zeta * omega]])
    B = np.array([0.0, omega ** 2])
    M = np.zeros((4, 4))
    M[:2, :2], M[:2, 2], M[2, 3] = A, B, 1.0
    E = linalg.expm(M * dt)
    return E[:2, :2], E[:2, 2], E[:2, 3]


def simulate(g: np.ndarray, dt: float, zeta: float, omega: float, r0=None, v0=None, method: str = "exact") -> np.ndarray:
    """Pen position r tracking the reference ``g`` (n, d) sampled every ``dt``:
    r'' + 2ζω r' + ω² (r - g) = 0, per coordinate. ``r0``/``v0`` default to g[0] and 0.
    "exact": matrix-exponential discretization, exact for g linear between samples and stable at
    any dt; "rk4": fixed-step Runge-Kutta on the same linear interpolation of g."""
    g = np.asarray(g, float)
    one_d = g.ndim == 1
    g = g[:, None] if one_d else g
    r = np.array(g[0] if r0 is None else r0, float).reshape(-1) * np.ones(g.shape[1])
    v = np.zeros(g.shape[1]) if v0 is None else np.array(v0, float).reshape(-1) * np.ones(g.shape[1])
    out = np.empty_like(g)
    out[0] = r
    if method == "exact":
        Phi, G0, G1 = _discretize(zeta, omega, dt)
        x = np.stack([r, v])  # (2, d)
        for k in range(len(g) - 1):
            x = Phi @ x + np.outer(G0, g[k]) + np.outer(G1, (g[k + 1] - g[k]) / dt)
            out[k + 1] = x[0]
    elif method == "rk4":
        def f(r, v, gk):
            return v, -2 * zeta * omega * v - omega ** 2 * (r - gk)
        for k in range(len(g) - 1):
            ga, gm, gb = g[k], 0.5 * (g[k] + g[k + 1]), g[k + 1]
            k1r, k1v = f(r, v, ga)
            k2r, k2v = f(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v, gm)
            k3r, k3v = f(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v, gm)
            k4r, k4v = f(r + dt * k3r, v + dt * k3v, gb)
            r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            out[k + 1] = r
    else:
        raise ValueError(method)
    return out[:, 0] if one_d else out


def invert(r: np.ndarray, dt: float, zeta: float, omega: float, derivative: str = "savgol", window: int = 15,
           lam: float | None = None) -> np.ndarray:
    """Reference implied by an observed path: g = r + (r'' + 2ζω r') / ω². Second derivatives of
    scanned ink are noisy, so derivatives come from ``derivatives`` (smoothed, not "finite")."""
    p, d1, d2, _ = derivatives(r, dt, derivative, window, lam=lam)
    return p + (d2 + 2 * zeta * omega * d1) / omega ** 2


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: indices of the corners of a polyline within ``eps`` of ``pts``."""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        n = np.linalg.norm(ab)
        seg = pts[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / n if n > 0 else np.linalg.norm(seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.nonzero(keep)[0]


ZETAS = (0.1, 0.3, 0.5, 0.707, 1.0, 1.5)
OMEGAS = (1.0, 2.0, 5.0, 10.0, 20.0)


def polyline_reference(r: np.ndarray, eps: float) -> np.ndarray:
    """The prior for g: straight pieces between the path's corners, traversed with the path's
    timing. Corners rounded by pen lag are what ζ and ω are fitted to."""
    idx = rdp(r, eps)
    k = np.arange(len(r))
    return np.c_[np.interp(k, idx, r[idx, 0]), np.interp(k, idx, r[idx, 1])]


def simulate_grid(g: np.ndarray, dt: float, zetas, omegas, r0=None) -> np.ndarray:
    """``simulate`` (exact method, starting at rest) for every (ζ, ω) pair at once:
    g (n, d) -> (len(zetas) * len(omegas), n, d), ζ varying slowest."""
    pairs = [(z, w) for z in zetas for w in omegas]
    mats = [_discretize(z, w, dt) for z, w in pairs]
    Phi = np.stack([m[0] for m in mats])                    # (C, 2, 2)
    G0 = np.stack([m[1] for m in mats])[:, :, None]         # (C, 2, 1)
    G1 = np.stack([m[2] for m in mats])[:, :, None]
    X = np.zeros((len(pairs), 2, g.shape[1]))
    X[:, 0] = g[0] if r0 is None else r0
    out = np.empty((len(pairs),) + g.shape)
    out[:, 0] = X[:, 0]
    dg = np.diff(g, axis=0) / dt
    for k in range(len(g) - 1):
        X = np.einsum("cij,cjd->cid", Phi, X) + G0 * g[k] + G1 * dg[k]
        out[:, k + 1] = X[:, 0]
    return out


def fit_dynamics_many(rs: list[np.ndarray], dt: float, refs: list[np.ndarray], zetas=ZETAS, omegas=OMEGAS) -> list[dict]:
    """Grid fit of (ζ, ω) so that simulate(reference) ≈ r, for several paths of equal length at
    once. Each result has the best pair, its RMS tracking error, the error grid, and
    ``identifiability`` = 1 - best / median error over the grid: near 0 means the data barely
    prefer any setting. All of it is relative to the reference prior; with a wrong prior it
    means nothing."""
    if not rs:
        return []
    R = np.concatenate(rs, axis=1)
    sim = simulate_grid(np.concatenate(refs, axis=1), dt, zetas, omegas, r0=R[0])
    sq = (sim - R[None]) ** 2                                # (C, n, 2S)
    err = np.sqrt(sq.reshape(sq.shape[0], sq.shape[1], -1, 2).sum(3).mean(1))  # (C, S)
    out = []
    for e in err.T:
        grid = e.reshape(len(zetas), len(omegas))
        i, j = np.unravel_index(np.argmin(grid), grid.shape)
        out.append(dict(zeta=float(zetas[i]), omega=float(omegas[j]), error=float(grid[i, j]), grid=grid,
                        identifiability=float(1 - grid[i, j] / max(np.median(grid), 1e-12))))
    return out


def fit_dynamics(r: np.ndarray, dt: float, reference: np.ndarray, zetas=ZETAS, omegas=OMEGAS) -> dict:
    """``fit_dynamics_many`` for one path."""
    return fit_dynamics_many([r], dt, [reference], zetas, omegas)[0]


DYNAMIC_NAMES = ["dyn_zeta", "dyn_log_omega", "dyn_error", "dyn_ident", "dyn_overshoot", "ref_corners"]


def dynamic_features_many(kins: list[Kinematics | None], eps: float = 0.05) -> list[dict]:
    """Pen-model fit per segment, against the straight-pieces prior (``polyline_reference``)."""
    have = [k for k in kins if k is not None]
    refs = [polyline_reference(k.xy, eps) for k in have]
    fits = iter(fit_dynamics_many([k.xy for k in have], have[0].t[1] - have[0].t[0], refs) if have else [])
    out = []
    for k in kins:
        if k is None:
            out.append(dict.fromkeys(DYNAMIC_NAMES, 0.0))
            continue
        fit = next(fits)
        z = fit["zeta"]
        out.append(dict(dyn_zeta=z, dyn_log_omega=math.log(fit["omega"]), dyn_error=fit["error"],
                        dyn_ident=fit["identifiability"], dyn_overshoot=resonance(z, 1.0)["overshoot"],
                        ref_corners=float(len(rdp(k.xy, eps)) - 2)))
    return out


# --------------------------------------------------------------------------------------------
# 5. Frequency response of the pen model
# --------------------------------------------------------------------------------------------


def transfer(zeta: float, omega: float, Omega: np.ndarray) -> np.ndarray:
    """H(jΩ) = ω² / (ω² - Ω² + j 2ζωΩ)."""
    Omega = np.asarray(Omega, float)
    return omega ** 2 / (omega ** 2 - Omega ** 2 + 2j * zeta * omega * Omega)


def bode(zeta: float, omega: float, Omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude (dB) and phase (degrees) of H(jΩ)."""
    H = transfer(zeta, omega, Omega)
    return 20 * np.log10(np.abs(H)), np.degrees(np.unwrap(np.angle(H)))


def resonance(zeta: float, omega: float) -> dict:
    """Closed-form properties of the second-order system: resonant peak (only for ζ < 1/√2),
    -3 dB bandwidth, damped frequency, step overshoot, 2% settling time (≈ 4 / ζω)."""
    peak_freq = omega * math.sqrt(1 - 2 * zeta ** 2) if zeta < 1 / math.sqrt(2) else 0.0
    peak_gain = 1 / (2 * zeta * math.sqrt(1 - zeta ** 2)) if zeta < 1 / math.sqrt(2) else 1.0
    bw = omega * math.sqrt(1 - 2 * zeta ** 2 + math.sqrt(4 * zeta ** 4 - 4 * zeta ** 2 + 2))
    under = zeta < 1
    return dict(peak_freq=peak_freq, peak_gain=peak_gain, bandwidth=bw,
                damped_freq=omega * math.sqrt(1 - zeta ** 2) if under else 0.0,
                overshoot=math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)) if under else 0.0,
                settling_time=4 / (zeta * omega))


def attenuated_energy(kin: Kinematics, zeta: float, omega: float) -> float:
    """Share of the path's (pseudo-time) spectral energy the pen model would remove:
    1 - Σ|H(jΩ)|²|X(Ω)|² / Σ|X(Ω)|². Uses the assumed timing, so it inherits that assumption."""
    x = kin.xy - kin.xy.mean(0)
    X = np.fft.rfft(x, axis=0)
    Omega = 2 * np.pi * np.fft.rfftfreq(len(x), kin.t[1] - kin.t[0])
    e = np.sum(np.abs(X) ** 2, 1)
    h2 = np.abs(transfer(zeta, omega, Omega)) ** 2
    return float(1 - np.sum(h2 * e) / max(np.sum(e), 1e-12))


def plot_bode(params: list[tuple[float, float]], Omega: np.ndarray | None = None, ax=None):
    """Bode plots for several (ζ, ω). Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    Omega = np.logspace(-1, 2, 400) if Omega is None else Omega
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    else:
        fig = ax[0].figure
    for z, w in params:
        mag, ph = bode(z, w, Omega)
        ax[0].semilogx(Omega, mag, label=f"ζ={z:g}, ω={w:g}")
        ax[1].semilogx(Omega, ph)
    ax[0].set_ylabel("magnitude (dB)")
    ax[1].set_ylabel("phase (°)")
    ax[1].set_xlabel("Ω (rad / unit time)")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    return fig


# --------------------------------------------------------------------------------------------
# Frequency-domain features of the stroke geometry
# --------------------------------------------------------------------------------------------


def fourier_descriptors(g: Geometry, k: int = 6, n: int = 64) -> np.ndarray:
    """|Z_m| for m = ±1..±k of z(s) = x + iy at n even arc-length points, in writing units.
    Magnitudes ignore rotation and start point; orientation is in the chord/axis features. Open
    paths are mirrored (there and back) so the FFT doesn't see a jump at the ends."""
    u = np.linspace(g.s[0], g.s[-1], n)
    z = np.interp(u, g.s, g.xy[:, 0]) + 1j * np.interp(u, g.s, g.xy[:, 1])
    if not g.closed:
        z = np.r_[z, z[::-1]]
    Z = np.fft.fft(z - z.mean()) / len(z)
    return np.abs(np.r_[Z[1:k + 1], Z[-k:][::-1]])


def frequency_features(g: Geometry | None, kin: Kinematics | None, k: int = 6) -> dict:
    names = [f"fd_pos{m}" for m in range(1, k + 1)] + [f"fd_neg{m}" for m in range(1, k + 1)] + \
            ["curv_low", "curv_high", "curv_centroid", "attenuation"]
    if g is None:
        return dict.fromkeys(names, 0.0)
    fd = fourier_descriptors(g, k)
    kappa = np.interp(np.linspace(g.s[0], g.s[-1], 64), g.s, g.curvature)
    P = np.abs(np.fft.rfft(kappa - kappa.mean())) ** 2
    total = max(P.sum(), 1e-12)
    m = np.arange(len(P))
    f = {name: float(v) for name, v in zip(names, fd)}
    f.update(curv_low=float(P[1:4].sum() / total), curv_high=float(P[8:].sum() / total),
             curv_centroid=float((m * P).sum() / total),
             attenuation=attenuated_energy(kin, 0.707, 10.0) if kin is not None else 0.0)
    return f


# --------------------------------------------------------------------------------------------
# Transitions between segments
# --------------------------------------------------------------------------------------------


def _end_tangent(pts: np.ndarray, at_start: bool, unit: float, reach: float = 0.15) -> np.ndarray:
    """Unit direction leaving the segment's start (or end) node, over ``reach`` units."""
    p = pts if at_start else pts[::-1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] / unit
    q = p[min(np.searchsorted(d, reach), len(p) - 1)]
    v = (q - p[0]) * np.array([1, -1])
    return v / max(np.linalg.norm(v), 1e-9)


def transition_features(sk: Skeleton, unit: float) -> list[dict]:
    """Per segment: how smoothly it continues into its best neighbour at each node (0 = a straight
    continuation, π = doubling back), how many segments meet there, and the gap to the next
    component along the line."""
    ends = {}  # node -> [(segment index, outward tangent)]
    for i, s in enumerate(sk.segments):
        if s.kind == "path":
            ends.setdefault(s.start, []).append((i, _end_tangent(s.points, True, unit)))
            ends.setdefault(s.end, []).append((i, _end_tangent(s.points, False, unit)))
    comp_x = {}
    for s in sk.segments:
        lo, hi = s.points[:, 0].min(), s.points[:, 0].max()
        a, b = comp_x.get(s.component, (lo, hi))
        comp_x[s.component] = (min(a, lo), max(b, hi))
    order = sorted(comp_x, key=lambda c: comp_x[c][0])
    nxt_gap = {c: (comp_x[order[k + 1]][0] - comp_x[c][1]) / unit if k + 1 < len(order) else 0.0
               for k, c in enumerate(order)}
    out = []
    for i, s in enumerate(sk.segments):
        f = dict(free_ends=0.0, junction_ends=0.0, best_continuation=0.0, worst_continuation=0.0,
                 junction_degree=0.0, next_gap=float(nxt_gap.get(s.component, 0.0)))
        if s.kind == "path":
            devs, degs = [], []
            for node in (s.start, s.end):
                here = ends.get(node, [])
                mine = [t for j, t in here if j == i]
                others = [t for j, t in here if j != i]
                degs.append(len(here))
                if not others:
                    f["free_ends"] += 1
                    continue
                f["junction_ends"] += 1
                t = mine[0]
                # Outward tangents of a smooth continuation point in opposite directions.
                devs.append(min(math.acos(np.clip(-float(t @ o), -1, 1)) for o in others))
            if devs:
                f["best_continuation"], f["worst_continuation"] = min(devs), max(devs)
            f["junction_degree"] = float(max(degs))
        out.append(f)
    return out


# --------------------------------------------------------------------------------------------
# 6. Feature vectors
# --------------------------------------------------------------------------------------------

GROUPS = ("geometry", "kinematic", "dynamic", "frequency", "transition")


@dataclass
class DSRConfig:
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    unit: float | None = None           # pixels per writing unit; None = image height
    smoothing: str = "savgol"           # "savgol", "spline" or "finite"
    timing: str = "power_law"           # see TIMINGS
    fourier_k: int = 6
    rdp_eps: float = 0.05               # units: corner tolerance of the dynamics prior
    groups: tuple[str, ...] = GROUPS

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DSRResult:
    skeleton: Skeleton
    names: list[str]
    groups: dict[str, list[str]]
    features: np.ndarray                # (segments, features)
    meta: list[dict]                    # per segment: kind, component, bbox, nodes
    config: dict


def extract_features(gray: np.ndarray, cfg: DSRConfig | None = None) -> DSRResult:
    """A line image -> one feature vector per skeleton segment, with metadata and the config."""
    cfg = cfg or DSRConfig()
    sk = extract_strokes(gray, cfg.extract)
    unit = cfg.unit or float(gray.shape[0])
    trans = transition_features(sk, unit) if "transition" in cfg.groups else [{}] * len(sk.segments)
    rows, meta, groups = [], [], {}
    geos = [geometry(seg, unit, cfg.smoothing) for seg in sk.segments]
    kins = [kinematics(g, cfg.timing) if g is not None else None for g in geos]
    dyns = dynamic_features_many(kins, cfg.rdp_eps) if "dynamic" in cfg.groups else [{}] * len(kins)
    for seg, tr, g, kin, dyn in zip(sk.segments, trans, geos, kins, dyns):
        parts = {"geometry": geometric_features(seg, g, unit, gray.shape[0]),
                 "kinematic": kinematic_features(kin),
                 "dynamic": dyn,
                 "frequency": frequency_features(g, kin, cfg.fourier_k),
                 "transition": tr}
        row = {}
        for name in cfg.groups:
            groups.setdefault(name, list(parts[name]))
            row.update(parts[name])
        rows.append(row)
        x0, y0 = seg.points.min(0)
        x1, y1 = seg.points.max(0)
        meta.append(dict(kind=seg.kind, component=seg.component, bbox=[float(x0), float(y0), float(x1), float(y1)],
                         start=seg.start, end=seg.end, n_points=len(seg.points)))
    names = [n for grp in cfg.groups for n in groups.get(grp, [])]
    if not names:  # an empty line: take the names from a dot's features
        names = [n for grp in cfg.groups for n in _empty_names(grp, cfg)]
        groups = {grp: _empty_names(grp, cfg) for grp in cfg.groups}
    X = np.array([[r[n] for n in names] for r in rows], np.float32).reshape(len(rows), len(names))
    return DSRResult(sk, names, groups, X, meta, cfg.to_dict())


def _empty_names(group: str, cfg: DSRConfig) -> list[str]:
    seg = Segment(np.zeros((1, 2)), np.ones(1), 0, -1, -1, "dot")
    return list({"geometry": geometric_features(seg, None, 1.0, 1.0), "kinematic": kinematic_features(None),
                 "dynamic": dict.fromkeys(DYNAMIC_NAMES, 0.0), "frequency": frequency_features(None, None, cfg.fourier_k),
                 "transition": dict(free_ends=0, junction_ends=0, best_continuation=0, worst_continuation=0,
                                    junction_degree=0, next_gap=0)}[group])


class FeatureScaler:
    """Robust per-feature standardization (median / IQR), clipped, fitted on training lines only."""

    def __init__(self, names: list[str], centre=None, scale=None, clip: float = 5.0):
        self.names, self.clip = list(names), clip
        self.centre = None if centre is None else np.asarray(centre, np.float32)
        self.scale = None if scale is None else np.asarray(scale, np.float32)

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        q1, med, q3 = np.percentile(X, [25, 50, 75], axis=0)
        self.centre = med.astype(np.float32)
        self.scale = np.maximum(np.maximum((q3 - q1) / 1.349, np.std(X, 0) * 0.1), 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.centre) / self.scale, -self.clip, self.clip).astype(np.float32)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(dict(names=self.names, centre=self.centre.tolist(), scale=self.scale.tolist(), clip=self.clip), f)

    @classmethod
    def load(cls, path) -> "FeatureScaler":
        with open(path) as f:
            d = json.load(f)
        return cls(d["names"], d["centre"], d["scale"], d["clip"])


def column_features(res: DSRResult, width: int, stride: int = 4, X: np.ndarray | None = None) -> np.ndarray:
    """Per-segment features spread onto the recognizer's frames (one per ``stride`` px of width),
    as the ink-weighted mean of the segments in each frame, plus a last column of ink coverage.
    ``X`` overrides ``res.features`` (e.g. scaled ones). Returns (frames, features + 1)."""
    X = res.features if X is None else X
    T = (width + stride - 1) // stride
    acc = np.zeros((T, X.shape[1]), np.float32)
    weight = np.zeros(T, np.float32)
    for row, seg in zip(X, res.skeleton.segments):
        frames = np.clip((seg.points[:, 0] // stride).astype(int), 0, T - 1)
        counts = np.bincount(frames, minlength=T).astype(np.float32)
        acc += counts[:, None] * row[None]
        weight += counts
    out = acc / np.maximum(weight, 1)[:, None]
    return np.c_[out, np.minimum(weight / stride, 1)]


# --------------------------------------------------------------------------------------------
# Synthetic variation and visualization
# --------------------------------------------------------------------------------------------


def vary_path(path: np.ndarray, zeta: float, omega: float, timing: str = "uniform", n: int = 200,
              tail: float = 0.1) -> np.ndarray:
    """Redraw a reference pen path (n, 2) as a pen with dynamics (ζ, ω) would trace it, over a
    duration of 1 plus a ``tail`` for the pen to settle. A variation, not real handwriting:
    check that the shorthand still reads the same (e.g. Topaz reads it back) before using it."""
    s, path = resample(np.asarray(path, float), _arc_length(path) / 400)
    heading = np.unwrap(np.arctan2(*np.gradient(path, axis=0)[:, ::-1].T))
    curvature = np.gradient(heading) / max(s[1] - s[0], 1e-12)
    t_s = time_parameterize(s, curvature, timing)
    t = np.linspace(0, 1, n)
    g = np.c_[np.interp(t, t_s, path[:, 0]), np.interp(t, t_s, path[:, 1])]
    extra = int(round(tail * n))
    g = np.r_[g, np.repeat(g[-1:], extra, 0)]
    return simulate(g, t[1] - t[0], zeta, omega)


def plot_extraction(res: DSRResult, feature: str = "mean_abs_curv"):
    """Original, binarized, skeleton, segments with nodes, curvature along each segment, and the
    per-segment feature matrix. Observed images on top; model-derived estimates below."""
    import matplotlib.pyplot as plt

    sk = res.skeleton
    unit = res.config["unit"] or sk.image.shape[0]
    fig, ax = plt.subplots(3, 2, figsize=(14, 7))
    ax[0, 0].imshow(sk.image, cmap="gray"); ax[0, 0].set_title("original (observed)")
    ax[0, 1].imshow(sk.ink, cmap="gray_r"); ax[0, 1].set_title("binarized")
    ax[1, 0].imshow(sk.skeleton, cmap="gray_r"); ax[1, 0].set_title("skeleton")
    ax[1, 1].imshow(sk.image, cmap="gray", alpha=0.35)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(sk.segments):
        if s.kind == "dot":
            ax[1, 1].plot(*s.points[0], "o", color="k", ms=4)
        else:
            ax[1, 1].plot(s.points[:, 0], s.points[:, 1], color=cmap(i % 20), lw=1.5)
    if len(sk.nodes):
        deg = sk.node_degree
        ax[1, 1].scatter(*sk.nodes[deg == 1].T, s=12, c="green", label="ends")
        ax[1, 1].scatter(*sk.nodes[deg >= 3].T, s=18, c="red", label="junctions")
        ax[1, 1].legend(fontsize=7, loc="lower right")
    ax[1, 1].set_title("segments and nodes")
    ax[2, 0].imshow(sk.image, cmap="gray", alpha=0.3)
    for s in sk.segments:
        g = geometry(s, unit, res.config["smoothing"])
        if g is None:
            continue
        xy = np.c_[g.xy[:, 0], -g.xy[:, 1]] * unit
        ax[2, 0].scatter(xy[:, 0], xy[:, 1], c=np.clip(g.curvature, -15, 15), cmap="coolwarm", s=1, vmin=-15, vmax=15)
    ax[2, 0].set_title("curvature (estimated, red = counterclockwise)")
    Xs = (res.features - res.features.mean(0)) / (res.features.std(0) + 1e-9)
    ax[2, 1].imshow(Xs.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax[2, 1].set_title("features per segment (z-scored; model-derived)")
    ax[2, 1].set_xlabel("segment (left to right)")
    for a in ax.flat[:5]:
        a.set_axis_off()
    fig.tight_layout()
    return fig
