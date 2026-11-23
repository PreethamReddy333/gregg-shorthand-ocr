"""Draw Grascii as Gregg shorthand.

Each consonant is an arc: a chord direction, a total turn (+ = counterclockwise) and a length,
in units of roughly one short stroke. Circle vowels are loops between the strokes they join,
written inside curves and outside angles; the O and OO hooks are small half-turns. ``Style``
holds one writer's proportions (the clean default follows the Manual's), so the same code draws
textbook outlines for people to read and varied synthetic handwriting for training Topaz.

The geometry is an approximation of the Manual, checked by having Topaz read the drawings back
(``scripts/write_gregg.py --check``), not a facsimile of the engraved plates.
"""

import math
import random
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .text import PUNCT

UP, DOWN = 35.0, 250.0  # chord directions of T/D and of the downstrokes, degrees
# stroke -> (chord direction, turn, length, size class used by Style). Turns, lengths and circle
# sizes were tuned so that Topaz reads drawn book lines back with the fewest stroke errors; LD,
# long I, the diphthong hooks and disjoining were then tuned stroke by stroke (scripts/stroke_check.py).
CONSONANTS = {
    "K": (0, -80, 1.04, "short"), "G": (0, -80, 2.07, "long"),
    "R": (0, 80, 1.04, "short"), "L": (0, 80, 2.07, "long"),
    "N": (0, 0, 0.78, "short"), "M": (0, 0, 1.75, "long"),
    "T": (UP, 0, 0.72, "short"), "D": (UP, 0, 1.75, "long"),
    "P": (DOWN + 5, 90, 1.17, "short"), "B": (DOWN + 5, 90, 2.07, "long"),
    "F": (DOWN + 5, -90, 1.17, "short"), "V": (DOWN + 5, -90, 2.07, "long"),
    "SH": (DOWN - 5, 0, 0.46, "short"), "CH": (DOWN - 5, 0, 1.1, "short"), "J": (DOWN - 5, 0, 1.75, "long"),
    "NG": (-25, 0, 0.91, "short"), "NK": (-25, 0, 1.64, "long"),
    "TN": (UP, -80, 1.31, "long"), "DN": (UP, -80, 1.31, "long"), "TM": (UP, -80, 2.29, "long"), "DM": (UP, -80, 2.29, "long"),
    "NT": (UP, 80, 1.31, "long"), "ND": (UP, 80, 1.31, "long"), "MT": (UP, 80, 2.29, "long"), "MD": (UP, 80, 2.29, "long"),
    "MN": (0, -10, 2.4, "long"), "MM": (0, -10, 3.05, "long"),
    "TD": (UP, 0, 2.84, "long"), "DT": (UP, 0, 2.84, "long"), "DD": (UP, 0, 2.84, "long"),
    "LD": (30, 100, 2.8, "long"),
    "DF": (-50, -200, 2.18, "long"), "DV": (-50, -200, 2.18, "long"), "TV": (-50, -200, 2.18, "long"),
    "PNT": (-25, 170, 1.75, "long"), "PND": (-25, 170, 1.75, "long"),
    "JNT": (-35, 170, 2.4, "long"), "JND": (-35, 170, 2.4, "long"),
}
# S and TH come in two directions, chosen by the joining rules (``_directions``).
SMALL_CURVES = {"S": (DOWN, 137, 0.57), "Z": (DOWN, 137, 0.57), "TH": (UP, 98, 0.8), "X": (-60, 120, 0.45)}
CIRCLES = {"A": 0.32, "I": 0.6, "A&E": 0.32, "A&'": 0.32, "E": 0.165}
HOOKS = {"O": (225, 183, 0.42), "U": (45, -183, 0.42)}  # start direction, turn, length
# Diphthongs are a circle and a hook; the hook is marked '*' so it can be sized on its own.
DIPHTHONGS = {"AU": ["A", "U*"], "EU": ["E", "U*"], "OE": ["O*", "E"]}
DIPHTHONG_HOOK = 1.8   # length of a diphthong's hook relative to a plain O or OO hook
I_INDENT = 1.0         # length of the dent that tells long I from A, in radii (0 = a plain circle)
DISJOIN = (0.2, 0.2)   # where a disjoined stroke (^) starts, relative to the end of the last one


@dataclass
class Style:
    """One writer. The defaults are clean textbook proportions."""
    unit: float = 26.0          # pixels per unit
    width: float = 2.2          # pen width, pixels
    short: float = 1.0          # length scale of short strokes (T, N, K, R, P, F...)
    long: float = 1.0           # ... and of long ones (D, M, G, L, B, V...)
    circle: float = 1.0         # circle vowel size
    curve: float = 1.0          # depth of curves (scales the turn)
    slant: float = 0.0          # degrees, + leans right
    lag: float = 0.0            # pen lag: 0 draws the path exactly, ~0.3 rounds corners and overshoots
    jitter: float = 0.0         # per-stroke random variation of length and direction
    gap: float = 0.7            # space between outlines, units
    ink: tuple = (20, 20, 20)
    paper: tuple = (255, 255, 255)

    @classmethod
    def writer(cls, rng: random.Random) -> "Style":
        """A random synthetic writer: proportions drift away from the Manual's."""
        return cls(unit=rng.uniform(18, 34), width=rng.uniform(1.4, 3.5),
                   short=rng.uniform(0.8, 1.4), long=rng.uniform(0.75, 1.15),
                   circle=rng.uniform(0.7, 1.4), curve=rng.uniform(0.6, 1.3),
                   slant=rng.gauss(0, 8), lag=rng.uniform(0.0, 0.5), jitter=rng.uniform(0.02, 0.12),
                   gap=rng.uniform(0.5, 1.1))


@dataclass
class Drawing:
    """Polylines (pen-down paths) and dots, in units with y pointing up."""
    paths: list = field(default_factory=list)
    dots: list = field(default_factory=list)

    def points(self):
        pts = [p for path in self.paths for p in path] + list(self.dots)
        return np.array(pts) if pts else np.zeros((0, 2))

    def shifted(self, dx, dy):
        return Drawing([[(x + dx, y + dy) for x, y in p] for p in self.paths], [(x + dx, y + dy) for x, y in self.dots])


def _norm(a):
    return (a + 180) % 360 - 180


def parse(outline: str) -> list[str]:
    """Topaz-style Grascii ('K-A-T', "'-O-P", 'K-^-S') -> stroke tokens, '^' for a pen lift."""
    toks = []
    for t in outline.lstrip("=").split("-"):
        t = t.rstrip("_").replace("\\", "")
        if not t:
            continue
        toks += DIPHTHONGS.get(t, [t])
    return toks


def _directions(toks: list[str]) -> list[str]:
    """Pick the direction of S, TH, O and U from their neighbours, after the Manual's rules
    (as implemented in grascii.outline): S takes the direction of the curve it joins, forms a
    sharp angle with straight lines, and is clockwise in S + vowel words; TH is clockwise except
    next to O, R, L; O lies on its side before N, M, R, L; OO lies on its side after N, M."""
    out = list(toks)
    cons = [i for i, t in enumerate(toks) if t in CONSONANTS or t in SMALL_CURVES]

    def neighbour(i, step):
        j = i + step
        while 0 <= j < len(toks) and toks[j] not in CONSONANTS and toks[j] not in SMALL_CURVES:
            if toks[j] == "^":
                return None
            j += step
        return toks[j] if 0 <= j < len(toks) else None

    for i, t in enumerate(toks):
        if t in ("S", "Z", "X", "SS", "XS"):
            prev, nxt = neighbour(i, -1), neighbour(i, 1)
            other = prev if prev in CONSONANTS else nxt if nxt in CONSONANTS else None
            if i == 0 and i + 1 < len(toks) and toks[i + 1] == "O":
                d = "R"
            elif other is None:
                d = "R"
            else:
                chord, turn, _, _ = CONSONANTS[other]
                if turn > 0:
                    d = "L"
                elif turn < 0:
                    d = "R"
                elif chord % 360 > 180:      # SH, CH, J
                    d = "R"
                else:                         # T, D, N, M: the sharp angle
                    d = "L" if other is prev else "R"
            out[i] = t + d
        elif t == "TH":
            near = {toks[j] for j in (i - 1, i + 1) if 0 <= j < len(toks)}
            out[i] = "THR" if near & {"O", "R", "L"} else "THL"
        elif t == "O" and i + 1 < len(toks) and toks[i + 1] in ("N", "M", "MN", "MM", "R", "L"):
            prev = toks[i - 1] if i else None
            if not (prev in CONSONANTS and CONSONANTS[prev][0] % 360 > 180):
                out[i] = "OL"
        elif t == "U" and i and toks[i - 1] in ("N", "M", "MN", "MM"):
            out[i] = "UR"
    return out


def _shape(tok: str, style: Style, rng: random.Random):
    """A consonant or hook -> (start direction, turn, length), in this writer's proportions."""
    if tok in CONSONANTS:
        chord, turn, length, size = CONSONANTS[tok]
        length *= style.short if size == "short" else style.long
    elif tok[:2] in ("SS", "XS"):
        # "ses" and "xes": the first curve, then an S the other way (drawn by draw_outline).
        base = "S" if tok[:2] == "SS" else "X"
        chord, turn, length = SMALL_CURVES[base]
        turn = turn if tok[-1] == "L" else -turn
        length *= style.short
    elif tok[:-1] in SMALL_CURVES:
        chord, turn, length = SMALL_CURVES[tok[:-1]]
        if tok[:-1] == "TH":
            turn = -turn if tok[-1] == "L" else turn
        else:
            turn = turn if tok[-1] == "L" else -turn
        length *= style.short
    elif tok[0] in HOOKS:
        start, turn, length = HOOKS[tok[0]]
        if tok.endswith("*"):
            length *= DIPHTHONG_HOOK
        if tok == "OL":
            start -= 90
        elif tok == "UR":
            start += 90
        chord = start + turn / 2
    else:
        raise KeyError(tok)
    turn *= style.curve
    if style.jitter:
        length *= math.exp(rng.gauss(0, style.jitter))
        chord += rng.gauss(0, 40 * style.jitter)
        turn *= math.exp(rng.gauss(0, style.jitter))
    return chord - turn / 2, turn, length


def _arc(path, heading, turn, length, step=0.04):
    """Extend ``path`` along an arc; returns the end heading."""
    n = max(2, int(length / step))
    x, y = path[-1]
    for k in range(n):
        h = math.radians(heading + turn * (k + 0.5) / n)
        x += math.cos(h) * length / n
        y += math.sin(h) * length / n
        path.append((x, y))
    return heading + turn


def draw_outline(outline: str, style: Style | None = None, rng: random.Random | None = None) -> Drawing:
    """One Grascii outline -> a Drawing, starting at the origin."""
    style = style or Style()
    rng = rng or random.Random(0)
    toks = _directions(parse(outline))
    shapes = {i: _shape(t, style, rng) for i, t in enumerate(toks) if t not in CIRCLES and t not in ("'", "^")}
    drawing, path, heading = Drawing(), [(0.0, 0.0)], None
    pending_dots = []

    def turn_of(i):
        return shapes[i][1] if i in shapes else None

    for i, t in enumerate(toks):
        if t == "^":
            if len(path) > 1:
                drawing.paths.append(path)
            x, y = path[-1]
            path, heading = [(x + DISJOIN[0], y + DISJOIN[1])], None
            continue
        if t == "'":
            # H before a vowel: a dot above the start; -ing and the rest: a dot after the outline.
            pending_dots.append("start" if i == 0 else "end")
            continue
        if t in CIRCLES:
            r = CIRCLES[t] * style.circle / 2
            nxt = next((j for j in range(i + 1, len(toks)) if j in shapes), None)
            if nxt is not None and any(toks[k] == "^" for k in range(i + 1, nxt)):
                nxt = None
            h_out = shapes[nxt][0] if nxt is not None else None
            prev = next((j for j in range(i - 1, -1, -1) if j in shapes), None) if heading is not None else None
            # Inside a curve, outside an angle; on a single straight line, clockwise.
            curve = turn_of(prev) if prev is not None and turn_of(prev) else turn_of(nxt) if nxt is not None else None
            if curve:
                sign = 1 if curve > 0 else -1
            elif heading is not None and h_out is not None:
                delta = _norm(h_out - heading)
                sign = (1 if -45 <= _norm(heading) <= 45 else -1) if abs(delta) < 15 else (-1 if delta > 0 else 1)
            else:
                sign = -1
            if heading is None and h_out is None:      # a lone circle
                heading, sweep = 0.0, 360.0
            elif heading is None:                      # word-initial
                sweep = 330.0
                heading = h_out - sign * sweep
            elif h_out is None:                         # word-final
                sweep = 330.0
            else:
                sweep = (sign * (h_out - heading)) % 360
                if sweep < 90:
                    sweep += 360
            if t == "I" and I_INDENT:
                # Long I: a large circle broken by a small inward dent where it starts.
                d = I_INDENT * r
                heading = _arc(path, heading - sign * 60, sign * 120, d)
            heading = _arc(path, heading, sign * sweep, math.radians(sweep) * r)
            continue
        start, turn, length = shapes[i]
        heading = _arc(path, start, turn, length)
        if t[:2] in ("SS", "XS"):
            s_chord, s_turn, s_len = SMALL_CURVES["S"]
            heading = _arc(path, heading, -turn if t[:2] == "SS" else -math.copysign(s_turn, turn) * style.curve,
                           (length if t[:2] == "SS" else s_len * style.short))
    if len(path) > 1:
        drawing.paths.append(path)
    pts = drawing.points()
    for where in pending_dots:
        if where == "start" and len(pts):
            x0, y0 = drawing.paths[0][0] if drawing.paths else (0, 0)
            drawing.dots.append((x0 - 0.15, y0 + 0.45))
        elif len(pts):
            x1, y1 = drawing.paths[-1][-1] if drawing.paths else (0, 0)
            drawing.dots.append((x1 + 0.3, y1 - 0.1))
        else:
            drawing.dots.append((0.0, 0.15))  # "a", "an": a dot on its own
    if outline.startswith("=") and len(pts):
        # A capital: two short upward dashes under the outline.
        x0, y0 = pts[:, 0].min() + 0.1, pts[:, 1].min() - 0.3
        drawing.paths += [[(x0, y0), (x0 + 0.25, y0 + 0.08)], [(x0 + 0.05, y0 - 0.14), (x0 + 0.3, y0 - 0.06)]]
    return _finish(drawing, style)


def _finish(drawing: Drawing, style: Style) -> Drawing:
    """Slant, then pen lag: a spring-damped pen following the path rounds corners and overshoots."""
    sh = math.tan(math.radians(style.slant))
    paths = [[(x + sh * y, y) for x, y in p] for p in drawing.paths]
    dots = [(x + sh * y, y) for x, y in drawing.dots]
    if style.lag > 0:
        k, c = 1.0 - 0.85 * style.lag, 0.55 + 0.3 * style.lag
        lagged = []
        for p in paths:
            pos, vel, out = np.array(p[0], float), np.zeros(2), []
            for q in p + [p[-1]] * 8:
                vel = (1 - c) * vel + k * (np.array(q) - pos)
                pos = pos + vel
                out.append(tuple(pos))
            lagged.append(out)
        paths = lagged
    return Drawing(paths, dots)


def _punctuation(tok: str) -> Drawing:
    if tok == ".":
        return Drawing([[(0.0, 0.35), (0.3, 0.05)]])     # the slanting dash
    if tok == "?":
        return Drawing([[(0.0, 0.3), (0.3, 0.0)], [(0.0, 0.0), (0.3, 0.3)]])
    if tok == ",":
        return Drawing([[(0.05, 0.1), (0.1, 0.0), (0.0, -0.2)]])
    return Drawing()


@dataclass
class Line:
    drawings: list            # Drawings placed on one line, in units
    items: list               # the Grascii items they came from


def layout(grascii: str, style: Style | None = None, max_width: float = 1e9, rng=None) -> list[Line]:
    """A Grascii line ('K-A-T S-E .') -> placed outlines, wrapped at ``max_width`` units.
    Each outline's first consonant rests on the line, as in the Manual."""
    style = style or Style()
    rng = rng or random.Random(0)
    lines, cur, items, x = [], [], [], 0.0
    for item in grascii.split():
        if item in PUNCT:
            d = _punctuation(item)
        elif re.search(r"[a-z0-9]", item):
            d = Drawing()  # figures and longhand words are drawn as text by ``render``
            d.text = item
        else:
            d = draw_outline(item, style, rng)
        pts = d.points()
        if len(pts):
            first = np.array(d.paths[0]) if d.paths else pts
            dy = -first[:, 1].min() if item not in PUNCT else 0.0
            if item not in PUNCT and first[:, 1].max() - first[:, 1].min() > 1.2:
                dy = -first[:, 1].min() - 0.3
            d = d.shifted(-pts[:, 0].min(), dy)
            w = pts[:, 0].max() - pts[:, 0].min()
        else:
            w = 0.3 * len(getattr(d, "text", "")) or 0.2
        if cur and x + w > max_width:
            lines.append(Line(cur, items))
            cur, items, x = [], [], 0.0
        text = getattr(d, "text", None)
        d = d.shifted(x, 0)
        if text:
            d.text, d.pos = text, (x, 0.0)
        cur.append(d)
        items.append(item)
        x += w + (style.gap * 0.5 if item in PUNCT else style.gap)
    if cur:
        lines.append(Line(cur, items))
    return lines


def render(lines: list[Line], style: Style | None = None, margin: float = 1.0, line_gap: float = 3.4,
           captions: list[str] | None = None, width_units: float | None = None, scale: int = 4) -> Image.Image:
    """Placed lines -> an image. ``captions``: text printed under each line (e.g. the English key)."""
    style = style or Style()
    u = style.unit * scale
    all_pts = [d.points() for ln in lines for d in ln.drawings if len(d.points())]
    xmax = max((p[:, 0].max() for p in all_pts), default=1.0)
    ymax = max((p[:, 1].max() for p in all_pts), default=1.0)
    ymin = min((p[:, 1].min() for p in all_pts), default=0.0)
    cap_h = 0.9 if captions else 0.0
    W = (width_units or xmax) + 2 * margin
    top = max(ymax, 1.5)
    row = max(line_gap, top - min(ymin, -1.0)) + cap_h
    H = margin * 2 + row * len(lines)
    img = Image.new("RGB", (int(W * u), int(H * u)), style.paper)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(0.45 * u))
    cap_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(0.42 * u))
    pw = max(1, int(style.width * scale))
    for li, ln in enumerate(lines):
        base = margin + li * row + top

        def px(p):
            return ((margin + p[0]) * u, (base - p[1]) * u)

        for dr in ln.drawings:
            for path in dr.paths:
                pts = [px(p) for p in path]
                d.line(pts, fill=style.ink, width=pw, joint="curve")
                for p in (pts[0], pts[-1]):
                    d.ellipse([p[0] - pw / 2, p[1] - pw / 2, p[0] + pw / 2, p[1] + pw / 2], fill=style.ink)
            for p in dr.dots:
                x, y = px(p)
                r = pw * 0.9
                d.ellipse([x - r, y - r, x + r, y + r], fill=style.ink)
            if getattr(dr, "text", None):
                x, y = px(dr.pos)
                d.text((x, y), dr.text, font=font, fill=style.ink, anchor="ls")
        if captions and li < len(captions):
            d.text((margin * u, (base + row - top - cap_h - 0.2) * u + 0.6 * u), captions[li], font=cap_font,
                   fill=(110, 110, 110))
    return img.resize((img.width // scale, img.height // scale), Image.LANCZOS)


def english_items(text: str, renderer) -> list[tuple[str, list[str]]]:
    """English -> (Grascii item, the words it writes) with ``synth.Renderer``. A word with no
    outline is kept as itself and drawn in longhand, as writers do with unfamiliar names."""
    from .text import tokenize

    return [(o if o is not None else ws[0], ws) for o, ws in renderer.render_items(tokenize(text))]
