"""Read phone photos and screenshots of handwritten Gregg, where transcribe.py's line finder struggles
(ruled paper, uneven light, curved pages, white-on-dark tablet drawings).

Try transcribe.py first: when its lines come out right it reads better. On the keyed handwritten
letter it scores 65.0% word error (trained model v2) against 72-80% here. Use this script when
transcribe.py finds no lines, merges lines, or reads them empty.

    python scripts/read_photo.py page.jpg                         # dark ink on paper
    python scripts/read_photo.py page.jpg --crop 70 600 180 1120  # x0 x1 y0 y1: skip headers and margins
    python scripts/read_photo.py drawing.png --white              # light strokes on a dark screen
    python scripts/read_photo.py page.jpg --scales 1.5 2 2.5 -o runs/photo/

Steps, each written to the output folder so they can be checked by eye:
1. **Ink.** Darkness against the local background (a 20 px blur), kept by hysteresis: strokes seeded
   where dark > 0.45 and grown into dark > 0.22, so faint pencil stays joined and paper texture goes.
2. **Deskew** by the angle that makes the row profile sharpest.
3. **Ruling.** Long thin horizontal runs are removed, except where a stroke crosses them.
4. **Lines.** Ink components are grouped into writing lines by their height on the page, following
   each line's neighbours by x so bent lines on curved paper stay together; dots join the nearest line.
5. **Strokes.** Each line is thinned to its skeleton and redrawn at an even width, then read by Topaz at
   several scales. Gregg tells N from M and T from D by length, so a reading that changes with
   scale is a reading not to trust: the report marks outlines that agree across scales.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))


def ink_map(rgb: np.ndarray, white: bool) -> np.ndarray:
    """RGB float -> bool ink mask, deskewed."""
    if white:
        ink = rgb.min(-1) > 175
    else:
        g = rgb.min(-1)  # ruling is light blue or red: its darkest channel stays fairly bright
        dark = np.clip(1 - g / np.maximum(cv2.GaussianBlur(g, (0, 0), 20), 1), 0, 1)
        n, lab, _, _ = cv2.connectedComponentsWithStats((dark > 0.22).astype(np.uint8), 8)
        good = np.zeros(n, bool)
        good[np.unique(lab[dark > 0.45])] = True
        good[0] = False
        ink = good[lab]
    H, W = ink.shape
    rot = lambda a, im: cv2.warpAffine(im, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1), (W, H))
    u8 = ink.astype(np.uint8)
    angle = max(np.arange(-8, 8.01, 0.5), key=lambda a: np.var(rot(a, u8).sum(1)))
    ink = rot(angle, u8) > 0
    # ruling: long thin horizontal runs, unless a stroke passes through
    h = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    vert = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 1), np.uint8)) > 0
    ink &= ~(cv2.dilate(h, np.ones((3, 1), np.uint8)) > 0) | vert
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.array([k > 0 and st[k][4] >= 6 and not (st[k][3] <= 4 and st[k][2] > 30) for k in range(n)])
    return keep[lab]


def group_lines(ink: np.ndarray) -> list[np.ndarray]:
    """Ink mask -> one mask crop per writing line, top to bottom."""
    p = ink.sum(1).astype(np.float32)
    p -= p.mean()
    pitch = 35 + int(np.argmax([np.dot(p[:-k], p[k:]) for k in range(35, 90)]))
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    comps = [k for k in range(1, n) if st[k][4] >= 6]

    def ymid(line, x):  # the line's height near x, so bent lines are followed
        near = sorted(line, key=lambda k: abs(cen[k][0] - x))[:3]
        return np.median([cen[k][1] for k in near])

    lines = []
    for k in sorted((k for k in comps if st[k][4] >= 40), key=lambda k: cen[k][1]):
        best = min(lines, key=lambda L: abs(ymid(L, cen[k][0]) - cen[k][1]), default=None)
        if best is not None and abs(ymid(best, cen[k][0]) - cen[k][1]) < 0.45 * pitch:
            best.append(k)
        else:
            lines.append([k])
    for k in comps:
        if st[k][4] < 40 and lines:  # dots and ticks
            min(lines, key=lambda L: abs(ymid(L, cen[k][0]) - cen[k][1])).append(k)
    lines.sort(key=lambda L: np.median([cen[k][1] for k in L]))
    crops = []
    for L in lines:
        m = np.isin(lab, L)
        ys, xs = np.where(m)
        crops.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return crops


def even_strokes(mask: np.ndarray, scale: float) -> np.ndarray:
    """Line mask -> uint8 crop (ink dark) at ``scale``, strokes redrawn 4 px wide from the skeleton."""
    from skimage.morphology import skeletonize

    big = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sk = skeletonize(big > 0).astype(np.uint8)
    ink = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    return np.pad(np.where(ink > 0, 0, 255).astype(np.uint8), 30, constant_values=255)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--white", action="store_true", help="light strokes on a dark background")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--t5-model", type=Path, default=ROOT / "models" / "gregg-t5-v2")
    ap.add_argument("--topaz", type=Path, default=ROOT / "models" / "topaz-61m",
                    help="Topaz model folder, e.g. models/topaz-ft/control")
    ap.add_argument("-o", "--out", type=Path, help="folder for the cleaned page, line crops and report")
    args = ap.parse_args()

    from topaz import Topaz
    from gregg_ocr.beam import BeamDecoder
    from gregg_ocr.lexicon import Lexicon
    from gregg_ocr.lm import TrigramLM
    from gregg_ocr.t5 import T5Decoder

    rgb = np.array(Image.open(args.image).convert("RGB")).astype(np.float32)
    if args.crop:
        x0, x1, y0, y1 = args.crop
        rgb = rgb[y0:y1, x0:x1]
    ink = ink_map(rgb, args.white)
    lines = group_lines(ink)
    out = args.out or ROOT / "runs" / "photo" / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).save(out / "clean.png")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(args.topaz, device)
    readings = {s: [topaz.read_line(even_strokes(m, s)) for m in lines] for s in args.scales}
    for i, m in enumerate(lines):
        Image.fromarray(even_strokes(m, args.scales[0])).save(out / f"line{i:02d}.png")

    # per line, the reading most scales agree on; outlines shared by every scale are the trustworthy ones
    chosen, report = [], [f"# {args.image.name}: {len(lines)} lines, scales {args.scales}", ""]
    for i in range(len(lines)):
        reads = [readings[s][i] for s in args.scales]
        pick = Counter(reads).most_common(1)[0][0]
        chosen.append(pick)
        stable = set.intersection(*(set(r.split()) for r in reads))
        report.append(f"line {i:02d}: {' '.join(o if o in stable else o + '?' for o in pick.split())}")
        report += [f"    x{s}: {r}" for s, r in zip(args.scales, reads)]
    report += ["", "## English, trained model", *T5Decoder(args.t5_model, device).decode(chosen)]
    lm = ROOT / "models" / "trigram.pkl"
    if lm.exists():  # built locally by train_lm.py, not downloadable
        report += ["", "## English, beam search", *BeamDecoder(TrigramLM.load(lm), Lexicon.load(ROOT)).decode(chosen)]
    report += ["", "Outlines marked ? read differently at different scales: treat them as guesses."]
    text = "\n".join(report)
    print(text)
    (out / "report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
