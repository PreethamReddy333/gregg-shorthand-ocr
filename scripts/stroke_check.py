"""Per-stroke read-back check of the renderer: draw book lines that contain a given stroke from
their correct Grascii, have Topaz read the drawings, and report how often that stroke comes back
right, what it is read as instead, and the stroke error of the whole lines (so a fix for one
stroke that breaks others shows up).

    stroke_check.py I AU EU OE SS LD MN ^ --n 80          # clean textbook style
    stroke_check.py I --n 80 --writers 3                  # also 3 random synthetic writers per line

Lines come from the training books (not FM2 or the 1929 Manual), with a fixed seed so runs
compare like for like.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
from eval_topaz import edit, outlines, strokes  # noqa: E402
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.render import Style, layout, render  # noqa: E402
from gregg_ocr.synth import _align  # noqa: E402

HELD_OUT = {"fm2", "manual"}


def has(g: str, s: str) -> bool:
    return any(s in o.split("-") for o in outlines(g)) if s != "^" else "^" in g


def book_lines() -> pd.Series:
    books = [p for p in sorted((PROCESSED / "topaz").glob("*.parquet")) if p.stem not in HELD_OUT]
    g = pd.concat([pd.read_parquet(p, columns=["grascii"]).grascii for p in books], ignore_index=True)
    return g[g.str.split().str.len().between(3, 14)]


def check(topaz, lines: list[str], target: str, styles: list) -> dict:
    hit = miss = 0
    instead = Counter()
    err = ref = 0
    for g in lines:
        for style in styles:
            img = np.array(render(layout(g, style, rng=random.Random(0)), style).convert("L"))
            crops = topaz.find_lines(img)
            pred = " ".join(topaz.read_line(c) for c in crops)
            gs = strokes(g)
            err += edit(gs, strokes(pred))
            ref += len(gs)
            go, po = outlines(g), outlines(pred)
            if len(go) != len(po):
                continue
            for a, b in zip(go, po):
                for x, y in _align(a.split("-"), b.split("-")):
                    if x == target:
                        if y == x:
                            hit += 1
                        else:
                            miss += 1
                            instead[y or "missing"] += 1
    n = hit + miss
    return dict(recall=hit / n if n else float("nan"), n=n, instead=instead.most_common(4), line_ter=err / max(ref, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strokes", nargs="+")
    ap.add_argument("--n", type=int, default=80, help="lines per stroke")
    ap.add_argument("--writers", type=int, default=0, help="also draw each line by this many random writers")
    args = ap.parse_args()
    from topaz import Topaz

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(ROOT / "models" / "topaz-61m", device)
    pool = book_lines()
    rng = random.Random(0)
    styles = [Style()] + [Style.writer(random.Random(rng.randrange(1 << 30))) for _ in range(args.writers)]
    for s in args.strokes:
        lines = pool[pool.map(lambda g: has(g, s))].sample(frac=1, random_state=0).head(args.n).tolist()
        r = check(topaz, lines, s, styles)
        instead = ", ".join(f"{k} {v}" for k, v in r["instead"])
        print(f"{s:4} read right {r['recall']:.0%} of {r['n']:4}   instead: {instead:40}  line stroke error {r['line_ter']:.1%}",
              flush=True)


if __name__ == "__main__":
    main()
