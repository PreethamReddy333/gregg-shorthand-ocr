"""Synthetic handwritten Gregg lines for fine-tuning Topaz: training-book lines redrawn from their
correct Grascii by random synthetic writers (gregg_ocr.render.Style.writer).

    make_synth_lines.py --n 12000 --out data/synth_lines/writers.parquet
    make_synth_lines.py --vary sln --out data/synth_lines/sln.parquet    # sigma-lognormal hands
    make_synth_lines.py --vary dsr --out data/synth_lines/dsr.parquet    # second-order pen hands

--vary replaces the renderer's ad-hoc pen lag with a motor model (gregg_ocr.pen_models), with a
random hand per line; the writer's proportions, slant and stroke jitter stay the same.

Each drawing is cut into a line crop by Topaz's own line finder, as at inference, and stored
ink-high at height 128 like the book crops. Topaz reads every crop back. Lines whose reading is
further than --max-ter from the label are dropped: the renderer drew them unrecognizably, and
training on them would teach its mistakes. The read-back stroke error is kept, so synthetic lines
can be compared with real handwriting for how hard they are.
"""

import argparse
import dataclasses
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
from eval_topaz import edit, strokes  # noqa: E402
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.pen_models import Hand, redraw_lines  # noqa: E402
from gregg_ocr.render import Style, layout, render  # noqa: E402
from topaz import Topaz  # noqa: E402
from topaz.pipeline import prepare  # noqa: E402

HELD_OUT = {"fm2", "manual"}  # test and dev books: never drawn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--max-ter", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vary", choices=["none", "sln", "dsr"], default="none", help="motor model for the pen")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth_lines" / "writers.parquet")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    books = [p for p in sorted((PROCESSED / "topaz").glob("*.parquet")) if p.stem not in HELD_OUT]
    df = pd.concat([pd.read_parquet(p, columns=["book", "grascii"]) for p in books], ignore_index=True)
    df = df[df.grascii.str.split().str.len() >= 2].sample(frac=1, random_state=args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(ROOT / "models" / "topaz-61m", device)
    rng = random.Random(args.seed)
    rows, dropped, split, t0 = [], 0, 0, time.time()
    for k, r in enumerate(df.itertuples()):
        if len(rows) >= args.n:
            break
        writer = rng.randrange(1 << 30)
        style = Style.writer(random.Random(writer))
        try:
            if args.vary == "none":
                lines = layout(r.grascii, style, rng=random.Random(writer))
            else:
                style = dataclasses.replace(style, lag=0.0)
                wr = random.Random(writer)
                lines = layout(r.grascii, style, rng=wr)
                hand = Hand.sample(args.vary, random.Random(writer + 1))
                lines = redraw_lines(lines, hand, wr)
            img = np.array(render(lines, style).convert("L"))
        except Exception:  # an outline the renderer can't draw
            dropped += 1
            continue
        crops = topaz.find_lines(img)
        if len(crops) != 1:
            split += 1
            continue
        pred = topaz.read_line(crops[0])
        gs = strokes(r.grascii)
        ter = edit(gs, strokes(pred)) / max(len(gs), 1)
        if ter > args.max_ter:
            dropped += 1
            continue
        crop = prepare(crops[0], 128)
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format="PNG")
        rows.append(dict(book=r.book, grascii=r.grascii, writer=writer, ter_topaz=ter, ctc_pred=pred,
                         height=crop.shape[0], width=crop.shape[1], png=buf.getvalue()))
        if len(rows) % 500 == 0:
            print(f"{len(rows)} kept / {k + 1} drawn, {dropped} dropped, {split} not one line, "
                  f"{(time.time() - t0) / (k + 1):.2f} s/line, mean read-back error "
                  f"{np.mean([x['ter_topaz'] for x in rows]):.1%}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(args.out)
    print(f"{len(out)} lines -> {args.out}; {dropped} dropped, {split} not one line; "
          f"read-back stroke error {out.ter_topaz.mean():.1%}")


if __name__ == "__main__":
    main()
