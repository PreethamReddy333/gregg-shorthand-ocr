"""English (or Grascii) -> Gregg shorthand image, and a check of the drawings with Topaz.

    write_gregg.py "Of course I will be glad to see you." -o note.png --key
    write_gregg.py --file passage.txt -o page.png --key              # a practice sheet to copy by hand
    write_gregg.py --grascii "K-A-T S-E" -o out.png
    write_gregg.py "..." --writer 7 -o hand.png                      # a random synthetic writer
    write_gregg.py --check fm2 --n 100 [--writer 7]                  # Topaz reads book lines drawn from their Grascii

The check draws held-out book lines from their correct Grascii and has Topaz read the drawing
back: the stroke error is how far the drawings are from what Topaz learned Gregg looks like.
"""

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.render import Style, english_items, layout, render  # noqa: E402


def style_for(args) -> Style:
    style = Style.writer(random.Random(args.writer)) if args.writer is not None else Style()
    if args.unit:
        style.width *= args.unit / style.unit
        style.unit = args.unit
    return style


def write(args):
    style = style_for(args)
    text = Path(args.file).read_text() if args.file else args.text
    if args.grascii:
        lines = [ln for row in text.splitlines() if row.strip() for ln in layout(row, style, args.width)]
        captions = [" ".join(ln.items) for ln in lines] if args.key else None
    else:
        from gregg_ocr.lexicon import Lexicon
        from gregg_ocr.synth import PhoneticSpeller, Renderer

        renderer = Renderer(Lexicon.load(ROOT).word_to_outline(), phrase_rate=1.0, speller=PhoneticSpeller())
        lines, captions = [], []
        for para in (p for p in text.splitlines() if p.strip()):
            # tokenize() lower-cases; keep the original spelling for the key and for capital marks
            original = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?;:!]", para.replace("’", "'"), re.I)
            items = capitalize(english_items(para, renderer), original)
            english = {i: ws for i, (_, ws) in enumerate(items)}
            placed = layout(" ".join(o for o, _ in items), style, args.width)
            k = 0
            for ln in placed:
                words = [w for j in range(k, k + len(ln.items)) for w in english[j]]
                k += len(ln.items)
                lines.append(ln)
                captions.append(detok(words))
            longhand = [ws[0] for o, ws in items if o.lower() == ws[0].lower() and o.isalpha()]
            if longhand:
                print("written in longhand (no outline):", ", ".join(longhand), file=sys.stderr)
        for ln, cap in zip(lines, captions):
            print(" ".join(ln.items), "|", cap)
        captions = captions if args.key else None
    img = render(lines, style, captions=captions, width_units=args.width)
    if args.tablet:
        from PIL import ImageOps
        img = ImageOps.invert(img)
    img.save(args.out)
    print(args.out)


def capitalize(items, original):
    """Mark capitalized names with '=' (sentence starts and 'I' aren't marked in Gregg), and put
    the original spelling back into each item's words."""
    out, k, sentence_start = [], 0, True
    for o, ws in items:
        orig = original[k:k + len(ws)]
        k += len(ws)
        first = orig[0] if orig else ""
        if first[:1].isupper() and not sentence_start and first != "I" and o != ws[0] and o[:1] != "=":
            o = "=" + o
        abbreviation = ws == ["."] and out and out[-1][1][-1].lower() in ("mr", "mrs", "ms", "dr", "st")
        sentence_start = ws[-1] in (".", "?", "!") and not abbreviation
        out.append((o, orig or ws))
    return out


def detok(words):
    out = ""
    for w in words:
        out += w if (w in ".,?;:!" or not out) else " " + w
    return out


def check(args):
    """Draw held-out book lines from their correct Grascii; Topaz reads them back."""
    import torch
    from eval_topaz import edit, outlines, strokes
    from gregg_ocr.synth import _align
    from topaz import Topaz

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(ROOT / "models" / "topaz-61m", device)
    df = pd.read_parquet(PROCESSED / "topaz" / f"{args.check}.parquet")
    df = df[df.grascii.str.split().str.len() >= 3].sample(args.n, random_state=0)
    rng = random.Random(args.writer or 0)
    tot = Counter()
    confusions = Counter()
    for k, (_, r) in enumerate(df.iterrows()):
        style = Style.writer(rng) if args.writer is not None else Style()
        lines = layout(r.grascii, style)
        img = np.array(render(lines, style).convert("L"))
        crops = topaz.find_lines(img)
        pred = " ".join(topaz.read_line(c) for c in crops)
        gs, ps = strokes(r.grascii), strokes(pred)
        tot["se"] += edit(gs, ps)
        tot["sr"] += len(gs)
        go, po = outlines(r.grascii), outlines(pred)
        tot["oe"] += edit(go, po)
        tot["or"] += len(go)
        if len(go) == len(po):
            for a, b in zip(go, po):
                for x, y in _align(a.split("-"), b.split("-")):
                    if x != y:
                        confusions[(x or "-", y or "-")] += 1
        if k < args.show:
            print(f"gold: {r.grascii}\nread: {pred}\n")
            if args.save:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                render(lines, style, captions=[r.grascii]).save(Path(args.save) / f"check_{k:02d}.png")
    print(f"{args.n} lines from {args.check}, {'writer seed ' + str(args.writer) if args.writer is not None else 'clean style'}")
    print(f"  stroke error:  {tot['se'] / tot['sr']:.1%}")
    print(f"  outline error: {tot['oe'] / tot['or']:.1%}")
    print("  most common stroke misreadings (gold -> read; '-' = missing):")
    for (a, b), c in confusions.most_common(args.top):
        print(f"    {c:3d}  {a:4} -> {b}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="English to write (or Grascii with --grascii)")
    ap.add_argument("--file", help="read the text from a file; one paragraph per line")
    ap.add_argument("--grascii", action="store_true", help="the input is Grascii, one line per row")
    ap.add_argument("-o", "--out", default="gregg.png")
    ap.add_argument("--key", action="store_true", help="print the English under each line")
    ap.add_argument("--width", type=float, default=28.0, help="line width in units (~a short stroke)")
    ap.add_argument("--unit", type=float, help="pixels per unit (default 26: text about the size of a book's)")
    ap.add_argument("--writer", type=int, help="draw as a random synthetic writer with this seed")
    ap.add_argument("--tablet", action="store_true", help="white strokes on black, like a tablet screenshot")
    ap.add_argument("--check", help="a book in data/processed/topaz (e.g. fm2): Topaz reads drawn lines back")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", type=int, default=5, help="print this many examples")
    ap.add_argument("--save", help="with --check: save the shown examples' drawings here")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    if args.check:
        check(args)
    elif args.text or args.file:
        write(args)
    else:
        ap.error("give text, --file or --check")


if __name__ == "__main__":
    main()
