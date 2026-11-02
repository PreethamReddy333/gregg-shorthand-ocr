"""Print what we have: row counts, label vocabulary, image sizes, and a contact sheet.

    python scripts/inspect_data.py
Writes data/samples.png with a few examples from each source.
"""

import io
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gregg_ocr import DATA  # noqa: E402
from gregg_ocr.data import load_gregg1916, load_grascii, load_topaz  # noqa: E402


def open_image(row) -> Image.Image:
    if isinstance(row.get("png"), bytes):
        return Image.open(io.BytesIO(row["png"])).convert("L")
    return Image.open(row["path"]).convert("L")


def grascii_tokens(s: str) -> list[str]:
    return [t for word in s.split() for t in word.split("-") if t]


def describe(name: str, df: pd.DataFrame):
    print(f"\n== {name}: {len(df):,} rows")
    if df.empty:
        return
    print(f"   levels: {dict(df['level'].value_counts())}")
    print(f"   unique english labels: {df['english'].str.lower().nunique():,}")
    if df["grascii"].notna().any():
        toks = Counter(t for g in df["grascii"].dropna() for t in grascii_tokens(g))
        print(f"   grascii symbols: {len(toks)} distinct; most common {toks.most_common(12)}")
    sample = df.sample(min(300, len(df)), random_state=0)
    sizes = pd.DataFrame([open_image(r).size for r in sample.to_dict("records")], columns=["w", "h"])
    print(f"   image size (sampled): w median {sizes.w.median():.0f} [{sizes.w.min()}-{sizes.w.max()}], "
          f"h median {sizes.h.median():.0f} [{sizes.h.min()}-{sizes.h.max()}]")
    if "book" in df:
        print(f"   lines per book: {dict(df['book'].value_counts())}")
        print(f"   english words total: {df['english'].str.split().str.len().sum():,}")


def contact_sheet(sources: dict[str, pd.DataFrame], path: Path, per_source: int = 4):
    tiles = []
    for name, df in sources.items():
        if df.empty:
            continue
        for r in df.sample(min(per_source, len(df)), random_state=1).to_dict("records"):
            im = open_image(r)
            scale = 64 / im.height if r["level"] != "line" else 1
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            caption = f"[{name}] {r['english'][:70]}  |  {r['grascii'] or '-'}"[:140]
            tiles.append((im, caption))
    width = min(1400, max(im.width for im, _ in tiles) + 20)
    height = sum(im.height + 30 for im, _ in tiles) + 10
    sheet = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(sheet)
    y = 10
    for im, caption in tiles:
        draw.text((10, y), caption, fill=0)
        sheet.paste(im.crop((0, 0, min(im.width, width - 20), im.height)), (10, y + 16))
        y += im.height + 30
    sheet.save(path)
    print(f"\ncontact sheet -> {path}")


def main():
    sources = {"grascii": load_grascii(), "gregg1916": load_gregg1916(), "topaz": load_topaz()}
    for name, df in sources.items():
        describe(name, df)

    words = sources["grascii"].query("level == 'word'")
    a = set(words.query("edition == 'anniversary'")["english"].str.lower())
    p = set(words.query("edition == 'preanniversary'")["english"].str.lower())
    print(f"\nanniversary vs pre-anniversary word overlap: {len(a & p):,} shared, "
          f"{len(a - p):,} anniversary-only, {len(p - a):,} pre-anniversary-only")

    contact_sheet(sources, DATA / "samples.png")


if __name__ == "__main__":
    main()
