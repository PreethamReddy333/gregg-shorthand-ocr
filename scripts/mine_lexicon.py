"""Mine outlines the Topaz dictionary lacks from keyed book lines (gold Grascii + English).

    python scripts/mine_lexicon.py

Uses every converted Topaz book except the held-out ones (fm2, manual).
Writes models/mined_lexicon.json: {outline: [[words, count], ...]}.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.lexicon import load_topaz_dictionary, mine  # noqa: E402

HELD_OUT_BOOKS = {"fm2", "manual"}


def main():
    lines = []
    for f in sorted((PROCESSED / "topaz").glob("*.parquet")):
        if f.stem not in HELD_OUT_BOOKS:
            lines += list(pd.read_parquet(f, columns=["grascii", "english"]).dropna().itertuples(index=False))
    dictionary = load_topaz_dictionary(ROOT / "models" / "topaz-61m" / "dictionary.json")
    mined = mine(lines, dictionary)
    out = ROOT / "models" / "mined_lexicon.json"
    out.write_text(json.dumps(mined, indent=0))
    by_word = {w: o for o, ws in mined.items() for w, _ in ws}
    print(f"{len(lines):,} lines -> {sum(len(v) for v in mined.values()):,} new outline/word pairs -> {out}")
    for w in ["had", "as", "him", "us", "made", "may", "away", "years", "four", "boy", "used", "i will not"]:
        print(f"  {w!r}: {by_word.get(w)}")


if __name__ == "__main__":
    main()
