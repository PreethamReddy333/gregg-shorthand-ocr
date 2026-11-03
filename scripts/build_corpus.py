"""Build the English text corpus the local decoders learn from.

    python scripts/build_corpus.py

Sources (all public domain):
  - Topaz's keyed Gregg books (the English of each plate line), the closest match in style.
    Held out: fm2 (the test book) and manual (the 1929 Manual, which the sample pages in
    data/inputs/ come from).
  - The literature texts that came with the Topaz data, plus a list of Project Gutenberg books.
    The Bible is deliberately absent so the Genesis test set stays honest.

Writes data/corpus/sentences.txt (one sentence per line, tokenized: lower-case words and
. , ? ; : ! as separate tokens) and data/corpus/gregg_lines.txt (the Topaz book lines, kept
separate so they can be weighted up).
"""

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gregg_ocr import DATA, PROCESSED, RAW  # noqa: E402
from gregg_ocr.text import tokenize  # noqa: E402

HELD_OUT_BOOKS = {"fm2", "manual"}
GUTENBERG_IDS = [
    20203, 1342, 98, 1661, 76, 74, 345, 84, 2944, 219, 1260, 768, 2591, 1232, 2814,
    158, 120, 35, 36, 1184, 205, 64317, 5200, 2600, 1400, 730, 766, 580, 3176, 4300,
]
OUT = DATA / "corpus"


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    out = []
    for s in re.split(r"(?<=[.?!])\s+", text):
        toks = tokenize(s)
        if 3 <= len(toks) <= 80:
            out.append(" ".join(toks))
    return out


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def gutenberg_texts() -> list[str]:
    cache = RAW / "gutenberg"
    cache.mkdir(parents=True, exist_ok=True)
    texts = []
    for gid in GUTENBERG_IDS:
        path = cache / f"pg{gid}.txt"
        if not path.exists():
            try:
                urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", path)
            except Exception as e:  # a missing book only shrinks the corpus
                print(f"  skipped Gutenberg {gid}: {e}")
                continue
        texts.append(strip_gutenberg(path.read_text(encoding="utf-8", errors="ignore")))
    bundled = RAW / "topaz" / "gregg-anniversary" / "processed" / "gutenberg"
    texts += [strip_gutenberg(p.read_text(encoding="utf-8", errors="ignore")) for p in sorted(bundled.glob("*.txt"))]
    return texts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gregg = []
    for f in sorted((PROCESSED / "topaz").glob("*.parquet")):
        if f.stem in HELD_OUT_BOOKS:
            continue
        for line in pd.read_parquet(f, columns=["english"]).english.dropna():
            toks = tokenize(line)
            if toks:
                gregg.append(" ".join(toks))
    (OUT / "gregg_lines.txt").write_text("\n".join(gregg) + "\n")

    sents = [s for t in gutenberg_texts() for s in sentences(t)]
    (OUT / "sentences.txt").write_text("\n".join(sents) + "\n")
    n_words = sum(len(s.split()) for s in sents)
    print(f"gregg book lines: {len(gregg):,} ({sum(len(s.split()) for s in gregg):,} tokens)")
    print(f"literature sentences: {len(sents):,} ({n_words:,} tokens)")


if __name__ == "__main__":
    main()
