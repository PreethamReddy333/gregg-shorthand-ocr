"""Tune the beam decoder's weights on a development book, never on the test sets.

    python scripts/tune_beam.py

Dev set: the 1929 Manual's keyed lines (held out of the LM corpus). Input is the Grascii
stored with each line (``ctc_pred``, read by an earlier Topaz checkpoint), so no image model
runs here. Lines are decoded a paragraph at a time, so context crosses line breaks.
Prints word error for each setting and the best one.
"""

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.beam import BeamConfig, BeamDecoder  # noqa: E402
from gregg_ocr.lm import TrigramLM  # noqa: E402
from gregg_ocr.text import PUNCT, tokenize  # noqa: E402
from gregg_ocr.lexicon import Lexicon  # noqa: E402


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def words(s: str) -> list[str]:
    return [t for t in tokenize(s) if t not in PUNCT]


def main():
    lm = TrigramLM.load(ROOT / "models" / "trigram.pkl")
    dictionary = Lexicon.load(ROOT)
    df = pd.read_parquet(PROCESSED / "topaz" / "manual.parquet", columns=["page", "para", "ctc_pred", "english"])
    df = df[df.english.fillna("").str.strip() != ""]
    paras = [g for _, g in df.groupby(["page", "para"], sort=False)]
    ref_total = sum(len(words(e)) for e in df.english)

    top1 = BeamDecoder(lm, dictionary, BeamConfig(lm_weight=0.0, beam=1))
    err = sum(edit(words(h), words(r)) for g in paras for h, r in zip(top1.decode(list(g.ctc_pred)), g.english))
    print(f"top-1 dictionary (no LM): {err / ref_total:.1%} over {len(df)} lines")

    results = []
    for lm_w, rank, fuzzy in itertools.product([0.6, 1.0, 1.5], [0.3, 0.8, 1.5], [1.0, 3.0]):
        dec = BeamDecoder(lm, dictionary, BeamConfig(lm_weight=lm_w, rank_cost=rank, fuzzy_cost=fuzzy))
        err = sum(edit(words(h), words(r)) for g in paras for h, r in zip(dec.decode(list(g.ctc_pred)), g.english))
        results.append((err / ref_total, lm_w, rank, fuzzy))
        print(f"  lm_weight={lm_w} rank_cost={rank} fuzzy_cost={fuzzy}: {err / ref_total:.1%}", flush=True)
    best = min(results)
    print(f"best: {best[0]:.1%} with lm_weight={best[1]} rank_cost={best[2]} fuzzy_cost={best[3]}")


if __name__ == "__main__":
    main()
