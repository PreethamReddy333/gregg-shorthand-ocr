"""Tune the LLM-guided decoder's weights on the dev book (the 1929 Manual), never on test sets.

    python scripts/tune_fused.py [--lines 150]

Lines are read in order within each paragraph; each is decoded with the earlier lines of its
paragraph (as read by the trigram decoder) as the LLM's context, as in eval_topaz.py --fused.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.beam import BeamDecoder  # noqa: E402
from gregg_ocr.fused import FusedDecoder, FusionConfig, LLMScorer, detokenize  # noqa: E402
from gregg_ocr.lexicon import Lexicon  # noqa: E402
from gregg_ocr.lm import TrigramLM  # noqa: E402
from tune_beam import edit, words  # noqa: E402

CONTEXT_LINES = 3
SETTINGS = [(0.3, 1.0), (0.0, 1.0), (0.6, 1.0), (0.3, 0.5), (0.3, 2.0)]  # (trigram_weight, llm_weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=150, help="dev lines to score (the first N, in order)")
    ap.add_argument("--llm", default=None, help="Hugging Face id of the causal LM (default SmolLM2-360M)")
    args = ap.parse_args()

    beam = BeamDecoder(TrigramLM.load(ROOT / "models" / "trigram.pkl"), Lexicon.load(ROOT))
    scorer = LLMScorer(args.llm) if args.llm else LLMScorer()
    df = pd.read_parquet(PROCESSED / "topaz" / "manual.parquet", columns=["page", "para", "ctc_pred", "english"])
    df = df[df.english.fillna("").str.strip() != ""].head(args.lines)

    # (line Grascii, English key, context text) with context from earlier lines in the paragraph
    items = []
    for _, g in df.groupby(["page", "para"], sort=False):
        read = beam.decode(list(g.ctc_pred))
        for k, (pred, english) in enumerate(zip(g.ctc_pred, g.english)):
            context = detokenize(" ".join(read[max(0, k - CONTEXT_LINES):k]).split())
            items.append((pred, english, context))
    ref = sum(len(words(e)) for _, e, _ in items)

    err = sum(edit(words(beam.decode([p])[0]), words(e)) for p, e, _ in items)
    print(f"{len(items)} dev lines; trigram beam alone: {err / ref:.1%}", flush=True)
    for tw, lw in SETTINGS:
        t = time.time()
        dec = FusedDecoder(beam, scorer, FusionConfig(trigram_weight=tw, llm_weight=lw))
        err = sum(edit(words(dec.decode([p], context=c)[0]), words(e)) for p, e, c in items)
        print(f"  trigram_weight={tw} llm_weight={lw}: {err / ref:.1%}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
