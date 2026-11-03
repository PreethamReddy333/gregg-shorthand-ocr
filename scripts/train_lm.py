"""Train the trigram language model on data/corpus (run scripts/build_corpus.py first).

    python scripts/train_lm.py

The Gregg book lines are counted 3x: they're the closest match to what the decoder reads.
Writes models/trigram.pkl.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gregg_ocr import DATA  # noqa: E402
from gregg_ocr.lm import TrigramLM  # noqa: E402


def read(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def main():
    t = time.time()
    lm = TrigramLM.train([read(DATA / "corpus" / "sentences.txt"), read(DATA / "corpus" / "gregg_lines.txt")],
                         weights=[1, 3])
    out = ROOT / "models" / "trigram.pkl"
    lm.save(out)
    print(f"{len(lm.c3):,} trigrams, vocab {lm.vocab_size:,}, {time.time() - t:.0f}s -> {out}")
    for s in ["i will not go", "i will knot go", "he can go there in an hour", "he can go their in an hour"]:
        toks = ["<s>", "<s>"] + s.split()
        print(f"  {sum(lm.logprob(*toks[i - 2:i + 1]) for i in range(2, len(toks))):7.2f}  {s}")


if __name__ == "__main__":
    main()
