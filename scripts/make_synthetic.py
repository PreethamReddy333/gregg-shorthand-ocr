"""Generate Grascii -> English training pairs for the seq2seq model.

    python scripts/make_synthetic.py [--synthetic 400000]

Each example is one line of shorthand plus up to two lines before it as context:
    source: "context: <line before last> / <previous line> line: <this line's Grascii>"
    target: this line's English (lower-case tokens, punctuation kept)

Sources:
  real      Topaz book lines (not fm2/manual) as Topaz's earlier checkpoint read them (real noise),
            and the same lines with their correct Grascii. Counted twice.
  synthetic Literature from data/corpus, plus modern encyclopedia text (WikiText-103, CC BY-SA),
            cut into plate-length lines and written as Grascii with the lexicon (unknown
            words, mostly names, spelled from their pronunciation), then corrupted with the
            noise model fitted on the real lines, sometimes with handwriting-style confusions.
  dev       The 1929 Manual (held out everywhere else), as Topaz read it.

Writes data/synth/{train,dev}.jsonl.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))
from gregg_ocr import DATA, PROCESSED  # noqa: E402
from gregg_ocr.lexicon import Lexicon  # noqa: E402
from gregg_ocr.synth import NoiseModel, PhoneticSpeller, Renderer, model_source  # noqa: E402
from gregg_ocr.text import tokenize  # noqa: E402

HELD_OUT_BOOKS = {"fm2", "manual"}
CONTEXT_DROP = 0.2  # share of examples shown without context, so the model also works alone
CONTEXT_LINES = 2
HAND_SHARE = 0.4  # share of synthetic lines given handwriting-style stroke confusions
WIKI_SHARE = 0.4  # share of synthetic lines from WikiText rather than the literature corpus


def example(context: list[str], line: str, english: str) -> dict:
    return {"source": model_source(context, line, CONTEXT_LINES), "target": " ".join(tokenize(english))}


def wiki_tokens(path: Path, limit: int) -> list[str]:
    """Article text from WikiText-103 as our tokens, headings and markup removed."""
    out = []
    for text in pd.read_parquet(path).text:
        text = text.strip()
        if not text or text.startswith("="):
            continue
        text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace("<unk>", "")
        out += tokenize(text)
        if len(out) >= limit:
            break
    return out


def book_lines(books) -> pd.DataFrame:
    frames = []
    for f in sorted((PROCESSED / "topaz").glob("*.parquet")):
        if f.stem in books:
            df = pd.read_parquet(f, columns=["page", "para", "grascii", "ctc_pred", "english"])
            df = df[df.english.fillna("").str.strip() != ""].dropna()
            for col in ("grascii", "ctc_pred"):
                for k in (1, 2):
                    df[f"prev{k}_{col}"] = df.groupby(["page", "para"], sort=False)[col].shift(k).fillna("")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=400_000, help="synthetic lines to generate")
    args = ap.parse_args()
    rng = random.Random(0)
    out = DATA / "synth"
    out.mkdir(parents=True, exist_ok=True)

    all_books = {f.stem for f in (PROCESSED / "topaz").glob("*.parquet")}
    real = book_lines(all_books - HELD_OUT_BOOKS)
    noise = NoiseModel().fit(list(zip(real.grascii, real.ctc_pred)))
    print("noise model:", noise.summary())

    train = []
    for r in real.itertuples():
        for _ in range(2):
            drop = rng.random() < CONTEXT_DROP
            train.append(example([] if drop else [r.prev2_ctc_pred, r.prev1_ctc_pred], r.ctc_pred, r.english))
            train.append(example([] if drop else [r.prev2_grascii, r.prev1_grascii], r.grascii, r.english))
        if rng.random() < HAND_SHARE:  # the correct strokes, blurred the way handwriting is misread
            train.append(example([r.prev2_grascii, r.prev1_grascii],
                                 noise.apply(r.grascii, rng, 1.0, hand=rng.choice([0.03, 0.06, 0.1])), r.english))
    n_real = len(train)

    renderer = Renderer(Lexicon.load(ROOT).word_to_outline(), rng=rng, speller=PhoneticSpeller())
    sources = {"literature": (DATA / "corpus" / "sentences.txt").read_text().split(),
               "wikitext": wiki_tokens(DATA / "raw" / "wikitext" / "train0.parquet", 6_000_000)}
    targets = {"literature": round(args.synthetic * (1 - WIKI_SHARE)), "wikitext": round(args.synthetic * WIKI_SHARE)}
    for name, tokens in sources.items():
        i, history, made, failed = 0, [], 0, 0
        while made < targets[name] and i < len(tokens):
            n = rng.randint(5, 12)
            chunk = tokens[i:i + n]
            i += n
            clean = renderer.render(chunk)
            if clean is None:
                failed += 1
                history = []  # context must be the lines actually before this one
                continue
            # The fitted noise comes from a model reading its own training books (~4% per stroke);
            # Topaz on unseen print errs on ~10% of strokes and more on handwriting, so mostly scale up.
            scale = rng.choices([0.0, 1.0, 2.5, 4.0], weights=[0.15, 0.25, 0.35, 0.25])[0]
            hand = rng.choice([0.03, 0.06, 0.1]) if rng.random() < HAND_SHARE else 0.0
            noisy = noise.apply(clean, rng, scale, hand=hand) if scale or hand else clean
            if rng.random() < 0.3:  # Topaz sometimes misses every punctuation mark on a page
                noisy = " ".join(t for t in noisy.split() if t not in {".", ",", "?", ";", ":", "!"})
            context = [] if rng.random() < CONTEXT_DROP else history
            train.append(example(context, noisy, " ".join(chunk)))
            history = (history + [noisy])[-CONTEXT_LINES:]
            made += 1
        print(f"{name}: {made:,} synthetic lines ({failed:,} chunks skipped: unwritable words)")
    print(f"real examples: {n_real:,}; words spelled from pronunciation: {renderer.spelled:,}")

    rng.shuffle(train)
    with open(out / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    dev = book_lines({"manual"})
    with open(out / "dev.jsonl", "w") as f:
        for r in dev.itertuples():
            f.write(json.dumps(example([r.prev2_ctc_pred, r.prev1_ctc_pred], r.ctc_pred, r.english)) + "\n")
    print(f"train {len(train):,} -> {out / 'train.jsonl'}; dev {len(dev):,} -> {out / 'dev.jsonl'}")
    for ex in train[:3]:
        print("  ", ex)


if __name__ == "__main__":
    main()
