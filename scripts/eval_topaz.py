"""Score Topaz on line images that have a known answer.

    python scripts/eval_topaz.py --book fm2 --n 300        # held-out printed book (Topaz never trained on it)
    python scripts/eval_topaz.py --genesis                 # handwritten Genesis, scored per verse
    python scripts/eval_topaz.py --folder data/handwritten # your own handwritten lines

--genesis reads data/handwritten/genesis (scripts/build_genesis_testset.py). Each verse's lines
are read one at a time and joined, then compared with the KJV verse. Verses cut off at the end
of a document are skipped.

Folder format (one line of shorthand per image):
    name.png        the line image (dark ink on light paper)
    name.txt        what it says in English
    name.grascii    optional: the strokes in Grascii, enables stroke error

Metrics:
    stroke error   edit distance over stroke tokens / reference length (Topaz's own metric)
    outline error  same, over whole outlines
    word error     English WER when each outline is read as its top dictionary suggestion.
                   This is a floor: the LLM step should do much better by using context.
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))
from gregg_ocr import DATA, PROCESSED  # noqa: E402


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def outlines(g: str) -> list[str]:
    return [o.lstrip("=") for o in g.split() if any(c.isalpha() for c in o)]


def strokes(g: str) -> list[str]:
    return [s for o in outlines(g) for s in o.split("-")]


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


CONTEXT_LINES = 3  # how many earlier lines of the same paragraph context-using decoders see


def load_book(book: str, n: int) -> list[dict]:
    df = pd.read_parquet(PROCESSED / "topaz" / f"{book}.parquet")
    # Earlier lines of the same paragraph, oldest first, for decoders that use context.
    groups = df.groupby(["page", "para"], sort=False).png
    df["context_pngs"] = [list(x) for x in zip(*(groups.shift(k) for k in range(CONTEXT_LINES, 0, -1)))]
    df = df.sample(min(n, len(df)), random_state=0)
    df = df[df.english.fillna("").str.strip() != ""]  # a few crops have no aligned key text

    # Topaz's training crops are stored ink-high (white on black); read_line expects dark ink.
    def img(png):
        return 255 - np.array(Image.open(io.BytesIO(png)).convert("L"))

    return [{"id": f"{book}:{r.page}:{r.para}:{i}", "images": [img(r.png)],
             "context": [img(p) for p in r.context_pngs if isinstance(p, bytes)],
             "english": r.english, "grascii": r.grascii}
            for i, r in enumerate(df.itertuples())]


def load_genesis() -> list[dict]:
    root = DATA / "handwritten" / "genesis"
    verses = pd.read_csv(root / "verses.csv")
    # Context for a verse: the last few lines before it in the same document.
    lines_before, context = {}, []
    for v in verses.itertuples():
        seen = lines_before.setdefault(v.doc, [])
        context.append(seen[-CONTEXT_LINES:])
        seen += v.lines.split()
    verses["context_lines"] = context

    def img(line_id):
        return np.array(Image.open(root / "lines" / f"{line_id}.png").convert("L"))

    return [{"id": f"{v.doc}:{v.verse}", "images": [img(lid) for lid in v.lines.split()],
             "context": [img(lid) for lid in v.context_lines],
             "english": v.kjv, "grascii": None}
            for v in verses.query("complete").itertuples()]


def load_folder(folder: Path) -> list[dict]:
    items = []
    for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
        key, gk = img.with_suffix(".txt"), img.with_suffix(".grascii")
        if not key.exists():
            print(f"skipping {img.name}: no {key.name}")
            continue
        gray = np.array(Image.open(img).convert("L"))
        # Anything taller than a few lines of writing is a page: main() splits it into lines.
        items.append({"id": img.name, "images": [gray], "page": gray.shape[0] > 300,
                      "english": key.read_text().strip(),
                      "grascii": gk.read_text().strip() if gk.exists() else None})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--book", help="a converted Topaz book in data/processed/topaz, e.g. fm2")
    src.add_argument("--genesis", action="store_true", help="handwritten Genesis test set, scored per verse")
    src.add_argument("--folder", type=Path, help="folder of line images with .txt keys")
    ap.add_argument("--n", type=int, default=300, help="lines sampled from --book")
    ap.add_argument("--limit", type=int, help="score only the first N items (e.g. a cheap smoke test)")
    ap.add_argument("--llm", action="store_true",
                    help="also turn Topaz's output into English with Claude (needs Anthropic credentials)")
    ap.add_argument("--beam", action="store_true",
                    help="also decode locally with the trigram beam decoder (alone and with earlier lines)")
    ap.add_argument("--t5", action="store_true",
                    help="also decode with the trained model in models/gregg-t5 (alone and with the previous line)")
    ap.add_argument("--fused", action="store_true",
                    help="also decode with the beam search guided by a local LLM, given the earlier lines")
    ap.add_argument("--out", type=Path, help="write per-item results to this CSV")
    args = ap.parse_args()

    from topaz import Topaz  # third-party model code, imported only when actually scoring

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(ROOT / "models" / "topaz-61m", device)
    if args.book:
        items, source = load_book(args.book, args.n), args.book
    elif args.genesis:
        items, source = load_genesis(), "handwritten Genesis"
    else:
        items, source = load_folder(args.folder), args.folder
    items = items[: args.limit]
    for it in items:
        if it.get("page"):
            it["images"] = topaz.find_lines(it["images"][0])

    totals = dict(stroke_err=0, stroke_ref=0, outline_err=0, outline_ref=0, word_err=0, word_ref=0,
                  llm_err=0)
    local = {}  # local decoder name -> total word errors
    if args.beam or args.fused:
        from gregg_ocr.beam import BeamDecoder
        from gregg_ocr.lexicon import Lexicon
        from gregg_ocr.lm import TrigramLM

        beam = BeamDecoder(TrigramLM.load(ROOT / "models" / "trigram.pkl"), Lexicon.load(ROOT))
    if args.beam:
        local.update({"beam": 0, "beam_context": 0})
    if args.fused:
        from gregg_ocr.fused import FusedDecoder, LLMScorer, detokenize

        fused = FusedDecoder(beam, LLMScorer(device=device))
        local["fused_context"] = 0
    if args.t5:
        from gregg_ocr.t5 import T5Decoder

        t5 = T5Decoder(ROOT / "models" / "gregg-t5", device)
        local.update({"t5": 0, "t5_context": 0})
    rows = []
    for n, it in enumerate(items):
        if n and n % 20 == 0:
            print(f"  {n}/{len(items)}", file=sys.stderr, flush=True)
        line_preds = [topaz.read_line(img) for img in it["images"]]
        context_preds = [topaz.read_line(img) for img in it.get("context", [])]
        local_out = {}
        if args.beam:
            local_out["beam"] = " ".join(beam.decode(line_preds))
            # Decode the context line too, then keep only this item's lines.
            local_out["beam_context"] = " ".join(beam.decode(context_preds + line_preds)[len(context_preds):])
        if args.t5:
            local_out["t5"] = " ".join(t5.decode(line_preds, use_context=False))
            # The model was trained with one previous line as context.
            local_out["t5_context"] = " ".join(t5.decode(line_preds, context=context_preds[-1] if context_preds else ""))
        if args.fused:
            # The LLM conditions on the earlier lines' English, read cheaply by the trigram decoder.
            context_text = detokenize(" ".join(beam.decode(context_preds)).split()) if context_preds else ""
            local_out["fused_context"] = " ".join(fused.decode(line_preds, context=context_text))
        pred = " ".join(line_preds)
        line = topaz.lookup(pred)
        top1 = " ".join(o["words"][0] if o.get("words") else "" for o in line.outlines)
        we, wr = edit(words(top1), words(it["english"])), len(words(it["english"]))
        totals["word_err"] += we
        totals["word_ref"] += wr
        row = {"id": it["id"], "english": it["english"], "pred_grascii": pred, "top1_english": top1,
               "word_error": we / max(wr, 1),
               # What the LLM sees: each line's outlines with their dictionary candidates.
               "intermediate": "\n".join(topaz.lookup(p).intermediate() for p in line_preds)}
        if it["grascii"]:
            se, sr = edit(strokes(pred), strokes(it["grascii"])), len(strokes(it["grascii"]))
            oe, orf = edit(outlines(pred), outlines(it["grascii"])), len(outlines(it["grascii"]))
            totals["stroke_err"] += se
            totals["stroke_ref"] += sr
            totals["outline_err"] += oe
            totals["outline_ref"] += orf
            row.update(gold_grascii=it["grascii"], stroke_error=se / max(sr, 1))
        for name, english in local_out.items():
            e = edit(words(english), words(it["english"]))
            local[name] += e
            row.update({f"{name}_english": english, f"{name}_word_error": e / max(wr, 1)})
        rows.append(row)

    if args.llm:
        from concurrent.futures import ThreadPoolExecutor

        from gregg_ocr.decode import EnglishDecoder

        decoder = EnglishDecoder((ROOT / "models" / "topaz-61m" / "prompt.txt").read_text().rstrip(),
                                 cache_path=ROOT / "runs" / "llm_cache.jsonl")
        import anthropic

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                decoded = list(pool.map(decoder.decode, [r["intermediate"] for r in rows]))
        except anthropic.AuthenticationError as e:
            sys.exit(f"Anthropic API rejected the credentials ({e.message}). "
                     "Set ANTHROPIC_API_KEY to an API key from console.anthropic.com and re-run.")
        answered = dict(llm_err=0, top1_err=0, ref=0, n=0)  # the same items, minus declined ones
        for r, english in zip(rows, decoded):
            ref = words(r["english"])
            e = edit(words(english), ref)
            totals["llm_err"] += e
            r.update(llm_english=english, llm_word_error=e / max(len(ref), 1), llm_declined=not english)
            if english:
                answered["llm_err"] += e
                answered["top1_err"] += edit(words(r["top1_english"]), ref)
                answered["ref"] += len(ref)
                answered["n"] += 1

    print(f"\n{len(rows)} {'verses' if args.genesis else 'lines'} from {source}")
    if totals["stroke_ref"]:
        print(f"  stroke error:  {totals['stroke_err'] / totals['stroke_ref']:.1%}")
        print(f"  outline error: {totals['outline_err'] / totals['outline_ref']:.1%}")
    print(f"  word error (top-1 dictionary, no LLM): {totals['word_err'] / max(totals['word_ref'], 1):.1%}")
    for name, err in local.items():
        print(f"  word error ({name}):{' ' * (26 - len(name))}{err / max(totals['word_ref'], 1):.1%}")
    if args.llm:
        print(f"  word error (Claude decoding):          {totals['llm_err'] / max(totals['word_ref'], 1):.1%}"
              f"  (declined items count as fully wrong)")
        if answered["n"] < len(rows):
            print(f"  {len(rows) - answered['n']} declined by the safety classifier; on the other {answered['n']}:"
                  f" top-1 {answered['top1_err'] / max(answered['ref'], 1):.1%},"
                  f" Claude {answered['llm_err'] / max(answered['ref'], 1):.1%}")
    for r in rows[:4]:
        print(f"\n  key:   {r['english']}\n  pred:  {r['pred_grascii']}\n  top-1: {r['top1_english']}")
        if args.llm:
            print(f"  llm:   {r['llm_english']}")
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nper-line results -> {args.out}")


if __name__ == "__main__":
    main()
