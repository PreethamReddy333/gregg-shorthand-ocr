"""Smoke-test the full word pipeline with the off-the-shelf grascii/gregg-vision model.

    python scripts/baseline_gregg_vision.py --n 200

image --(gregg-vision)--> Grascii --(Grascii dictionary search)--> English candidates

gregg-vision was trained on the pre-anniversary word images, so its score on that
set is a training-set number. The anniversary set is a different edition, so read
that number as "how far off-the-shelf is for the edition we care about".
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from grascii.searchers import GrasciiSearcher
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gregg_ocr.data import load_grascii  # noqa: E402

MODEL_ID = "grascii/gregg-vision-v0.2.1"
DICTIONARY = {"anniversary": ":anniversary", "preanniversary": ":preanniversary"}


def norm(g: str) -> str:
    return g.replace("-", "").replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="words sampled per edition")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    words = load_grascii().query("level == 'word'")
    for edition, group in words.groupby("edition"):
        sample = group.sample(min(args.n, len(group)), random_state=0)
        searcher = GrasciiSearcher(dictionaries=[DICTIONARY[edition]])
        stroke_hits = english_hits = 0
        examples = []
        for row in sample.itertuples():
            img = np.array([Image.open(io.BytesIO(row.png)).convert("L")])
            pixels = processor(img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixels, max_new_tokens=12)[0]
            pred = tokenizer.decode(ids, skip_special_tokens=True)

            results = searcher.sorted_search(grascii=norm(pred)) or []
            candidates = [r.entry.translation for r in results[: args.topk]]
            stroke_hits += norm(pred) == norm(row.grascii)
            english_hits += row.english.lower() in (c.lower() for c in candidates)
            if len(examples) < 6:
                examples.append((row.english, row.grascii, pred, candidates))

        n = len(sample)
        print(f"\n== {edition} words (n={n})")
        print(f"   exact Grascii match:          {stroke_hits / n:.1%}")
        print(f"   correct word in top-{args.topk} lookup: {english_hits / n:.1%}")
        for english, gold, pred, cands in examples:
            print(f"   {english!r:18} gold={gold:14} pred={pred:14} -> {cands}")


if __name__ == "__main__":
    main()
