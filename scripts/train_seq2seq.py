"""Fine-tune flan-t5-small to turn Grascii (plus the previous line as context) into English.

    python scripts/train_seq2seq.py [--steps 15000] [--batch 32]

Reads data/synth/{train,dev}.jsonl (scripts/make_synthetic.py). Every --eval-every steps it
decodes the dev set (the held-out 1929 Manual) and keeps the checkpoint with the lowest word
error in models/gregg-t5/. Runs on the Apple GPU (MPS) when available.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gregg_ocr import DATA  # noqa: E402
from gregg_ocr.text import PUNCT  # noqa: E402

BASE = "google/flan-t5-small"
MAX_SOURCE, MAX_TARGET = 192, 64


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def words(s: str) -> list[str]:
    return [t for t in s.split() if t not in PUNCT]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@torch.no_grad()
def dev_wer(model, tok, dev, device, batch=64) -> float:
    model.eval()
    err = ref = 0
    for i in range(0, len(dev), batch):
        chunk = dev[i:i + batch]
        enc = tok([d["source"] for d in chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_SOURCE).to(device)
        out = model.generate(**enc, max_new_tokens=MAX_TARGET, num_beams=1)
        for d, hyp in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
            err += edit(words(hyp), words(d["target"]))
            ref += len(words(d["target"]))
    model.train()
    return err / max(ref, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = ROOT / "models" / "gregg-t5"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE).to(device)
    train, dev = read(DATA / "synth" / "train.jsonl"), read(DATA / "synth" / "dev.jsonl")
    print(f"{len(train):,} train, {len(dev)} dev, device {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = 500

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(args.steps - warmup, 1)))

    rng = random.Random(0)
    order, pos = list(range(len(train))), len(train)
    best, t0, running = float("inf"), time.time(), 0.0
    print(f"step 0 dev WER {dev_wer(model, tok, dev, device):.1%}", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            rng.shuffle(order)
            pos = 0
        batch = [train[k] for k in order[pos:pos + args.batch]]
        pos += args.batch
        enc = tok([b["source"] for b in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_SOURCE).to(device)
        labels = tok([b["target"] for b in batch], return_tensors="pt", padding=True, truncation=True,
                     max_length=MAX_TARGET).input_ids.to(device)
        labels[labels == tok.pad_token_id] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 200 == 0:
            print(f"step {step} loss {running:.3f} lr {lr_at(step):.2e} {(time.time() - t0) / step:.2f}s/step",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            wer = dev_wer(model, tok, dev, device)
            note = ""
            if wer < best:
                best = wer
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
                note = " (saved)"
            print(f"step {step} dev WER {wer:.1%}{note}", flush=True)
    print(f"best dev WER {best:.1%} -> {out_dir}")


if __name__ == "__main__":
    main()
