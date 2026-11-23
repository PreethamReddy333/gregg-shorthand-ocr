"""Fine-tune Topaz's line recognizer on the book lines with basic augmentation (stretch, slant,
stroke width, blur, contrast, noise), optionally with synthetic lines and grid distortion mixed in.

    finetune_topaz.py --arm control --steps 1500 --amp
    finetune_topaz.py --arm writers_sln --synth data/synth_lines/sln.parquet --grid 0.7 --steps 1500 --amp

Training starts from Topaz-61M's weights and uses all books except FM2 (the test book) and the
1929 Manual (the dev book), so the control arm separates the effect of fine-tuning itself from
the effect of synthetic data. Batch-norm statistics stay frozen (batches are small). The model is
saved as a full Topaz directory under models/topaz-ft/<arm>, so eval_topaz.py --topaz can score it. The last step is kept: no selection on any test set.
"""

import argparse
import io
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
from eval_topaz import edit, strokes  # noqa: E402
from gregg_ocr import PROCESSED  # noqa: E402
from gregg_ocr.augment import basic, grid_distort  # noqa: E402
from topaz.model import ModelConfig, Recognizer  # noqa: E402
from topaz.tokenizer import PUNCT, Tokenizer  # noqa: E402

BASE = ROOT / "models" / "topaz-61m"
HELD_OUT = {"fm2", "manual"}
MAX_WIDTH = 1600


def encode(grascii: str, stoi: dict) -> list[int] | None:
    """Grascii -> Topaz token ids (the inverse of Tokenizer.decode); None if a stroke is unknown."""
    ids = []
    try:
        for item in grascii.split():
            if ids:
                ids.append(stoi[" "])
            if item.isdigit():
                ids += [stoi[c] for c in item]
            elif item in PUNCT:
                ids.append(stoi[item])
            else:
                if item.startswith("="):
                    ids.append(stoi["="])
                    item = item[1:]
                for part in item.split("-"):
                    if part.endswith("_") and len(part) > 1:
                        ids += [stoi[part[:-1]], stoi["_"]]
                    else:
                        ids.append(stoi[part])
    except KeyError:
        return None
    return ids


def decode_png(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("L"))


def load_lines(frames: list[pd.DataFrame], tok: Tokenizer) -> list[tuple[bytes, list[int], str]]:
    stoi = {t: i for i, t in enumerate(tok.itos)}
    out, bad = [], 0
    for df in frames:
        for png, g, w in zip(df.png, df.grascii, df.width):
            ids = encode(g, stoi) if isinstance(g, str) else None
            # CTC needs at least one frame per token (plus blanks between repeats).
            if not ids or len(ids) * 2 > min(w, MAX_WIDTH) // 4:
                bad += 1
                continue
            out.append((png, ids, g))
    print(f"  {len(out)} lines ({bad} skipped: unknown stroke or too long for the width)", flush=True)
    return out


def batch(items, rng, grid_p=0.0, train=True):
    imgs = []
    for png, _, _ in items:
        im = decode_png(png)
        if im.shape[1] > MAX_WIDTH:
            im = np.array(Image.fromarray(im).resize((MAX_WIDTH, im.shape[0])))
        if train:
            im = basic(im, rng, MAX_WIDTH)
            if rng.random() < grid_p:
                im = grid_distort(im, rng)
        imgs.append(im)
    W = max(im.shape[1] for im in imgs)
    x = np.zeros((len(imgs), 1, 128, W), np.float32)
    for i, im in enumerate(imgs):
        x[i, 0, :, :im.shape[1]] = im / 255.0
    widths = torch.tensor([im.shape[1] for im in imgs])
    targets = [torch.tensor(ids) for _, ids, _ in items]
    return torch.from_numpy(x), widths, targets


@torch.no_grad()
def stroke_error(model, tok, items, device, bs=16) -> float:
    model.eval()
    err = ref = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x, w, _ = batch(chunk, None, train=False)
        for ids, (_, _, g) in zip(model.greedy(x.to(device), w.to(device)), chunk):
            gs = strokes(g)
            err += edit(gs, strokes(tok.decode(ids)))
            ref += len(gs)
    return err / max(ref, 1)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def save(model, out: Path, info: dict):
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    for f in ("config.json", "segmenter.safetensors", "vocab.json", "dictionary.json", "prompt.txt"):
        if not (out / f).exists():
            shutil.copy(BASE / f, out / f)
    (out / "finetune.json").write_text(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--synth", type=Path, help="synthetic lines (make_synth_lines.py)")
    ap.add_argument("--synth-share", type=float, default=0.5, help="share of each batch that is synthetic")
    ap.add_argument("--grid", type=float, default=0.0, help="probability of grid distortion per line")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the forward pass")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg = json.loads((BASE / "config.json").read_text())
    tok = Tokenizer(BASE / "vocab.json")
    model = Recognizer(ModelConfig.from_dict(cfg["recognizer"]))
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(BASE / "model.safetensors")))
    model.to(device)

    print("book lines:", flush=True)
    books = [p for p in sorted((PROCESSED / "topaz").glob("*.parquet")) if p.stem not in HELD_OUT]
    real = load_lines([pd.read_parquet(p, columns=["png", "grascii", "width"]) for p in books], tok)
    synth = []
    if args.synth:
        print("synthetic lines:", flush=True)
        synth = load_lines([pd.read_parquet(args.synth, columns=["png", "grascii", "width"])], tok)
    print("dev (1929 Manual):", flush=True)
    dev = load_lines([pd.read_parquet(PROCESSED / "topaz" / "manual.parquet", columns=["png", "grascii", "width"])], tok)
    dev = random.Random(1).sample(dev, min(300, len(dev)))

    log = [dict(step=0, dev_ter=stroke_error(model, tok, dev, device))]
    print(f"step 0 dev stroke error {log[0]['dev_ter']:.2%}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    n_synth = round(args.batch * args.synth_share) if synth else 0
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        model.train()
        freeze_bn(model)
        items = rng.sample(real, args.batch - n_synth) + (rng.sample(synth, n_synth) if n_synth else [])
        x, w, targets = batch(items, rng, args.grid)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=args.amp):
            logits, lengths = model(x.to(device), w.to(device))
        logp = logits.float().log_softmax(-1).transpose(0, 1)  # T, B, V
        loss = F.ctc_loss(logp, torch.cat(targets).to(device), lengths, torch.tensor([len(t) for t in targets]),
                          blank=0, zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if device == "mps" and step % 50 == 0:
            torch.mps.empty_cache()
        if step % 100 == 0:
            print(f"step {step} loss {np.mean(losses[-100:]):.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ter = stroke_error(model, tok, dev, device)
            log.append(dict(step=step, dev_ter=ter, loss=float(np.mean(losses[-100:]))))
            print(f"step {step} dev stroke error {ter:.2%}", flush=True)
    out = ROOT / "models" / "topaz-ft" / args.arm
    save(model, out, dict(args={k: str(v) for k, v in vars(args).items()}, log=log,
                          real_lines=len(real), synth_lines=len(synth)))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
