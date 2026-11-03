"""Random search over the renderer's geometry (gregg_ocr/render.py), scored by Topaz reading drawn
book lines (fm1, 100 lines) back: `tune_render.py 120`. The best values are copied into the tables."""
import random, sys, json, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
import gregg_ocr.render as R
from eval_topaz import edit, strokes
from topaz import Topaz

topaz = Topaz(ROOT / "models" / "topaz-61m", "mps")
df = pd.read_parquet(ROOT / "data/processed/topaz/fm1.parquet")
df = df[df.grascii.str.split().str.len() >= 3].sample(100, random_state=1)
BASE = (copy.deepcopy(R.CONSONANTS), copy.deepcopy(R.SMALL_CURVES), dict(R.CIRCLES), dict(R.HOOKS))

def apply(p):
    C, SC, CI, H = copy.deepcopy(BASE[0]), copy.deepcopy(BASE[1]), dict(BASE[2]), dict(BASE[3])
    for k in ("K", "G", "R", "L"):
        ch, t, l, sz = C[k]; C[k] = (ch, (1 if t > 0 else -1) * p["kr_turn"], l, sz)
    for k in ("P", "B", "F", "V"):
        ch, t, l, sz = C[k]; C[k] = (ch, (1 if t > 0 else -1) * p["pf_turn"], l, sz)
    for k, (ch, t, l, sz) in C.items():
        C[k] = (ch, t, l * (p["short"] if sz == "short" else p["long"]), sz)
    for k in ("T", "D", "TN", "DN", "TM", "DM", "NT", "ND", "MT", "MD", "TD", "DT", "DD"):
        ch, t, l, sz = C[k]; C[k] = (p["up"], t, l, sz)
    SC["S"] = SC["Z"] = (R.DOWN, p["s_turn"], p["s_len"])
    SC["TH"] = (p["up"], p["th_turn"], p["th_len"])
    for k in CI: CI[k] = p["a"] if k != "E" else p["e"]
    CI["I"] = p["a"] * p["i"]
    H["O"] = (225, p["hook_turn"], p["hook_len"]); H["U"] = (45, -p["hook_turn"], p["hook_len"])
    R.CONSONANTS, R.SMALL_CURVES, R.CIRCLES, R.HOOKS = C, SC, CI, H

def score(p):
    apply(p)
    se = sr = 0
    for g in df.grascii:
        img = np.array(R.render(R.layout(g)).convert("L"))
        pred = " ".join(topaz.read_line(c) for c in topaz.find_lines(img))
        a, b = strokes(g), strokes(pred)
        se += edit(a, b); sr += len(a)
    return se / sr

best = dict(kr_turn=80, pf_turn=60, short=1.0, long=1.0, up=50, s_turn=120, s_len=0.42, th_turn=90, th_len=0.75,
            a=0.45, e=0.2, i=1.22, hook_turn=170, hook_len=0.65)
RANGES = dict(kr_turn=(50, 110), pf_turn=(35, 90), short=(0.75, 1.3), long=(0.8, 1.25), up=(35, 65), s_turn=(70, 170),
              s_len=(0.25, 0.6), th_turn=(60, 140), th_len=(0.5, 1.0), a=(0.3, 0.65), e=(0.12, 0.3), i=(1.0, 1.4),
              hook_turn=(140, 230), hook_len=(0.4, 0.9))
best_s = score(best); print("start", round(best_s, 4), flush=True)
rng = random.Random(0)
for it in range(int(sys.argv[1])):
    cand = dict(best)
    for k in rng.sample(list(RANGES), 3):
        lo, hi = RANGES[k]
        cand[k] = min(hi, max(lo, cand[k] + rng.gauss(0, (hi - lo) * 0.15)))
    s = score(cand)
    if s < best_s:
        best, best_s = cand, s
        print(it, round(s, 4), json.dumps({k: round(v, 3) for k, v in best.items()}), flush=True)
print("best", round(best_s, 4), json.dumps({k: round(v, 3) for k, v in best.items()}))
