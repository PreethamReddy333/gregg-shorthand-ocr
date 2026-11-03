"""How stable are the DSR features? Line-level feature summaries under image perturbations
and under changes to the modelling assumptions, on printed (FM2) and handwritten (Genesis) lines.

    dsr_stability.py --n 120 --out runs/dsr

For each line, each feature is summarized as its length-weighted mean over the line's segments.
Stability of a feature under a change = Spearman correlation, across lines, between its value on
the original and on the changed line (1 = ranks lines the same way). A feature that is unstable
under small image noise can't help the recognizer; one that is unstable under a change of
*assumption* (timing model, skeletonizer) measures the assumption as much as the ink.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gregg_ocr import dynamical_stroke_representation as dsr  # noqa: E402


def load_lines(n: int, seed: int = 0) -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    fm2 = pd.read_parquet(ROOT / "data" / "processed" / "topaz" / "fm2.parquet")
    fm2 = [np.array(Image.open(io.BytesIO(b)).convert("L")) for b in fm2.png.iloc[rng.permutation(len(fm2))[:n]]]
    gen = sorted((ROOT / "data" / "handwritten" / "genesis" / "lines").glob("*.png"))
    gen = [np.array(Image.open(gen[i]).convert("L")) for i in rng.permutation(len(gen))[:n]]
    return {"fm2 (printed)": fm2, "genesis (handwritten)": gen}


def noise(gray, rng):
    g = gray.astype(float) + rng.normal(0, 12, gray.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def rotate(gray, deg):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), borderValue=int(np.median(gray)), flags=cv2.INTER_LINEAR)  # paper colour


def scale(gray, f):
    h, w = gray.shape
    return cv2.resize(gray, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def variants(base: dsr.DSRConfig):
    """name -> (image transform, config). Image changes first, then changes of assumption."""
    def cfg(**kw):
        c = dsr.DSRConfig(**{**base.__dict__, **kw})
        return c
    ex = base.extract
    return {
        "noise σ=12": (lambda g, rng: noise(g, rng), base),
        "rotate +3°": (lambda g, rng: rotate(g, 3), base),
        "rotate -3°": (lambda g, rng: rotate(g, -3), base),
        "scale 0.8": (lambda g, rng: scale(g, 0.8), base),
        "scale 1.25": (lambda g, rng: scale(g, 1.25), base),
        "skeleton: lee": (None, cfg(extract=dsr.ExtractConfig(**{**ex.__dict__, "skeleton": "lee"}))),
        "skeleton: medial axis": (None, cfg(extract=dsr.ExtractConfig(**{**ex.__dict__, "skeleton": "medial_axis"}))),
        "smoothing: spline": (None, cfg(smoothing="spline")),
        "timing: uniform": (None, cfg(timing="uniform")),
        "timing: min jerk": (None, cfg(timing="min_jerk")),
    }


def summarize(res: dsr.DSRResult) -> np.ndarray:
    """Length-weighted mean of each feature over the line's segments (dots weigh one pen width)."""
    if not len(res.features):
        return np.full(len(res.names), np.nan)
    w = np.array([max(m["n_points"], 3) for m in res.meta], float)
    return (res.features * w[:, None]).sum(0) / w.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="lines per set")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "dsr")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = dsr.DSRConfig()
    sets = load_lines(args.n)
    report = {"config": base.to_dict(), "n_lines": args.n, "sets": {}}
    for set_name, lines in sets.items():
        rng = np.random.default_rng(1)
        t0 = time.perf_counter()
        base_res = [dsr.extract_features(g, base) for g in lines]
        per_line = (time.perf_counter() - t0) / len(lines)
        names, groups = base_res[0].names, base_res[0].groups
        B = np.array([summarize(r) for r in base_res])
        segs = [len(r.features) for r in base_res]
        rows = {}
        for vname, (transform, cfg) in variants(base).items():
            imgs = [transform(g, rng) for g in lines] if transform else lines
            V = np.array([summarize(dsr.extract_features(g, cfg)) for g in imgs])
            rho = []
            for j in range(len(names)):
                ok = np.isfinite(B[:, j]) & np.isfinite(V[:, j])
                if ok.sum() < 10 or np.std(B[ok, j]) == 0 or np.std(V[ok, j]) == 0:
                    rho.append(np.nan)
                else:
                    rho.append(stats.spearmanr(B[ok, j], V[ok, j]).statistic)
            rows[vname] = rho
            print(f"{set_name:22} {vname:24} " + "  ".join(
                f"{g}={np.nanmedian([rho[names.index(n)] for n in groups[g]]):.2f}" for g in groups), flush=True)
        # Redundancy: features nearly duplicating another (|ρ| > 0.95 across lines).
        C = stats.spearmanr(B, nan_policy="omit").statistic
        dup = sorted({tuple(sorted((names[i], names[j]))) for i in range(len(names)) for j in range(len(names))
                      if i < j and abs(C[i, j]) > 0.95})
        report["sets"][set_name] = dict(seconds_per_line=per_line, segments_per_line=float(np.mean(segs)),
                                        stability={v: dict(zip(names, map(float, r))) for v, r in rows.items()},
                                        groups=groups, near_duplicates=dup)
        print(f"{set_name}: {per_line * 1000:.0f} ms/line, {np.mean(segs):.1f} segments/line, "
              f"{len(dup)} near-duplicate feature pairs", flush=True)
        fig = dsr.plot_extraction(base_res[0])
        fig.savefig(args.out / f"extraction_{set_name.split()[0]}.png", dpi=110)
    with open(args.out / "stability.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(args.out / "stability.json")


if __name__ == "__main__":
    main()
