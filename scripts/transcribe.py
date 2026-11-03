"""Transcribe photos or scans of Gregg Anniversary shorthand into English.

    python scripts/transcribe.py page.jpg [more pages...]       # local beam decoder (default)
    python scripts/transcribe.py --decoder t5 page.jpg           # trained model, models/gregg-t5
    python scripts/transcribe.py --decoder claude page.jpg       # Claude (needs ANTHROPIC_API_KEY)
    python scripts/transcribe.py --decoder none page.jpg         # strokes + dictionary candidates only
    python scripts/transcribe.py -o out/ page.jpg                # also write <page>.txt files

Pipeline: Topaz finds and straightens the lines and reads each into Grascii; the decoder turns
the strokes into English, reading each line in the context of the ones before it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "topaz-61m"))


def make_decoder(name: str, topaz, device: str, t5_model: Path):
    """-> function(list of Grascii lines) -> English text, or None for --decoder none."""
    if name == "beam":
        from gregg_ocr.beam import BeamDecoder
        from gregg_ocr.lexicon import Lexicon
        from gregg_ocr.lm import TrigramLM

        beam = BeamDecoder(TrigramLM.load(ROOT / "models" / "trigram.pkl"), Lexicon.load(ROOT))
        return lambda lines: "\n".join(beam.decode(lines))
    if name == "t5":
        from gregg_ocr.t5 import T5Decoder

        t5 = T5Decoder(t5_model, device)
        return lambda lines: "\n".join(t5.decode(lines))
    if name == "claude":
        from gregg_ocr.decode import EnglishDecoder

        claude = EnglishDecoder(topaz.prompt, cache_path=ROOT / "runs" / "llm_cache.jsonl")

        def run(lines):
            refused_before = claude.refusals
            english = claude.decode("\n".join(topaz.lookup(line).intermediate() for line in lines))
            if claude.refusals > refused_before:
                return ("(Claude declined this transcript: a safety-classifier false positive. "
                        "Try --decoder beam.)")
            return english

        return run
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--decoder", choices=["beam", "t5", "claude", "none"], default="beam")
    ap.add_argument("--t5-model", type=Path, default=ROOT / "models" / "gregg-t5", help="model folder for --decoder t5")
    ap.add_argument("--lines", action="store_true", help="each image is a single line, not a page")
    ap.add_argument("-o", "--out", type=Path, help="write <image>.txt files here")
    args = ap.parse_args()

    from topaz import Topaz

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    topaz = Topaz(ROOT / "models" / "topaz-61m", device)
    decode = make_decoder(args.decoder, topaz, device, args.t5_model)

    for path in args.images:
        gray = np.array(Image.open(path).convert("L"))  # PIL reads webp, which cv2 may not
        t = topaz.transcribe(gray, lines=args.lines)
        parts = [f"# {path.name}: {len(t.lines)} lines", "", "## Strokes (Grascii)", t.grascii,
                 "", "## Outlines with dictionary candidates", t.intermediate()]
        if decode:
            parts += ["", f"## English ({args.decoder})", decode([line.grascii for line in t.lines])]
        text = "\n".join(parts)
        print(text + "\n")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{path.stem}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
