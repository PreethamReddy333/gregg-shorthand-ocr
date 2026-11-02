"""Transcribe photos or scans of Gregg Anniversary shorthand into English.

    python scripts/transcribe.py page.jpg [more pages...]
    python scripts/transcribe.py --no-llm page.jpg     # strokes + dictionary candidates only
    python scripts/transcribe.py -o out/ page.jpg      # also write <page>.txt files

Pipeline: Topaz finds and straightens the lines and reads each into Grascii; its dictionary
lists candidate words per outline; Claude reads the whole page and writes the English.
The LLM step needs Anthropic credentials (ANTHROPIC_API_KEY).
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--no-llm", action="store_true", help="skip the Claude step")
    ap.add_argument("--lines", action="store_true", help="each image is a single line, not a page")
    ap.add_argument("-o", "--out", type=Path, help="write <image>.txt files here")
    args = ap.parse_args()

    from topaz import Topaz

    topaz = Topaz(ROOT / "models" / "topaz-61m", "mps" if torch.backends.mps.is_available() else "cpu")
    decoder = None
    if not args.no_llm:
        from gregg_ocr.decode import EnglishDecoder

        decoder = EnglishDecoder(topaz.prompt, cache_path=ROOT / "runs" / "llm_cache.jsonl")

    for path in args.images:
        gray = np.array(Image.open(path).convert("L"))  # PIL also reads webp/heic-less formats cv2 may not
        t = topaz.transcribe(gray, lines=args.lines)
        intermediate = t.intermediate()
        parts = [f"# {path.name}: {len(t.lines)} lines", "", "## Strokes (Grascii)", t.grascii,
                 "", "## Outlines with dictionary candidates", intermediate]
        if decoder:
            refused_before = decoder.refusals
            english = decoder.decode(intermediate)
            if decoder.refusals > refused_before:
                english = ("(Claude declined this transcript: a safety-classifier false positive. "
                           "Use the outlines and candidates above.)")
            parts += ["", "## English", english]
        text = "\n".join(parts)
        print(text + "\n")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{path.stem}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
