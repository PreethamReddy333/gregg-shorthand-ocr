"""The trained Grascii -> English model (scripts/train_seq2seq.py), for decoding Topaz output."""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .synth import model_text


class T5Decoder:
    def __init__(self, path: Path, device: str | None = None, num_beams: int = 4):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval()
        self.num_beams = num_beams

    @torch.no_grad()
    def decode(self, lines: list[str], context: str = "", use_context: bool = True) -> list[str]:
        """Grascii lines of one passage -> English per line. Each line is read with the line
        before it as context (the first with ``context``), unless ``use_context`` is False."""
        sources = []
        for i, line in enumerate(lines):
            prev = (lines[i - 1] if i else context) if use_context else ""
            sources.append(f"context: {model_text(prev)} line: {model_text(line)}")
        enc = self.tok(sources, return_tensors="pt", padding=True, truncation=True, max_length=192).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=64, num_beams=self.num_beams)
        return self.tok.batch_decode(out, skip_special_tokens=True)
