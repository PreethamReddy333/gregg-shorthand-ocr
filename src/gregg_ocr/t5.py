"""The trained Grascii -> English model (scripts/train_seq2seq.py), for decoding Topaz output."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .synth import model_source


class T5Decoder:
    def __init__(self, path: Path, device: str | None = None, num_beams: int = 4):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval()
        self.num_beams = num_beams
        # How many earlier lines the model was trained to read (models before this file existed: 1).
        config = Path(path) / "gregg_config.json"
        self.n_context = json.loads(config.read_text())["context_lines"] if config.exists() else 1

    @torch.no_grad()
    def decode(self, lines: list[str], context: str | list[str] = "", use_context: bool = True) -> list[str]:
        """Grascii lines of one passage -> English per line. Each line is read with the lines
        before it as context (the first ones with ``context``, the lines before the passage),
        unless ``use_context`` is False."""
        before = [context] if isinstance(context, str) else list(context)
        sources = []
        for i, line in enumerate(lines):
            prev = (before + lines[:i]) if use_context else []
            sources.append(model_source(prev, line, self.n_context))
        enc = self.tok(sources, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=64, num_beams=self.num_beams)
        return self.tok.batch_decode(out, skip_special_tokens=True)
