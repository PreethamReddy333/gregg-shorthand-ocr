"""Beam search over dictionary candidates, guided by a local open LLM ("shallow fusion").

The trigram beam decoder only sees two words of context. Here a small causal language model
(SmolLM2, Apache-2.0, runs on the Mac) scores every partial reading after each outline, given
the paragraph decoded so far, so the choice between candidates uses real knowledge of English:

    score = channel (candidate rank) + trigram_weight * trigram + llm_weight * log P_llm(reading | context)

Like the beam decoder it can only output words the lexicon offered, so it can't invent text.
"""

import math
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .beam import BeamDecoder
from .lm import BOS

DEFAULT_LLM = "HuggingFaceTB/SmolLM2-360M"


def detokenize(tokens: list[str]) -> str:
    """Our lower-case token stream -> ordinary-looking text for the LLM."""
    text = " ".join(tokens)
    text = re.sub(r" ([.,?;:!])", r"\1", text)
    text = re.sub(r"\bi\b", "I", text)
    return re.sub(r"(^|[.?!] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


@dataclass
class FusionConfig:
    beam: int = 12
    prune: int = 48          # expansions scored by the LLM per step
    # Tuned on 150 lines of the 1929 Manual dev set (scripts/tune_fused.py): 15.4% word error,
    # against 18.6% for the trigram beam alone; (0.3, 2.0) tied, (0.3, 1.0) gave 15.8%.
    trigram_weight: float = 0.6
    llm_weight: float = 1.0
    context_chars: int = 600  # earlier text the LLM sees


class LLMScorer:
    def __init__(self, name: str = DEFAULT_LLM, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def logprobs(self, context: str, texts: list[str]) -> list[float]:
        """log P(text | context) for each text (natural log, summed over the text's tokens)."""
        prefix = self.tok.bos_token or ""
        ctx_ids = self.tok(prefix + context, add_special_tokens=False).input_ids
        seqs = [self.tok(prefix + context + (" " if context else "") + t, add_special_tokens=False).input_ids
                for t in texts]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        token_lp = logp.gather(-1, ids[:, 1:].to(self.device).unsqueeze(-1)).squeeze(-1).cpu()
        out = []
        start = max(len(ctx_ids) - 1, 0)  # score only the text's tokens, not the context's
        for i, s in enumerate(seqs):
            out.append(float(token_lp[i, start:len(s) - 1].sum()))
        return out


class FusedDecoder:
    def __init__(self, beam_decoder: BeamDecoder, scorer: LLMScorer, config: FusionConfig | None = None):
        self.base = beam_decoder  # supplies candidates and the trigram LM
        self.scorer = scorer
        self.cfg = config or FusionConfig()

    def decode(self, lines: list[str], context: str = "") -> list[str]:
        """Grascii lines of one passage -> English per line. ``context``: English of the lines
        before this passage (e.g. the rest of the paragraph), which the LLM conditions on.

        Lines are read one at a time, each with the lines already read as context, so the LLM's
        inputs stay short on long passages (a whole page as one search ran into swap)."""
        out = []
        for line in lines:
            out.append(self._decode_line(line, context)[0])
            context = detokenize((context + " " + out[-1]).split())[-self.cfg.context_chars:]
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return out

    def _decode_line(self, line: str, context: str) -> list[str]:
        lines = [line]
        items = [(i, tok) for i, line in enumerate(lines) for tok in line.split()]
        lm = self.base.lm
        # hypothesis: (channel+trigram score, llm score, u, v, tokens per item)
        hyps = [(0.0, 0.0, BOS, BOS, [])]
        for _, item in items:
            expanded = []
            for base, llm, u, v, out in hyps:
                for toks, channel in self.base.options(item):
                    s, a, b = base + channel, u, v
                    for w in toks:
                        s += self.cfg.trigram_weight * lm.logprob(a, b, w)
                        a, b = b, w
                    expanded.append([s, llm, a, b, out + [toks], bool(toks)])
            # Cheap pre-prune with the parent's LLM score, then score the survivors properly.
            expanded.sort(key=lambda h: -(h[0] + self.cfg.llm_weight * h[1]))
            expanded = expanded[: self.cfg.prune]
            changed = [h for h in expanded if h[5]]
            if changed:
                texts = [detokenize([w for t in h[4] for w in t]) for h in changed]
                for h, lp in zip(changed, self.scorer.logprobs(context, texts)):
                    h[1] = lp
            merged = {}
            for s, llm, a, b, out, _ in expanded:
                key = tuple(w for t in out for w in t)
                if key not in merged or s + self.cfg.llm_weight * llm > merged[key][0] + self.cfg.llm_weight * merged[key][1]:
                    merged[key] = (s, llm, a, b, out)
            hyps = sorted(merged.values(), key=lambda h: -(h[0] + self.cfg.llm_weight * h[1]))[: self.cfg.beam]
        best = hyps[0]
        return self.base._per_line(len(lines), items, best[4])
