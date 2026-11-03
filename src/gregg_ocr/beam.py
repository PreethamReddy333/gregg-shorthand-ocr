"""Grascii -> English without an LLM: beam search over dictionary candidates with a trigram LM.

Each outline Topaz reads has a ranked list of candidate words from the Topaz dictionary
(exact matches first, then words of nearby outlines). The decoder picks one candidate per
outline so that the whole passage scores best under

    lm_weight * log P_trigram(words) - rank_cost * rank - fuzzy_cost * [candidate is a near match]

The language-model state carries across lines, so each line is read in the context of the
ones before it. It can only ever output words the dictionary offered.
"""

from dataclasses import dataclass

from .lm import BOS, EOS, TrigramLM
from .text import PUNCT, tokenize


@dataclass
class BeamConfig:
    beam: int = 24
    candidates: int = 8
    # Tuned on the 1929 Manual dev set (scripts/tune_beam.py); results there are flat around these.
    lm_weight: float = 1.0
    rank_cost: float = 0.8
    fuzzy_cost: float = 1.0


class BeamDecoder:
    def __init__(self, lm: TrigramLM, dictionary, config: BeamConfig | None = None):
        """``dictionary``: a topaz.lookup.Dictionary (outline -> ranked candidates)."""
        self.lm = lm
        self.dictionary = dictionary
        self.cfg = config or BeamConfig()
        self._cache = {}

    def options(self, item: str) -> list[tuple[list[str], float]]:
        """One Grascii item -> [(tokens, channel log-score)]; punctuation passes through."""
        if item in self._cache:
            return self._cache[item]
        if item in PUNCT or item.isdigit():
            opts = [([item], 0.0)]
        elif not any(c.isalpha() for c in item) and item.lstrip("=") != "'":
            # Paragraph marks, dashes, quotes: nothing to say. A lone ' is the dot for a/an.
            opts = [([], 0.0)]
        else:
            hit = self.dictionary.lookup(item, k=self.cfg.candidates)
            penalty = 0.0 if hit["exact"] else self.cfg.fuzzy_cost
            opts = [(tokenize(w), -self.cfg.rank_cost * r - penalty) for r, w in enumerate(hit["words"])]
            opts = [(t, s) for t, s in opts if t] or [([], -self.cfg.fuzzy_cost * 2)]
        self._cache[item] = opts
        return opts

    def _search(self, lines: list[str], beam: int, history: int):
        """Beam search; hypotheses ending in the same ``history`` words are merged.
        -> (items, [(final score, output tokens per item)] best first)."""
        items = [(i, tok) for i, line in enumerate(lines) for tok in line.split()]
        # beam entry: (score, u, v, output tokens per item)
        beams = [(0.0, BOS, BOS, [])]
        for _, item in items:
            nxt = {}
            for score, u, v, out in beams:
                for toks, channel in self.options(item):
                    s, a, b = score + channel, u, v
                    for w in toks:
                        s += self.cfg.lm_weight * self.lm.logprob(a, b, w)
                        a, b = b, w
                    new = out + [toks]
                    # history=2 is exact for a trigram LM; longer keeps more varied alternatives.
                    key = tuple(w for t in new for w in t)[-history:] if history > 2 else (a, b)
                    if key not in nxt or s > nxt[key][0]:
                        nxt[key] = (s, a, b, new)
            beams = sorted(nxt.values(), key=lambda x: -x[0])[:beam]
        final = [(s + self.cfg.lm_weight * self.lm.logprob(a, b, EOS), out) for s, a, b, out in beams]
        return items, sorted(final, key=lambda x: -x[0])

    @staticmethod
    def _per_line(n_lines: int, items, out) -> list[str]:
        per_line = [[] for _ in range(n_lines)]
        for (i, _), toks in zip(items, out):
            per_line[i] += toks
        return [" ".join(t) for t in per_line]

    def decode(self, lines: list[str]) -> list[str]:
        """Grascii lines of one passage -> English per line (decoded jointly, split back by line)."""
        items, final = self._search(lines, self.cfg.beam, history=2)
        return self._per_line(len(lines), items, final[0][1])

    def nbest(self, lines: list[str], k: int = 30) -> list[tuple[float, list[str]]]:
        """The ``k`` best distinct readings of a passage, as (score, English per line)."""
        items, final = self._search(lines, max(self.cfg.beam, 2 * k), history=4)
        seen, out = set(), []
        for score, toks in final:
            per_line = self._per_line(len(lines), items, toks)
            if tuple(per_line) not in seen:
                seen.add(tuple(per_line))
                out.append((score, per_line))
        return out[:k]
