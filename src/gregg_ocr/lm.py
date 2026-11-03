"""A word trigram language model with interpolated Kneser-Ney smoothing.

Small and dependency-free: it scores how likely a word is given the two before it, which is
the context the beam decoder uses to choose between dictionary candidates.
"""

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS = "<s>", "</s>"
D = 0.75  # absolute discount


class TrigramLM:
    def __init__(self):
        self.c3 = Counter()            # (u, v, w) -> count
        self.c3_ctx = Counter()        # (u, v) -> total count
        self.c3_types = Counter()      # (u, v) -> distinct w
        self.cc2 = Counter()           # (v, w) -> distinct u before it
        self.cc2_ctx = Counter()       # v -> sum of cc2 over w
        self.cc2_types = Counter()     # v -> distinct w
        self.cc1 = Counter()           # w -> distinct v before it
        self.cc1_total = 0
        self.vocab_size = 0

    @classmethod
    def train(cls, sentences, weights=None) -> "TrigramLM":
        """``sentences``: iterables of token lists; ``weights``: repeat counts per iterable."""
        lm = cls()
        weights = weights or [1] * len(sentences)
        for source, weight in zip(sentences, weights):
            for toks in source:
                seq = [BOS, BOS] + toks + [EOS]
                for i in range(2, len(seq)):
                    lm.c3[(seq[i - 2], seq[i - 1], seq[i])] += weight
        bigram_left = defaultdict(set)
        for (u, v, w), c in lm.c3.items():
            lm.c3_ctx[(u, v)] += c
            lm.c3_types[(u, v)] += 1
            bigram_left[(v, w)].add(u)
        unigram_left = defaultdict(set)
        for (v, w), us in bigram_left.items():
            lm.cc2[(v, w)] = len(us)
            lm.cc2_ctx[v] += len(us)
            lm.cc2_types[v] += 1
            unigram_left[w].add(v)
        for w, vs in unigram_left.items():
            lm.cc1[w] = len(vs)
        lm.cc1_total = sum(lm.cc1.values())
        lm.vocab_size = len(lm.cc1)
        return lm

    def _p1(self, w: str) -> float:
        # Unseen words keep a small share so they are unlikely but not impossible.
        return (self.cc1.get(w, 0) + 0.01) / (self.cc1_total + 0.01 * (self.vocab_size + 1))

    def _p2(self, v: str, w: str) -> float:
        ctx = self.cc2_ctx.get(v, 0)
        if not ctx:
            return self._p1(w)
        return (max(self.cc2.get((v, w), 0) - D, 0) / ctx
                + D * self.cc2_types[v] / ctx * self._p1(w))

    def prob(self, u: str, v: str, w: str) -> float:
        ctx = self.c3_ctx.get((u, v), 0)
        if not ctx:
            return self._p2(v, w)
        return (max(self.c3.get((u, v, w), 0) - D, 0) / ctx
                + D * self.c3_types[(u, v)] / ctx * self._p2(v, w))

    def logprob(self, u: str, v: str, w: str) -> float:
        return math.log(self.prob(u, v, w))

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TrigramLM":
        # Only ever loads a file this project wrote itself (scripts/train_lm.py).
        lm = cls()
        with open(path, "rb") as f:
            lm.__dict__.update(pickle.load(f))
        return lm
