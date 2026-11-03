"""Synthetic training data for the Grascii -> English model.

``Renderer`` writes English as Grascii outlines from the lexicon (using phrase outlines such as
I-L-N "I will not" where they exist, and simple suffix rules for words the lexicon lacks).
``NoiseModel`` learns how Topaz misreads strokes, by comparing its real readings of book lines
with their correct Grascii, and applies the same kinds of mistakes to clean text.
"""

import random
from collections import Counter

from .text import PUNCT

# Grascii endings for common English suffixes, tried when a word isn't in the lexicon.
SUFFIXES = [("'s", "S"), ("ies", "E-S"), ("es", "S"), ("s", "S"), ("ied", "E-D"), ("ed", "D"), ("d", "D"),
            ("ing", "'"), ("ly", "E"), ("er", "R"), ("est", "S-T"), ("ness", "N-S"), ("ment", "M")]
TOKEN_SAFE = {"^": "Q"}  # characters the T5 tokenizer can't represent


def model_text(grascii: str) -> str:
    """Grascii as fed to the seq2seq model. Capital marks are dropped: synthetic text has none."""
    grascii = grascii.replace("=", "")
    for a, b in TOKEN_SAFE.items():
        grascii = grascii.replace(a, b)
    return grascii


class Renderer:
    def __init__(self, word_to_outline: dict[str, str], phrase_rate: float = 0.7, rng=None):
        self.single = {w: o for w, o in word_to_outline.items() if " " not in w}
        self.phrases = {tuple(w.split()): o for w, o in word_to_outline.items() if 1 < len(w.split()) <= 3}
        self.phrase_rate = phrase_rate
        self.rng = rng or random.Random(0)

    def outline(self, word: str) -> str | None:
        if word in self.single:
            return self.single[word]
        for suffix, stroke in SUFFIXES:
            base = word[: -len(suffix)]
            if word.endswith(suffix) and len(base) >= 2:
                for stem in (base, base + "e", base[:-1] if base[-1:] == base[-2:-1] else base):
                    if stem in self.single:
                        return f"{self.single[stem]}-{stroke}"
        return None

    def render(self, tokens: list[str]) -> str | None:
        """English tokens -> Grascii, or None if some word can't be written."""
        out, i = [], 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in PUNCT or tok.isdigit():
                out.append(tok)
                i += 1
                continue
            for n in (3, 2):
                key = tuple(tokens[i:i + n])
                if len(key) == n and key in self.phrases and self.rng.random() < self.phrase_rate:
                    out.append(self.phrases[key])
                    i += n
                    break
            else:
                o = self.outline(tok)
                if o is None:
                    return None
                out.append(o)
                i += 1
        return " ".join(out)


def _align(a: list[str], b: list[str]) -> list[tuple[str | None, str | None]]:
    """Edit-distance alignment of two stroke lists -> pairs (None = insertion/deletion)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    return pairs[::-1]


class NoiseModel:
    """Stroke substitutions/deletions/insertions, outline merges/splits, dropped punctuation."""

    def fit(self, pairs: list[tuple[str, str]]) -> "NoiseModel":
        """``pairs``: (correct Grascii, Topaz's reading) for the same line."""
        self.sub, self.dele, self.ins, self.total = Counter(), Counter(), Counter(), Counter()
        outlines = merges = splits = punct = punct_kept = 0
        for gold, pred in pairs:
            g = [t for t in gold.split() if t not in PUNCT]
            p = [t for t in pred.split() if t not in PUNCT]
            g_punct = sum(t in PUNCT for t in gold.split())
            punct += g_punct
            punct_kept += min(g_punct, sum(t in PUNCT for t in pred.split()))
            outlines += len(g)
            if len(p) < len(g):
                merges += len(g) - len(p)
            elif len(p) > len(g):
                splits += len(p) - len(g)
            else:
                for go, po in zip(g, p):
                    gs, ps = go.lstrip("=").split("-"), po.lstrip("=").split("-")
                    for s in gs:
                        self.total[s] += 1
                    for x, y in _align(gs, ps):
                        if x is None:
                            self.ins[y] += 1
                        elif y is None:
                            self.dele[x] += 1
                        elif x != y:
                            self.sub[(x, y)] += 1
        self.merge_rate = merges / max(outlines, 1)
        self.split_rate = splits / max(outlines, 1)
        self.punct_drop = 1 - punct_kept / max(punct, 1)
        self.ins_rate = sum(self.ins.values()) / max(sum(self.total.values()), 1)
        self.subs_from = {}
        for (x, y), c in self.sub.items():
            self.subs_from.setdefault(x, []).append((y, c))
        return self

    def summary(self) -> str:
        n = max(sum(self.total.values()), 1)
        return (f"per stroke: sub {sum(self.sub.values()) / n:.1%}, del {sum(self.dele.values()) / n:.1%}, "
                f"ins {self.ins_rate:.1%}; per outline: merge {self.merge_rate:.1%}, split {self.split_rate:.1%}; "
                f"punctuation dropped {self.punct_drop:.0%}")

    def apply(self, grascii: str, rng: random.Random, scale: float = 1.0) -> str:
        """Corrupt clean Grascii the way Topaz does (``scale`` > 1 makes it noisier)."""
        out = []
        for item in grascii.split():
            if item in PUNCT:
                if rng.random() >= self.punct_drop * scale:
                    out.append(item)
                continue
            strokes = []
            for s in item.split("-"):
                t = max(self.total.get(s, 0), 1)
                r = rng.random()
                if r < scale * self.dele.get(s, 0) / t:
                    continue
                if r < scale * (self.dele.get(s, 0) + sum(c for _, c in self.subs_from.get(s, []))) / t:
                    ys, cs = zip(*self.subs_from[s])
                    s = rng.choices(ys, weights=cs)[0]
                strokes.append(s)
                if rng.random() < scale * self.ins_rate:
                    strokes.append(rng.choice(list(self.ins.elements()) or [s]))
            if not strokes:
                continue
            if len(strokes) > 1 and rng.random() < scale * self.split_rate:
                k = rng.randrange(1, len(strokes))
                out += ["-".join(strokes[:k]), "-".join(strokes[k:])]
            elif out and out[-1] not in PUNCT and rng.random() < scale * self.merge_rate:
                out[-1] = out[-1] + "-" + "-".join(strokes)
            else:
                out.append("-".join(strokes))
        return " ".join(out)
