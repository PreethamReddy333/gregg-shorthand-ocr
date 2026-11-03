"""Outline <-> English lexicon: Topaz's dictionary plus outlines mined from keyed book lines.

Topaz's dictionary (from the 1930 Anniversary dictionary and phrase lists) leaves out many of
the commonest words (had, as, him, made, may, ...), which the books teach as brief forms and
rules rather than listing. ``mine`` recovers them from lines where both the strokes and the
English are known: it aligns outlines to words with a small dynamic program anchored on
pairs the dictionary already knows, and keeps the unknown pairs that recur consistently.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .text import PUNCT, tokenize

MAX_WORDS = 3  # one outline can stand for a phrase of up to this many words


def strip_outline(item: str) -> str:
    return item.lstrip("=")


def load_topaz_dictionary(path: Path) -> dict[str, list[tuple[str, float]]]:
    return {o: [(w, s) for w, s in ws] for o, ws in json.loads(Path(path).read_text()).items()}


def align(outlines: list[str], words: list[str], known: set[tuple[str, str]]) -> list[tuple[str, str]] | None:
    """Best split of ``words`` into one chunk per outline; None if no split exists."""
    n, m = len(outlines), len(words)
    neg = float("-inf")
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, min(MAX_WORDS, j) + 1):
                prev = best[i - 1][j - k]
                if prev == neg:
                    continue
                phrase = " ".join(words[j - k:j])
                score = prev + (2.0 if (outlines[i - 1], phrase) in known else -0.5 * k)
                if score > best[i][j]:
                    best[i][j], back[i][j] = score, k
    if best[n][m] == neg:
        return None
    pairs, j = [], m
    for i in range(n, 0, -1):
        k = back[i][j]
        pairs.append((outlines[i - 1], " ".join(words[j - k:j])))
        j -= k
    return pairs[::-1]


def mine(lines: list[tuple[str, str]], dictionary: dict, min_count: int = 3, min_share: float = 0.6
         ) -> dict[str, list[tuple[str, int]]]:
    """(gold Grascii, English) lines -> {outline: [(words, count)]} for pairs the dictionary lacks."""
    known = {(o, w.lower()) for o, ws in dictionary.items() for w, _ in ws}
    pair_counts, word_totals = Counter(), Counter()
    for grascii, english in lines:
        outlines = [strip_outline(t) for t in grascii.split() if any(c.isalpha() for c in t)]
        words = [t for t in tokenize(english) if t not in PUNCT]
        if not outlines or not words or len(words) > 3 * len(outlines):
            continue
        pairs = align(outlines, words, known)
        if not pairs:
            continue
        # Only trust lines the dictionary mostly explains; the rest are likely misaligned keys.
        if sum(p in known for p in pairs) < 0.6 * len(pairs):
            continue
        for o, w in pairs:
            if (o, w) not in known:
                pair_counts[(o, w)] += 1
                word_totals[w] += 1
    mined = defaultdict(list)
    for (o, w), c in pair_counts.items():
        if c >= min_count and c >= min_share * word_totals[w]:
            mined[o].append((w, c))
    return {o: sorted(ws, key=lambda x: -x[1]) for o, ws in mined.items()}


class Lexicon:
    """Candidate lookup for the decoders: Topaz's lookup, plus mined outlines merged in."""

    def __init__(self, topaz_dictionary, topaz_raw: dict, mined: dict[str, list[tuple[str, int]]]):
        self.topaz = topaz_dictionary  # a topaz.lookup.Dictionary, for its phrase and near-match search
        self.topaz_raw = topaz_raw     # the same dictionary.json as {outline: [(words, score)]}
        self.mined = mined

    @classmethod
    def load(cls, root: Path) -> "Lexicon":
        from topaz.lookup import Dictionary  # third-party (MIT), shipped with the Topaz model

        path = root / "models" / "topaz-61m" / "dictionary.json"
        mined_path = root / "models" / "mined_lexicon.json"
        mined = json.loads(mined_path.read_text()) if mined_path.exists() else {}
        return cls(Dictionary(path), load_topaz_dictionary(path), mined)

    def lookup(self, outline: str, k: int = 8) -> dict:
        hit = self.topaz.lookup(outline, k=k)
        extra = [w for w, _ in self.mined.get(strip_outline(outline), [])]
        if not extra:
            return hit
        if hit["exact"]:
            # Interleave: a mined word is often the right reading of a common outline.
            words = hit["words"][:1] + [w for w in extra if w not in hit["words"]] + hit["words"][1:]
        else:
            words = extra + [w for w in hit["words"] if w not in extra]
        return {"outline": outline, "words": words[:k], "exact": True}

    def word_to_outline(self) -> dict[str, str]:
        """English word or phrase -> its most likely outline, for generating training data."""
        best = {}
        for o, ws in self.topaz_raw.items():
            for w, s in ws:
                w = w.lower()
                if w not in best or s > best[w][1]:
                    best[w] = (o, s)
        for o, ws in self.mined.items():
            for w, c in ws:
                if w not in best:
                    best[w] = (o, c)
        return {w: o for w, (o, _) in best.items()}
