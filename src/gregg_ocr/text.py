"""Shared tokenization for the corpus, the language model and the decoders."""

import re

PUNCT = {".", ",", "?", ";", ":", "!"}


def tokenize(text: str) -> list[str]:
    """Lower-case words (keeping apostrophes), numbers, and . , ? ; : ! as separate tokens."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?;:!]", text.lower())
