"""Loaders that bring every source into one table shape.

Every loader returns a DataFrame with at least:
    source   which dataset the row came from
    level    "word", "phrase" or "line"
    grascii  stroke transcription (None where the source has none, e.g. Gregg-1916)
    english  the English the shorthand stands for
    png      image as PNG bytes (Gregg-1916 rows carry `path` instead)
"""

import pandas as pd

from . import PROCESSED, RAW

GRASCII_FILES = {
    "gregg-anniversary-words": ("anniversary", "word"),
    "gregg-preanniversary-words": ("preanniversary", "word"),
    "gregg-preanniversary-phrases": ("preanniversary", "phrase"),
}


def load_grascii() -> pd.DataFrame:
    frames = []
    for name, (edition, level) in GRASCII_FILES.items():
        df = pd.read_parquet(RAW / "grascii" / f"{name}.parquet")
        frames.append(pd.DataFrame({
            "source": name,
            "edition": edition,
            "level": level,
            "grascii": df["grascii_normalized"],
            "english": df["longhand"],
            "png": df["image"].map(lambda im: im["bytes"]),
        }))
    return pd.concat(frames, ignore_index=True)


def load_gregg1916() -> pd.DataFrame:
    paths = sorted((RAW / "gregg1916" / "extracted").rglob("*.png"))
    return pd.DataFrame({
        "source": "gregg1916",
        "edition": "preanniversary",
        "level": "word",
        "grascii": None,
        "english": [p.stem for p in paths],
        "path": [str(p) for p in paths],
    })


def load_topaz() -> pd.DataFrame:
    files = sorted((PROCESSED / "topaz").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.insert(0, "source", "topaz")
    df.insert(1, "edition", "anniversary")
    df.insert(2, "level", "line")
    return df
