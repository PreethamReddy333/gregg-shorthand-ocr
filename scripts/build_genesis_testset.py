"""Turn Nathan Tonning's handwritten Gregg Genesis (CC0) into a line/verse test set.

    python scripts/build_genesis_testset.py

Source: github.com/NPTprogrammer/GreggShorthandStuff (CC0), downloaded to data/raw/npt_genesis/
    01_GEN_001_007.pdf          Genesis 1:1-5:3, 7 pages, two ruled columns per page
    WORKING_DOC_Genesis_1.pdf   an earlier draft of Genesis 1:1-13 (a second writing, same writer)
Key: King James Version (public domain), data/raw/npt_genesis/kjv_genesis.json

Writes data/handwritten/genesis/:
    lines/<doc>_p<page>_c<col>_<n>.png   one ruled line each, ruling removed
    lines.csv                            line -> verse it belongs to
    verses.csv                           verse -> KJV text and its lines (the scoring unit)
    debug/<doc>_p<page>.png              detected lines (green) and verse starts (red), for checking

Verses are found from the small superscript numbers at the start of a line, and chapters from
the lone centred chapter numerals. The sequence is checked against the KJV verse counts, so a
missed or extra verse number fails loudly instead of silently shifting every key.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "npt_genesis"
OUT = ROOT / "data" / "handwritten" / "genesis"
SCALE = 3  # PDF points -> pixels; line spacing comes out around 90 px

# Lines the verse-number detector gets wrong, checked by eye against the debug pages.
# (doc, page, column, band index between rules) -> what the line really is.
OVERRIDES = {
    ("gen", 5, 2, 3): "verse_start",   # 4:2, the "2" is faint
    ("gen", 5, 2, 21): "verse_start",  # 4:10, the "10" runs into the first outline
    ("gen", 6, 1, 12): "continuation", # inside 4:15; the line opens with a small upright mark
    ("gen", 6, 2, 8): "verse_start",   # 4:23, the "23" runs into the first outline
}

DOCS = {
    # doc id: (file, first verse as (chapter, verse))
    "gen": ("01_GEN_001_007.pdf", (1, 1)),
    "draft": ("WORKING_DOC_Genesis_1.pdf", (1, 1)),
}


def render(path: Path) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(path)
    return [np.array(page.render(scale=SCALE, grayscale=True).to_pil().convert("L")) for page in pdf]


def find_rules(gray: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[int]], np.ndarray]:
    """Long horizontal rules -> (column x-ranges, rule ys per column, rule mask)."""
    ink = (gray < 220).astype(np.uint8)  # the ruling is light grey
    width = gray.shape[1]
    mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (width // 6, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    segs = [stats[i] for i in range(1, n)]  # x, y, w, h, area
    # Group segments into columns by their x-extent.
    columns: list[tuple[int, int]] = []
    for x, y, w, h, _ in sorted(segs, key=lambda s: s[0]):
        for i, (c0, c1) in enumerate(columns):
            if abs(x - c0) < width * 0.05:
                columns[i] = (min(c0, x), max(c1, x + w))
                break
        else:
            columns.append((x, x + w))
    columns = [c for c in columns if c[1] - c[0] > width * 0.25]
    rules = {}
    for c in columns:
        # A rule is a row inked across nearly the whole column; writing never is.
        coverage = (gray[:, c[0]:c[1]] < 230).mean(axis=1)
        rows = np.nonzero(coverage > 0.85)[0]
        merged = []  # centre of each run of rule rows
        for y in rows:
            if merged and y - merged[-1][-1] <= 2:
                merged[-1].append(y)
            else:
                merged.append([y])
        merged = [int(np.mean(r)) for r in merged]
        # The ruling is evenly spaced; fill in any rule hidden under heavy writing.
        if len(merged) > 2:
            step = float(np.median(np.diff(merged)))
            filled = [merged[0]]
            for y in merged[1:]:
                k = round((y - filled[-1]) / step)
                filled += [round(filled[-1] + (y - filled[-1]) * j / k) for j in range(1, k)] + [y]
            merged = filled
        rules[c] = merged
    mask = np.zeros_like(ink)
    for (c0, c1), ys in rules.items():
        for y in ys:
            mask[max(0, y - 3):y + 4, c0:c1] = ink[max(0, y - 3):y + 4, c0:c1]
    return columns, rules, mask


def components(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= 4]


def classify_band(ink: np.ndarray, spacing: int, pad: int) -> tuple[str, int | None]:
    """A band between two rules -> ("empty"|"chapter"|"verse_start"|"continuation", digit x-end).

    ``ink`` starts ``pad`` px above the band's top rule, because verse numbers often sit on that
    rule. Gaps left by erasing the rule are closed so a digit cut in two stays one component.
    """
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))
    # Shift y so 0 is the top rule; keep only ink that reaches below it.
    comps = [(x, y - pad, cw, ch) for x, y, cw, ch in components(closed) if y + ch > pad + 3]
    if not comps:
        return "empty", None
    w = ink.shape[1]
    x0 = min(c[0] for c in comps)
    x1 = max(c[0] + c[2] for c in comps)
    # A chapter heading is a lone small numeral near the middle of the column.
    if x0 > 0.3 * w and x1 < 0.7 * w and (x1 - x0) < 0.15 * w:
        return "chapter", None

    # A verse number is a small superscript at the left margin: one digit, or two side by side.
    # Digits stand upright; a flat hook of the same size is the start of an outline.
    def digit_like(c):
        return (c[1] < 0.3 * spacing and c[1] + c[3] < 0.6 * spacing
                and 0.12 * spacing <= c[3] < 0.45 * spacing and c[2] < 0.05 * w)

    digits = [c for c in comps if c[0] < 0.035 * w and digit_like(c)]
    # Verse numbers are the leftmost ink on their line; a small hook with writing further left is not one.
    if digits and all(c[0] >= min(d[0] for d in digits) for c in comps):
        end = max(c[0] + c[2] for c in digits)
        digits += [c for c in comps if c not in digits and end - 2 <= c[0] < end + 0.02 * w and digit_like(c)]
        return "verse_start", max(c[0] + c[2] for c in digits)
    return "continuation", None


def line_crop(clean: np.ndarray, top: int, bottom: int, x0: int, x1: int, spacing: int) -> np.ndarray:
    """The writing of one ruled line. Strokes rise above the top rule and dip below the bottom
    one, so crop generously, then blank out pieces of the neighbouring lines: a component
    belongs to this line when its vertical centre lies between the two rules."""
    y0, y1 = max(0, top - int(0.3 * spacing)), min(clean.shape[0], bottom + int(0.35 * spacing))
    crop = clean[y0:y1, x0:x1].copy()
    n, labels, _, centroids = cv2.connectedComponentsWithStats((crop < 160).astype(np.uint8))
    lo, hi = top - y0, bottom - y0 + 0.1 * spacing
    own = [i for i in range(1, n) if lo <= centroids[i][1] <= hi]
    # Keep this line's strokes with their anti-aliased edges; whiten everything else.
    keep = cv2.dilate(np.isin(labels, own).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    crop[keep == 0] = 255
    return crop


def build():
    kjv = json.loads((SRC / "kjv_genesis.json").read_text())
    verse_text = {(int(c["chapter"]), int(v["verse"])): v["text"]
                  for c in kjv["chapters"] for v in c["verses"]}
    n_verses = {int(c["chapter"]): len(c["verses"]) for c in kjv["chapters"]}

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "lines").mkdir(parents=True)
    (OUT / "debug").mkdir()

    line_rows, verse_lines = [], {}
    for doc, (fname, (chapter, verse)) in DOCS.items():
        chapter, verse = chapter, verse - 1  # incremented at the first verse number
        for p, gray in enumerate(render(SRC / fname), 1):
            columns, rules, rule_mask = find_rules(gray)
            clean = gray.copy()
            clean[rule_mask > 0] = 255  # erase ruling; strokes crossing a rule lose a sliver
            debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for c, (cx0, cx1) in enumerate(sorted(columns), 1):
                ys = rules[(cx0, cx1)]
                spacing = int(np.median(np.diff(ys))) if len(ys) > 1 else 90
                n = 0
                for band, (top, bottom) in enumerate(zip(ys[:-1], ys[1:])):
                    y0, y1 = max(0, top - int(0.3 * spacing)), min(gray.shape[0], bottom + int(0.35 * spacing))
                    pad = min(top, int(0.25 * spacing))
                    band_ink = (clean[top - pad:bottom + 3, cx0:cx1] < 160).astype(np.uint8)
                    kind, digits_end = classify_band(band_ink, spacing, pad)
                    forced = OVERRIDES.get((doc, p, c, band))
                    if forced == "verse_start" and kind != forced:
                        kind, digits_end = forced, int(0.045 * (cx1 - cx0))
                    elif forced == "continuation":
                        kind, digits_end = forced, None
                    if kind == "empty":
                        continue
                    if kind == "chapter":
                        if verse != n_verses[chapter]:
                            cv2.imwrite(str(OUT / "debug" / f"{doc}_p{p}.png"), debug)
                            sys.exit(f"{doc} p{p} c{c}: chapter {chapter} ended at verse {verse}, "
                                     f"KJV has {n_verses[chapter]}")
                        chapter, verse = chapter + 1, 0
                        cv2.rectangle(debug, (cx0, top), (cx1, bottom), (255, 0, 0), 3)
                        continue
                    if kind == "verse_start":
                        verse += 1
                        if verse > n_verses[chapter]:
                            cv2.rectangle(debug, (cx0, top), (cx1, bottom), (0, 0, 255), 8)
                            cv2.imwrite(str(OUT / "debug" / f"{doc}_p{p}.png"), debug)
                            sys.exit(f"{doc} p{p} c{c}: verse {chapter}:{verse} does not exist; detection is off "
                                     f"(see debug/{doc}_p{p}.png)")
                    if verse == 0:
                        sys.exit(f"{doc} p{p} c{c}: writing before the first verse number")
                    n += 1
                    crop_x0 = cx0 + (digits_end + 4 if digits_end else 0)
                    # Writing sometimes runs a little past the end of the ruling.
                    crop = line_crop(clean, top, bottom, crop_x0, min(gray.shape[1], cx1 + spacing // 2), spacing)
                    line_id = f"{doc}_p{p}_c{c}_{n:02d}"
                    cv2.imwrite(str(OUT / "lines" / f"{line_id}.png"), crop)
                    ref = f"{chapter}:{verse}"
                    line_rows.append({"line_id": line_id, "doc": doc, "page": p, "column": c, "n": n,
                                      "verse": ref, "verse_start": kind == "verse_start"})
                    verse_lines.setdefault((doc, chapter, verse), []).append(line_id)
                    color = (0, 0, 255) if kind == "verse_start" else (0, 160, 0)
                    cv2.rectangle(debug, (crop_x0, y0), (cx1, y1), color, 2)
                    cv2.putText(debug, ref, (cx1 - 110, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imwrite(str(OUT / "debug" / f"{doc}_p{p}.png"), debug)

    # The last verse of each document may stop mid-way; it is kept but not scored.
    last = {}
    for doc, ch, v in verse_lines:
        last[doc] = max(last.get(doc, (0, 0)), (ch, v))
    verse_rows = [{"doc": doc, "verse": f"{ch}:{v}", "kjv": verse_text[(ch, v)],
                   "lines": " ".join(ids), "complete": (ch, v) != last[doc]}
                  for (doc, ch, v), ids in verse_lines.items()]

    for name, rows in (("lines.csv", line_rows), ("verses.csv", verse_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    for doc in DOCS:
        chapters = {}
        for (d, ch, v) in verse_lines:
            if d == doc:
                chapters.setdefault(ch, []).append(v)
        for ch, vs in sorted(chapters.items()):
            gaps = sorted(set(range(1, max(vs) + 1)) - set(vs))
            print(f"{doc} ch{ch}: verses {min(vs)}-{max(vs)} of {n_verses[ch]}" + (f"  MISSING {gaps}" if gaps else ""))
    print(f"{len(line_rows)} lines, {len(verse_rows)} verses ({sum(r['complete'] for r in verse_rows)} complete) -> {OUT}")


if __name__ == "__main__":
    build()
