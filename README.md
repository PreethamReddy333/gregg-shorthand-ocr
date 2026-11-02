# Gregg shorthand → English

OCR for **Gregg Anniversary** shorthand (1929 edition), in two stages:

1. **Image → Grascii.** A vision model reads the strokes into
   [Grascii](https://github.com/grascii/grascii), an ASCII spelling of Gregg outlines (e.g. `E-S-P-R-E-M`).
2. **Grascii → English.** The Grascii dictionary lists candidate words for each outline,
   and a language model picks the right ones from the context of the sentence.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[train]"
.venv/bin/python scripts/download_data.py      # ~4 GB download; ~1-2 GB kept on disk
.venv/bin/python scripts/inspect_data.py       # stats + data/samples.png
.venv/bin/python scripts/baseline_gregg_vision.py
```

## Data

| Source | Level | Rows | Labels | Edition |
|---|---|---|---|---|
| `grascii/gregg-anniversary-words` | word | 17,766 | Grascii + English | Anniversary |
| `grascii/gregg-preanniversary-words` | word | 15,832 | Grascii + English | Pre-Anniversary |
| `grascii/gregg-preanniversary-phrases` | phrase | 3,477 | Grascii + English | Pre-Anniversary |
| Gregg-1916 (Zhai et al. 2018) | word | 15,280 | English only | Pre-Anniversary |
| Topaz line crops (`amongglue/topaz-shorthand-data`) | line | see `inspect_data.py` | Grascii + English | Anniversary |

Everything here is printed book material, not real handwriting. Topaz files are pickles,
so `src/gregg_ocr/safe_pickle.py` only lets them rebuild numpy arrays. They're converted
to parquet with PNG bytes in `data/processed/topaz/`.

## Baseline (off-the-shelf `grascii/gregg-vision-v0.2.1`, 200 words each)

| Edition | Exact Grascii | Right word in top-5 dictionary lookup |
|---|---|---|
| Pre-Anniversary (its training data) | 95.0% | 94.5% |
| Anniversary (unseen edition) | 60.0% | 60.0% |

## Target: handwritten notes

We build on [Topaz-61M](https://huggingface.co/amongglue/topaz-gregg-shorthand-61m) (MIT, in `models/topaz-61m`)
instead of training a recognizer from scratch. Topaz is trained and tested only on printed
textbook plates, so the work is closing the handwriting gap:

1. **Handwritten test set first.** Aim for 200+ lines from at least 5 writers, never trained on.
   - **Genesis (built):** Nathan Tonning's handwritten Anniversary Gregg of Genesis 1:1–5:3 plus
     an earlier draft of 1:1–12 ([CC0](https://github.com/NPTprogrammer/GreggShorthandStuff)),
     keyed to the KJV. 307 lines, 119 scorable verses, 1 writer.
     `scripts/build_genesis_testset.py` builds it (verse counts are checked against the KJV;
     4 lines are hand-corrected in `OVERRIDES`), and `scripts/eval_topaz.py --genesis` scores it per verse.
     The translation is inferred from the shape of the outlines, not stated by the writer;
     confirm it from Topaz's first run.
   - **Your own lines:** `data/handwritten/` as `name.png` + `name.txt` English key (optional
     `name.grascii`); `scripts/eval_topaz.py --folder data/handwritten` scores them.
   - **Next candidate:** Pvt. Charles Malcolm Smith's 1918 Gregg field diary (Australian War
     Memorial, 21 pages, pre-Anniversary) with his own typed transcript. Copyrighted: private
     evaluation only.
2. **Automatic English step.** Replace Topaz's copy-paste prompt with an LLM call and report
   English word error rate, not just stroke error.
3. **Handwritten training data.** Ranked by value per hour:
   - Handwritten practice notes whose English is already known: people's notes on dictation
     material from keyed books (Speed Studies, Dictation Studies, Review Letters…).
   - Writing printed key sentences by hand on purpose (one writer → hundreds of lines a day).
   - Stronger synthetic handwriting: Topaz already simulates pen dynamics; push it further
     (per-writer proportion drift, sloppier joins, dropped vowels, speed-writing shortcuts).
4. **Fine-tune Topaz** on real handwritten lines + augmented book lines, judged on the
   handwritten test set by writer, not by line.
5. **Later editions.** Many surviving handwritten notes are Simplified or Diamond Jubilee,
   which Topaz doesn't read. The keyed books for those editions are the data for that step.

### Topaz-61M results (2026-11-02)

| Test set | Unit | Stroke error | Outline error | Word error, top-1 dictionary (no LLM) |
|---|---|---|---|---|
| Functional Method 2, printed, held out (300 lines) | line | 10.2% | 24.6% | 43.0% |
| Handwritten Genesis, all (119 verses) | verse | n/a | n/a | 61.5% |
| ↳ chapter 1 / 2 / 3 / 4 | verse | | | 46% / 63% / 72% / 74% |
| ↳ 1:1–11, final copy vs earlier draft | verse | | | 40.9% / 40.9% |

Top-1 word error is a floor: it takes the first dictionary suggestion per outline with no
context. Part of the rise across chapters is vocabulary the dictionary lacks (3% of KJV words
in chapter 1, 15% in chapter 4: Cain, Abel, Enoch, begat…), which an LLM step can recover.
Genesis has no Grascii key, so stroke error can't be measured there.

### With Claude decoding (`--llm`, Claude Opus 5.5, effort medium, structured output)

| Test set | Items | Declined by safety classifier | Word error, top-1 → Claude (answered items) |
|---|---|---|---|
| Functional Method 2, printed, held out | 271 lines (29 without key text skipped) | 7 | 41.0% → **24.2%** |
| Handwritten Genesis | 119 verses | 10 | 59.7% → 6.0% |

FM2 is the honest number: obscure 1930s prose, decoded one line at a time with no
surrounding context. Genesis is flattering, because Claude knows the KJV nearly by heart
and recognizes verses from a few words. It shows the pipeline works end to end, not how
well it reads unfamiliar text.

**Safety-classifier false positives.** Some transcripts are declined (`stop_reason:
"refusal"`, category `bio`) even though they're innocuous; Grascii strings plus odd
candidate words seem to look like encoded text. Server-side fallback (Opus 4.8, Opus 5),
Sonnet 5.5 and Haiku 5.5 decline the same inputs. `transcribe.py` reports a declined page
and still prints the strokes and candidates. A local decoder (n-gram or small open LM over
the candidates) would avoid this entirely.

Transcribe your own pages with `scripts/transcribe.py page.jpg` (needs `ANTHROPIC_API_KEY`).

## Layout

```
src/gregg_ocr/      package: paths, safe pickle loader, dataset loaders
scripts/            download, inspect, baselines, build_genesis_testset, eval_topaz
data/               downloaded + processed data (git-ignored)
models/topaz-61m/   Topaz weights + inference code (MIT)
```
