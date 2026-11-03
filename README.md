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
2. **Automatic English step (done).** Claude and three local decoders, compared below.
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

### Grascii → English: every decoder compared (2026-11-03)

Word error rate (lower is better). The strokes are Topaz's in every column; only the step
from strokes to English changes.

| Decoder | Context | FM2, printed (271 lines) | Handwritten Genesis (119 verses) | Your two pages (176 words) |
|---|---|---|---|---|
| Dictionary only (top-1 per outline) | none | 41.3% | 61.5% | 45.5% |
| Beam search + trigram LM | none | 28.0% | 53.6% | 21.0% |
| Beam search + trigram LM | 3 earlier lines | 27.6% | 53.6% | 21.0% |
| Trained model (flan-t5-small) | none | 28.8% | 45.6% | 25.6% |
| Trained model (flan-t5-small) | previous line | 28.1% | 45.4% | 21.0% |
| LLM-guided beam (SmolLM2-360M) | 3 earlier lines | **24.7%** | pending | pending |
| Claude Opus 5.5 (declines count as all wrong) | none | 25.9% | 13.7% | 51% on one page; the other declined |
| Claude Opus 5.5, answered items only | none | 24.1% (264 lines) | 5.8% (109 verses) | |

Everything above the Claude rows runs locally on the Mac with no API.

- **Dictionary only** takes each outline's first dictionary word. **Beam search**
  (`gregg_ocr/beam.py`) picks among each outline's 8 candidates with a trigram language
  model trained on 5.6M words of public-domain English and Gregg book keys
  (`scripts/build_corpus.py`, `train_lm.py`), plus 2,292 outline → word pairs mined
  from the book keys (`scripts/mine_lexicon.py`; *had*, *him*, *made*, *I will not*…).
- **Trained model** (`gregg_ocr/t5.py`) is flan-t5-small fine-tuned on 530k Grascii → English
  lines: 130k real Topaz readings of book lines and 400k synthetic ones, made by writing
  English sentences in Grascii and adding the stroke errors Topaz actually makes
  (`scripts/make_synthetic.py`, `train_seq2seq.py`). Unlike the others it can write words
  no dictionary offered, which is why it's best of the local decoders on Genesis
  (*garden*, *heads*), and also why it sometimes writes non-words (*uponered*).
- **LLM-guided beam** (`gregg_ocr/fused.py`) is the beam search with a small open LLM
  rescoring each partial reading, given the earlier lines. Like the beam it can only choose
  dictionary words. It reads a line in about 10 s on an M4.
- **Claude** is the earlier run (`--llm`); its safety classifier declines some transcripts
  (see below).

**Tuning, all on the 1929 Manual (the dev set, never on test sets).** Manual and FM2 are
excluded from the corpus and the mined lexicon.

| Decoder | Dev word error |
|---|---|
| Dictionary only | 36.4% |
| Beam search (`scripts/tune_beam.py`) | 20.5% |
| Trained model, best of 12k steps | 20.0% |
| Beam, first 150 lines | 18.6% |
| LLM-guided beam, first 150 lines (`scripts/tune_fused.py`; trigram 0.6, LLM 1.0) | 15.4% |

**How to read these numbers.**
- **FM2 is the honest test.** It's held out from everything, and it's obscure 1930s prose. Some lines
  are word-list drills where context can't help.
- **Where the remaining error comes from.** With FM2's *correct* strokes the beam search
  scores 17.7% instead of 28.0%, so Topaz's stroke errors cost about 10 points, and word choice costs
  the rest. The right word is among the 8 candidates for 88% of words (94% with correct
  strokes). That's the ceiling for every decoder except the trained model.
- **Genesis flatters Claude.** It knows the KJV almost by heart and recognizes verses from a
  few words. SmolLM2 has likely seen it too. The local decoders also suffer from KJV vocabulary
  the dictionary lacks (*Cain*, *begat*, *thence*) and from handwriting Topaz reads less well.
- **Your two pages are from the 1929 Manual**, which Topaz was probably trained on and which is
  this project's dev set. Treat them as a demo, not a test.
- **Real handwriting is the open problem.** One photographed page of a personal letter
  (143 words, in `data/handwritten/letter`, not committed): dictionary 94.4%, beam 88.8%,
  LLM-guided 89.5%, trained model 66.4%. The strokes are the bottleneck: Topaz confuses
  outlines that differ only in length (T/D, P/B, F/V: *tried → drew*, *behind → pine*), and
  longhand names break the line they're on.
- Context barely moves the beam. The trigram LM already carries state across a passage, and
  books mostly start a new sentence on a new line anyway.

**Safety-classifier false positives.** Some transcripts are declined (`stop_reason:
"refusal"`, category `bio`) even though they're innocuous; Grascii strings plus odd
candidate words seem to look like encoded text. Server-side fallback (Opus 4.8, Opus 5),
Sonnet 5.5 and Haiku 5.5 decline the same inputs. The local decoders avoid this.

### Transcribe your own pages

```bash
.venv/bin/python scripts/transcribe.py page.jpg                     # beam search (default)
.venv/bin/python scripts/transcribe.py --decoder t5 page.jpg        # trained model
.venv/bin/python scripts/transcribe.py --decoder claude page.jpg    # needs ANTHROPIC_API_KEY
```

Reproduce the comparison with `scripts/eval_topaz.py {--book fm2 | --genesis | --folder data/inputs}
--beam --t5 --fused [--llm]`.

## Layout

```
src/gregg_ocr/      package: data loaders; decoders (beam, fused, t5, decode = Claude);
                    lm (trigram), lexicon (dictionary + mined pairs), synth (synthetic data)
scripts/            download, inspect, baselines, build_genesis_testset, eval_topaz, transcribe;
                    build_corpus, train_lm, mine_lexicon, make_synthetic, train_seq2seq, tune_*
data/               downloaded + processed data (git-ignored)
models/topaz-61m/   Topaz weights + inference code (MIT)
models/             also trigram.pkl, mined_lexicon.json, gregg-t5/ (built by the scripts; git-ignored)
```
