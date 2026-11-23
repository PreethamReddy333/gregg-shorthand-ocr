# Reading Gregg shorthand with a computer

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Runs locally on a Mac" src="https://img.shields.io/badge/runs-locally%20on%20Apple%20silicon-lightgrey.svg">
</p>

I'm learning **Gregg shorthand**, the curvy script that secretaries, reporters and students wrote
by the million for most of the 20th century. Every attic, archive and second-hand bookshop has
notebooks full of it, and almost nobody can read them anymore. I wanted a tool that could look at
a page of Gregg and tell me what it says.

This repo is everything I built and measured on the way: a pipeline that reads Gregg from images,
four ways of turning the strokes into English, a renderer that **writes** English as Gregg, a
reading game, and a lot of honest benchmarks, including the experiments that didn't work.

**The short version:**

- **Printed textbook Gregg: works.** About 23% word error on a held-out 1930s book, fully local.
- **Clean handwriting: half works.** 41–45% word error on one writer's handwritten Genesis.
- **Messy real-world photos: not yet.** Phone photos of notebooks, pencil pages and tablet sketches
  mostly come out as fluent nonsense. The section [What I learned](#what-i-learned) explains why,
  and what would fix it.

If you have **handwritten Gregg with a known transcription**, that's the most valuable thing you
could contribute. See [Help wanted](#help-wanted).

---

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Benchmarks](#benchmarks)
- [Writing English as Gregg](#writing-english-as-gregg)
- [Things I tried that didn't help](#things-i-tried-that-didnt-help)
- [Real-world photos: a field report](#real-world-photos-a-field-report)
- [What I learned](#what-i-learned)
- [Repo layout](#repo-layout)
- [Help wanted](#help-wanted)
- [Credits and licenses](#credits-and-licenses)

## How it works

```
 photo / scan ──► Topaz-61M ──► Grascii strokes ──► decoder ──► English
                 (vision model)   "E-S-P-R-E-M"    (dictionary + language model)
```

1. **Image → strokes.** [Topaz-61M](https://huggingface.co/amongglue/topaz-gregg-shorthand-61m)
   finds the lines on a page and reads each outline into
   [Grascii](https://github.com/grascii/grascii), an ASCII spelling of Gregg (`K-A-T` is "cat").
2. **Strokes → English.** This is the hard part. Gregg drops most vowels, and many words share one
   outline: `T` alone can be *it* or *at*. So the strokes go to a decoder that picks words from
   context. I built and compared four:
   - **Dictionary only:** the first dictionary word for each outline. The floor.
   - **Beam search + trigram language model** (`gregg_ocr/beam.py`): picks among each outline's 8
     candidates. The LM is trained on 5.6M words of public-domain English and Gregg book keys.
   - **Trained model** (`gregg_ocr/t5.py`): flan-t5-small fine-tuned on 543k Grascii → English lines.
     Unlike the others, it can produce words the dictionary never offered.
   - **Claude** (`gregg_ocr/decode.py`): an LLM given the strokes and dictionary candidates. Needs
     an API key. Everything else runs locally.

Everything here targets **Gregg Anniversary** (1929), the edition most surviving notes use.

## Quick start

```bash
git clone https://github.com/PreethamReddy333/gregg-shorthand-ocr
cd gregg-shorthand-ocr
python3 -m venv .venv
.venv/bin/pip install -e ".[train,dsr]"

# the Topaz reader (MIT), about 250 MB
.venv/bin/hf download amongglue/topaz-gregg-shorthand-61m --local-dir models/topaz-61m
.venv/bin/pip install -r models/topaz-61m/requirements.txt
```

**Read a page:**

```bash
.venv/bin/python scripts/transcribe.py page.jpg                    # strokes + English (beam search)
.venv/bin/python scripts/transcribe.py --decoder none page.jpg     # strokes + dictionary candidates only
.venv/bin/python scripts/transcribe.py --decoder claude page.jpg   # needs ANTHROPIC_API_KEY
.venv/bin/python scripts/read_photo.py notebook.jpg                # fallback for messy phone photos
```

The beam and trained-model decoders need models you build once on your own machine. The
trained weights aren't in this repo.

```bash
.venv/bin/python scripts/download_data.py      # ~4 GB download, ~1-2 GB kept
.venv/bin/python scripts/build_corpus.py && .venv/bin/python scripts/train_lm.py   # trigram LM
.venv/bin/python scripts/mine_lexicon.py                                           # outline → word pairs from book keys
.venv/bin/python scripts/make_synthetic.py && .venv/bin/python scripts/train_seq2seq.py   # trained decoder
```

**Write English as Gregg, or play the reading game:**

```bash
.venv/bin/python scripts/write_gregg.py "Thank you for your letter." -o note.png --key
.venv/bin/python scripts/gregg_web.py          # http://localhost:8765, type English, watch it turn into Gregg
open web/gregg-comprehension/index.html        # read outlines and type them against the clock
```

## Benchmarks

All numbers are **word error rate**: lower is better, and 0% is perfect. The strokes come from
Topaz in every row; only the decoder changes. I tuned on the 1929 Manual only and never on a test
set.

### The test sets

| Test set | What it is | Size |
|---|---|---|
| **FM2** | *Functional Method* book 2, printed. Held out from everything. Obscure 1930s prose. | 271–300 lines |
| **Genesis** | Nathan Tonning's handwritten Gregg of Genesis 1–5, keyed to the KJV ([CC0](https://github.com/NPTprogrammer/GreggShorthandStuff)). One writer, neat. | 119 verses |
| **Letter** | One phone photo of a real handwritten letter (private, not included). | 143 words |
| **Manual pages** | Two photographed pages of the 1929 Manual. This is the dev book, so treat it as a demo. | 176 words |

### Off-the-shelf baseline

The existing `grascii/gregg-vision-v0.2.1` word model, 200 words each:

| Edition | Exact strokes | Right word in top-5 |
|---|---|---|
| Pre-Anniversary (its training data) | 95.0% | 94.5% |
| Anniversary (unseen edition) | 60.0% | 60.0% |

Hence Topaz, which is trained on Anniversary books.

### Topaz on its own

| Test set | Stroke error | Outline error | Word error, top-1 dictionary word |
|---|---|---|---|
| FM2, printed | 10.2% | 24.6% | 43.0% |
| Genesis, handwritten | n/a (no stroke key) | n/a | 61.5% |

### Strokes → English: every decoder

| Decoder | Context | FM2 (printed) | Genesis (handwritten) | Manual pages |
|---|---|---|---|---|
| Dictionary only | none | 41.3% | 61.5% | 45.5% |
| Beam search + trigram LM | 3 lines | 27.6% | 53.6% | 21.0% |
| Trained model v1 | previous line | 28.1% | 45.4% | 21.0% |
| **Trained model v2** | 2 lines | **23.4%** | 44.8% | **18.2%** |
| LLM-guided beam (SmolLM2-360M) | 3 lines | 24.7% | | |
| Claude Opus 5.5 | none | 25.9% | 13.7%* | 51% on one page; declined the other |

\* Genesis flatters Claude: it knows the KJV nearly by heart and recognises verses from a few
words. On printed FM2, my local trained model beats it.

On the dev book (1929 Manual), from worst to best:

| Dictionary | Beam | Trained model | Trained v2 + context | LLM-guided beam (150 lines) |
|---|---|---|---|---|
| 36.4% | 20.5% | 20.0% | 16.6% | 15.4% |

**Where the remaining error lives.** Give the beam search FM2's *correct* strokes and it scores
17.7% instead of 28.0%. So Topaz's misreadings cost about 10 points, and choosing between words
costs the rest. The right word is among the 8 candidates for 88% of words.

### Fine-tuning Topaz

Topaz fine-tuned on the training books (never FM2 or the Manual), with and without synthetic
handwriting from my renderer. Word error after the trained decoder v2:

| Test set | Original Topaz | Plain fine-tune | + synthetic writers | + sigma-lognormal pens | + DSR pens |
|---|---|---|---|---|---|
| FM2, printed | 23.4% | 23.3% | 23.8% | 23.1% | 23.9% |
| FM2 stroke error | 10.2% | 10.0% | 10.5% | 10.4% | 10.4% |
| **Genesis, handwritten** | 44.8% | **41.3%** | 41.1% | 41.3% | 41.3% |
| Letter | 65.0% | **55.9%** | 65.0% | 65.7% | 62.2% |
| Manual pages | 18.2% | 16.5% | 18.2% | 17.6% | 17.6% |

Plain fine-tuning gives **−3.5 points on handwritten Genesis**. None of the synthetic-handwriting
variants add anything beyond that: every pairwise difference on Genesis is within about ±1.5
points (95% bootstrap). The letter moves a lot, but it's a single page.

### The handwriting gap, in one table

| Same pipeline, trained decoder v2 | Word error |
|---|---|
| Printed book (FM2) | 23.4% |
| Neat handwriting, one writer (Genesis) | 41.3% (fine-tuned) |
| A real handwritten letter, phone photo | 55.9% (fine-tuned) |
| Phone photos of notebook pages, no key | unreadable (see the field report below) |

## Writing English as Gregg

`scripts/write_gregg.py`, `scripts/gregg_web.py` and `gregg_ocr/render.py` go the other way: type
English, get Gregg.

The renderer looks words up in the Grascii dictionary and its phrase forms. Words it can't find
are spelled from their pronunciation (CMU dictionary). Each stroke is drawn as an arc and joined
by the Manual's rules: circles inside curves and outside angles, S and TH turned to suit their
neighbours, O/OO hooks, disjoined strokes, capitals marked with two dashes. I tuned stroke
proportions until Topaz could read the drawings back (`tune_render.py`), then went stroke by
stroke (`stroke_check.py`).

| Drawn from correct strokes, read back by Topaz (FM2, 150 lines) | Stroke error |
|---|---|
| Clean textbook style | 15.0% |
| Random synthetic writers (`--writer N`) | 25.5% |
| Topaz on the real printed lines, for comparison | 10.2% |

It's useful for practice sheets: type a passage, print it, copy it by hand. Every page you copy
becomes a **keyed handwritten line**, exactly the data this project needs.

**Can't rules build Gregg words on their own?** I checked. Turning each sound into its stroke
(with blends) matches the textbook outline for only **17.2% of 5,000 words**. The rest need the
Manual's habits: dropping minor vowels, prefix and suffix signs (*con-* → K, *-ing* → a dot), and
the abbreviating principle (*inspecting* → N-S-P-dot). The ~300 brief forms are simply memorised.
That's why the renderer uses the 34k-word dictionary first and rules only as a fallback.

## Things I tried that didn't help

I want these on record, because negative results save other people weeks.

- **Synthetic handwriting to fine-tune Topaz.** Four ways of varying my renderer's drawings:
  per-writer proportions, grid distortion
  (Wigington et al., ICDAR 2017), sigma-lognormal motor
  models (Plamondon's kinematic theory), and a second-order pen. All landed at the same Genesis
  score as plain fine-tuning. My drawings vary in ways real hands don't, and not in the ways they do.
- **Dynamical Stroke Representation (DSR).** A feature library that turns line images into
  stroke-level geometry, pseudo-kinematics and a fitted pen-dynamics model (`docs/DSR.md`,
  35 tests). The stability study found geometry, frequency and transition features stable
  (ρ ≥ 0.85 under noise, rotation and scale). The fitted pen damping and frequency mostly measure
  my *timing assumption*: they drop to ρ ≈ 0.6 when the assumed timing changes. No recognizer
  has been trained on DSR yet.
- **More context for the beam search.** Three earlier lines instead of none moved FM2 by 0.4
  points. The trigram already carries the passage along.
- **Thinning and multi-scale reading on photos** (`read_photo.py`). It rescues pages where
  Topaz's line finder returns nothing. But on a photo the line finder *can* handle, it's worse:
  70–80% vs 65% word error on the letter.

## Real-world photos: a field report

I fed the pipeline about twenty images people actually have: typing-test screenshots, phone
photos of notebooks, a crumpled ruled page, a yellowed 1915 page, green-ink line crops, pencil,
and white-on-black tablet drawings. No keys, so no word error rates, but the pattern was clear:

| Kind of image | Result |
|---|---|
| Gregg drawn by software (typing-test screenshots) | **Perfect.** "What shall I do? He could not do that." |
| Neat handwriting, photographed flat | Some phrases right; most lines change reading when zoomed |
| Notebook photos: ruled, curved, thick pen, small outlines | Empty or unstable readings. Even Claude's vision model declined to guess. |
| Pencil, crumpled paper, old yellowed pages | Mostly noise after cleanup |
| White-on-dark tablet drawings | A few words; proportions drift with finger drawing |
| A German page | DEK (German shorthand), not Gregg. Topaz can't read it. |

Two lessons. Don't call a page "not Gregg" too fast: 1915-era writers use brief forms (comma-like
TH, a tick for *it*, dots for *a*) that look foreign at first. And the most common failure isn't
the decoder. It's the strokes.

## What I learned

**Gregg encodes meaning in relative size.** N and M, T and D, K and G, and the A and E circles
differ only in length or size. Topaz learned absolute sizes from printed plates, where every N is
about the same number of pixels. Real hands write at their own scale, so the same outline reads
differently at 1.5× and 2.5× zoom. I watched that happen on page after page.

**Small stroke errors become most words wrong.** An outline is about three strokes. At the 25–45%
stroke error Topaz makes on handwriting, most outlines have at least one wrong stroke. A wrong
stroke is usually a *different real word*, and then the language model writes a fluent sentence
around it.

**People learn Gregg differently from how the model did.** A student learns *rules* ("M is long
compared with N") from a few hundred examples, with a lifetime of knowing what pens and paper do.
They can also write the outline themselves, and they go back and re-read when a sentence makes no
sense. The model got examples only, one-way, from print. Ideas worth trying, roughly by value:

1. **Real keyed handwriting.** Even 50–100 lines per writer, to fine-tune and to measure.
   Synthetic variation didn't substitute for it.
2. **Normalise each writer's scale** before reading, by measuring their typical stroke length and
   circle size.
3. **Whole-word matching against rule-drawn outlines** (analysis by synthesis): compare a word
   blob with the renderer's drawings of likely words, so size is judged within the word.
4. **Close the loop.** Let the decoder send doubtful words back for a second look.
5. **Start from a big pretrained vision model** instead of a small one trained from scratch.

**The Claude decoder has a quirk.** Some transcripts get declined by a safety classifier (Grascii
plus odd candidate words seems to look like encoded text). Every model tier declines the same
inputs. The local decoders don't have this problem.

## Repo layout

```
src/gregg_ocr/
  beam.py, lm.py, lexicon.py      beam search, trigram LM, dictionary + mined outline→word pairs
  t5.py, fused.py, decode.py      trained decoder, LLM-guided beam, Claude decoder
  render.py, synth.py             drawing Gregg; synthetic training data; pronunciation speller
  augment.py, pen_models.py       augmentation and motor models for fine-tuning Topaz
  dynamical_stroke_representation.py   DSR feature library (docs/DSR.md)
  data.py, safe_pickle.py, text.py
scripts/
  transcribe.py, read_photo.py           read pages and photos
  eval_topaz.py, build_genesis_testset.py, baseline_gregg_vision.py   benchmarks
  build_corpus.py, train_lm.py, mine_lexicon.py, make_synthetic.py, train_seq2seq.py, tune_*.py
  finetune_topaz.py, make_synth_lines.py, stroke_check.py, dsr_stability.py
  write_gregg.py, gregg_web.py           English → Gregg
web/gregg-comprehension/                 the reading game (one HTML file)
tests/                                   DSR numerics against closed forms
docs/DSR.md                              DSR design, stability results, experiment plan
```

`data/`, `models/` and `runs/` are git-ignored. They hold downloads, trained weights and results.

## Help wanted

- **Handwritten Gregg with its English.** Your own practice pages, old dictation notes with the
  book key, anything. Put a page at `data/handwritten/<name>.png` with `<name>.txt` next to it, and
  `eval_topaz.py --folder data/handwritten` scores it. Open an issue if you're willing to share.
- **Simplified, Diamond Jubilee or Series 90 data.** Many surviving notes use later editions, which
  Topaz doesn't read.
- **Any of the five ideas in [What I learned](#what-i-learned).** Number 3 especially is a
  self-contained project.

## Credits and licenses

This project's code is [MIT](LICENSE). It stands on:

- **Topaz-61M** by amongglue (MIT): the image → Grascii reader, downloaded separately.
- **Grascii** by chanicpanic (MIT): the Gregg spelling scheme, dictionary and word datasets. The
  outline images in the reading game come from it (`web/gregg-comprehension/THIRD_PARTY_LICENSE.txt`).
- **Genesis in Gregg** by Nathan Tonning (CC0).
- **Gregg-1916** word dataset (Zhai et al., 2018), CMU Pronouncing Dictionary, WikiText-103,
  flan-t5-small, SmolLM2.
- And John Robert Gregg, whose 1929 Manual is still the best-designed writing system I've ever
  tried to learn.
