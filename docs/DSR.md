# Dynamical Stroke Representation (DSR): status and plan

Research question: do physics-inspired stroke features (geometry, pseudo-kinematics, a
second-order pen model, frequency content, stroke transitions) improve recognition of handwritten
Anniversary Gregg compared with Topaz's image-only recognizer? A negative answer is a valid outcome.

## Built so far (2026-11-04)

| Piece | Where |
|---|---|
| Feature library: skeleton graph, geometry, pseudo-kinematics, pen-model fit, frequency and transition features, feature vectors, per-frame features for the recognizer | `src/gregg_ocr/dynamical_stroke_representation.py` |
| Second-order simulator (exact matrix-exponential discretization + RK4), regularized inversion, frequency response, resonance, Bode plots | same module, sections 4-5 |
| Synthetic variation of a reference path under (ζ, ω) | `vary_path` |
| Unit tests against closed forms (35 passing) | `tests/test_dsr.py` |
| Feature stability and cost study | `scripts/dsr_stability.py` → `runs/dsr/` |

Not built yet: the DSR-enhanced recognizer, the training framework, E1–E6, the dashboard.

### Design decisions

- **Reuse Topaz's preprocessing.** Topaz already finds, deskews, rectifies and normalizes lines
  (`topaz.find_lines`). DSR runs on its line crops, so features line up with its output frames
  (one per 4 px of width; `column_features`). Lengths are in units of the crop height, which
  Topaz scales to the writer's line spacing. That keeps size differences such as T vs D, which
  rotation- or scale-normalized descriptors would erase.
- **Skeleton segments are not Gregg strokes.** The skeleton is split at junctions, so an outline
  is several segments and a circle vowel is often a closed loop. Short spurs (< 1.5 pen widths)
  are pruned, and the staircase pixels that look like junctions are thinned away.
- **Writing direction is assumed.** A skeleton has no direction. Segments run left to right, or
  top to bottom when steep (Gregg's downstrokes). Signed features depend on this rule. For loops,
  whose direction can't be inferred, the signed features are zeroed.
- **Smoothing before measuring.** Arc length and curvature are measured on a Savitzky–Golay
  smoothed path (window ≥ 2.5 pen widths). Measured on the pixel staircase, a slanted line comes
  out up to 8% long.
- **Pseudo-kinematics need a timing assumption.** Three are supported:
  - uniform speed;
  - the two-thirds power law (speed ∝ curvature^(-1/3), Lacquaniti et al. 1983);
  - a minimum-jerk speed profile (Flash & Hogan 1985).

  All are labelled as estimates. With uniform timing, acceleration equals v²κ exactly: the
  "kinematics" are geometry restated.
- **ζ and ω are not identifiable from a still image.** For any (ζ, ω), the inversion
  g = r + (r'' + 2ζωr')/ω² gives a reference the pen would trace exactly as observed. They are
  fitted only against an explicit prior: g is made of straight pieces between the path's
  Ramer–Douglas–Peucker corners. Each fit reports an identifiability score (1 − best/median grid
  error). They are shape descriptors under that prior, not measurements of a writer.
- **The inversion needs regularized derivatives.** In the tests, Savitzky–Golay and smoothing
  splines (with the smoothing chosen by generalized cross-validation) recover g with under a
  tenth of the error of finite differences, at 0.1% positional noise.

### Feature stability (60 printed FM2 lines, 60 handwritten Genesis lines)

The numbers are Spearman ρ across lines between each feature's line-level value before and after
a change. The table shows the median over each group's features.

| Change | Geometry | Kinematic | Dynamic | Frequency | Transition |
|---|---|---|---|---|---|
| Noise σ = 12 (printed / handwritten) | 0.75 / 0.96 | 0.74 / 0.94 | 0.59 / 0.91 | 0.77 / 0.96 | 0.58 / 0.95 |
| Rotate ±3° | 0.86 / 0.86 | 0.78 / 0.79 | 0.75 / 0.77 | 0.92 / 0.90 | 0.86 / 0.88 |
| Scale 0.8 / 1.25 | 0.87 / 0.87 | 0.82 / 0.77 | 0.76 / 0.79 | 0.89 / 0.90 | 0.83 / 0.87 |
| Skeletonizer: Lee / medial axis | 0.92 / 0.95 | 0.80 / 0.80 | 0.74 / 0.85 | 0.95 / 0.96 | 0.91 / 0.93 |
| Smoothing: spline instead of Savitzky–Golay | 0.99 / 0.99 | 0.88 / 0.77 | 0.85 / 0.77 | 1.00 / 1.00 | 1.00 / 1.00 |
| Timing: uniform | 1.00 / 1.00 | 0.94 / 0.91 | 0.85 / 0.91 | 1.00 / 1.00 | 1.00 / 1.00 |
| Timing: minimum jerk | 1.00 / 1.00 | 0.94 / 0.93 | **0.60 / 0.62** | 1.00 / 1.00 | 1.00 / 1.00 |

Cost: 55–65 ms per line on one CPU core, about 25–34 segments per line.

What this says:
1. **Geometry, frequency and transition features are the stable ones.** Most hold ρ ≥ 0.85 under
   realistic image changes and under the choice of skeletonizer.
2. **Curvature statistics are fragile.** Mean and max curvature, turning, curvature rate and
   curvature acceleration fall to ρ 0.3–0.5 on handwriting. The cause is second derivatives of a
   skeleton.
3. **The fitted ζ and ω mostly measure the timing assumption.** Switching to minimum-jerk timing
   drops them to ρ ≈ 0.6, and they're among the least stable features under image noise. The
   tracking error and identifiability score are more stable (ρ ≥ 0.8).
4. **Redundancy.** Fourier descriptor magnitudes at ±m are near-duplicates (|ρ| > 0.95), as are
   length/log-length and mean acceleration/normal acceleration. 14–17 pairs in all, to prune
   before training.
5. **Printed plates are noisier than handwriting under pixel noise.** Engraved hairlines break
   when binarized; handwriting has thicker strokes.

The prior expectation, before any training: geometric and transition features are the plausible
contributors. The dynamical group is the least likely to add information, because under every
timing model it is a deterministic function of the same geometry. Any gain from it would be an
inductive bias, not new information. The experiments below are designed to measure that, not to
assume it.

## Experiment plan

### Stage 0: probe (cheap; run first)

Does DSR carry information about strokes that Topaz's own encoder doesn't already have?

1. Force-align each training line's Grascii label to Topaz's output frames using Topaz's CTC
   output. This gives one stroke label per frame.
2. Train small classifiers (logistic regression or a 2-layer MLP) to predict the frame label
   from (a) Topaz's encoder output, (b) DSR column features, and (c) both.
3. Report frame accuracy overall and on the confusable pairs: T/D, P/B, F/V, K/G, N/M, A/E.
4. Splits: train on book lines, test on held-out books and on Genesis. Genesis has no Grascii
   key, so there we can only align to the v2 decoder's output; that's a weaker check.

If (c) isn't better than (a) beyond seed noise, fusion is unlikely to help and we stop
cheaply, with that as the negative result. This costs hours, not days.

### Stage 1: E1 vs E6, three seeds each

- **E1:** Topaz fine-tuned image-only, with the same data, steps and seeds as the DSR arm. This
  keeps fine-tuning itself from being credited to DSR.
- **E6:** Topaz plus a DSR branch. Column features go through an MLP and are added to the
  encoder input after `proj` (intermediate fusion). The branch's last layer starts at zero, so
  E6 starts out exactly equal to Topaz.
- **Data:** Topaz's book lines (28k, in `data/processed/topaz`), excluding FM2 (test) and the
  1929 Manual (dev).
- **Features:** extracted once per line and cached. That's about 30 minutes of CPU time; on the
  fly it would be about 4 s per batch.
- **Augmentation:** a fixed set of K augmented copies per line, features computed on each copy.
- **Metrics:**
  - stroke error (CER on Grascii) on FM2 and on held-out lines;
  - WER through the v2 decoder on Genesis and the letter (same decoder for every arm);
  - sequence accuracy;
  - mean ± std over seeds.
- **Writer-independent evaluation:** only two handwriting writers exist so far (Genesis and the
  letter), both test-only. Claims about unseen writers are limited until more writers are
  collected.

### Stage 2, only if E6 beats E1

- **E2–E5:** one feature group at a time.
- **Other fusion points:** early (feature maps as extra CNN channels), late (before the CTC
  head), and cross-attention from frames to segment tokens.
- **Losses:**
  - L_geometry: an auxiliary head predicting DSR features from the encoder. It needs no DSR
    at test time.
  - L_motion: applied only to synthetic lines, where the renderer knows the true path and
    timing. Never to scans, which have no temporal ground truth.

### Synthetic variation

`vary_path` redraws a reference path under (ζ, ω). The renderer being built in parallel
(`src/gregg_ocr/render.py`) already has a discrete spring-damped pen lag. The plan:
1. Calibrate it against the (ζ, ω) model.
2. For each setting, check structural preservation: Topaz reads the varied drawing back, and
   the stroke error against the label is measured. Settings that change the reading are
   excluded.

### Compute estimate (M4)

| Stage | Cost |
|---|---|
| Feature caching | ~30 min |
| Stage 0 | a few hours |
| Stage 1 | 6 fine-tuning runs; at ~2 h each for ~3k steps, about 12 h |
| Stage 2 | a further ~24–36 h if justified |

## Limitations to keep in view

- **Training data is printed.** Topaz's labelled lines are engraved plates. Real handwriting
  dynamics appear only in the test sets, so any DSR gain on handwriting has to transfer from
  print or synthetic data.
- **Few handwriting writers.** Two, both test-only. The keyed handwriting collection
  (`data/handwritten/`) is the main lever for every generalization claim.
- **ζ and ω are not measurements.** Under every timing assumption they are model-dependent
  descriptors.
