# The label suggester

```bash
trackanno suggest --train      # build features, fit, cross-validate, save
trackanno suggest --report     # print the last report
trackanno suggest --ablate     # what each feature group is worth
trackanno suggest --list-features  # columns by group, with importance
```

Trains a classifier on the labels so far and displays its prediction on the next
clip. With `suggest.select_rule: "off"`, the default, it does not choose which
clip comes next either, so the labelled set stays a random sample and every
statistic computed over it is a statement about the data. Turning selection on
(see *Which clip comes next*, above) spends that property to buy the model's
attention, and writes the trade into every row rather than leaving it to be
inferred.

**CatBoost is the default**, and an ordinary dependency, for two properties of
this data:

* **The label distribution is lopsided.** One state dominates and the interesting
  states are rare, so an unweighted model scores well by predicting the majority.
  `suggest.balance_classes` enables `auto_class_weights="Balanced"`.
* **NaN means undefined**, for example no beat detected or the straightness of a
  stationary track. CatBoost splits on missingness natively; filling with 0.0
  would present those as real extremes, straightness 0.0 being a perfect circle.

**The hyperparameter defaults are CatBoost's own** - `iterations: 1000`,
`depth: 6`, `l2_leaf_reg: 3.0` - rather than numbers picked here, which were
guesses tuned against one dataset where CatBoost's are tuned against many.
`learning_rate: auto` is the one that cannot be written as a number: CatBoost
derives the rate from the row count and the iteration count, and the right rate
for 200 labelled clips is not the right rate for 5000. Give it a number and that
goes away.

The cost is real and worth knowing: 1000 iterations is about **four times the
fitting time** of the 400 this used to default to, on every cross-validation fold
and every background refit. Refits happen on a worker thread and annotation
carries on through them, but a project that retrains often can set
`suggest.iterations` back down.

CatBoost trees are symmetric, so `suggest.depth` alone sets the width at
2<sup>depth</sup> leaves - 64 at the default 6. Past that a few-hundred-clip
hand-labelled set is memorised rather than learned, which is a failure the
reliability gate catches rather than one that reaches the annotator. Feature
importances come from the fit rather than from column shuffling.
`suggest.model: hist` selects scikit-learn's HistGradientBoosting with matched
hyperparameters, `auto` there meaning sklearn's own default rate.

**`suggest.device`** is `cpu`, `gpu` or `auto`, and applies to CatBoost only;
`gpu` with `model: hist` is refused. CatBoost's GPU implementation uses 128
feature borders against the CPU's 254, so the two fit different models on the
same data. It is also slower below roughly 50 000 rows: measured at 1000
iterations over 177 columns, 2.9x slower at 200 rows, 2.2x at 20 000, and 4.1x
faster at 100 000. A hand-labelled set is in the range where the CPU wins.
`auto` uses a GPU where CatBoost sees one, so the same project can be fitted two
ways on two machines; `gpu` raises where there is none rather than falling back.

`suggest.balance_classes` is deliberately **not** a CatBoost default: it
describes this data rather than the estimator, and is on because one behaviour
dominates a behaviour log.

**Feature families.** `suggest.feature_groups` selects what the model may see.
The Configure page disables families this project cannot compute and states why.

| group | contents | requires |
|---|---|---|
| `kinematic` | speed, straightness, turning, box area and aspect | nothing |
| `pose` | body and flagellum length, beat amplitude, body-anchored translocation | `task: pose` |
| `waveform` | beat-shape descriptors, below | `task: pose` |
| `context` | the kinematic features over `suggest.context_frames`, below | `context_frames > 0` |
| `emb_<name>` | an embedding from `suggest.embeddings`, below | see below |

**Context.** `suggest.context_frames` (0, off) adds the `context` family: the
kinematic features again, measured over that many frames centred on each clip or
dense window and clipped to the track, as `ctx_*` columns. Add `context` to
`feature_groups` to use it. The model records the value it was fitted with, and
prediction under another value is refused.

**Embeddings.** `suggest.embeddings` lists per-detection vector inputs, off by
default. Each has a `name` and a `source`:

* `detector`: the YOLO head vectors in `instance_features.npy` (needs
  `features.enabled`).
* `file`: one table per movie at `path` (with `{movie_id}`, relative to the
  config), parquet or csv, with columns `track_id`, `frame` and any numeric
  vector columns (a saved `Unnamed: 0` index column is ignored). A detection
  with no row is NaN. A movie with no file, a file none of whose rows match a
  detection, or empty `track_id`/`frame` cells are refused.

```yaml
suggest:
  embeddings:
    - {name: ae, source: file, path: embeddings/{movie_id}.parquet}
  feature_groups: [kinematic, emb_ae]
```

One PCA per embedding (`suggest.pca_components` wide) is fitted over every
detection of the project, without labels, when the clip features are built, and
saved to `embedding_pca.joblib` and into the trained model with its source.
Dense prediction reads each embedding the model uses from that source in that
basis, whatever the config now declares, and refuses if the data is gone. The
clip table is stamped with its bases and `context_frames`; a model whose table
was rebuilt differently (`--features-only`) refuses to load until retrained.
`--ablate` builds its table in memory. Each clip and dense window gets the mean, std
and frame-to-frame delta of the codes, `emb_<name>_<i>_{mean,std,delta}`.
Cross-validation and the saved model use the same columns. Embeddings can
re-identify individuals, so judge them by the leave-movie-out score.

An old `cnn` entry in `feature_groups` is read as `{name: det, source:
detector}`; models fitted on `cnn*` columns must be retrained.

**Single columns.** `suggest.feature_columns` is the finest of the three cuts:
the columns themselves, by name. `null`, the default, means every column the
families and the scope selected. A list narrows that set rather than replacing
it, so every name has to belong to a family that is still ticked, and the loader
refuses a name that does not, a name no feature table has (offering the nearest
spelling) and an empty list.
`trackanno suggest --list-features` prints the names by family, marks the
ones the current config uses, and shows the trained model's importance per
category group next to its top columns.

Set it in `config.yaml`; the Model page's top-features table says which columns
the last fit leaned on. Leave it `null` rather than writing today's full list, so
a project that has not narrowed anything follows the families wherever they go.

`--ablate` ignores this key and prices whole families, which is what makes its
rows comparable.

**Beat waveform features** implement the low-dimensional description in Geyer,
Howard and Sartori, *Ciliary beating patterns map onto a low-dimensional
behavioural space*, Nature Physics 18, 193-200 (2022). The flagellum is
parameterised by its tangent angle against normalised arclength in the co-swimming
frame; the time average is the static mode, and the remainder is fitted as
`a(s) cos(2 pi f t + phi(s))`. Projecting `a(s)` and `phi(s)` onto shifted
Legendre polynomials gives the axes the paper found sufficient for ~80% of the
variation across every perturbation tested.

| column | paper | meaning |
|---|---|---|
| `wave_freq_hz` | f | beat frequency |
| `wave_amp_mean` | a0 | mean amplitude |
| `wave_amp_asym` | a1/a0 | amplitude asymmetry |
| `wave_amp_para` | a2/a0 | amplitude parabolicity |
| `wave_phase_asym` | phi1 | phase asymmetry |
| `wave_phase_para` | phi2 | phase parabolicity |
| `wave_length_px` | L | flagellum length |

`wave_static_*` is the intrinsic curvature that makes a cell circle rather than
swim straight. `wave_wavelength_ratio` is lambda/L, reported by the paper as
0.97 ± 0.08 across every condition, which makes it a check on the tracking;
measured here, median 1.19 over 30 cells, with the frequency agreeing with the
independent `beat_freq_hz` estimator to within 0.2 Hz.

`wave_fundamental_frac` is the quality readout and a feature in its own right,
not an internal threshold. It runs near 0.47 on intact cells in brightfield
against the paper's >0.9 for isolated axonemes, so read it before trusting a row.
The Machin number requires bending rigidity and drag coefficient and is absent.

**Wave direction** comes twice. `wave_direction` (`+1` = Base->Tip) is read at
the beat's spectral bin, so it needs a beat `spectral.window` can resolve; where
`wave_censored` is 1 it was read at a frequency the cell is not beating at.
`wave_flow_direction` is the same quantity from the phase gradient alone, over
`spectral.flow_frames` (5), with `wave_flow_coherence` as its confidence. Gate
on the coherence, not on `wave_excess`.

**Guardrails.** Cross-validation splits on whole movies, since ungrouped folds
would score re-identification rather than behaviour. The model stays silent until
its chance-corrected agreement (Cohen's kappa) clears zero at the 5th bootstrap
percentile and macro-F1 clears `suggest.min_macro_f1`. A share of tracks
(`suggest.audit_fraction`) is never suggested on and never trained on, keeping
human-vs-model agreement measurable; all clips of a held-out track are held
out. With selection on, audit clips are never ranked, so the labelled part of
that holdout stays a random sample of it.

Where no suggestion appears the cause is one of: a held-out audit clip,
confidence below `suggest.min_confidence`, a group gated off by the report, or a
stale model. `suggest.audit_visible: true` makes the panel state which.

**Repairing orphaned labels.** `clip_id` embeds the track id, so re-running
detection renumbers the tracks and orphans every label written beforehand;
`train` inner-joins on `clip_id` and drops those rows silently. The Model page
counts them, and `trackanno repair-links --config config.yaml` matches orphans
back onto the current clips by movie, frame window and crop position
(`--dry-run` reports without writing). It declines ambiguous cases: a nearest clip that
leads the next one by less than `min_margin_px` (3 px, the crop's own movement
between runs) is not close enough to name, two orphans landing on one clip are
both left alone, and a clip already
carrying a newer label is never overwritten. The CSV is copied to a numbered
`.bak` before writing, and unmatched labels are reported rather than dropped.

---
