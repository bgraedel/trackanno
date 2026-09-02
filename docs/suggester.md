# The label suggester

```bash
uv run suggest.py --train      # build features, fit, cross-validate, save
uv run suggest.py --report     # print the last report
uv run suggest.py --ablate     # kinematic vs CNN vs both
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
| `cnn` | PCA of the detector's own embeddings | `features.enabled` |

The CNN vectors re-identify individuals at 100%, so they reward memorising
animals over learning behaviour, which makes dropping `cnn` a modelling decision.

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

**Guardrails.** Cross-validation splits on whole movies, since ungrouped folds
would score re-identification rather than behaviour. The model stays silent until
its chance-corrected agreement (Cohen's kappa) clears zero at the 5th bootstrap
percentile and macro-F1 clears `suggest.min_macro_f1`. A share of clips
(`suggest.audit_fraction`) is never suggested on and never trained on, keeping
human-vs-model agreement measurable. With selection on, the labelled part of that
holdout also has to stay a random sample of it, which is what
`suggest.select_audit_share` reserves slots in every batch for.

Where no suggestion appears the cause is one of: a held-out audit clip,
confidence below `suggest.min_confidence`, a group gated off by the report, or a
stale model. `suggest.audit_visible: true` makes the panel state which.

**Repairing orphaned labels.** `clip_id` embeds the track id, so re-running
detection renumbers the tracks and orphans every label written beforehand;
`train` inner-joins on `clip_id` and drops those rows silently. The Model page's
**Repair label links** matches orphans back onto the current clips by movie,
frame window and crop position. It declines ambiguous cases: a nearest clip that
leads the next one by less than `min_margin_px` (3 px, the crop's own movement
between runs) is not close enough to name, two orphans landing on one clip are
both left alone, and a clip already
carrying a newer label is never overwritten. The CSV is backed up before writing,
and unmatched labels are reported rather than dropped.

---
