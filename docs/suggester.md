# The label suggester

```bash
trackanno suggest --train              # build features, fit, cross-validate, save
trackanno suggest --train --rebuild-features
trackanno suggest --report             # print the last report
trackanno suggest --ablate             # what each feature group is worth
trackanno suggest --features-only      # build the clip feature table and stop
trackanno suggest --list-features      # columns by group, with importance
```

A classifier trained on the labels so far shows its prediction beside the next
clip. It never pre-selects a value. The Model page runs the same training
(**Train suggester**) and refits in the background every
`suggest.retrain_every` new labels while you annotate.

Outputs in `work/`: `suggest_model.joblib`, `suggest_report.json`,
`cv_predictions.parquet`, `clip_features.parquet` (reused until its inputs
change, keyed by `clip_features.key`) and `embedding_pca.joblib`.

## Model

**CatBoost** (`suggest.model: catboost`, default) with balanced class weights;
it handles NaN natively, which matters because many features are undefined on
some clips (no beat, a stationary track). `suggest.model: hist` uses
scikit-learn's HistGradientBoosting. `suggest.iterations` (1000),
`suggest.learning_rate` (`auto`) and `suggest.l2_leaf_reg` (3.0) are CatBoost's
own defaults; lower `iterations` if background refits are slow.
`suggest.device` (`cpu`, `gpu`, `auto`) applies to CatBoost only. The CPU is
faster on hand-labelled set sizes.

## Feature groups

`suggest.feature_groups` selects the families the model may see. The Configure
page disables families the project cannot compute.

| group | contents | requires |
|---|---|---|
| `kinematic` | speed, straightness, turning, box area and aspect | nothing |
| `pose` | body and flagellum length, keypoint shape, tip angle, body-anchored translocation and rotation | `task: pose` |
| `waveform` | the beat per 0.75 s window, as the MATLAB pipeline measures it (below) | `task: pose` |
| `beat` | the beat read in every frame, summarised over the clip | `task: pose` |
| `context` | kinematic features over `suggest.context_frames` | `context_frames > 0` |
| `emb_<name>` | an embedding from `suggest.embeddings` | see below |

**Context.** `suggest.context_frames` (0, off) measures the kinematic features
again over that many frames centred on each clip or dense window, clipped to the
track, as `ctx_*` columns. Add `context` to `feature_groups` to use them. A model
refuses to predict under a different `context_frames` than it was fitted with.

**Embeddings** are per-detection vectors, off by default. Each entry has a
`name` (letters and digits) and a `source`:

* `detector`: the YOLO head vectors in `instance_features.npy` (needs
  `features.enabled`, on by default).
* `file`: one parquet or csv table per movie at `path` (containing
  `{movie_id}`, relative to the config) with columns `track_id`, `frame` and
  numeric vector columns. A detection without a row is NaN; a missing file is
  refused.

```yaml
suggest:
  embeddings:
    - {name: det, source: detector}
    - {name: ae, source: file, path: embeddings/{movie_id}.parquet}
  feature_groups: [kinematic, pose, emb_det, emb_ae]
```

One PCA per embedding (32 components) is fitted over every detection of the
project, without labels, and stored in the model. Each clip or dense window gets
the mean, std and frame-to-frame delta of the codes as
`emb_<name>_<i>_{mean,std,delta}`. Prediction uses the stored basis. Embeddings
can re-identify individuals, so judge them by the leave-movie-out score.

**Single columns.** `suggest.feature_columns` (`null`, every column) narrows the
groups to named columns. Every name must exist and belong to a selected group;
the loader suggests the nearest spelling otherwise. `--list-features` prints the
columns by group, marks the ones in use and shows the trained model's top
features per category group. The Model page's report shows the same top-features
table. `--ablate` ignores `feature_columns` and compares whole groups.

## Beat waveform

The `waveform` group measures the beat as the flagella-analysis-pipeline (Fochler et al. 2025)
does: the Fourier spectrum of the flagellum's tangent angle, with amplitude and phase profiles at
the dominant frequency reduced to Legendre coefficients. The `beat` group reads the same profiles in
every frame from a wavelet transform and takes their median over the clip.
Definitions, validation against the MATLAB pipeline and the synthetic Leishmania simulator, and
limits are in [measurements.md](measurements.md).

## Guardrails

* Cross-validation holds out whole movies: each one alone up to 5 movies, 5
  stratified folds above that. Every fold trains on every class, so a class
  needs labels in at least two movies.
* A suggestion is shown only once Cohen's kappa clears zero at the 5th
  percentile of a bootstrap over tracks and macro-F1 clears 0.45. Classes with
  fewer than `suggest.min_per_class` labels are not modelled.
* `suggest.audit_fraction` (0.10) of tracks form the audit holdout: never
  trained on, never shown a suggestion. Agreement on them is the honest measure,
  reported as `audit_accuracy`.
* `suggest.min_confidence` hides suggestions below that confidence.

## Settings

| key | default | meaning |
|---|---|---|
| `suggest.enabled` | `true` | train and show suggestions |
| `suggest.model` | `catboost` | `catboost` or `hist` |
| `suggest.device` | `cpu` | `cpu`, `gpu` or `auto` |
| `suggest.feature_groups` | `[kinematic, pose, waveform, beat]` | families the model may use |
| `suggest.embeddings` | `[]` | embedding inputs |
| `suggest.context_frames` | `0` | context span; 0 is off |
| `suggest.feature_columns` | `null` | narrow to these columns |
| `suggest.retrain_every` | `10` | background refit interval, in labels |
| `suggest.min_per_class` | `5` | smallest class modelled |
| `suggest.audit_fraction` | `0.10` | share of tracks held out |
| `suggest.min_confidence` | `0.0` | hide suggestions below this |
| `suggest.select_rule` | `"off"` | `off` or `margin` ([clip order](annotation.md#clip-order)) |
| `suggest.iterations`, `suggest.learning_rate`, `suggest.l2_leaf_reg` | `1000`, `auto`, `3.0` | estimator settings |

## Repairing orphaned labels

`clip_id` contains the track id, so re-running detection orphans earlier labels,
and training ignores them. The Model page counts them and shows the command:

```bash
trackanno repair-links --config config.yaml --dry-run
trackanno repair-links --config config.yaml
```

It matches orphans to current clips by movie, frame window and crop position. It
declines a match whose runner-up is within 3 px, two orphans landing on one clip,
and a clip that already carries a newer label. The CSV is copied to
`<csv>.bak-N` first; unmatched labels are reported and kept.
