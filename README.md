# trackanno

Behaviour annotation for individual tracked organisms in large grayscale
microscopy movies (`.tif`, `.nd2`), with the labels mapped back onto the original
recording.

The unit of work is a **clip**: `clips.frames` consecutive frames of one track,
cropped around that track. The clip loops, one keypress labels it, and the view
advances.

## Install

```bash
uv tool install --with catboost .     # or git+https://github.com/<you>/trackanno
trackanno                             # the whole pipeline in one window
```

`uv tool install` puts `trackanno` on PATH in an isolated environment. Add
`--with catboost` for the label suggester; without it the suggester falls back to
scikit-learn.

From a clone:

```bash
uv sync                    # add --extra cuda for GPU torch
uv run gui.py
```

Each step is also a CLI entry point, for a headless GPU box or a batch script:

```bash
uv run prepare.py          # 1. tiled YOLO detection, tracking, clip cache
uv run annotate.py         # 2. the annotation screen
uv run suggest.py --train  # 3. train the label suggester
uv run predict.py          # 4. label every clip with the trained model
uv run render.py           # 5. labelled boxes back onto the original movie
```

Both paths call the same functions (`prepare_all`, `train`, `predict_all`,
`render_all`), so the GUI and the CLI cannot drift apart. `uv tool install` also
exposes them as `trackanno-prepare`, `trackanno-annotate`, `trackanno-suggest`,
`trackanno-predict` and `trackanno-render`.

Everything is driven by `config.yaml` in the project folder.

---

## 0 · The workflow window

```bash
trackanno                      # reopens the last project
trackanno path/to/project
```

Seven steps down the left, in the order the work happens. Each carries its state
(`*` done, `>` next, `!` something missing) and a count.

| step | purpose |
|---|---|
| **1 Project** | pick the folder to work in |
| **2 Configure** | point it at the movies, choose the labels |
| **3 Detect & crop** | find the animals, cut them into clips |
| **4 Annotate** | watch each clip, press one key |
| **5 Model** | train on the labels, suggest the next one |
| **6 Distribution** | inspect the label distribution before exporting |
| **7 Export** | put the labels back onto the original movies |

Each page states its prerequisites up front and offers a **Next:** bar at the
bottom.

**A project** is a folder holding `config.yaml`, `work/` (the clip cache) and
`annotations.csv`. *New project* writes a fully commented `config.yaml` with the
paths filled in.

**Configure** has four tabs. *Basics* holds the fifteen settings that must be
decided (movie location, model, clip length); *Labels* is the button editor;
*Keypoints* is the pose table, active only for `task: pose`; *Advanced* holds
detector, tracker and cropping parameters. Settings marked `*` describe how the
clips were cut, so changing one requires re-running step 3, which the page
states.

**Settings autosave** ~0.7 s after typing stops, and navigating away flushes.
Values are rewritten in place, line by line, preserving every comment and the
file order, so the two thirds of `config.yaml` that document each knob survive.
Each autosave is parsed by the real loader before it replaces anything; an edit
that would not load is refused with the loader's own message and the file is
untouched. Paths that resolve to nothing are marked amber and annotated, since a
model on an offline share should not block unrelated edits.

**Theme.** `gui/theme.py` holds one palette and a stylesheet generated from it.
Widgets request a role (`muted`, `danger`, `heading`) rather than a colour.
`ui.theme` takes `dark` (default), `light` or `system`.

**Long jobs run on a worker thread.** Detection reports movie, stage and frame
count; Cancel takes effect at the next frame.

---

## 1 · `prepare.py`: detection and the clip cache

```bash
uv run prepare.py --list                       # visible movies
uv run prepare.py
uv run prepare.py --movies MOVIE_A --force     # redo one
uv run prepare.py --limit-frames 100           # smoke test
uv run prepare.py --self-test --tile 320       # check the tiling merge
```

**Device.** `detection.device: auto` selects the first CUDA device, then MPS,
then CPU, and logs the choice. A config naming a specific GPU is a claim about
the machine it was written on, so an absent device is reported and the run
continues on CPU. The Configure page reads the available hardware on a worker
thread, since importing torch costs a second or two.

**Tiled inference.** Each frame is cut into overlapping tiles (`tiling.tile`,
`tiling.overlap`), run as one batch, mapped back to full-frame coordinates and
merged with a class-agnostic NMS. Detections clipped by an interior tile border
are dropped in favour of the neighbouring tile's complete copy, with a safety net
that re-admits any object too large to be complete in any tile. Frames fitting a
single tile skip the mechanism.

**Tracking.** `model.track()` owns the inference stream and would track tiles
rather than frames, so ultralytics' `BYTETracker` is driven directly with the
merged full-frame boxes. `tracking.backend: greedy` selects a nearest-centroid
tracker, adequate at 100 fps where per-frame motion is ~1 px.

**Output** in `work/<movie_id>/`:

| file | contents |
|---|---|
| `tracks.parquet` | every detection, in original full-frame coordinates |
| `clips.parquet` | one row per clip, with its crop rectangle and scale |
| `clip_boxes.parquet` | overlay geometry (target and neighbours) in cache coordinates |
| `clips.dat` | pre-cropped pixels, uncompressed `uint8` memmap |
| `thumbs.dat` | optional full-frame thumbnails for the minimap |
| `norm.json` | the movie's fixed intensity mapping |
| `prepare_done.json` | resume marker; re-running skips finished movies |

**The cache is uncompressed** so that loading a clip is one contiguous ~1 MB read
with no decoding. Keep `paths.work_dir` on a local disk; over a network share
the per-clip latency dominates, and `prepare.py` warns on a UNC path.

**Crop geometry.** The crop is centred on the track and sized to contain the
animal on every frame of the window plus `clips.pad` of margin, which is what
lets the GUI re-centre and zoom at display time from one cache. Clips needing
more than `clips.crop_size` px are downscaled with their `scale` recorded, so
every cached pixel maps back to an exact original coordinate. The crop is
zero-padded where it overhangs the movie border; clamping it inside would park
every border-swimming animal against the edge of its own crop.

**One commit per movie.** Pixels, manifest, box table and cache header are
written beside their real names and renamed into place at the end, so a
cancelled or killed run leaves the previous cache byte-for-byte intact. The
resume marker is cleared before any of it starts, so an interrupted movie is
retried. `ClipStore` independently refuses any movie whose manifest disagrees
with its header or whose `clips.dat` is short, since pairing a manifest row with
another clip's pixels looks normal during annotation.

### Pose models

Set `detection.task: pose`. A checkpoint stores `kpt_shape`, so the keypoint
count is read from the model; it stores no names, so placeholders `kp0`, `kp1`,
… are generated and can be renamed under `keypoints.nodes`.

**Keypoint order is the contract.** Pose features address keypoints by position:
**0** is the head, **1** the base and body-frame origin, **1** onwards the
flagellum chain. A model predicting them in a different order still produces
body-length and beat numbers, and they are meaningless. The names are the only
written record of that ordering.

The Configure page's **Keypoints** tab reads the count from the model file, takes
the names, and propagates a rename through the skeleton links. *Connect in order*
wires 0-1, 1-2, …, the shape of a flagellum chain. A `keypoints.nodes` that
disagrees with the checkpoint raises, since the wrong count misaligns every
keypoint column. `edges` cannot be inferred, stays empty until written, and
affects display only.

### ND2 files

ND2 movies are read natively. They carry P (multipoint), T, C and Z axes and are
reduced to one 2-D time series:

* each position `P` becomes its own movie, id `<stem>__P00`, `__P01`, …
  (`nd2.positions: all` or a list);
* `nd2.channel` takes an index or a channel name (`"BF"`);
* `nd2.z` takes an index, or `max` / `mean` / `min` for a projection;
* `nd2.fps: auto` reads the frame interval from the ND2 metadata.

TIFF carries no frame rate, so set `tiff.fps`.

### Annotation volume

`clips.window_mode`:

| value | meaning |
|---|---|
| `first` | only the movie's first window, one clip per track present there |
| `first_per_track` | the first full window of each track, wherever it starts |
| `all` | every non-overlapping window of every track (default) |

`clips.stride` sets the spacing (`= clips.frames` gives no overlap) and
`clips.max_clips_per_track` caps per-track volume.

---

## 2 · `annotate.py`: the annotation screen

```bash
uv run annotate.py
uv run annotate.py --backend ndv         # embed ndv.ArrayViewer
uv run annotate.py --csv other.csv       # a second annotator's log
```

The screen the workflow window hosts on its Annotate page, in a window of its
own.

`ui.backend: simple` (default) is a built-in player and the only backend that
draws the marker as a separate vector layer with centring and zoom as view
transforms. `ndv` takes a finished array, so both are burned into pixels there.

**One keypress labels and advances.** With `ui.auto_advance: true`, picking a
value in a single-choice group commits the row and moves on once every
`required: true` group is filled. Ticking a multi-choice box never advances.

### Shortcuts

| key | action |
|---|---|
| `1`…`5`, `q`/`w`/`e` | category values (defined per option in `config.yaml`) |
| `→` / `Enter` | commit and next clip |
| `←` | previous clip |
| `Tab` | next unannotated clip |
| `Ctrl+U` | first unannotated clip |
| `Space` | play / pause |
| `o` | marker on/off |
| `s` | cycle the view: centred → smooth → locked |
| `c` | outline neighbouring tracks |
| `+` / `-` / `0` | zoom in / out / reset |
| `m` | context minimap |
| `u` | mark uncertain |
| `Backspace` | clear this clip's annotation |
| `Ctrl+Z` | undo the last annotation |
| `Ctrl+S` | force-save the current clip |

All bindings are configurable under `keys:`. The config refuses to start if a
category shortcut collides with a navigation one.

**The marker** identifies the one animal being judged. With ~50 organisms per
movie a padded crop usually contains a neighbour; `c` outlines those on demand.
It is painted as vector graphics over the grayscale frame, four amber corner
brackets with a dark halo, one hairline wide at any zoom, with no line across
the flagellum. On a frame where the detector missed the animal it is drawn
dimmed and dashed at the last known position.

**View modes (`s`)** all read the same cached pixels, so switching is free:

* **centred** (default), one view for the whole clip, centred on the median of
  the track's own path. All motion visible, and the only mode in which
  "swimming" and "immotile" differ.
* **smooth**, following a moving average of the path (`ui.smooth_window`), so
  drift is removed while the beat survives.
* **locked**, re-centred every frame. Best for body shape and flagellar beat;
  translation is invisible by construction.

Centring is a display transform with the outside padded, so a clip cropped at the
movie border still shows the animal at the exact centre. Frames where the track
dropped out are filled from the nearest detected frame. `ui.view_zoom` and
`+`/`-`/`0` magnify into the crop as a transform, resampling nothing.

If the three modes look identical, the cache is the reason: a cache cut with
`clips.follow: track` and a tight `clips.pad` has already removed the
translation. Pressing `s` reports this with the numbers for that clip. Re-cut
with `clips.follow: fixed` and a larger `pad` to recover it.

### Curvature kymograph

On a `task: pose` project a panel under the clip shows the flagellum's signed
curvature along arclength (vertical, base at top) against time (horizontal). A
travelling wave appears as diagonal stripes, a standing wave as vertical ones, a
mistracked frame as a gap. Clicking scrubs the clip to that frame.

Curvature is a second derivative, and finite differences on seven noisy points
are dominated by tracking jitter, so a smoothing spline is fitted through the
keypoint chain once per frame and differentiated analytically. The chain order
comes from `keypoints.edges` (the longest path through that graph), since walking
a bend from the opposite end negates it.

**Sign convention.** A 2D image cannot distinguish a cell from its mirror image;
the two give curvature fields differing by an overall factor of -1. Every even
functional of curvature is mirror-invariant, so amplitude, spectrum, wave speed
and propagation direction cannot resolve it. The one odd quantity with a physical
interpretation is the static mode, the time-averaged shape remaining once the
beat cancels. `kymo_sign: canonical` flips each clip to make that mode positive,
subject to two guards: the mode must be large relative to the beat, and the two
halves of the clip must independently agree on its sign. Where either fails the
panel reports the clip as undecided and leaves it as measured.

| key | meaning |
|---|---|
| `ui.show_kymograph` | show the panel (default `true`) |
| `ui.kymo_field` | `curvature` (default) or `tangent` |
| `ui.kymo_sign` | `canonical` (default), `raw` or `absolute` |
| `ui.kymo_interpolate` | interpolate between measured nodes (default `true`) |
| `ui.kymo_smooth_px` | spline tolerance against the keypoints, px (default `0.6`) |

These take effect on the open annotator without reloading the project. The colour
scale is diverging with neutral grey at zero.

**Resuming.** The CSV is the state: any clip whose `clip_id` appears in it is
done. On startup the GUI jumps to the first unannotated clip
(`ui.resume: last_position` returns to the previous position). Every annotation
is appended and `fsync`ed immediately, so a crash costs at most the clip on
screen. Annotated clips reopen with their labels loaded, and edits rewrite the
CSV atomically.

### The CSV

```
clip_id, movie_id, source_file, series, channel, track_id, f_start, f_end,
crop_x0, crop_y0, crop_w, crop_h, scale,
<one column per category group>, uncertain, notes, annotator, timestamp_utc, dwell_s
```

Multi-choice groups are `|`-joined. The identity and geometry columns, with
`tracks.parquet`, map any label back onto exact pixels in the original movie:

```python
original_x = crop_x0 + cache_x / scale
original_y = crop_y0 + cache_y / scale
```

---

## 3 · `suggest.py`: the label suggester

```bash
uv run suggest.py --train      # build features, fit, cross-validate, save
uv run suggest.py --report     # print the last report
uv run suggest.py --ablate     # kinematic vs CNN vs both
```

Trains a classifier on the labels so far and displays its prediction on the next
clip without pre-selecting it, so labels stay statistically independent of the
model trained on them.

**CatBoost is the default**, for two properties of this data:

* **The label distribution is lopsided.** One state dominates and the interesting
  states are rare, so an unweighted model scores well by predicting the majority.
  `suggest.balance_classes` enables `auto_class_weights="Balanced"`.
* **NaN means undefined**, for example no beat detected or the straightness of a
  stationary track. CatBoost splits on missingness natively; filling with 0.0
  would present those as real extremes, straightness 0.0 being a perfect circle.

CatBoost trees are symmetric, so `suggest.depth` alone sets the width at
2<sup>depth</sup> leaves. The default 5 is 32 leaves; 6 is 64, which overfits a
few-hundred-clip hand-labelled set quickly. Feature importances come from the fit
rather than from column shuffling. `suggest.model: hist` selects scikit-learn's
HistGradientBoosting with matched hyperparameters, and is the automatic fallback
when catboost is absent.

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
human-vs-model agreement measurable.

Where no suggestion appears the cause is one of: a held-out audit clip,
confidence below `suggest.min_confidence`, a group gated off by the report, or a
stale model. `suggest.audit_visible: true` makes the panel state which.

**Repairing orphaned labels.** `clip_id` embeds the track id, so re-running
detection renumbers the tracks and orphans every label written beforehand;
`train` inner-joins on `clip_id` and drops those rows silently. The Model page's
**Repair label links** matches orphans back onto the current clips by movie,
frame window and crop position. It declines ambiguous cases: two candidate clips
equidistant from the old crop centre are both left alone, and a clip already
carrying a newer label is never overwritten. The CSV is backed up before writing,
and unmatched labels are reported rather than dropped.

---

## 4 · `predict.py`: label every clip with the model

```bash
uv run predict.py                          # every clip -> predictions.parquet
uv run predict.py --coverage manifest      # only clips cut for annotation
uv run predict.py --include-labelled       # also score clips judged by hand
```

Labels the whole dataset in one pass for export, plotting, or comparison against
the hand labels. Three rules:

* **Predictions never reach the annotation CSV.** They go to
  `predictions.parquet`, or the next model trains on its own output and the
  ground truth is lost. Every row carries `human_labelled`.
* **A group the report gated off is skipped**, and named as such. The decision
  comes from the saved report rather than from the presence of a fitted model.
* **The file is replaced atomically**, since everything downstream reads it as
  authoritative.

`predict.coverage` selects the clips. `manifest` scores exactly the annotation
clips, leaving frames between two annotation windows unlabelled and unpainted.
`dense` (default) lays its own grid over every frame of every track, so a
`clips.stride` of 50 no longer leaves 49 frames blank in the exported movie. It
reads no pixels and re-detects nothing; features come from `tracks.parquet`, the
saved embeddings and the frame rate. A window too thin to measure still gets a
row carrying `n_present` and `skip_reason`, since an absent row is
indistinguishable from a declined one.

The Model page runs the same code with a progress bar. The CNN PCA is loaded
from beside the model and never refitted, since a refit yields a different
orthogonal basis silently, invalidating the model's input space.

---

## 5 · Distribution

One bar chart per category group, with a text summary flagging:

* values with nothing labelled yet;
* values below `suggest.min_per_class`, dropped at training time;
* values appearing in only one movie, unscoreable by grouped cross-validation;
* values present in the CSV but no longer in the config;
* any value exceeding 90% of the labels, where a constant predictor scores well.

---

## 6 · `render.py`: labelled boxes on the original movie

```bash
uv run render.py --list                          # movies with labels
uv run render.py --movie MOVIE_A
uv run render.py --group quality --hide-unlabelled
uv run render.py --source both                   # hand labels and predictions
uv run render.py --movie MOVIE_A --tif --max-frames 200
```

Reads `tracks.parquet` and the annotation CSV, and colours each track by the
category of the clip covering that frame, using the colours from `config.yaml`.
Works on a partially annotated CSV. Unlabelled tracks are drawn grey or hidden.
Output goes to `render.out_dir`.

`--source` selects `labels` (default), `predictions` or `both`. The categories
already own the colours, so provenance is carried by line style instead: a model
value is drawn as a dashed outline with its confidence beside it, and the output
file name records the source, so a model-coloured movie cannot overwrite a
hand-labelled one.

---

## Tests

```bash
uv run pytest
uv run pytest tests/test_kymograph.py -v
uv run pytest -k "prediction or orphan"
```

Ten suites, 112 tests, 910 named checks. `uv sync` installs pytest. Nothing is
skipped, and no test reads data this repository does not carry: every suite that
needs a project generates one (config, synthetic movie, real clip cache) into a
temporary directory and deletes it afterwards (`tests/fixtures.py`). Real
projects can be added with `pytest --project-root PATH`, which runs the
window-independence checks over every clip of a prepared `work/`.

Checks are named claims (`tests/harness.py`) because a suite is typically one
scenario making a dozen related statements, and a failure reports the claim
rather than the expression. A run therefore reports checks as well as tests.

| suite | coverage |
|---|---|
| `test_pipeline.py` | clip planning, crop geometry, cache round-trip, box table, `window_mode` |
| `test_nd2.py` | ND2 axis, position, channel and z index arithmetic, against a mock `ND2File` |
| `test_gui_headless.py` | the annotation window built offscreen and driven with real input |
| `test_workflow.py` | projects, config editing, background jobs, the workflow window |
| `test_regressions.py` | one test per defect found by adversarial review |
| `test_trackseries.py` | the per-frame series against closed-form motion, at every window |
| `test_wavespectra.py` | the sliding beat spectrum against a wave of known f, lambda, direction |
| `test_kymograph.py` | curvature against shapes with analytically known curvature |
| `test_predict.py` | provenance of a prediction at every stage |
| `test_reassociate.py` | label repair, largely the cases where it must decline |

`test_pipeline.py` asserts the reassociation contract directly, mapping cached
clip coordinates back through `crop_x0 + cache_x / scale` and comparing against
ground truth. `test_gui_headless.py` renders the canvas into a pixmap and counts
amber pixels, so the marker is measured where it reaches the screen.
`test_trackseries.py` and `test_wavespectra.py` compare against closed forms (a
constant-speed track, a circle of known angular rate, a travelling wave of known
frequency and wavelength), since agreement between two implementations
establishes only that they agree.

To compare the tiled path against a whole-frame run:

```bash
uv run prepare.py --self-test --tile 320 --movies <id>
```

## Notes

* `requires-python >=3.10,<3.13`; `.python-version` pins 3.12, since torch has no
  3.13 wheels. `uv` fetches the interpreter.
* `uv sync` installs CPU torch; `uv sync --extra cuda` installs the CUDA build.
  On a small GPU, lower `detection.batch` and `tiling.tile` first.
* `opencv-python` is overridden out of the resolution in favour of
  `opencv-python-headless`. ultralytics pulls the former, which ships its own Qt
  plugins and conflicts with PyQt at startup.
* `uv sync --extra sleap` with `prepare.py --sleap` writes a SLEAP `.h5` per
  movie for keypoint-MoSeq compatibility.
* Project data is gitignored: `work/`, `annotations.csv`, `predictions.parquet`
  and `config.yaml`, which carries absolute paths to the recordings.
