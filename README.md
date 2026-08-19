# trackanno

Annotate the behaviour of individual tracked organisms in large grayscale
microscopy movies (`.tif` / `.nd2`), then get the labels back onto the original
movie.

The unit of work is a **clip**: 50 consecutive frames of **one track**, cropped
around that track. You watch it loop, press one key, and it advances.

**Install it once, then just run `trackanno`:**

```bash
uv tool install --with catboost git+https://github.com/<you>/trackanno   # or: from a clone
uv tool install --with catboost .

trackanno                  # everything, in one window
```

`uv tool install` puts `trackanno` on your PATH in its own isolated environment -
no venv to activate, no working directory to be in. Add `--with catboost` for
the label suggester (it works without, using scikit-learn). Later:
`uv tool upgrade trackanno`.

To work on the code instead, run it from a clone:

```
uv sync                    # once (add --extra cuda for GPU torch)
uv run gui.py              # same window, from the source tree
```

`gui.py` is the whole pipeline: create a project, edit its settings, run
detection with a progress bar and a stop button, annotate, train the suggester,
export the overlay movies. Nothing in the workflow needs a terminal or a text
editor.

The individual steps are still separate CLI entry points, which is what you
want on a headless GPU box or in a batch script:

```
uv run prepare.py          # 1. tiled YOLO detection + tracking + clip cache
uv run annotate.py         # 2. just the annotation screen
uv run suggest.py --train  # 3. train the label suggester
uv run predict.py          # 4. label every clip with the trained model
uv run render.py           # 5. labelled boxes back onto the original movie
```

Both paths call the same functions (`prepare_all`, `train`, `predict_all`,
`render_all`), so the GUI cannot quietly do something different from the CLI.
`uv tool install` also puts each of them on PATH as `trackanno-prepare`,
`trackanno-annotate`, `trackanno-suggest`, `trackanno-predict` and
`trackanno-render`.

Everything is driven by `config.yaml`, which lives in the project folder.

---

## 0 · `trackanno` - the workflow window

```bash
trackanno                      # reopens the last project
trackanno path/to/project      # or a specific one
```

Seven numbered steps down the left, in the order the work happens. The sidebar
is a checklist, not a menu: each step carries a one-line description, its own
state (`*` done, `>` this one next, `!` something is missing) and a count, so you
can see where the project is without opening anything.

| step | what it does |
|---|---|
| **1 Project** | Pick the folder to work in |
| **2 Configure** | Point it at your movies and choose your labels |
| **3 Detect & crop** | Find the animals and cut them into clips |
| **4 Annotate** | Watch each clip and press one key |
| **5 Model** | Learn from your labels and suggest the next one |
| **6 Distribution** | See what you have actually labelled, before exporting it |
| **7 Export** | Put the labels back onto the original movies |

A **Next:** bar sits along the bottom of every page with a button that takes you
there, so the guidance is never on the page you just left. Each page also states
its own prerequisites up front instead of failing when you press the button.

**Configure has four tabs, not one wall of settings.** *Basics* is the fifteen
things you have to decide (where the movies are, which model, how long a clip
is). *Labels* is the button editor. *Keypoints* is the pose table, greyed out
unless the task is `pose`. *Advanced* is the detector, tracker and cropping
knobs, whose defaults are fine. Settings marked `*` describe how the
clips were **cut**, so changing one needs a re-run of step 3 before it takes
effect - the page says so rather than leaving you to wonder.

**A project** is a folder containing `config.yaml`, `work/` (the clip cache) and
`annotations.csv`. "New project" writes a fully commented `config.yaml` with the
paths filled in; recently opened projects are remembered.

**Settings save themselves.** There is no Save button: edits are written ~0.7 s
after you stop typing, and navigating away flushes rather than prompting. That
is not just convenience - the old "unsaved changes?" dialog was how you ended up
on step 3 wondering why it still could not find your movies.

**Saving does not destroy the file, and cannot break the project.**
`config.yaml` is two thirds comment, and those comments are the documentation
for each knob. The Configure page rewrites values *in place* - line by line,
keeping every comment and the file's order - rather than round-tripping through
a YAML dumper. Settings the form does not show are left untouched, not dropped.

Every autosave is parsed by the real loader **before** it replaces anything, so
an edit that would not load never reaches disk; you get a red banner quoting the
loader's own complaint and the file is untouched. Paths that point at nothing
are treated differently: they are amber outlines and a note, not errors, because
a model on a share that happens to be offline must not stop you editing anything
else.

**One theme, one set of tokens.** The look comes from `gui/theme.py`: a small
palette plus a stylesheet generated from it, applied once to the application.
Widgets ask for a *role* (`muted`, `danger`, `heading`) rather than a colour, so
nothing carries its own hardcoded hex. The button in the status bar switches
light/dark instantly and remembers the choice in the project; `ui.theme` takes
`dark` (default), `light`, or `system` to follow the desktop.

**Long jobs run on a worker thread.** Detection reports which movie, which
stage, and how many frames in; Cancel takes effect at the next frame rather than
at the next movie.

---

## 1 · `prepare.py` - detection and the clip cache

```bash
uv run prepare.py --list                       # what movies can it see?
uv run prepare.py                              # all of them
uv run prepare.py --movies MOVIE_A --force     # redo one
uv run prepare.py --limit-frames 100           # quick smoke test
uv run prepare.py --self-test --tile 320       # check the tiling merge
```

**Which GPU.** `detection.device: auto` picks the first CUDA device, then MPS,
then the CPU, and says in the log what it chose. A config naming a specific GPU
is a promise about the machine it was written on, so if that device is not here
the run reports it and continues on the CPU rather than dying inside
ultralytics. The Configure page fills its dropdown from the actual hardware
(`0: NVIDIA GeForce MX450 (2.1 GB)`), read on a worker thread since importing
torch to ask costs a second or two.

**Tiled inference.** Movies are larger than the model's input, so each frame is
cut into overlapping tiles (`tiling.tile`, `tiling.overlap`), run as one batch,
mapped back to full-frame coordinates and merged with a class-agnostic NMS.
Detections clipped by an *interior* tile border are dropped in favour of the
neighbouring tile's complete copy - with a safety net that re-admits any object
too large to be complete in any tile. Frames that fit in a single tile skip all
of this automatically, so small movies pay nothing.

**Tracking.** `model.track()` can't be used with tiling (it owns the inference
stream and would track tiles, not frames), so ultralytics' `BYTETracker` is
driven directly with the merged full-frame boxes. `tracking.backend: greedy`
switches to a nearest-centroid tracker, which is adequate at 100 fps where
per-frame motion is ~1 px.

**Output** - `work/<movie_id>/`:

| file | contents |
|---|---|
| `tracks.parquet` | every detection, in **original** full-frame coordinates |
| `clips.parquet` | one row per clip + its crop rectangle and scale |
| `clip_boxes.parquet` | overlay geometry (target + neighbours) in cache coords |
| `clips.dat` | the pre-cropped pixels, uncompressed `uint8` memmap |
| `thumbs.dat` | optional full-frame thumbnails for the GUI minimap |
| `norm.json` | the movie's fixed intensity mapping |
| `prepare_done.json` | resume marker - re-running skips finished movies |

**Why the cache is uncompressed.** Loading a clip is one contiguous ~1 MB read
with zero decoding. Combined with background prefetch, navigation is instant.
Keep `paths.work_dir` on a **local disk**: over a network share the per-clip
latency is what you feel, and `prepare.py` warns if you point it at a UNC path.

**Crop geometry.** The crop is centred on the track and sized to contain the
animal on every frame of the window, with `clips.pad` of margin. That margin is
what lets the GUI re-centre and zoom at display time - one cache, every viewing
mode, no extra storage. Clips needing more than `clips.crop_size` px are
downscaled and their `scale` recorded, so every cached pixel still maps back to
an exact original coordinate.

**One commit per movie.** The pixels, the manifest, the box table and the cache
header only make sense together, so they are written beside their real names and
renamed into place at the end. A run that is cancelled, killed, or loses its
network share leaves the previous cache byte-for-byte intact rather than half
replaced, and the resume marker is cleared before any of it starts so an
interrupted movie is retried rather than trusted. `ClipStore` independently
refuses any movie whose manifest disagrees with its header or whose `clips.dat`
is short, because pairing a manifest row with another clip's pixels looks
completely normal while you label it.

The crop is deliberately **not** slid back inside the frame when it overhangs
the movie border. Clamping it there parked every animal that swims near an edge
against the side of its own crop and filled the opposite half with interior
background nobody asked for. The crop is zero-padded instead, so the only pixels
lost are the ones that genuinely do not exist.

### Pose models

Set `detection.task: pose` and the keypoints look after themselves. A checkpoint
stores `kpt_shape`, so the **count** is read from the model; it stores no
keypoint *names*, so placeholders `kp0`, `kp1`, ... are generated and you can
rename them under `keypoints.nodes` afterwards. Order is what has to match the
model; the names are only labels.

The Configure page has a **Keypoints** tab: press *Read from model file* to get
the count, then type the names into the table. Renaming a keypoint follows
through into the skeleton links, and *connect in order* wires 0-1, 1-2, ...
which is what a flagellum chain looks like. `prepare.py` does the same
automatically when it loads the model, and says so.

**Order is the contract, names are labels.** The pose features address keypoints
purely by position: **0** is the head, **1** the base and body-frame origin, and
**1** onwards the flagellum chain. A model that predicts them in a different
order still produces body-length and beat numbers, and they mean nothing. The
names are the only place that ordering is written down, which is why they are
worth typing once.

If `keypoints.nodes` is already set and disagrees with the checkpoint, that is an
error rather than something to paper over: the wrong count would silently
misalign every keypoint column.

`edges` (the skeleton drawn over the animal) cannot be inferred and stays empty
until you write it; it only affects display.

### ND2 files

ND2 movies are read natively. They can carry P (multipoint), T, C and Z axes,
and we need one 2-D time series, so:

* **each position `P` becomes its own movie**, with id `<stem>__P00`, `__P01`, …
  (`nd2.positions: all` or a list);
* `nd2.channel` takes an index or a channel name (`"BF"`);
* `nd2.z` takes an index, or `max` / `mean` / `min` for a projection;
* `nd2.fps: auto` reads the real frame interval out of the ND2 metadata.

TIFF carries no frame rate, so set `tiff.fps` yourself.

### How much gets annotated

`clips.window_mode` controls the volume:

| value | meaning |
|---|---|
| `first` | only the movie's **first 50 frames** - one clip per track present there |
| `first_per_track` | the first full window of each track, wherever it starts |
| `all` | every non-overlapping window of every track (default) |

`clips.stride` sets the spacing (`= clips.frames` means non-overlapping), and
`clips.max_clips_per_track` caps per-track volume.

---

## 2 · `annotate.py` - the annotation screen

```bash
uv run annotate.py
uv run annotate.py --backend ndv         # embed ndv.ArrayViewer instead
uv run annotate.py --csv other.csv       # a second annotator's log
```

The same screen the workflow window hosts on its Annotate page - this entry
point just gives it a window of its own.

The default `ui.backend: simple` is a built-in player, and now for a stronger
reason than "ndv is heavy": it is the only backend that can draw the marker as a
separate vector layer and treat centring and zoom as view transforms. `ndv`
takes a finished array, so with it both go back to being burned into pixels.

**One keypress labels and advances.** With `ui.auto_advance: true`, picking a
value in a single-choice group commits the row and jumps to the next clip as
soon as every `required: true` group is filled. Ticking a multi-choice box (e.g.
`quality`) never advances on its own.

### Shortcuts

| key | action |
|---|---|
| `1`…`5`, `q`/`w`/`e` | category values (defined per option in `config.yaml`) |
| `→` / `Enter` | commit + next clip |
| `←` | previous clip |
| `Tab` | next **unannotated** clip |
| `Ctrl+U` | jump to the first unannotated clip |
| `Space` | play / pause |
| `o` | marker on/off |
| `s` | cycle the view: centred → smooth → locked |
| `c` | also outline neighbouring tracks |
| `+` / `-` / `0` | zoom into the crop / out / reset |
| `m` | context minimap |
| `u` | mark uncertain |
| `Backspace` | clear this clip's annotation |
| `Ctrl+Z` | undo the last annotation |
| `Ctrl+S` | force-save the current clip |

All bindings are configurable under `keys:`; the config refuses to start if a
category shortcut collides with a navigation one.

**The marker** answers one question - "which animal am I judging?" - so it marks
exactly one animal. With ~50 organisms per movie a padded crop usually contains
a neighbour, and outlining them all defeats the purpose; `c` turns the
neighbours on when you actually want them.

It is **not rasterised**. The clip cache stays grayscale and the marker is
painted as vector graphics over the frame: four amber corner brackets with a
dark halo, one hairline wide at any zoom, and no line drawn across the
flagellum. Burning a 1 px `cv2.rectangle` into the array and then upscaling it
4× with nearest neighbour - the old path - is what made the box chunky. On a
frame where the detector missed the animal the marker is drawn dimmed and dashed
at the last known position, rather than vanishing and strobing at playback
speed.

The `ndv` backend takes a finished array and has no overlay layer, so with
`ui.backend: ndv` the marker goes back to being burned in.

**What the view follows (`s`).** Three answers to three different questions, all
reading the *same* cached pixels - the crop is never re-cut, so switching is
free:

* **centred** (default) - one view for the whole clip, centred on the median of
  the track's own path rather than on the middle of the cached crop. All motion
  visible; the only view in which "swimming" and "immotile" genuinely differ.
* **smooth** - the view follows a moving average of the path (`ui.smooth_window`),
  so slow drift is removed while the beat and local excursion survive.
* **locked** - re-centred on the animal every frame. Best for body shape and
  flagellar beat; translation is invisible by construction.

None of them clamp. Centring is a display transform with the outside padded, so
a clip cropped at the movie border still shows the animal dead centre, and
"locked" puts it at the exact centre on every frame instead of "as close as the
crop allows". Frames where the track dropped out are filled from the nearest
detected frame, so a missed detection no longer makes the animal jump sideways
and back.

`ui.view_zoom` (and `+` / `-` / `0`) magnifies into the crop, again as a
transform - no pixels are resampled to achieve it.

**If the three modes look identical, the cache is why.** They can only move the
view within the cached crop, so a cache cut with `clips.follow: track` (crop
re-centred on the animal at prepare time) and a tight `clips.pad` leaves them
nothing to do - the translation was removed before the viewer ever saw the
pixels. Pressing `s` says so explicitly, with the numbers for that clip. To get
the distinction back, re-cut with `clips.follow: fixed` and a larger `pad`.

### The curvature kymograph

On a `task: pose` project a second panel sits under the clip: the flagellum's
signed curvature along its arclength (vertical, base at the top) against time
(horizontal). A travelling wave is diagonal stripes, a standing wave vertical
ones, a mistracked frame a gap. Click anywhere in it to scrub the clip to that
frame.

It follows the tracked points and invents nothing. A smoothing spline is fitted
through the keypoint chain once per frame and differentiated analytically -
curvature is a second derivative, and finite differences on seven noisy points
are almost pure noise. The chain order comes from `keypoints.edges` (the longest
path through that graph), because walking a bend from the other end negates it.

**The sign is a real ambiguity, handled rather than hidden.** A 2D image cannot
tell a cell from its own mirror image; the two give curvature fields differing by
an overall factor of -1. Every *even* functional of curvature is
mirror-invariant, so amplitude, spectrum, wave speed and direction of travel
cannot decide it. The one odd quantity with a physical meaning is the static
mode - the time-averaged shape left once the beat cancels - so `kymo_sign:
canonical` flips each clip to make that positive, but only when the static mode
is big enough relative to the beat *and* the two halves of the clip agree on its
sign. When they do not, the panel says so and leaves the clip exactly as
measured rather than flipping on a coin toss.

| key | meaning |
|---|---|
| `ui.show_kymograph` | show the panel at all (default `true`) |
| `ui.kymo_field` | `curvature` (default) or `tangent`, the tangent angle it came from |
| `ui.kymo_sign` | `canonical` (default), `raw`, or `absolute` |
| `ui.kymo_interpolate` | smooth the image between the measured nodes (default `true`) |
| `ui.kymo_smooth_px` | how far the spline may pull from the keypoints, in px (default `0.6`) |

Changing any of these takes effect on the open annotator, without reloading the
project. The colour scale is diverging with a neutral grey at zero, so the sign
of a bend is readable and zero curvature is not a colour.

**Resuming.** The CSV *is* the state: any clip whose `clip_id` appears in it is
done. On startup the GUI jumps to the first unannotated clip
(`ui.resume: last_position` returns to where you stopped instead). Every
annotation is appended and `fsync`ed immediately, so a crash costs at most the
clip on screen. Already-annotated clips reopen with their labels loaded and can
be edited; edits rewrite the CSV atomically.

### The CSV

```
clip_id, movie_id, source_file, series, channel, track_id, f_start, f_end,
crop_x0, crop_y0, crop_w, crop_h, scale,
<one column per category group>, uncertain, notes, annotator, timestamp_utc, dwell_s
```

Multi-choice groups are `|`-joined. The identity + geometry columns, together
with `tracks.parquet`, map any label back onto exact pixels in the original
movie:

```python
original_x = crop_x0 + cache_x / scale
original_y = crop_y0 + cache_y / scale
```

---

## 3 · `suggest.py` - the label suggester

```bash
uv run suggest.py --train      # build features, fit, cross-validate, save
uv run suggest.py --report     # print the last report
uv run suggest.py --ablate     # kinematic vs CNN vs both
```

Trains a classifier on the labels so far and **displays** its prediction on the
next clip. It never pre-selects: you still press the real key, so your labels
stay statistically independent of the model trained on them.

**CatBoost is the default**, for one property this data has:

* **the label distribution is lopsided.** Real behaviour scoring is not
  balanced - one state dominates and the interesting ones are rare. An
  unweighted model answers that by predicting the majority class and scoring
  well for it, which is exactly the suggestion you don't want.
  `suggest.balance_classes` turns on `auto_class_weights="Balanced"`, so missing
  a rare class costs the model something.
* **NaN means "undefined" here**, not zero - no beat detected, straightness of a
  stationary track. CatBoost splits on missingness natively; filling with 0.0
  would disguise those as real extremes (straightness 0.0 is a perfect circle).

CatBoost trees are **symmetric**, so `suggest.depth` alone sets the width -
2<sup>depth</sup> leaves, with no separate width knob. 5 is the default and 6 is
already 64 leaves, which overfits a few-hundred-clip hand-labelled set fast.
Feature importances come from the fit itself rather than from column shuffling.

`suggest.model: hist` falls back to scikit-learn's HistGradientBoosting with the
same hyperparameters, so switching backend doesn't silently change the fit too.
It is also the automatic fallback if catboost isn't installed.

**Which features the model may use.** `suggest.feature_groups` picks the
families it is allowed to see, and the Configure page disables the ones this
project cannot compute, with the reason on the control:

| group | what it is | needs |
|---|---|---|
| `kinematic` | speed, straightness, turning, box area and aspect | nothing |
| `pose` | body and flagellum length, beat amplitude, body-anchored translocation | `task: pose` |
| `waveform` | beat-shape descriptors, below | `task: pose` |
| `cnn` | PCA of the detector's own embeddings | `features.enabled` |

Dropping `cnn` is a real modelling choice rather than tidiness: those vectors
re-identify individuals at 100%, so they reward memorising animals over
learning behaviour.

**Beat waveform features** implement the low-dimensional description in Geyer,
Howard and Sartori, *Ciliary beating patterns map onto a low-dimensional
behavioural space*, Nature Physics 18, 193-200 (2022). The flagellum is
parameterised by its tangent angle against normalised arclength in the
co-swimming frame; the time average is the static mode, and the remainder is fit
as `a(s) cos(2 pi f t + phi(s))`. Projecting `a(s)` and `phi(s)` onto shifted
Legendre polynomials gives the axes the paper found sufficient for about 80% of
the variation across every perturbation it tested:

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
swim straight, and `wave_wavelength_ratio` is lambda/L, which the paper reports
as 0.97 +- 0.08 across every condition, so it doubles as a check on the
tracking. Measured here: median 1.19 over 30 real cells, and the frequency
agrees with the independent `beat_freq_hz` estimator to within 0.2 Hz.

**Read `wave_fundamental_frac` before trusting a row.** It runs near 0.47 on
intact cells in brightfield against the paper's >0.9 for isolated axonemes, so
it is the honest quality readout, and it is a feature in its own right rather
than an internal threshold. The Machin number the paper uses needs the bending
rigidity and drag coefficient, so it is deliberately absent.

**The guardrails matter more than the model.** Cross-validation splits on whole
**movies** - the CNN features re-identify individuals almost perfectly, so
ungrouped folds would score re-identification rather than behaviour. The model
stays silent unless its chance-corrected agreement (Cohen's kappa) clears zero
at the 5th bootstrap percentile *and* macro-F1 clears `suggest.min_macro_f1`. A
share of clips (`suggest.audit_fraction`) is never suggested on and never
trained on, so human-vs-model agreement stays measurable.

**When no suggestion appears,** the reason is one of: the clip is a held-out
audit clip, the model's confidence is below `suggest.min_confidence`, the group
was gated off by the report, or the model is stale. Set
`suggest.audit_visible: true` and the panel says which, instead of leaving a
blank space that looks like a bug.

**Repairing orphaned labels.** `clip_id` embeds the track id, so re-running
detection renumbers the tracks and orphans every label written before it -
`train` inner-joins on `clip_id` and drops those rows silently. The Model page's
**Repair label links** matches orphans back onto the current clips by movie,
frame window and crop position, and refuses rather than guesses: two candidate
clips equidistant from the old crop centre are both left alone, and a clip that
already carries a newer label is never overwritten. It backs the CSV up before
writing, and reports what it could not match instead of dropping it.

---

## 4 · `predict.py` - label every clip with the model

```bash
uv run predict.py                          # score every clip -> predictions.parquet
uv run predict.py --coverage manifest      # only the clips cut for annotation
uv run predict.py --include-labelled       # also score clips you judged yourself
```

Once the model is good enough, this labels the whole dataset in one pass so it
can be exported, plotted or compared against the hand labels. Three rules:

* **Predictions never touch the annotation CSV.** They go to
  `predictions.parquet` and nowhere else, or the next model trains on its own
  output and the ground truth is gone. Every row carries `human_labelled` so a
  consumer merging the two can tell them apart.
* **A group the report gated off is not predicted**, and is named as skipped.
  The saved report decides, not the presence of a fitted model.
* **The file is replaced atomically**, since everything downstream reads it as
  authoritative.

`predict.coverage` decides which clips get a row. `manifest` scores exactly the
clips cut for annotation, which leaves every frame between two annotation
windows unlabelled and unpainted. `dense` (the default) lays its own grid over
every frame of every track, so a `clips.stride` of 50 no longer means 49 frames
of nothing in the exported movie. It reads **no pixels** and re-detects nothing:
the features come from `tracks.parquet`, the saved embeddings and the frame rate.
A window too thin to measure still gets a row, carrying `n_present` and
`skip_reason`, because an absent row is indistinguishable from one the model
declined to label.

The Model page runs the same thing with a progress bar, and the CNN PCA is
**loaded** from beside the model rather than refitted - a refit gives a different
orthogonal basis, with no error and no warning, and the model's input space
quietly stops meaning anything.

---

## 5 · Distribution - what you have actually labelled

The page between Model and Export, and worth opening before either. One bar
chart per category group plus a plain-language read of it: which values have
nothing labelled yet, which are below `suggest.min_per_class` and so will be
dropped at training time, which appear in only one movie and therefore cannot be
scored by grouped cross-validation, which appear in the CSV but no longer in the
config, and whether one value is over 90% of the labels - the case where a model
that always answers it would look accurate.

---

## 6 · `render.py` - labelled boxes on the original movie

```bash
uv run render.py --list                          # which movies have labels?
uv run render.py --movie MOVIE_A
uv run render.py --group quality --hide-unlabelled
uv run render.py --source both                   # your labels and the model's
uv run render.py --movie MOVIE_A --tif --max-frames 200
```

Reads `tracks.parquet` + the annotation CSV and colours each track by the
category of the clip covering that frame, using the colours from `config.yaml`.
Works at any time, including on a partially annotated CSV. Unlabelled tracks are
drawn grey or hidden. Output goes to `render.out_dir`.

`--source` picks what to colour by: `labels` (default), `predictions`, or
`both`. **A prediction never looks like a hand label.** Colour cannot carry the
difference - the categories already own the colours, and a viewer three frames
into a clip has no legend - so a model value is drawn as a **dashed** outline
with its confidence beside it, and the file name carries the provenance too, so
a model-coloured movie cannot overwrite or be mistaken for one you coloured
yourself.

---

## Tests

```bash
uv run pytest                       # everything
uv run pytest tests/test_kymograph.py -v
uv run pytest -k "prediction or orphan"
```

Ten suites, 112 tests, 910 named checks. `uv sync` installs pytest; nothing else
is needed and nothing is skipped.

**No test reads data this repository does not carry.** Every suite that needs a
project generates one - config, synthetic movie, real clip cache - into a
temporary directory and deletes it afterwards (`tests/fixtures.py`). That is not
tidiness: loading whatever project happened to be on the machine meant the
suite passed by skipping on a clean checkout and failed on the developer's box
whenever a network share went away, and in neither case was it saying anything
about the code. Real projects can be added as an opt-in extra with
`pytest --project-root PATH`, which runs the window-independence checks over
every clip of a prepared `work/`.

Tests are written as **named claims** rather than bare asserts
(`tests/harness.py`): most of these are one scenario making a dozen related
statements, and "`window.position == 1` is False" does not say which one broke.
That is why a run reports checks as well as tests.

| suite | what it covers |
|---|---|
| `test_pipeline.py` | clip planning, crop geometry, the cache round-trip, the box table, `window_mode` |
| `test_nd2.py` | ND2 axis/position/channel/z index arithmetic, against a mock `ND2File` |
| `test_gui_headless.py` | the annotation window built offscreen and driven with real input |
| `test_workflow.py` | projects, config editing, background jobs, the workflow window |
| `test_regressions.py` | one test per defect an adversarial review turned up |
| `test_trackseries.py` | the per-frame series against closed-form motion, at every window |
| `test_wavespectra.py` | the sliding beat spectrum against a wave of known f, lambda and direction |
| `test_kymograph.py` | curvature against shapes whose curvature is known exactly |
| `test_predict.py` | provenance: a prediction stays recognisable as one, at every stage |
| `test_reassociate.py` | label repair, most of it checking that it refuses rather than guesses |

A few of these are worth knowing about individually:

`test_pipeline.py` asserts the reassociation contract directly: it maps cached
clip coordinates back through `crop_x0 + cache_x / scale` and compares against
ground truth, and checks that "locked" puts the animal at the exact centre of
the view on every frame.

`test_gui_headless.py` renders the canvas into a pixmap and counts amber pixels,
so "the marker is painted" is checked against what actually reaches the screen
rather than against a flag.

`test_regressions.py` names each test for the bug it prevents. The interesting
ones assert things that are easy to regress and hard to notice: that a manifest
disagreeing with its cache is refused rather than silently mis-associating
labels, that a locked CSV produces a message instead of aborting the process,
that the prefetch thread is joined before the memory it reads is unmapped, and
that a config edit does not grow the file every time you pause typing.

`test_trackseries.py` and `test_wavespectra.py` compare against **closed forms**,
not against another implementation: a constant-speed track, a circle of known
angular rate, a travelling wave of known frequency and wavelength. Agreement
between two implementations only proves they agree.

To check the tiled path against a whole-frame run on data small enough to do
both:

```bash
uv run prepare.py --self-test --tile 320 --movies <id>
```

## Notes

* `requires-python >=3.10,<3.13`; `.python-version` pins 3.12 (torch has no 3.14
  wheels). `uv` fetches the interpreter for you.
* `uv sync` installs **CPU** torch. `uv sync --extra cuda` installs the CUDA
  build. On a small GPU, lower `detection.batch` and `tiling.tile` first.
* `opencv-python` is overridden out of the resolution in favour of
  `opencv-python-headless`: ultralytics pulls the former, which ships its own Qt
  plugins and fights with PyQt at startup.
* Optional `uv sync --extra sleap` + `prepare.py --sleap` writes a SLEAP `.h5`
  per movie, so the existing keypoint-MoSeq notebook keeps working.
* A project's own data - `work/`, `annotations.csv`, `predictions.parquet` and
  `config.yaml`, which carries absolute paths to somebody's movies - is
  gitignored. The repository carries code, not anybody's recordings.
