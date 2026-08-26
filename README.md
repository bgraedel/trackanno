# trackanno

Behaviour annotation for individual tracked organisms in large grayscale
microscopy movies (`.tif`, `.nd2`), with the labels mapped back onto the original
recording.

The unit of work is a **clip**: `clips.frames` consecutive frames of one track,
cropped around that track. The clip loops, one keypress labels it, and the view
advances.

## Install

```bash
uv tool install "trackanno[cuda]"     # NVIDIA, Linux or Windows
uv tool install "trackanno[mps]"      # Apple Silicon
uv tool install trackanno             # CPU

trackanno                             # the pipeline in one window
```

Replace `trackanno` with `.` from a clone, or with
`git+https://github.com/bgraedel/trackanno`.

`uv tool install` puts `trackanno` on PATH in an isolated environment.

**The GPU backend is chosen at install time**, because torch ships one wheel per
backend rather than loading one at runtime. The bracketed extras select it;
`--with` installs an additional package and cannot. `[cuda]` takes torch from
PyTorch's cu124 index and is unavailable on macOS. `[mps]` takes the ordinary
PyPI wheel, which on Apple Silicon already carries Metal support, and requires
macOS 14 or later. Without either, torch comes from PyPI for the running
platform.

`detection.device: auto` (the default in a new project) then selects the first
CUDA device, else MPS, else the CPU, so a `config.yaml` moves between a Mac and a
CUDA box unchanged.

From a clone:

```bash
uv sync                    # add --extra cuda or --extra mps
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

**Templates.** Tuning the detector, the crop geometry and the category buttons
is most of the setup for a dataset, and the next dataset usually wants the same
answers. *Save as template* on the Project page stores the open project's
settings under a name; *New project* then offers them under **start from**.

A template is a whole `config.yaml`, comments included, kept outside any project
so it outlives the one it came from. It carries everything except the three
paths that describe one project on one disk: `video_dir`, `work_dir` and `csv`
are reset, since a template pointing at last month's movie folder is worse than
one pointing nowhere. `paths.model` is kept, because the same weights are
normally reused, and the New Project form overrides it when a model is chosen.
`ui.annotator` is cleared, a template being a protocol rather than a person.

Saving copies the file rather than re-serialising it, so the comments survive,
and the result is parsed by the real loader before it is stored: a template that
would not load cannot be saved. They live under the state directory
(`%APPDATA%	rackanno	emplates`, or `~/.config/trackanno/templates`), one
`.yaml` each, so a template can be handed to somebody else as a file.

From the API:

```python
from trackanno.project import create_project
from trackanno.templates import save_template, list_templates

save_template("Trypanosome 100 fps", "path/to/config.yaml")
create_project("new-project", video_dir="movies/", template="Trypanosome 100 fps")
```

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
then CPU. A config naming a specific GPU is a claim about the machine it was
written on, so an absent device is reported and the run continues on CPU. Detect
& crop logs what `auto` actually resolved to, card name and all, so a slow run is
never a guess.

One probe answers both the list and the resolution, which is not free: torch
will tell you CUDA is *available* and separately that it can enumerate *no
device*, because those questions are answered by different layers and only one
of them reads `CUDA_VISIBLE_DEVICES`. The list used to count devices while `auto`
only asked whether CUDA existed and then assumed device 0, so the page could
offer no GPU while the run picked one anyway. `cuda_devices()` is now the single
answer, and `auto` takes the first entry on the list you were actually shown.

The Configure page runs that probe on a worker thread, since importing torch
costs the better part of ten seconds. Until it lands the list holds only `auto`
and `cpu`, so the row says which it is doing - "checking this machine for GPUs"
while it looks, then one line naming what `auto` picks here. A CPU answer says
*why*, because `torch.cuda.is_available()` returns the same `False` for a
CPU-only wheel, an absent driver and a laptop card asleep behind hybrid
graphics, and those need different things from you. **Re-check** beside the row
asks again without restarting: torch caches a device count only once it is
non-zero, so a card that has since woken up is found.

**Tiled inference.** Each frame is cut into overlapping tiles (`tiling.tile`,
`tiling.overlap`), run as one batch, mapped back to full-frame coordinates and
merged with a class-agnostic NMS. Detections clipped by an interior tile border
are dropped in favour of the neighbouring tile's complete copy, with a safety net
that re-admits any object too large to be complete in any tile. Frames fitting a
single tile skip the mechanism.

**Tracking.** `model.track()` owns the inference stream and would track tiles
rather than frames, so an ultralytics tracker is driven directly with the merged
full-frame boxes. `tracking.backend` defaults to `tracktrack`, built for crowded
scenes with frequent occlusion, which is what ~50 organisms in one field of view
produce; `bytetrack`, `botsort`, `ocsort`, `deepocsort` and `fasttrack` are also
available, and `greedy` selects a nearest-centroid tracker that is adequate at
100 fps where per-frame motion is ~1 px. `tracktrack` needs ultralytics >= 8.4.

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

### Shared models

Weights do not have to be a file somebody copied onto this machine.
**Choose shared model...**, in the detection model row on the Configure page,
lists the models a GitHub release offers, downloads the one you pick into a
shared cache, and applies the settings it was trained with. Only models this
project can run are listed; the rest of the registry is counted below the list.

```yaml
model:
  source: trypanosome-pose   # "" uses paths.model, a .pt on this disk
  version: v2
  sha256: 9f3c...            # written on the first fetch, pins it thereafter
```

`model.source` outranks `paths.model`, which is why the field is greyed out and
labelled while a shared model is in use. **Browse** stays live and is the way
back: picking a `.pt` clears the block above and runs the file you picked, since
choosing a file is the one unambiguous statement that the shared model is no
longer wanted.

The index is one release asset, so
`https://github.com/bgraedel/leishmania-models/releases/latest/download/index.json`
always names the current set while each entry points at one immutable asset:

```json
{"models": [{
  "id": "leishmania-pose", "version": "v2",
  "framework": "ultralytics", "task": "pose", "modality": "brightfield",
  "url": "https://github.com/bgraedel/leishmania-models/releases/download/v2/best_pose.pt",
  "sha256": "9f3c...", "size": 52428800,
  "notes": "8-point flagellum, 20x",
  "config": {
    "detection": {"imgsz": 1280, "conf": 0.25},
    "tiling": {"tile": 640, "overlap": 96},
    "keypoints": {"nodes": ["Head", "Base", "Flag1", "..."],
                  "edges": [["Head", "Base"], ["Base", "Flag1"]]}},
  "preprocess": {},
  "assumes": {"fps": 100, "pixel_size_um": [0.2, 0.65]}}]}
```

`config` is keyed by consumer. trackanno reads `detection`, `tiling`,
`tracking`, `clips` and `spectral` and ignores the rest, so a cellpose entry can
carry `config.cellpose` in the same index. `preprocess` says how pixels reach the
model, for frameworks that do not handle that themselves.

The value is in what a checkpoint cannot report: the keypoint **names**, the
skeleton **edges**, the resolution it was validated at, a tile size that suits
it. Those are applied to the project and reported, never changed silently.

`assumes` takes a number or a list, so a model trained across magnifications
states `[0.2, 0.65]` and a project is checked for containment rather than
equality. It is never applied, only warned about.

`framework` names what loads the file. trackanno runs `ultralytics` `detect` and
`pose` models; anything else is listed with what it is, marked as one it cannot
drive, and left to whatever does run it.

**Three rules, all about not running the wrong code.** Loading a checkpoint
executes the pickle inside it, so this is a security boundary rather than a
convenience feature.

* **The registry URL is a machine setting**, kept in the state directory beside
  the recent-projects list. A `config.yaml` somebody sends you may name a model
  id; it may not decide which server the weights come from.
* **https only, and the digest is checked.** A download whose sha256 disagrees
  with its manifest is deleted rather than cached. `model.sha256` is written back
  on the first success, so a release that later moves a tag to different weights
  is a loud mismatch instead of a silent substitution. Plain `http` is refused
  except to loopback, which is what makes the suite testable.
* **A cache hit never reaches the network.** The cache is content-addressed under
  the state directory, so two ids naming the same weights share one copy, and a
  miss with no connection fails naming the id, the digest and the directory it
  looked in.

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
uv run annotate.py --csv other.csv       # a second annotator's log
```

The screen the workflow window hosts on its Annotate page, in a window of its
own.

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

### Which clip comes next

By default the queue is a seeded shuffle of the whole manifest, filtered by movie
and by done-ness. Movie-by-movie order invites order effects, and the seed
(`ui.shuffle_seed`) keeps the shuffle reproducible.

**`suggest.select_rule` turns active selection on**, and `"off"` is the default:
no ranking runs, the queue is the one every earlier version produced, and nothing
about the screen or the CSV changes. Set it to a rule and, once a model has been
trained and cleared its reliability gate, the clips that model is least sure
about are moved to the front of the queue.

| key | default | meaning |
|---|---|---|
| `suggest.select_rule` | `"off"` | `off`, `margin`, `least_confidence` or `entropy`. Quote it: an unquoted `off` is a YAML boolean |
| `suggest.select_batch_size` | `20` | clips per batch, twice `retrain_every` so a batch outlasts the model that ranked it |
| `suggest.select_audit_share` | `auto` | share of each batch drawn at random from the audit holdout; `auto` is `suggest.audit_fraction` |
| `suggest.select_frame_gap` | `auto` | frames two ranked clips of one track must be apart; `auto` is `clips.frames`, which suppresses overlapping windows of one animal and nothing else |
| `suggest.select_target_quota` | `0` | labels a class needs before slots are spent hunting for it; off by default, because targeting skews the label marginals the reliability gate is computed against |

**What you will notice.** The session opens on the model's top pick rather than
on the first clip of the shuffle. Clips the selector said nothing about keep
their shuffle order behind the batch.

A batch is recomputed when the model changes - `suggest.retrain_every` labels
trigger a background refit, and the queue is re-ranked when that refit lands,
never once per label. **The clip on screen when it lands does not move.** The
fresh picks are spliced in immediately after it, which is the earliest place a
forward-only cursor can still reach; anything put in front of the cursor would
never be seen, because navigation only ever scans forward. The status line says
how many were re-queued. If the selector fails - a model trained at a different
window length, say - the status line says that instead and the queue falls back
to the shuffle, so a broken selector costs the ordering and never the ability to
annotate.

**The labelled set stops being a random sample of the clips the moment this is
on.** That is not a side effect to be managed, it is what selection is: the rows
that end up in the CSV are the ones the model asked for, which is to say the hard
ones, and an accuracy computed over them afterwards is an accuracy on a set
assembled to be hard - lower than the truth by an amount the file alone cannot
say.

Two things keep an unbiased number alive. `suggest.select_audit_share` of every
batch is drawn **uniformly at random** from the audit holdout
(`suggest.audit_fraction`) instead of being ranked, so the labelled part of that
holdout stays a random sample of it however good the selector gets - and that is
the sample `audit_accuracy` in the model report is computed over. Those draws are
spread through the batch rather than parked at its end, so stopping halfway
through still leaves them in proportion. And every row now records why its clip
was shown, which is what lets anyone separate the two groups later.

### The CSV

```
clip_id, movie_id, source_file, series, channel, track_id, f_start, f_end,
crop_x0, crop_y0, crop_w, crop_h, scale,
<one column per category group>, uncertain, notes, annotator, timestamp_utc,
dwell_s, suggested, suggest_conf, suggest_shown, selected_by
```

Multi-choice groups are `|`-joined. The identity and geometry columns, with
`tracks.parquet`, map any label back onto exact pixels in the original movie:

```python
original_x = crop_x0 + cache_x / scale
original_y = crop_y0 + cache_y / scale
```

`suggested` / `suggest_conf` / `suggest_shown` record the model's answer and
whether the annotator could see it, because agreement is not influence unless the
two are recorded apart.

**`selected_by` records why the clip was put in front of a human**, as one line of
JSON. It is the column that keeps the paragraph above computable. Four shapes,
and they are four different facts:

| cell | means |
|---|---|
| empty | **not recorded.** No selector had chosen anything in that session. This is *not* "chosen at random" and nothing may read it as such |
| `{"rule":"margin","score":0.83,"rank":0,...}` | the selector chose this clip: the rule that chose it, the score that earned it, its rank in the batch, the pool it was drawn from and the model's own prediction at the time |
| `{"rule":"audit_random",...}` | a reserved uniform draw from the audit holdout, carrying `"score": null` because a coin toss has no score and `0.0` would read as certainty |
| `{"rule":"annotator_navigated"}` | a selector was running and did not choose this clip - a human navigated to it, with the arrow keys, `Tab`, or a filter change |

```python
from trackanno.annotate.store import decode_provenance

why = labels["selected_by"].map(decode_provenance)
unbiased = labels[why.map(lambda d: d.get("rule") == "audit_random")]
```

Keys are sorted and the encoding is ASCII, so re-recording the same decision is a
no-op in a diff and no rule name can put a newline inside a row. Rows written
before the column existed are empty; opening such a file says so once, out loud,
because filling those blanks in with "random" is exactly the conflation the column
was added to prevent.

---

## 3 · `suggest.py`: the label suggester

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

CatBoost trees are symmetric, so `suggest.depth` alone sets the width at
2<sup>depth</sup> leaves. The default 5 is 32 leaves; 6 is 64, which overfits a
few-hundred-clip hand-labelled set quickly. Feature importances come from the fit
rather than from column shuffling. `suggest.model: hist` selects scikit-learn's
HistGradientBoosting with matched hyperparameters.

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

### Behavioural boundaries

A dense window carries one label for every frame in it, and out of the box those
windows tile, so a change of behaviour can only ever be reported at a window
boundary. At `window_frames: 50` a cell that stops swimming at frame 137 is
called swimming until 150, or immotile from 100, and nothing downstream recovers
the 137. That is a bias in a bout duration, not noise around one. Five keys buy
a sharper boundary, and **every one of their defaults reproduces today's output
exactly** - `window_stride: auto` is `window_frames`, which is the tiling that
has always been there - so an existing project is unchanged until it opts in.

```yaml
predict:
  window_stride: auto     # = window_frames: the windows tile, one per frame
  decode: none            # none | min_bout | viterbi
  min_bout_frames: 0      # the shortest bout you would accept as real
  self_transition: auto   # viterbi only; = 1 - 1/min_bout_frames
  save_posteriors: false
```

**A `window_stride` below `window_frames` overlaps the windows.** Each frame is
then scored by several of them, and averaging their class probabilities onto the
frames they cover gives a posterior per frame instead of per window. That
removes the *quantisation* of a boundary to a multiple of the window length. It
does not remove the *blur*: a window straddling a transition genuinely contains
both behaviours and its features are a blend of the two, and no stride fixes
that. Boundary error against a planted transition, in frames:

| `window_stride` | mean abs. error | max |
|---|---|---|
| 50 (tiling) | 12.24 | 24 |
| 25 | 6.24 | 12 |
| 10 | 2.23 | 4 |
| 5 | 1.20 | 2 |

That is `stride/4` on average and `stride/2` at worst: the quantisation limit
with nothing left over. Worth stating, because it says exactly what a smaller
stride buys - halve the stride, halve the error, down to wherever the cost stops
being worth it.

**The cost is windows, not pixels.** Featurising one track costs about 37 ms
fixed plus about 4 ms per window (a 500-frame synthetic track, 8 keypoints,
`spectral.window: 75`), so the per-track arithmetic is linear in the window
count over a substantial constant.

Do not extrapolate that microbenchmark to a project. Measured end to end on a
real one - three movies, 500 frames each, `clips.frames: 25` - going from the
tiling default to `window_stride: 5` took **18s to 74s**, about 4x for 5x the
windows, including rebuilding the feature table at the finer stride. The
single-track figure predicts 2.8x and is wrong at project scale, because a run
pays per-movie and per-track costs the one-track measurement amortises away.
Budget from the 4x.

The dense stage reads no pixels and re-detects nothing, so this stays cheap in
absolute terms - it is arithmetic over `tracks.parquet` and the saved
embeddings, not another pass over the recording.

**Overlapping windows need a `decode`, and the config refuses to load without
one.** Several rows would otherwise cover the same frames of one track, and
`render.py` paints `range(f_start, f_end)` per row: the exported colour of an
overlapped frame would be decided by row order in the parquet rather than by the
model. A decode collapses the overlap into **bouts** - disjoint, variable-length
runs between inferred boundaries, one row each in `predictions.parquet`. It
needs `coverage: dense` for the same reason: `manifest` has a posterior only
inside the clips that were cut for annotation, so its bouts would be drawn
across the gaps between them.

**`min_bout` and `viterbi` are choosing different things**, which is worth
knowing before the first test on easy data suggests one of them is redundant.
`min_bout` is run-length post-processing: take the per-frame argmax, then absorb
every run shorter than `min_bout_frames` into whichever neighbouring class the
absorbed frames themselves score higher. `viterbi` picks the most likely label
*sequence* under a sticky transition prior, so a short run survives when its
evidence outweighs the cost of the two transitions it needs. On a clean
posterior the two agree, and they will keep agreeing for as long as the evidence
is clean. They diverge on noisy evidence, which is the case that matters:
`min_bout` drops a brief episode for being brief however confident the model
was, and `viterbi` keeps the ones the model was sure about while still smoothing
frame-to-frame flicker.

**`self_transition: auto` reads `min_bout_frames` as an average rather than a
floor.** Under a constant self-transition `a` a chain's holding time is
geometric, and a geometric holding time has expectation `1 / (1 - a)`. Setting
that equal to `min_bout_frames` and inverting gives `a = 1 - 1/min_bout_frames`.
So one number serves twice - a hard floor to `min_bout`, a mean bout length to
`viterbi` - and `auto` needs `min_bout_frames` of 2 or more, since at 1 the
identity returns a chain forbidden to stay anywhere and at 0 it divides by zero.

**`save_posteriors: true`** writes `frame_posteriors.parquet` into
`paths.work_dir`, beside `predictions.parquet`: the per-frame posterior the
decode was run on, kept so a decode can be inspected and re-tuned without
re-scoring the movie. Off by default because it is one row per frame per track
rather than one per window, and so is much the larger file.

**`window_scales` scores at several window lengths at once.** `window_stride`
changes where a window starts; this changes how long it is. One window length is
one commitment about the timescale of behaviour, and out of the box it is a
guess: an episode shorter than the window is invisible *inside* it, and a bout
much longer than the window is chopped into independent verdicts that can
disagree. Listing several lengths builds the dense grid once per length and lets
every row reach the same per-frame average.

```yaml
predict:
  window_scales: []            # [] = window_frames alone, exactly as before
  # window_scales: [25, 50, 100]
suggest:
  feature_scope: all           # all | length_invariant
```

`[]` and `[window_frames]` are the same run, byte for byte. The list must include
`window_frames`, which is the length the rest of the pipeline is planned at; two
keys naming the window length and disagreeing is not a configuration.

**What it buys is duration sensitivity, not a sharper boundary.** On the test
suite's planted movie a cell stops for 30 frames. No window of the 50-frame
tiling grid is ever more than 40% stopped, so a model that reads the moving
fraction back *exactly* still says "swimming" everywhere - the event is not
missed because the model is weak, it is missed because a 50-frame window cannot
represent it. Adding a 25-frame scale finds it; adding a 10-frame scale as well
puts its start on the frame it happened. Meanwhile the planted *transition* at
frame 137 is reported at 150 with one scale and at 150 with three, because every
scale's grid still cuts at its own window edges. `window_stride` is the knob for
*where* a boundary falls; `window_scales` is the knob for *what* can be seen at
all, and they are worth using together.

**Every scale carries the same weight**, and that is a decision rather than an
accident of the arithmetic. Each scale's stride is scaled with its length so all
of them cover a frame equally often, after which the plain mean already inside
the decode does the rest. The alternative - one stride for every scale - weights
a scale by its length: at 25/50/100 on a common stride the 100-frame window
covers an interior frame four times against the 25-frame window's once, so it
carries 4 of the 7 votes on its own, and the short scale you added to see a brief
event is outvoted by the window too blunt to see it. Equal weight has its own
cost, stated so it is not a surprise: a frame deep inside a long uniform bout
gets no extra confidence from the long window agreeing. It is exact in a track's
interior; near the ends the grid is clipped and the weights drift with the
covering counts, which is what `n_covering_mean` on each bout reports.

**The cost is a feature build per scale, but less than a whole one each.** All
the scales of a track are measured in one pass, so the per-track setup - the
series layout and the sliding spectral transform - is paid once and only the
per-window work repeats. On the repository's synthetic fixture, `[25, 50, 100]`
costs 3.3x the windows for 1.3x the wall time on a `task: detect` project and
2.2x on a `task: pose` one, where the spectral half dominates. Those are a
120-frame synthetic movie on one machine, so do not budget from them: the run
logs the window count, the per-scale breakdown and the multiple before it spends
any of it, and that line is the one to quote.

**Several scales need `decode`, and a model whose features survive a change of
length.** The decode for the same reason overlapping windows do - windows of
several lengths cover the same frames, and without a decode the exported colour
of a frame is decided by parquet row order. The second condition is the sharp
edge of the whole feature: window length changes what the features are *worth*,
not what they are *called*. `path_length` over 100 frames is simply twice the
`path_length` over 50 under the same column name, so a model fitted at one length
reads another length's rows on a scale it has never seen - at normal confidence,
with nothing in the output to show for it, and then averaged into the same
posterior as the rows that were right.

`suggest.feature_scope: length_invariant` is how a project earns the right to do
this. It keeps only the columns whose value does not scale with the window: the
`beat_*` and `wave_*` families, which are measured over `spectral.window` and not
over the requested window at all, plus the intensive means and fractions
(`speed_mean`, `area_mean`, `immotile_frac`, `cnn*_mean`). Out go the extensive
quantities (`path_length`, `net_displacement`, `radius_gyration`), everything
derived from them (`straightness`), the dispersions (`*_std`), the order
statistics (`speed_max`, `speed_p90`), `present_frac`, and the lag-based turn
columns - the lag itself is `min(n_frames // 4, 25)`, an explicit function of the
window length. It is an allowlist, so a feature added to the schema later is
treated as length-dependent until somebody says otherwise; that costs a little
signal and says so, where the other way round is silent.

It composes with `feature_groups` rather than replacing it: `feature_groups:
[pose, waveform]` with `feature_scope: length_invariant` is "the beat and
waveform columns only, length-invariant" without typing out a column list. The
subset costs real accuracy - those columns carry information, they are simply not
comparable between lengths - and how much is a per-project number. `uv run
suggest.py --ablate` prints it beside the full set; read the kappa 5th percentile
rather than the macro-F1, because on a few hundred labels a macro-F1 difference
of a few hundredths is not distinguishable from zero while the bootstrap bound
over movies is about whether the subset survives the next one.

**The incoherent combination is refused twice, on purpose.** `load_config`
refuses several scales against a `feature_scope` that is not `length_invariant` -
that is the config saying what the *next* model will be fitted on. A labelling
run then asks the model *on disk* what it was actually fitted on, by reading the
column names in its own payload, and names the offending columns if it cannot be
scored at several lengths. Neither check subsumes the other: retraining is
exactly what moves a project from failing the second to passing it, and editing a
config alone must not.


### Correcting a bout

A decoded boundary is a claim about when an animal changed what it was doing,
and the review window (**Model -> Review predictions...**) is where you can see
whether it is in the right place. You can also move it. What you say is written
to **`bout_corrections.csv`, beside `annotations.csv`** - never into
`predictions.parquet` and never into the annotation CSV.

That separation is the whole design, not filing tidiness. `annotations.csv` holds
one row per **fixed-length window**, keyed by `clip_id`, and every consumer of it
- the feature join, the suggester, the audit holdout, `reassociate` - assumes
that shape. Saying "frames 812 to 1043 of track 3 are swimming" is *more* than a
window label: it carries two boundaries, and no clip grid can express a boundary
that does not fall on a grid multiple. Flattened into clip rows it would be
destroyed at the moment it was recorded, and the clip table would quietly acquire
rows of a second kind that nothing downstream could tell from a hand label. It
sits beside the annotation log rather than under `work/` because `work/` is
deleted and rebuilt whenever detection is re-run, and a human judgement a rebuild
destroys is not a record.

Each row is an **assertion** - these frames are this class - plus who said it,
when, and what the model had said over those frames at the time. That last column
is what keeps "the reviewer confirmed the model" distinguishable from "the
reviewer overruled it" after the next prediction run has overwritten the
evidence.

```
correction_id, movie_id, track_id, group, f_start, f_end, label,
annotator, timestamp_utc, overruled
```

**One operation, four names.** The bouts of a track are a disjoint, gapless
run-length encoding, so the state is just where the cuts are and what each cell
is called. Dragging a boundary, splitting a bout, merging two and relabelling one
are all the same edit - *name a class over a frame range* - with different
arguments. Drag an edge in the record lane, or type a range and press **Set
class**; **Merge into previous/next** takes the neighbour's class over the
selection. Every edit is checked against the invariant and saved immediately, so
there is no Save button to forget and nothing to lose by closing the window.

**Only what you touched is written.** Correct one boundary and one or two rows
appear, not a transcript of the whole track. The rest of the track is still the
model's claim, and the window draws it that way: **each group gets two lanes**,
the model's bouts above and *the record* below - your spans filled solid and
outlined, spans still the model's faded and dashed. The record lane deliberately
has no bar height, because `conf` is a mean over exactly a bout's frames and a
bout you have cut in half no longer has one anybody measured. A boundary the
decode placed keeps its stride error band; a boundary you placed is drawn as a
hairline, because yours is exact to the frame you chose.

Some things it refuses, and why. A file of fixed windows cannot be corrected at
all - a window's edges are multiples the predictor chose, so moving one corrects
nothing about the animal, and a window relabelled by hand is a clip label, which
already has a file. A stretch no window could be featurised over cannot be
*extended*: "nothing scored this" is a fact about the run, not something a person
is in a position to assert. A class `config.yaml` does not declare is refused on
the way in - it could not be coloured, trained on or exported - while a class the
config has since *dropped* still loads, because a rename must not delete the
record of what somebody judged. And two rows claiming one frame stop the window
with both named: nothing can say what was judged while both stand.

**Turning corrections into training data is a separate step, and it is not taken
for you.** *Derived labels...* reports what your corrections would imply about
the fixed windows the training set is actually built from, under one rule:

> A window inherits a class **if and only if it lies entirely inside a single
> corrected span**. A window that overlaps a correction without being contained
> in one inherits nothing, and is listed with the frames that make it ambiguous.

Not a majority vote and not the midpoint. A window straddling a transition
genuinely contains both behaviours - that is the same blur a smaller
`window_stride` cannot remove - so labelling it either way teaches the next model
that a blend looks like a pure class, precisely at the transitions it is already
worst at. A majority rule would label the most misleading windows most
confidently. Straddlers are reported rather than dropped because they are the
most informative windows you have: they are where the boundaries are.

Nothing in that report is written anywhere. A derived label is not a hand label -
it is what your correction *implies* about a clip nobody looked at - and the
moment the two are indistinguishable in `annotations.csv`, nobody can say again
what a person actually judged.

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

**Open project folder** on the Export page shows the project in the file
manager; the rendered movies land in `render.out_dir` inside it.

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

Twenty-six suites, 315 tests, 2323 named checks. `uv sync` installs pytest. Nothing is
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
| `test_render.py` | which rows the overlay may paint over, and the one it refuses |
| `test_reassociate.py` | label repair, largely the cases where it must decline |
| `test_models.py` | the model registry, largely the cases where a fetch must be refused |
| `test_defaults_lock.py` | every knob this branch added is off in a project a new user makes |
| `test_multiscale.py` | one model at several window lengths: off is unchanged, the scales weigh equally, length-dependent columns are refused |
| `test_bout_corrections.py` | correcting a boundary: the partition stays gapless, and what a correction does not imply |
| `test_selection_wiring.py` | the candidate pool a selector ranks, and what must stay out of it |
| `test_selection_queue.py` | which clip the annotator is handed, and what the CSV records about why |
| `test_audit_stability.py` | the holdout is the same clips tomorrow that it is today |
| `test_discover.py` | sub-behaviours found where they were planted, and the batch effect that is not one |
| `test_bout_review.py` | reading decoded bouts back, including a file that holds windows instead |
| `test_bouts.py` | frame posteriors and the two decoders, against closed forms |
| `test_dense_grid.py` | the window grid at every stride, and the frames a plan must reach |
| `test_select.py` | which clip to annotate next, and what the audit draw must ignore |
| `test_subcluster.py` | structure recovered where it was planted, and refused where it was not |
| `test_store_provenance.py` | why a clip was chosen, surviving the CSV and its repairs |
| `test_suggest_window.py` | the window length a model was trained at, and the skew it closes |

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
* `uv sync --extra cuda` and `--extra mps` are declared as conflicting extras,
  so the lockfile carries a separate torch resolution for each. On a small GPU,
  lower `detection.batch` and `tiling.tile` first.
* `opencv-python` is overridden out of the resolution in favour of
  `opencv-python-headless`. ultralytics pulls the former, which ships its own Qt
  plugins and conflicts with PyQt at startup.
* `uv sync --extra sleap` with `prepare.py --sleap` writes a SLEAP `.h5` per
  movie for keypoint-MoSeq compatibility.
* Project data is gitignored: `work/`, `annotations.csv`, `predictions.parquet`
  and `config.yaml`, which carries absolute paths to the recordings.
