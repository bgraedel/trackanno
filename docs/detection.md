# Detection and the clip cache

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
and `cpu`, so the row says which it is doing: "checking this machine for GPUs"
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
| `instance_features.npy` | `(N, D)` CNN features, row-aligned with `tracks.parquet` |
| `clips.parquet` | one row per clip, with its crop rectangle and scale |
| `clip_boxes.parquet` | overlay geometry (target and neighbours) in cache coordinates |
| `clips.dat` | pre-cropped pixels, uncompressed `uint8` memmap |
| `thumbs.dat` | optional full-frame thumbnails for the minimap |
| `norm.json` | the movie's fixed intensity mapping |
| `detect_done.json` | the detection stage's marker, and what the clip stage cannot work out for itself |
| `prepare_done.json` | resume marker; re-running skips finished movies |

`tracks.parquet`, `instance_features.npy`, `norm.json` and `detect_done.json` are
everything the detection half produces, and the only files that have to travel
when detection runs on another machine. See [Detecting on one machine,
annotating on another](#detecting-on-one-machine-annotating-on-another).

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

**Discarding tracks at the frame edge.** `tracking.border_margin_px` is 0 by
default, which keeps every track. Set it and a track is discarded when its box
comes within that many pixels of any side of the frame.

The reason is pose, and it is not fixable downstream. At the edge the box is cut
off by the picture rather than by the animal ending, so a keypoint model places
its parts against a shape that is not all there. Everything built on those
keypoints, curvature and wave direction and body length, is then wrong in a way
nothing reports, because the detector is confident about the part it can see.
There is no confidence score for "the rest of the animal is outside the frame".

`tracking.border_max_frac` is how much of a track's life may be spent there
before the whole track goes. `0.0`, the default once the margin is on, discards
a track that reaches the edge at all; `0.1` lets one that passes through it
survive. Detection reports how many tracks went and how much of the worst one's
life was at the edge, so the pair can be tuned against a real movie rather than
guessed.

Rows are **not** removed from `tracks.parquet`, which stays row-aligned with
`instance_features.npy`; the track simply becomes no clips. The rule is applied
to dense prediction as well, which is the one place `tracking.min_track_length`
is deliberately *not* applied: that threshold is about annotation units, while
this one is about whether the measurement means anything at all.

**One commit per movie.** Pixels, manifest, box table and cache header are
written beside their real names and renamed into place at the end, so a
cancelled or killed run leaves the previous cache byte-for-byte intact. The
resume marker is cleared before any of it starts, so an interrupted movie is
retried. `ClipStore` independently refuses any movie whose manifest disagrees
with its header or whose `clips.dat` is short, since pairing a manifest row with
another clip's pixels looks normal during annotation.

## Shared models

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

## Detecting on one machine, annotating on another

Detection needs a GPU; annotation needs to be where you are. Detect & crop runs
in two halves, and only the first needs the model:

| stage | what it writes | needs |
|---|---|---|
| `--stage detect` | `norm.json`, `tracks.parquet`, `instance_features.npy`, `detect_done.json` | the model, a GPU |
| `--stage clips` | `clips.dat`, `clips.parquet`, `clip_boxes.parquet`, `cache.json`, `prepare_done.json` | the movie |

Only the first group crosses the network, a few MB of parquet plus the instance
features. `clips.dat` (`n_clips × clips.frames × crop²`, routinely gigabytes) is
written where it is read.

```bash
# on the GPU box - same registry, so model.source + sha256 give identical weights
export TRACKANNO_MODEL_REGISTRY="github://you/your-models"
trackanno-prepare --config /data/proj/config.yaml --stage detect --device 0

# back where you annotate: copy the four small files per movie, then just work
rsync -a server:/data/proj/work/ ./work/
trackanno
```

Without `--stage`, each movie gets whatever it is still missing, so step 3 finds
the tracks already there and only cuts the clips. The movie list marks it
`[detected]` and the button reads **Cut clips**. That half needs no model, so the
page opens on a machine that has no `.pt` at all.

Paths differ between machines, and `Browse` writes an absolute one. Override them
per run instead of editing `config.yaml`:

```bash
trackanno-prepare --config ./config.yaml --stage detect \
    --video-dir /mnt/scope/run7 --work-dir /scratch/run7
```

`--model` does the same for the weights and clears `model.source`, since naming a
file says the shared model is no longer wanted.

`detect_done.json` carries a digest of the `detection`, `tiling`, `tracking`,
`features`, `normalize`, `nd2` and `tiff` sections. Cutting clips against a
different one fails naming the section that changed. `detection.device` and
`detection.batch` are excluded, so a CUDA box and a laptop agree.

The handoff also carries the keypoint names, which a pose checkpoint resolves at
load time (`kpt_shape`) and `config.yaml` may never list. The machine cutting the
clips never loads the checkpoint, so without them a pose project would write a
box table with no keypoint columns and say nothing about it.

## Pose models

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

## ND2 files

ND2 movies are read natively. They carry P (multipoint), T, C and Z axes and are
reduced to one 2-D time series:

* each position `P` becomes its own movie, id `<stem>__P00`, `__P01`, …
  (`nd2.positions: all` or a list);
* `nd2.channel` takes an index or a channel name (`"BF"`);
* `nd2.z` takes an index, or `max` / `mean` / `min` for a projection;
* `nd2.fps: auto` reads the frame interval from the ND2 metadata.

TIFF carries no frame rate, so set `tiff.fps`.

## Annotation volume

`clips.window_mode`:

| value | meaning |
|---|---|
| `first` | only the movie's first window, one clip per track present there |
| `first_per_track` | the first full window of each track, wherever it starts |
| `all` | every non-overlapping window of every track (default) |

`clips.stride` sets the spacing (`= clips.frames` gives no overlap) and
`clips.max_clips_per_track` caps per-track volume.

---
