# Detection and the clip cache

```bash
trackanno prepare --list                       # visible movies
trackanno prepare                              # every movie not yet prepared
trackanno prepare --movies MOVIE_A --force     # redo one
trackanno prepare --limit-frames 100           # smoke test
trackanno prepare --self-test --tile 320       # compare tiled and whole-frame inference
```

`--device`, `--tile`, `--batch`, `--model`, `--video-dir` and `--work-dir`
override the config for one run.

**Device.** `detection.device: auto` takes the first CUDA device, then MPS, then
the CPU. A named device that is absent is reported and the run uses the CPU. The
log names the device actually used.

**Tiling.** Each frame is cut into overlapping tiles (`tiling.tile`,
`tiling.overlap`), run as one batch, mapped back to full-frame coordinates and
merged with a class-agnostic NMS (`tiling.merge_iou`). A box cut by an interior
tile border gives way to the neighbouring tile's complete copy. Frames that fit
one tile skip tiling.

## Tracking

The merged full-frame boxes drive an ultralytics tracker directly.
`tracking.backend` takes `tracktrack` (default, for crowded scenes with
occlusion), `bytetrack`, `botsort`, `ocsort`, `deepocsort`, `fasttrack`, or
`greedy`, a nearest-centroid tracker that suits high frame rates.
`tracking.overrides` passes parameters to the tracker. Tracks shorter than
`tracking.min_track_length` frames are dropped.

## Output

In `work/<movie_id>/`:

| file | contents |
|---|---|
| `tracks.parquet` | every detection, in full-frame coordinates |
| `instance_features.npy` | detector embeddings, row-aligned with `tracks.parquet` (`features.enabled`) |
| `norm.json` | the intensity mapping the detector sees, and the range the clip cache maps to 8 bits |
| `detect_done.json` | detection marker and fingerprint |
| `clips.parquet` | one row per clip, with its crop rectangle and scale |
| `clip_boxes.parquet` | overlay geometry in cache coordinates |
| `clips.dat` | cropped pixels, uncompressed `uint8` memmap |
| `cache.json` | cache header, with the raw intensities of 0 and 255 |
| `prepare_done.json` | clip-stage marker; re-running skips finished movies |

The cache is uncompressed so a clip loads with one contiguous read. Keep
`paths.work_dir` on a local disk; `trackanno prepare` warns when it is on a
network share.

Each movie is committed at once: files are written beside their final names and
renamed at the end, so a cancelled run leaves the previous cache intact.

## Crops

The crop is sized to contain the animal plus `clips.pad` of margin. Crops larger
than `clips.crop_size` are downscaled, with the `scale` recorded so every cached
pixel maps back to an original coordinate. A crop overhanging the movie border is
zero-padded. `clips.follow` sets how the crop moves within a clip:

| value | crop |
|---|---|
| `smooth` | moving average of the track centre (default) |
| `fixed` | one crop for the whole clip; all motion visible |
| `track` | re-centred every frame; translation invisible |

`clips.window_mode` sets how many clips are cut: `first` (the movie's first
window), `first_per_track` (each track's first full window) or `all` (every
window, default). `clips.stride` spaces them; equal to `clips.frames` they tile.
A window needs `clips.min_clip_frames` detected frames.

## The frame edge

`tracking.border_margin_px` (0, off) excludes detections whose box comes within
that many pixels of the frame edge. Keypoints on an animal cut off by the frame
are unreliable. Excluded frames count as missing; the track stays.
`tracking.border_max_frac` is the share of one clip's frames allowed at the edge
before the clip is dropped (`0.0` drops any clip that reaches it). Dense
prediction skips such windows with a `skip_reason`. Detection reports how many
tracks reached the edge.

## Shared models

**Choose shared model...** (Configure page and *New project*) lists the models a
registry offers, downloads the chosen one into a shared cache and applies the
settings it was trained with. Only models this project can run are listed.

```yaml
model:
  source: trypanosome-pose   # registry id or https URL; "" uses paths.model
  version: v2
  sha256: 9f3c...            # written on the first fetch, pins the weights
```

`model.source` outranks `paths.model`. Picking a `.pt` with **Browse** clears it.

The registry is a GitHub release holding `index.json`. Each entry names an
`id`, `version`, `framework`, `task`, `url`, `sha256` and optionally keypoint
names, edges, detection and tiling settings under `config`, and `assumes`
(frame rate, pixel size) that is checked and warned about, never applied.
trackanno runs `ultralytics` `detect` and `pose` models; other entries are listed
as not runnable.

Security rules, since loading a checkpoint runs code:

* The registry URL is a machine setting (state directory, or
  `TRACKANNO_MODEL_REGISTRY`), never a project one. It takes
  `github://owner/repo[@tag]` or an https URL to an `index.json`.
* https only; a download whose sha256 does not match is deleted.
* A cache hit never touches the network.

## Detecting on one machine, annotating on another

| stage | writes | needs |
|---|---|---|
| `--stage detect` | `norm.json`, `tracks.parquet`, `instance_features.npy`, `detect_done.json` | the model, a GPU |
| `--stage clips` | `clips.dat`, `clips.parquet`, `clip_boxes.parquet`, `cache.json`, `prepare_done.json` | the movie |

Only the first group has to travel.

```bash
# on the GPU box
trackanno prepare --config /data/proj/config.yaml --stage detect --device 0 \
    --video-dir /mnt/scope/run7 --work-dir /scratch/run7

# where you annotate: copy work/, then run Detect & crop
rsync -a server:/scratch/run7/ ./work/
```

Without `--stage`, each movie gets what it is missing. The Detect & crop page
marks such a movie `[detected]` and the button reads **Cut clips**, which needs
no model.

`detect_done.json` fingerprints the settings that change detection (detection,
tiling, tracker, features, normalisation, channel, z, fps, centerline) and the weights'
sha256; device, batch and `--tile` are left out. `prepare_done.json` fingerprints
the clip keys (`clips.*`, `tracking.min_track_length`, `tracking.border_*`).
A movie whose fingerprint does not match is reported with the changed keys,
marked `[ stale ]` and skipped, since new track ids or a new clip grid orphan the
labels. `--force` (or "redo movies that are already prepared") redoes it;
`--stage clips --force` re-cuts without detecting.

## Pose models

Set `detection.task: pose`. The keypoint count comes from the checkpoint; the
names go in `keypoints.nodes` (placeholders `kp0`, `kp1`, ... until renamed).
Order matters: keypoint 0 is the head, 1 the body base, 1 onwards the flagellum
chain. On the Keypoints tab, **Read from model file** takes the count from the
checkpoint, a rename carries through the skeleton links, and **connect in
order** wires 0-1, 1-2, and so on. `edges` affect display only.

## Centerline model

A centerline checkpoint can replace the pose flagellum. After tracking, each
tracked box is cropped as the model was trained: a stack of frames around it,
body and flagellum hint boxes from a 3-class detector (animal, body,
flagellum), and the model's own curves from earlier frames of the same track.
The output overwrites the keypoint columns: node 0 keeps the pose Head, nodes 1
onwards are the model's own points from Base to Tip (129 for the current
checkpoints), and each node's confidence is the model's visibility there.
`keypoints.nodes` is filled in with Head, Base, C1, ..., Tip on the first run.

```yaml
detection:
  task: pose                 # the pose model still tracks and gives the Head
tiff:
  pixel_size_um: 0.325       # the beat smoothing is in um; ND2 records it
centerline:
  enabled: true
  checkpoint: path/to/last.ckpt
  hint_model: path/to/detect3.pt
  code: path/to/centerline_regression          # unless the package is installed
```

In the GUI it is a third choice of the task on the Configure page, `pose +
centerline`, beside `pose` and `detect`; it is saved as the YAML above. Choosing
it shows both models and the TIFF pixel size under it, and the Keypoints tab then
shows the names the model writes instead of reading the pose model. It needs `timm`
(`trackanno[centerline]`). Every frame runs the hint detector plus one model
pass per box, so detection takes much longer. The hint detector is tiled like
the pose model: `centerline.hint_tile` native pixels per tile (640, the detect3
training tiles), seen at `centerline.hint_imgsz`, with `tiling.overlap` and
`tiling.merge_iou`; tiles are merged per class, so an animal box never
suppresses its own body. Sent whole (`hint_tile: 0`), a frame is shrunk to
`hint_imgsz`: 0.27x for a 2400 px 20x frame, where a third of the boxes found
no hint cell. For a movie well above 100 fps, set `centerline.stack_stride`
so the stack spans the time window the model was trained on. To compare with
the pose chain, prepare the same movies in two projects, one with the setting
on and one with it off.

The beat per frame (`inst_*`) smooths a centerline over 2 um rather than over
one keypoint segment, so every cell is smoothed alike whatever its length or
the magnification.

## ND2 files

ND2 movies are reduced to one 2D time series:

* each position becomes a movie, id `<stem>__P00`, `__P01`, ...
  (`nd2.positions: all` or a list);
* `nd2.channel` takes an index or a channel name (`"BF"`);
* `nd2.z` takes an index, or `max`, `mean` or `min` for a projection;
* `nd2.fps: auto` reads the frame interval from the file.

TIFF carries no frame rate, so set `tiff.fps`.
