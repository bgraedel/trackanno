# Configuration

`config.yaml` drives everything. *New project* writes it from the packaged
template (`src/trackanno/default_config.yaml`), with a comment on every key.
Relative paths resolve against the file's folder.

The loader warns about unknown sections and keys and ignores them. A malformed
value stops the load with a one-line message.

## Keys

| key | default | meaning |
|---|---|---|
| `paths.video_dir` | `./videos` | folder holding the movies |
| `paths.model` | `./best.pt` | YOLO weights, unless `model.source` is set |
| `paths.work_dir` | `./work` | clip cache and per-movie tables; keep it local |
| `paths.csv` | `./annotations.csv` | the annotation log |
| `paths.patterns` | `*.tif`, `*.tiff`, `*.nd2` | which files are movies |
| `nd2.channel`, `nd2.z`, `nd2.positions`, `nd2.fps` | `0`, `0`, `all`, `auto` | see [ND2 files](detection.md#nd2-files) |
| `tiff.fps`, `tiff.channel` | `100`, `0` | TIFF stores no frame rate |
| `tiff.pixel_size_um` | empty | TIFF pixel size; needed by a centerline project |
| `normalize.mode` | `per_frame` | detector input: `per_frame` or `fixed` (one mapping per movie) |
| `normalize.percentiles` | `[0.5, 99.5]` | mapped to 0 and 255 |
| `detection.task` | `detect` | `detect` (boxes) or `pose` (keypoints) |
| `detection.conf` | `0.25` | minimum confidence |
| `detection.device` | `auto` | `auto`, `cpu`, `mps` or a CUDA index |
| `detection.batch`, `detection.imgsz`, `detection.half` | `8`, `640`, `false` | inference settings |
| `model.source`, `model.version`, `model.sha256` | empty | a [shared model](detection.md#shared-models) |
| `tiling.enabled` | `auto` | tile only frames larger than one tile |
| `tiling.tile`, `tiling.overlap`, `tiling.merge_iou` | `640`, `0.25`, `0.6` | tile geometry and merge |
| `tracking.backend` | `tracktrack` | see [Tracking](detection.md#tracking) |
| `tracking.min_track_length` | `40` | shorter tracks are dropped |
| `tracking.border_margin_px`, `tracking.border_max_frac` | `0`, `0.0` | [frame edge](detection.md#the-frame-edge) |
| `tracking.overrides` | `{gmc_method: none}` | passed to the tracker |
| `keypoints.nodes`, `keypoints.edges` | `[]` | names in the model's order; skeleton links |
| `centerline.enabled`, `.checkpoint`, `.hint_model`, `.code` | off, empty | see [Centerline model](detection.md#centerline-model) |
| `centerline.stack_stride`, `.hint_conf`, `.hint_tile`, `.hint_imgsz`, `.batch` | `1`, `0.3`, `640`, `640`, `16` | crop stack and hint detector settings |
| `clips.frames` | `50` | frames per clip, and the window the model is fitted at |
| `clips.stride` | `= clips.frames` | frames between clip starts |
| `clips.window_mode` | `all` | `first`, `first_per_track` or `all` |
| `clips.follow` | `smooth` | `smooth`, `fixed` or `track` |
| `clips.pad`, `clips.crop_size` | `1.6`, `auto` | crop margin and cache size |
| `clips.min_clip_frames` | `40` | detected frames a window needs |
| `features.enabled` | `true` | save detector embeddings during detection |
| `spectral.window` | `75` | frames the beat spectrum is measured over |
| `suggest.*` | | see [the suggester](suggester.md#settings) |
| `predict.window_stride` | `auto` | frames between window starts; `auto` is `clips.frames` |
| `predict.min_bout_frames` | `0` | shorter runs join a neighbour |
| `predict.min_frames` | `2` | detections a window needs |
| `ui.theme` | `dark` | `dark`, `light` or `system`; applies on next start |
| `ui.fps` | `20` | playback rate |
| `ui.auto_advance` | `true` | one keypress labels and advances |
| `ui.shuffle` | `true` | seeded random clip order |
| `ui.annotator` | empty | written into every CSV row |
| `ui.kymo_sign` | `canonical` | `canonical` or `raw` |
| `render.fps`, `render.out_dir` | `auto`, `./renders` | overlay movie output |
| `render.hide_unlabelled`, `render.show_skeleton` | `false`, `false` | what overlays draw |

`categories` lists the label groups. Each group has a `name`, a `type`
(`single` or `multi`), `required`, and `options`, each with a `name`, a `key`
shortcut and a `color`. `keys` rebinds the navigation shortcuts
([list](annotation.md#shortcuts)). A category key that collides with a
navigation key is refused.

## Fixed settings and view toggles

Settings that are not configurable live as constants in `FIXED_SETTINGS` in
`src/trackanno/config.py` (for example `suggest.depth`,
`suggest.min_macro_f1`, `clips.smooth_window`, `ui.resume`). A config that
sets one loads with a warning naming the value in use.

Display toggles of the annotation window (`show_overlay`, `show_skeleton`,
`show_context_tracks`, `follow`, `view_zoom`, `show_contrast`, `contrast`,
`gamma`) are per user, kept in `view.ini` in the state directory (`VIEW_SETTINGS` in `config.py`). In
`config.yaml` they are ignored with a warning.

The state directory is `$TRACKANNO_STATE_DIR`, else `%APPDATA%\trackanno`, else
`$XDG_CONFIG_HOME/trackanno`, else `~/.config/trackanno`. It also holds the
recent-projects list, templates, the model registry URL and the model cache.
