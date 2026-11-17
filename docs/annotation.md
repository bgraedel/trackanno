# The annotation screen

The Annotate page of the workflow window. With `ui.auto_advance: true`, picking a
value in a single-choice group commits the row and moves on once every
`required: true` group is filled. A multi-choice box never advances.

## Shortcuts

| key | action | `keys:` name |
|---|---|---|
| category keys | set a value (per option in `categories`) | |
| `Right` / `Enter` | commit and next clip | `next` |
| `Left` | previous clip | `prev` |
| `Tab` | next unannotated clip; in review: next disagreement | `next_unannotated` |
| `Space` | play / pause | `play_pause` |
| `o` | marker on / off | `overlay` |
| `s` | cycle the view: centred / locked | `stabilise` |
| `c` | outline neighbouring tracks | `context` |
| `+` / `-` / `0` | zoom in / out / reset | `zoom_in`, `zoom_out`, `zoom_reset` |
| `u` | mark uncertain | `uncertain` |
| `n` | type a note (`Esc` or `Enter` to leave) | `notes` |
| `Backspace` | clear this clip's annotation | `clear` |
| `Ctrl+Z` | undo the last annotation | `undo` |

The `?` button lists the bindings in effect.

## The view

The **marker** outlines the animal being judged. **neighbours** outlines the
other tracks in the crop. **keypoints** draws the pose skeleton (pose projects
only).

**view (s)** switches between two display transforms of the same cached pixels:

* **centred**: one view for the whole clip, centred on the track's median
  position. All motion is visible.
* **locked**: re-centred every frame. Best for body shape and beat; translation
  is invisible.

Zoom magnifies the crop without resampling the cache.

These toggles (marker, neighbours, keypoints, view, zoom) are remembered per
user in `view.ini` in the state directory, not in `config.yaml`.

## Curvature kymograph

On a pose project a panel under the clip shows the flagellum's signed curvature
along arclength (vertical, base at top) against time (horizontal). A travelling
wave shows as diagonal stripes, a standing wave as vertical ones, a mistracked
frame as a gap. Clicking moves the clip to that frame.

A smoothing spline is fitted through the keypoint chain from Base to Tip
(ordered by `keypoints.edges`; the Head is left out) each frame and
differentiated analytically. On a centerline project the curve is first
smoothed over 2 um, the same for every cell and magnification, rather than with
a tolerance per point, which on 129 close points would erase the small bends.
`ui.kymo_sign: canonical` flips each clip so its time-averaged shape has
positive curvature, where that sign is clear; otherwise the clip is shown as
measured and the panel says "sign not determined". `raw` never flips.

## Modes

The **mode** combo switches between two queues with the same keys and widgets.

| mode | queue |
|---|---|
| **annotate** | clips still to do |
| **review labelled** | clips already labelled, oldest first |

In **review labelled**:

* a pick rewrites the label and the clip stays on screen; `Right` advances;
* the side panel shows the annotator, the timestamp, *marked uncertain*, and
  where the cross-validated model disagrees with the label;
* `Tab` jumps to the next clip where it disagrees.

The comparison uses the cross-validated predictions in
`work/cv_predictions.parquet` (each clip scored by a model fitted without its
movie), not the saved model, which was fitted on these labels. Clips in the
audit holdout, clips without features and clips labelled since the last fit have
no fold prediction and are skipped by `Tab`. Reviewing does not move the resume
point or change `selected_by`.

## Clip order

The annotate queue is a seeded shuffle of all clips (`ui.shuffle`), filtered by
the **movie** combo. On start it opens at the first unannotated clip.

`suggest.select_rule: "margin"` (quote it: unquoted `off` is a YAML boolean)
moves the clips the model is least sure about (smallest gap between the top two
classes) to the front, once a model has passed its reliability gate. A batch of
20 is ranked after each background refit (`suggest.retrain_every` labels) and
inserted right after the clip on screen. Audit-holdout clips are never ranked,
so the labelled part of the holdout stays a random sample. `"off"` (default)
keeps the shuffle. With selection on, the labelled set is not a random
sample; `selected_by` records which rows the selector chose.

## The CSV

The CSV is the state: a clip whose `clip_id` is in it is done. Each annotation
is appended and `fsync`ed immediately; edits rewrite the file atomically under a
`<csv>.lock` file.

```
clip_id, movie_id, source_file, series, channel, track_id, f_start, f_end,
crop_x0, crop_y0, crop_w, crop_h, scale,
<one column per category group>, uncertain, notes, annotator, timestamp_utc,
dwell_s, suggested, suggest_conf, suggest_shown, selected_by, audit_holdout
```

Multi-choice values are `|`-joined. With `tracks.parquet`, the identity and
geometry columns map a label back to the original movie:

```python
original_x = crop_x0 + cache_x / scale
original_y = crop_y0 + cache_y / scale
```

`suggested`, `suggest_conf` and `suggest_shown` record the model's answer and
whether the annotator saw it. `selected_by` is empty when the selector did not
choose the clip (not the same as "random"), or one line of JSON with the rule,
score, rank and the model's prediction:

```python
from trackanno.annotate.store import decode_provenance

why = labels["selected_by"].map(decode_provenance)
selected = labels[why.map(lambda d: d.get("rule") == "margin")]
```
