# The annotation screen

The screen the workflow window hosts on its Annotate page.

**One keypress labels and advances.** With `ui.auto_advance: true`, picking a
value in a single-choice group commits the row and moves on once every
`required: true` group is filled. Ticking a multi-choice box never advances.

## Shortcuts

| key | action |
|---|---|
| `1`…`5`, `q`/`w`/`e` | category values (defined per option in `config.yaml`) |
| `→` / `Enter` | commit and next clip |
| `←` | previous clip |
| `Tab` | next unannotated clip (in review mode: next label the model argues with) |
| `Space` | play / pause |
| `o` | marker on/off |
| `s` | toggle the view: centred / locked |
| `c` | outline neighbouring tracks |
| `+` / `-` / `0` | zoom in / out / reset |
| `u` | mark uncertain |
| `Backspace` | clear this clip's annotation |
| `Ctrl+Z` | undo the last annotation |

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
* **locked**, re-centred every frame. Best for body shape and flagellar beat;
  translation is invisible by construction.

Centring is a display transform with the outside padded, so a clip cropped at the
movie border still shows the animal at the exact centre. Frames where the track
dropped out are filled from the nearest detected frame. `ui.view_zoom` and
`+`/`-`/`0` magnify into the crop as a transform, resampling nothing.

## Curvature kymograph

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
| `ui.kymo_sign` | `canonical` (default) or `raw` |
| `ui.kymo_smooth_px` | spline tolerance against the keypoints, px (default `0.6`) |

These take effect on the open annotator without reloading the project. The colour
scale is diverging with neutral grey at zero.

**Resuming.** The CSV is the state: any clip whose `clip_id` appears in it is
done. On startup the GUI jumps to the first unannotated clip
(`ui.resume: last_position` returns to the previous position). Every annotation
is appended and `fsync`ed immediately, so a crash costs at most the clip on
screen. Annotated clips reopen with their labels loaded, and edits rewrite the
CSV atomically.

## Reviewing what you labelled

The **mode** combo at the top left of the annotation screen switches between
two queues over the same widgets, the same keys and the same video.

| mode | the queue |
|---|---|
| **annotate** | the clips still to do |
| **review labelled** | the ones already done, oldest first |

### review labelled

Where a session is read back. Four things change, and nothing else does.

* **The queue is the labels, in the order they were made.** The seeded shuffle
  is skipped, since reading three labels back would otherwise mean scrolling
  through the hundreds of clips between them. The selector's batch is not spliced in either,
  since nothing here is waiting to be chosen.
* **A keypress rewrites the label and stays put.** Auto-advance is for
  labelling; reviewing, the next thing you do after changing a label is look at
  the clip again, so the key that changed it must not take it away. `→`
  advances, as always.
* **The side panel says whose judgement this is**: the annotator and the
  timestamp off the row, plus *marked uncertain* where that was ticked. It also
  says **where the model would now say otherwise**, naming the class it would
  pick.
  Nothing is shown for a clip in the audit holdout: the suggester refuses to
  answer about those on purpose, and a model's opinion of the last unbiased
  sample there is must not reach a reviewer.
* **`Tab` jumps to the next label the model argues with.** That is where to
  start: a label the model agrees with is evidence about neither of them, and
  one it does not is either a mistake or the thing the model still has to learn.
  Pick a value and the note under the metadata says the two now agree, without
  the clip moving.

> **It has to be a held-out prediction, and this is the whole subtlety.** The
> saved model was fitted on every label there is. Ask it about one of them and
> it is answering about a row it has memorised, reproducing essentially all of
> them however wrong they are.
>
> So the comparison uses the **cross-validated** predictions instead: for each
> labelled clip, what a model fitted **without that clip's movie** said about
> it. Those are the same predictions the report's confusion matrix is a tally
> of. `cv_predictions.parquet` is where a training run leaves them; on a model
> trained before this build existed the screen says to train again rather than
> quietly falling back on the model's own memory.

The fold predictions are **re-read when a background refit lands**, and the
queue on screen is left where it is, because re-ordering under a reviewer
mid-clip loses their place.

`Tab` skips a clip **no fold scored**: one in the **audit holdout**, which is
never trained on, one whose features are missing, and one **labelled since the
last fit**, which the next fit will cover.

Reviewing does not move the resume point, and it does not touch `selected_by`.
That column is the selector's own account of why a clip was put in front of
somebody, and nobody revising a label months later has a selector to ask; an
edit that says nothing about it keeps what the row already said rather than
overwriting it with *not recorded*.

## Which clip comes next

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
| `suggest.select_rule` | `"off"` | `off` or `margin` (smallest gap between the top two classes). Quote it: an unquoted `off` is a YAML boolean |
| `suggest.select_batch_size` | `20` | clips per batch, twice `retrain_every` so a batch outlasts the model that ranked it |

**What you will notice.** The session opens on the model's top pick rather than
on the first clip of the shuffle. Clips the selector said nothing about keep
their shuffle order behind the batch.

A batch is recomputed when the model changes: `suggest.retrain_every` labels
trigger a background refit, and the queue is re-ranked when that refit lands,
never once per label. **The clip on screen when it lands does not move.** The
fresh picks are spliced in immediately after it, which is the earliest place a
forward-only cursor can still reach; anything put in front of the cursor would
never be seen, because navigation only ever scans forward. If the selector fails, say on a model trained at a
different window length, the status line says that instead and the queue falls back
to the shuffle, so a broken selector costs the ordering and never the ability to
annotate.

**The labelled set stops being a random sample of the clips the moment this is
on.** That is what selection does: the rows that end up in the CSV are the ones
the model asked for, which is to say the hard ones, so an accuracy computed over
them afterwards is an accuracy on a set assembled to be hard, lower than the
truth by an amount the file alone cannot say.

Two things keep an unbiased number alive. Clips in the audit holdout
(`suggest.audit_fraction`) are never ranked, so they reach the annotator only
through the shuffle behind the batch and the labelled part of the holdout stays
a random sample of it. That is the sample `audit_accuracy` in the model report
is computed over. And every row records why its clip was shown, which is what
lets anyone separate the two groups later.

## The CSV

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
JSON. It is the column that keeps the paragraph above computable:

| cell | means |
|---|---|
| empty | **not recorded.** The selector did not choose this clip. This is *not* "chosen at random" and nothing may read it as such |
| `{"rule":"margin","score":0.83,"rank":0,...}` | the selector chose this clip: the rule that chose it, the score that earned it, its rank in the batch, the pool it was drawn from and the model's own prediction at the time |

```python
from trackanno.annotate.store import decode_provenance

why = labels["selected_by"].map(decode_provenance)
selected = labels[why.map(lambda d: d.get("rule") == "margin")]
```

Keys are sorted and the encoding is ASCII, so re-recording the same decision is a
no-op in a diff and no rule name can put a newline inside a row. Rows written
before the column existed are empty; opening such a file says so once, out loud,
because filling those blanks in with "random" is exactly the conflation the column
was added to prevent.

---
