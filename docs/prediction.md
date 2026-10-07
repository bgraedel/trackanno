# Prediction, bouts and transitions

```bash
trackanno predict --config config.yaml
```

Labels every frame of every track with the trained model and writes decoded
bouts to `work/predictions.parquet`. The Model page runs the same code
(**Label all clips with the model**) and states the window count beforehand.

* Predictions never reach the annotation CSV.
* A category group the report gated off is skipped and named as such.
* The file is replaced atomically.

## How frames become bouts

1. Windows of `clips.frames` frames (the length the model was fitted at) start
   every `predict.window_stride` frames along each track. `auto` equals
   `clips.frames`, so windows tile; a smaller stride overlaps them.
2. Each window is featurised from `tracks.parquet` and the saved embeddings; no
   pixels are read. A window with fewer than `predict.min_frames` detections is
   not scored.
3. Each window's class probabilities are averaged onto the frames it covers,
   giving a posterior per frame.
4. The per-frame argmax is cut into runs. Runs shorter than
   `predict.min_bout_frames` are absorbed into the neighbouring class their
   frames score higher. 0 keeps every run.

With tiling windows a boundary can only fall on a window edge. A stride of
`s` frames places it to within about `s/2`. The cost grows linearly with the
number of windows.

## Output

One row per bout and category group:

| column | meaning |
|---|---|
| `bout_id`, `movie_id`, `track_id` | identity |
| `f_start`, `f_end`, `n_frames` | the frame span |
| `group` | the category group the bout belongs to |
| `<group>_pred`, `<group>_conf` | the label and its mean posterior (filled on that group's rows) |
| `n_covering_mean` | mean number of windows covering each frame |
| `skip_reason` | why frames have no label (no window could be featurised); empty otherwise |
| `human_overlap_frac`, `human_labelled` | share of frames a person labelled; whether it is above 0 |

## The Review page

Pick a movie and a track. One lane per category group shows the decoded bouts
along the frame axis; bar height is the bout's confidence, and a hatched stretch
is one no window could score (hover for the reason). Click or drag on the lanes
to move the footage there. The footage has play, a **view** combo and a
**marker** box. The page is read-only and says when the predictions may be out
of date.

## Transitions

```bash
trackanno transitions --config config.yaml
trackanno transitions --exclude-human-overlap   # drop bouts on hand-labelled frames
trackanno transitions --no-write                # print only
```

Reads the bouts in `predictions.parquet`, prints a summary and writes
`transitions.json` beside it.

A transition is counted only between touching bouts of one track. A bout ending
at a track end or a gap is right-censored and enters the survival estimate
(Kaplan-Meier) as a lower bound; a bout whose start was not seen is left out of
durations.

On top of the counts it fits the continuous-time Markov chain of
[Boggon et al. 2026](https://doi.org/10.1038/s41467-026-75076-8), Fig. 1h-i:

| key | meaning |
|---|---|
| `lifetime` | mean dwell time per state, exponential fit with censoring |
| `survival` | Kaplan-Meier curve with the fitted curve beside it |
| `fit_max_deviation` | widest gap between those two; above about 0.1 the exponential fit is poor |
| `jump` | P(next = j \| in i) |
| `rates` | jump probability over lifetime |
| `steady_state` | long-run share of time per state predicted by the chain |
| `occupancy_observed` | the same share measured directly |

`steady_state` is refused, with the reason, when the chain has no unique one.
Standard errors resample whole tracks, 200 times.

These are decoded predictions, so classifier confusion appears as transitions;
the report points at the confusion matrix in `suggest_report.json`.
`predict.min_bout_frames` puts a floor under every duration, so lifetimes are
biased upward.

Figures (state diagram, survival) are drawn by `trackanno.infer.figures`, used
from `notebooks/downstream_analysis.ipynb`. `notebooks/motifs.py` holds the
motif analysis.
