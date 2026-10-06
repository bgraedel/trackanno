# Prediction, bouts and transitions

```bash
trackanno predict                          # every frame -> bouts in predictions.parquet
```

Labels every frame of every track in one pass and cuts the result into bouts.
Three rules:

* **Predictions never reach the annotation CSV.** They go to
  `predictions.parquet`, or the next model trains on its own output and the
  ground truth is lost. Every bout carries `human_overlap_frac`, the share of
  its frames a person labelled, and `human_labelled` when that share is above 0.
* **A group the report gated off is skipped**, and named as such. The decision
  comes from the saved report rather than from the presence of a fitted model.
* **The file is replaced atomically**, since everything downstream reads it as
  authoritative.

Windows of `clips.frames` (the length the model was fitted at) are laid over
every frame of every track. It reads no pixels and re-detects nothing; features
come from `tracks.parquet`, the saved embeddings and the frame rate. A window
too thin to measure still gets a row carrying `n_present` and `skip_reason`, and
the frames only it covered become a bout with an empty label and a reason.

The Model page runs the same code with a progress bar and states the window
count beforehand. Embeddings are summarised in the PCA basis stored in the
model at training time, never refitted and never read from disk, since a
refit yields a different basis silently. A model without its basis is refused.

## Behavioural boundaries

Each window's class probabilities are averaged onto the frames it covers, and
the per-frame argmax is cut into **bouts** - disjoint, variable-length runs, one
row each in `predictions.parquet`. Out of the box the windows tile, so a change
of behaviour can only be reported at a window boundary: at `clips.frames: 50` a
cell that stops swimming at frame 137 is called swimming until 150.

```yaml
predict:
  window_stride: auto     # = clips.frames: the windows tile
  min_frames: 2           # detections a window needs to be featurised
  min_bout_frames: 0      # the shortest bout you would accept as real
```

**A `window_stride` below `clips.frames` overlaps the windows.** Each frame is
then scored by several of them, and their mean gives a posterior per frame
instead of per window. That removes the *quantisation* of a boundary to a
multiple of the window length. It does not remove the *blur*: a window
straddling a transition genuinely contains both behaviours and its features are
a blend of the two, and no stride fixes that. Boundary error against a planted transition, in frames:

| `window_stride` | mean abs. error | max |
|---|---|---|
| 50 (tiling) | 12.24 | 24 |
| 25 | 6.24 | 12 |
| 10 | 2.23 | 4 |
| 5 | 1.20 | 2 |

That is `stride/4` on average and `stride/2` at worst: the quantisation limit
with nothing left over. Worth stating, because it says exactly what a smaller
stride buys: halve the stride, halve the error, down to wherever the cost stops
being worth it.

**The cost is windows, not pixels.** Featurising one track costs a fixed
per-track term plus about 1.5 ms per window (a 500-frame synthetic track, 8
keypoints, `spectral.window: 75`), so the per-track arithmetic is linear in the
window count over a substantial constant.

The per-window term was 6.2 ms until the keypoint grid and the spectral series
were built once per track instead of once per window; both are sliced now, and
the values are unchanged. Scoring the rows is not the cost: CatBoost predicts
about 1 us per row on eight threads, so a 200 000-window run spends under a
second in the model and the rest in the arithmetic above.

Do not extrapolate that microbenchmark to a project. Measured end to end on a
real one (three movies, 500 frames each, `clips.frames: 25`), going from the
tiling default to `window_stride: 5` took **18s to 74s**, about 4x for 5x the
windows, including rebuilding the feature table at the finer stride. The
single-track figure predicts 2.8x and is wrong at project scale, because a run
pays per-movie and per-track costs the one-track measurement amortises away.
Budget from the 4x.

The dense stage reads no pixels and re-detects nothing, so this stays cheap in
absolute terms: arithmetic over `tracks.parquet` and the saved embeddings, and
no second pass over the recording.

**`min_bout_frames`** absorbs every run shorter than it into whichever
neighbouring class the absorbed frames themselves score higher. 0 keeps every
run of the argmax.


## Reviewing bouts

The **Review** page draws one track's decoded bouts along the frame axis, one
lane per group, beside the footage. Bar height is the bout's mean posterior for
its own class. It is read-only: nothing on the page writes a file.

Leaving the page is what lets go of the movie and the thread decoding it; coming
back puts the footage on the frame it was left on.

**The player beside the lanes.** It reads frames straight from the movie, in
chunks, and it has **no scrubber of its own**: the lanes are the frame axis and
they span the whole track, while a slider under the picture would span only the
chunk decoded so far - 50 frames, a number that means nothing in the project and
which, read beside a clip length of 25, looks like one. Play, the view mode and
the marker are its only controls.

Two things it is careful about, each of which was a visible fault:

* **The frame goes in with the pixels.** Handing the viewer a new chunk and then
  seeking paints frame 0 of that chunk first, so every chunk boundary crossed by
  a scrub flashed the wrong picture.
* **The end of a track loops to the start of the track.** The viewer loops the
  array it holds, which is one chunk, so playback used to restart part-way
  through, at whatever frame the last chunk began on, and at a different place
  depending only on how the chunks divided the track.

## Transitions, dwell times, and the chain they imply

```bash
trackanno transitions --config config.yaml
```

Reads `predictions.parquet` when it holds **bouts** and writes `transitions.json`
beside it, printing the same numbers. A file from an older version that holds
windows is refused: a transition is a change between two consecutive bouts.

**What is counted.** A pair is formed only inside one `(movie, track)` and only
across touching bouts; a gap, a track end and a movie end each break the pair
rather than joining across it. A bout running to a track end or into a gap is
right-censored, its true length being at least what was seen, and Kaplan-Meier
folds it in as a lower bound. A bout whose own start was never seen is dropped
from every duration. Every rate is emitted beside the count it came from.

**The chain.** On top of those counts the report fits the continuous-time Markov
chain of [Boggon et al. 2026](https://doi.org/10.1038/s41467-026-75076-8),
Fig. 1h and 1i:

| in `transitions.json` | what it is |
|---|---|
| `lifetime` | the mean of an exponential fitted to the durations by maximum likelihood: total time at risk over ends seen, so a censored bout contributes its time but not an end |
| `survival` | the measured Kaplan-Meier curve P(T > t), with the fitted curve alongside it point for point |
| `fit_max_deviation` | the widest gap between those two. Above about 0.1 the state does not leave at a constant rate, and one number cannot stand for its lifetime |
| `jump` | rho(i->j) = P(next = j \| in i), which is what the counts give directly |
| `rates` | q(i->j) = rho(i->j) / lifetime(i) |
| `steady_state` | the share of **time** each state holds in the long run: p(i) proportional to nu(i) x lifetime(i), with nu the jump chain's own stationary law |
| `occupancy_observed` | that same share, measured straight off the record |

The last two are the pair worth reading together. `steady_state` is what the
fitted chain predicts; `occupancy_observed` is what actually happened. They come
apart exactly when the chain does not describe these bouts, and neither number
says that on its own. A state entered rarely and held for a long time takes up
most of the record, which is why this is not a count of bouts.

A steady state is **refused rather than approximated** where the chain has no
unique one: a class with no observed successor, or classes splitting into groups
that never reach each other, each with a stationary law of its own. The refusal
names which.

**Every `se` resamples whole tracks**, 200 times, not bouts. The bouts within one
track are a dependent sequence, and resampling them would report an error bar far
narrower than the data supports.

The state diagram and the survival panel are drawn by
`trackanno.infer.figures`, from the downstream-analysis notebook.

**These are decoded predictions, not ground truth.** Two classes the classifier
confuses appear as transitions between them that the animal never made, and the
report points at the confusion matrix in `suggest_report.json` where there is
one. `predict.min_bout_frames` puts a floor under every duration, which lifts
every lifetime, so the rates are lower bounds. The report says so, with the
number.

---
