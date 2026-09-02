# Prediction, bouts and transitions

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

## Behavioural boundaries

A dense window carries one label for every frame in it, and out of the box those
windows tile, so a change of behaviour can only ever be reported at a window
boundary. At `window_frames: 50` a cell that stops swimming at frame 137 is
called swimming until 150, or immotile from 100, and nothing downstream recovers
the 137. That is a bias in a bout duration, not noise around one. Five keys buy
a sharper boundary, and **every one of their defaults reproduces today's output
exactly** (`window_stride: auto` is `window_frames`, the tiling that has always
been there), so an existing project is unchanged until it opts in.

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
stride buys: halve the stride, halve the error, down to wherever the cost stops
being worth it.

**The cost is windows, not pixels.** Featurising one track costs about 37 ms
fixed plus about 4 ms per window (a 500-frame synthetic track, 8 keypoints,
`spectral.window: 75`), so the per-track arithmetic is linear in the window
count over a substantial constant.

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
So one number serves twice, a hard floor to `min_bout` and a mean bout length
to `viterbi`, and `auto` needs `min_bout_frames` of 2 or more, since at 1 the
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
fraction back *exactly* still says "swimming" everywhere. A 50-frame window
cannot represent the event, however good the model reading it is. Adding a 25-frame scale finds it; adding a 10-frame scale as well
puts its start on the frame it happened. Meanwhile the planted *transition* at
frame 137 is reported at 150 with one scale and at 150 with three, because every
scale's grid still cuts at its own window edges. `window_stride` is the knob for
*where* a boundary falls; `window_scales` is the knob for *what* can be seen at
all, and they are worth using together.

**Every scale carries the same weight**, and that is a decision rather than an
accident of the arithmetic. Each scale's stride is scaled with its length so all
of them cover a frame equally often, after which the plain mean already inside
the decode does the rest. The alternative, one stride for every scale, weights
a scale by its length: at 25/50/100 on a common stride the 100-frame window
covers an interior frame four times against the 25-frame window's once, so it
carries 4 of the 7 votes on its own, and the short scale you added to see a brief
event is outvoted by the window too blunt to see it. Equal weight has its own
cost, stated so it is not a surprise: a frame deep inside a long uniform bout
gets no extra confidence from the long window agreeing. It is exact in a track's
interior; near the ends the grid is clipped and the weights drift with the
covering counts, which is what `n_covering_mean` on each bout reports.

**The cost is a feature build per scale, but less than a whole one each.** All
the scales of a track are measured in one pass, so the per-track setup (the
series layout and the sliding spectral transform) is paid once and only the
per-window work repeats. On the repository's synthetic fixture, `[25, 50, 100]`
costs 3.3x the windows for 1.3x the wall time on a `task: detect` project and
2.2x on a `task: pose` one, where the spectral half dominates. Those are a
120-frame synthetic movie on one machine, so do not budget from them: the run
logs the window count, the per-scale breakdown and the multiple before it spends
any of it, and that line is the one to quote.

**Several scales need `decode`, and a model whose features survive a change of
length.** The decode for the same reason overlapping windows do: windows of
several lengths cover the same frames, and without a decode the exported colour
of a frame is decided by parquet row order. The second condition is the sharp
edge of the whole feature: window length changes what the features are *worth*,
not what they are *called*. `path_length` over 100 frames is simply twice the
`path_length` over 50 under the same column name, so a model fitted at one length
reads another length's rows on a scale it has never seen, at normal confidence,
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
columns, whose lag is `min(n_frames // 4, 25)`, an explicit function of the
window length. It is an allowlist, so a feature added to the schema later is
treated as length-dependent until somebody says otherwise; that costs a little
signal and says so, where the other way round is silent.

It composes with `feature_groups` rather than replacing it: `feature_groups:
[pose, waveform]` with `feature_scope: length_invariant` is "the beat and
waveform columns only, length-invariant" without typing out a column list. The
subset costs real accuracy, since those columns carry information that is
simply not comparable between lengths, and how much is a per-project number. `uv run
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


## Correcting a bout

A decoded boundary is a claim about when an animal changed what it was doing,
and the **Review** page (step 6, or the *Review predictions* button on the Model
page) is where you can see whether it is in the right place. You can also move
it. What you say is written to **`bout_corrections.csv`, beside
`annotations.csv`**, never into `predictions.parquet` and never into the
annotation CSV.

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

Each row is an **assertion**, these frames are this class, plus who said it,
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
are all the same edit, *name a class over a frame range*, with different
arguments. Drag an edge in the record lane, or type a range and press **Set
class**; **Merge into previous/next** takes the neighbour's class over the
selection. Every edit is checked against the invariant and saved immediately, so
there is no Save button to forget and nothing to lose by navigating away.

It is a page rather than a window that floats over one because reviewing is a
step of the work and not a lookup: it wants the whole width for the frame axis,
it belongs in the sequence the sidebar states, and one panel per project is what
keeps two views of `bout_corrections.csv` from each holding their own idea of a
track and erasing the other's on save. Leaving the page is what lets go of the
movie and the thread decoding it; coming back puts the footage on the frame it
was left on.

**Only what you touched is written.** Correct one boundary and one or two rows
appear, not a transcript of the whole track. The rest of the track is still the
model's claim, and the window draws it that way: **each group gets two lanes**,
the model's bouts above and *the record* below, your spans filled solid and
outlined, spans still the model's faded and dashed. The record lane deliberately
has no bar height, because `conf` is a mean over exactly a bout's frames and a
bout you have cut in half no longer has one anybody measured. A boundary the
decode placed keeps its stride error band; a boundary you placed is drawn as a
hairline, because yours is exact to the frame you chose.

**The player beside the lanes.** It reads frames straight from the movie, in
chunks, and it has **no scrubber of its own**: the lanes are the frame axis and
they span the whole track, while a slider under the picture would span only the
chunk decoded so far - 50 frames, a number that means nothing in the project and
which, read beside a clip length of 25, looks like one. Play, the view mode and
the marker are its only controls.

Three things it is careful about, each of which was a visible fault:

* **The frame goes in with the pixels.** Handing the viewer a new chunk and then
  seeking paints frame 0 of that chunk first, so every chunk boundary crossed by
  a scrub flashed the wrong picture.
* **The end of a track loops to the start of the track.** The viewer loops the
  array it holds, which is one chunk, so playback used to restart part-way
  through, at whatever frame the last chunk began on, and at a different place
  depending only on how the chunks divided the track.
* **The caption is one elided line while there is video**, with the whole of it
  on hover. Wrapped, it was a different number of lines for every frame and
  every track, and each of those took its line out of the picture: the video
  changing size as you switched tracks reads as the recording changing.

**Going to where a held-out model disagreed with the label.** A confusion
matrix says a model confuses two classes. It does not say which clips, and the
clips are what a person can look at. The **`< disagreement`** and
**`disagreement >`** buttons walk every **annotation window** whose recorded
label is not what a model fitted *without that window's movie* predicted for it,
across every track and every movie, **surest first**. It is the same shortlist
the annotation screen's *review disagreements* mode queues, out of the same
`cv_predictions.parquet`.

Each stop selects the track, puts the playhead in the middle of the window, aims
the frame boxes at exactly its range so *Set class* is already loaded, and says
what each side claims and how sure the model is of its half. One of the two is
wrong, and the clip is the only thing that says which.

**Most confident first** is the order, because that is the order in which they
are worth opening. A disagreement at 0.95 is either a label to fix or a model
failure worth understanding; one at 0.35 is the model shrugging, and putting
those first would bury the rest.

The unit is the annotation window and not the decoded bout, so there is no frame
arithmetic between the two claims and nothing has to have been predicted: the
walk works as soon as there is a model and something labelled.

**Two numbers, not one.** The line above the lanes says how many labelled windows
disagree *out of how many the model was asked about*, and separately **how many
it would not answer about at all**. The suggester refuses a clip in the audit
holdout and a clip with no feature row, and a count with no denominator beside it
reads as a model that agrees with everything, including everything nobody asked
it.

The audit holdout being invisible here is deliberate. Those are the only labels
the model's own score is still honest over, and a reviewer shown its opinion of
them, who then revised one, would have spent the last unbiased sample the
project has.

The same comparison drives the annotation screen's **review disagreements**
mode, which is where a label is actually *changed*; this walk is for seeing where
the boundary sits around one. Both go through one definition, so they cannot
drift into two ideas of what a disagreement is:

```python
from trackanno.infer.agreement import differs, find, summarise
```

Some things it refuses, and why. **A file of fixed windows cannot be corrected
at all**, and offers no controls rather than dead ones: a window's edges are
multiples the predictor chose, so moving one corrects nothing about the animal,
and a window relabelled by hand is a clip label, which already has a file: the
annotation CSV, reached from the Annotate page. That is the whole answer to
"where do I edit the dense predictions": you do not, because a dense window is
not a claim about when anything happened. Set `predict.decode` and the same
pixels come back as bouts, which are. A stretch no window could be featurised over cannot be
*extended*: "nothing scored this" is a fact about the run, not something a person
is in a position to assert. A class `config.yaml` does not declare is refused on
the way in, since it could not be coloured, trained on or exported, while a class the
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
genuinely contains both behaviours, the same blur a smaller `window_stride`
cannot remove, so labelling it either way teaches the next model
that a blend looks like a pure class, precisely at the transitions it is already
worst at. A majority rule would label the most misleading windows most
confidently. Straddlers are reported rather than dropped because they are the
most informative windows you have: they are where the boundaries are.

Nothing in that report is written anywhere. A derived label is what your
correction *implies* about a clip nobody looked at, and the moment that is
indistinguishable from a hand label in `annotations.csv`, nobody can say again
what a person actually judged.

## Transitions, dwell times, and the chain they imply

```bash
python -m trackanno.infer.transitions --config config.yaml
python -m trackanno.infer.transitions --plot chain.png --plot-survival curves.png
```

Reads `predictions.parquet` when it holds **bouts** and writes `transitions.json`
beside it, printing the same numbers. A windows file is refused, naming the
settings that turn a decode on: a transition is a change between two consecutive
bouts, and a frame-to-frame count measures `predict.min_bout_frames` rather than
the animal.

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

**Two drawings**, both needing matplotlib, which the project does not depend on.
`--plot` is the state diagram: node area is the steady-state share of time, edge
width is rho, and each edge carries the rate behind it. `--plot-survival` is the
survival panel: the measured curve as points, the exponential as a line over it,
on a log axis where an exponential is a straight line and a state that does not
have one bends visibly away.

**These are decoded predictions, not ground truth.** Two classes the classifier
confuses appear as transitions between them that the animal never made, and the
report points at the confusion matrix in `suggest_report.json` where there is
one. `predict.min_bout_frames` puts a floor under every duration, which lifts
every lifetime, so the rates are lower bounds. The report says so, with the
number.

---
