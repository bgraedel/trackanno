# Tests and packaging

```bash
uv run pytest
uv run pytest tests/test_kymograph.py -v
uv run pytest -k "prediction or orphan"
```

35 suites, 447 tests, 3066 named checks. `uv sync` installs pytest. Nothing is
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
| `test_workflow.py` | projects, config editing, background jobs, the workflow window, the two prepare stages |
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
| `test_review_mode.py` | reading a labelling session back: the queue, the provenance that must survive an edit |
| `test_agreement.py` | which annotation windows the folds contradicted, that this is the confusion matrix's own off-diagonal, and the walk through them |
| `test_transitions.py` | transitions, dwell times and the fitted chain, against planted bouts |
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

# Notes

* `requires-python >=3.10,<3.13`; `.python-version` pins 3.12, since torch has no
  3.13 wheels. `uv` fetches the interpreter.
* `uv sync --extra cuda` and `--extra cpu` are declared as conflicting extras,
  so the lockfile carries a separate torch resolution for each. On a small GPU,
  lower `detection.batch` and `tiling.tile` first.
* The CUDA index is pinned to `cu128` rather than the newest, because CUDA 12.8
  runs on an older driver than CUDA 13 does. Bump it when the GPU boxes are
  current. Do not go back to `cu124`: that channel stopped at torch 2.6.0 while
  PyPI is many versions past it, so pinning it freezes the whole install.
* `opencv-python` is overridden out of the resolution in favour of
  `opencv-python-headless`. ultralytics pulls the former, which ships its own Qt
  plugins and conflicts with PyQt at startup.
* `uv sync --extra sleap` with `prepare.py --sleap` writes a SLEAP `.h5` per
  movie for keypoint-MoSeq compatibility.
* Project data is gitignored: `work/`, `annotations.csv`, `predictions.parquet`
  and `config.yaml`, which carries absolute paths to the recordings.
