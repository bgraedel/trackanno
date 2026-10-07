# Cleanup plan

Goal: cut scope back to the core pipeline (detect, clips, annotate, train, dense
bouts, transitions, export), fix the bugs found in the 2026-10-06 review, then
restructure what is left. One commit per checkbox where practical.

## Decisions

| Area | Decision |
|---|---|
| Discovery / subclustering | Cut entirely (GUI page, discover.py, subcluster.py, docs, tests) |
| Active learning | Keep the margin rule only |
| Multi-scale prediction | Cut `window_scales`, `feature_scope`, length-invariance |
| Longer context | New: one `context_frames` knob adding a context feature family |
| Feature column picker | YAML `feature_columns` + CLI `--list-features`; no GUI dialog |
| CNN features | Generalise into a pluggable embedding input (YOLO head = one provider, autoencoder later); off by default |
| Bout corrections | Cut; Review page becomes a read-only bout viewer |
| Review page | Keep minimal: movie/track pick, model lane, footage |
| Prediction output | Decoded dense bouts only (no per-clip, no window rows) |
| Decoders | Cut Viterbi; keep `min_bout` |
| motifs.py | Move to notebooks/ |
| transitions.py | Cut its own plots and long text summaries (figures.py draws them) |
| Agreement walk, `save_posteriors` | Cut |
| Annotation review | Modes: annotate + review labelled; Tab jumps to next CV disagreement |
| Annotation extras | Cut minimap/thumbnails, smooth follow, standalone window, nav duplicates (unannotated-only, Ctrl+U, Ctrl+S), kymograph extras |
| Hub | Fold Distribution into Model page; cut device probe, `hide_steps`, Detect device override, "Review predictions" button; repair links to CLI; theme config-only (no toggle). Keep templates, top-features table, plan line (shrunk to a window count) |
| Infra | Cut `render --tif`, SLEAP export, normalizer modes other than fixed + per_frame. Keep model registry and tracker backends |
| Config | Full rebuild: one metadata-driven schema, ~60 knobs to constants, view toggles to QSettings, form ~25-30 fields |
| Tests | Dedicated overhaul phase |
| Prose | Dedicated final pass |

## Status

Phases 0-4 merged (2026-10-07), including the Phase 4 review fixes; master at
f2e6c4d passes the suite (2584 checks). Real projects' prepared movies still read
as fresh. Phase 5 and the docs half of Phase 6 in flight; source prose after Phase 5.

Not changed on purpose: loader defaults for omitted keys (`detection.task` etc.)
match the shipped template; all real projects set them explicitly.

**Known follow-ups, not yet done:**
- `Suggester(cfg)` load in `AnnotationWidget.__init__` still on the UI thread.
- Threads outside `run_bg` (retrain, review loader, chunk decoder, TaskPanel) are
  not joined on close; `joblib.dump(model.joblib)` is not atomic.
- Commit can wait up to `LOCK_WAIT_S` on the UI thread; stale-lock race between
  two processes; undo overwrites another writer's later edit of the same clip.
- Prepare scan list keyed per config path only, so a `video_dir` change shows the
  old folder until reopen.

**Working method:** parallel agents in git worktrees on disjoint scopes, merged
into master by the main session; full suite (`uv run pytest -q`, ~5 min, prints
"N checks passed") plus an independent read-only verifier after each phase.
Commit messages: short imperative subject, 1-3 sentence body, no trailers.
Merged worktrees under `.claude/worktrees/` can be removed with
`git -c core.longpaths=true worktree remove` (`.claude/` is excluded via
`.git/info/exclude`).

## Phase 0: baseline
- [x] Checkpoint commit of the current working tree

## Phase 1: data-integrity fixes (before any cut)
- [x] Tracker id counter never reset under `tracktrack` (`detect/tracking.py:167`): call `BaseTrack.reset_id()`; test two movies in one process start at the same id
- [x] Audit holdout hashed per clip (`config.py:602`): hash `movie_id::track_id`; existing stamped labels keep their stamp
- [x] Detection fingerprint (`detect/prepare.py:133`): explicit list of detection-affecting keys, include model sha256, exclude clip-stage keys and fps; `--tile` must not change the fingerprint; stale handoff re-detects instead of FAILED; clip-stage fingerprint for `clips.*`
- [x] Clip cache check uses `<` (`annotate/clips.py:733`): require exact size; commit header/manifest last
- [x] `tip_angle_std` gap unwrap (`measure/trackseries.py:497`): unwrap valid samples only
- [x] Console scripts point at deleted modules (`pyproject.toml:85-89`): repoint now (dispatcher comes in Phase 3)
- [x] `sys.excepthook` in GUI `main()` so a slot exception logs instead of aborting
- [x] Configure: `maybe_save` always True and background reloads wipe refused edits (`gui/hub.py:996, 1241`); unguarded `float()` on percentiles (`gui/configform.py:915`)
- [x] Export "draw only annotated tracks" cannot turn off (`view/render.py:439`)
- [x] Dead code after `return` (`measure/wavespectra.py:814`)

## Phase 2: cuts (each a deletion commit, tests and docs for the cut code go with it)
- [x] Discovery / subclustering, incl. warm-decoder code in `TrackFootage`
- [x] Bout corrections (`annotate/corrections.py`, record lane, undo, derived labels, merge), agreement walk, stride bands, hatching, coverage counts, `save_posteriors` + posterior trace
- [x] Active learning down to margin rule (`learn/select.py`, `select_*` keys, batch splicing in `gui/app.py`)
- [x] Multi-scale: `window_scales`, `feature_scope`, invariance refusals, column-picker dialog (keep `feature_columns` in YAML)
- [x] Prediction: bouts only; drop `RowKind` clip/window, manifest coverage, `include_labelled`; drop Viterbi; shrink `DensePlan`/plan line to a window count
- [x] motifs.py to notebooks/; transitions plots, `summarise` prose, duplicate pair loop
- [x] Annotation screen extras (see Decisions)
- [x] Hub extras (see Decisions); repair links becomes a CLI command
- [x] `render --tif`, SLEAP export, extra normalizer modes

## Phase 3: new and generalised features
- [x] Single CLI: `trackanno <prepare|suggest|predict|render|transitions|repair-links>` (annotation stays in the GUI); delete repo-root shims
- [x] `trackanno suggest --list-features` (columns + importance)
- [x] Embedding input: per-detection `(movie, track, frame) -> vector` table, one summary path (mean/std/delta, one unsupervised PCA over all detections, the same for CV and the final model, stored in the model payload); YOLO head as first provider
- [x] `context_frames`: context feature family over N frames centred on the clip

## Phase 4: structure
- [x] Config rebuild: metadata-driven dataclasses, defaults in one place, test pinning `default_config.yaml`, one-line errors, warn on unknown sections/keys, removed keys load with a warning
- [x] Knobs to constants, view toggles to QSettings, Configure form to ~25-30 fields; form writes only changed keys (fixes spin-box rounding rewrites)
- [x] yamledit: block lists, 3-level keys, numpy scalars
- [x] Explicit Save + Revert; pages marked stale instead of full reload per autosave; remove per-page caches
- [x] `Page.run_bg()`: project-tagged jobs, stale results dropped, joined on shutdown (fixes model probe, scan, repair, download races)
- [x] Split `gui/hub.py` into one module per page
- [x] Split `gui/timeline.py`; footage as a plain chunk LRU with non-blocking stop (fixes right-edge click, track-switch stalls)
- [x] Suggester retrain: build off-thread, swap on UI thread, read snapshot not live CSV; cache features on a config/tracks hash
- [x] Remaining annotation bugs: suggestion outline in review modes, Ctrl+Z in review, refused commit wiping input at last clip
- [x] CSV: lock or re-read before rewrite to avoid two-writer loss
- [x] Bouts extend past detections: clip decode to the track's detected span
- [x] `recent_projects()` uses the timeout-guarded exists check

## Phase 5: test overhaul
- [ ] Plain asserts instead of `check()`; split mega-scenarios
- [ ] Merge `test_regressions.py` into module test files; dedupe bout/workflow fixtures
- [ ] Remove `inspect.getsource` and default-pin tests
- [ ] Add tests: tracker id reset, fingerprint staleness, two-writer CSV

## Phase 6: prose and docs
- [ ] Docstrings to 1-2 lines, no history; one-line error messages; shorter help/tooltips
- [ ] README and docs/ reflect the cut feature set
