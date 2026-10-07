# Development

## Tests

```bash
uv sync                                   # installs pytest with the dev group
uv run pytest -q                          # full suite, about 5 minutes
uv run pytest tests/test_config.py -v
uv run pytest -k "predict or bouts"
uv run pytest --project-root PATH         # also check a real prepared project
```

Tests need no external data: suites that need a project generate one (config,
synthetic movie, clip cache) in a temporary directory (`tests/fixtures.py`).
`tests/conftest.py` puts `src/` on the path, runs Qt offscreen and points
`TRACKANNO_STATE_DIR` at a temporary folder, so a run never touches the real
profile.

To compare tiled and whole-frame detection on a real movie:

```bash
trackanno prepare --self-test --tile 320 --movies <id>
```

## Layout of `src/trackanno`

| path | contents |
|---|---|
| `cli.py` | the `trackanno` command and its subcommand table |
| `config.py` | the config schema (one dataclass per section), `FIXED_SETTINGS`, `VIEW_SETTINGS` |
| `default_config.yaml`, `presets/` | the project template and the presets applied to it |
| `project.py`, `templates.py`, `yamledit.py` | projects, templates, comment-preserving config edits |
| `progress.py` | progress reporting and cancellation for long jobs |
| `video/` | movie reading (TIFF, ND2) and the shared model registry (`models.py`) |
| `detect/` | `prepare.py` (the `prepare` command), tiling, tracker backends, detector embeddings |
| `annotate/` | clip planning and cache (`clips.py`), the annotation CSV (`store.py`), `reassociate.py` (`repair-links`) |
| `measure/` | per-frame track series, beat spectra, kymograph, clip features, embeddings |
| `learn/` | `suggest.py` (the `suggest` command), `select.py` (margin selection) |
| `infer/` | dense windows, bout decoding, `predict.py`, `transitions.py`, `figures.py`, CV agreement |
| `view/` | overlay drawing and `render.py` (the `render` command) |
| `gui/hub/` | the workflow window: `window.py`, `base.py` (page base, background jobs), one module per page |
| `gui/review/` | the Review page: prediction reader, bout lanes, footage decoder |
| `gui/app.py` | the annotation screen |
| `gui/configform.py` | the Configure form, category and keypoint editors |
| `gui/viewprefs.py` | per-user view toggles in QSettings |
| `gui/` (other) | viewer, kymograph panel, distribution chart, task panel, prefetch, theme |

`notebooks/` holds the downstream-analysis notebook and `motifs.py`.

## Conventions

* The GUI and the CLI call the same functions; a pipeline step is never
  implemented twice.
* A new setting is a field in `config.py` with its default, help line and
  allowed values, plus a commented line in `default_config.yaml`. A setting
  nobody needs to change belongs in `FIXED_SETTINGS`.
* Hub pages run slow work through `Page.run_bg()`, never on the UI thread.
* Files other processes read (CSV, caches, `predictions.parquet`) are written
  beside their final name and renamed into place.
* Commits: short imperative subject, 1-3 sentence body, no trailers.
* Prose (docstrings, docs, GUI text): brief, no history, no em-dashes.

## Packaging notes

* `requires-python >=3.10,<3.13`; `.python-version` pins 3.12.
* The `cuda` and `cpu` extras are declared as conflicting, so the lockfile holds
  a torch resolution for each. The Windows CUDA index is pinned to `cu128`.
* `opencv-python` is overridden out of the resolution in favour of
  `opencv-python-headless`, which does not clash with PyQt's Qt plugins.
* Project data is gitignored: `work/`, `annotations.csv`, `predictions.parquet`
  and `config.yaml`.
