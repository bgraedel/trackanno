# trackanno

Behaviour annotation for individual tracked organisms in large grayscale
microscopy movies (`.tif`, `.nd2`), with the labels mapped back onto the original
recording.

The unit of work is a **clip**: `clips.frames` consecutive frames of one track,
cropped around that track. The clip loops, one keypress labels it, and the view
advances. A classifier trained on the labels suggests the next one and can then
label every frame of every track as behavioural bouts.

## Install

From GitHub:

```bash
uv tool install git+https://github.com/bgraedel/trackanno
trackanno
```

uv picks a Python the package supports (3.10 to 3.12) by itself. Append
`@<commit or tag>` to the URL to install a fixed version. Replace the URL with
`trackanno` to install from PyPI, or with `.` from a clone. From a clone,
`uv sync` and `uv run trackanno` work the same way.

`detection.device: auto` picks CUDA, then MPS, then the CPU at run time. The
plain install covers Linux (CUDA) and macOS (Metal). Two extras cover the rest;
with a URL, the package name and extra go in front of it:

```bash
# Windows with an NVIDIA GPU
uv tool install "trackanno[cuda] @ git+https://github.com/bgraedel/trackanno"
# Linux without a GPU: skips the CUDA wheels
uv tool install "trackanno[cpu] @ git+https://github.com/bgraedel/trackanno"
```

## Quick start

1. `trackanno` opens the workflow window.
2. **New project...** on the Project page: a name, a location, a video folder, a
   detection model and a preset.
3. Configure: check the paths and the category buttons, then **Save**.
4. Detect & crop: **Scan video folder**, tick movies, run.
5. Annotate: one keypress per clip.
6. Model: train the suggester, then **Label all clips with the model**.
7. Review the bouts, then Export overlay movies.

## The pipeline

A **project** is a folder holding `config.yaml`, `work/` (clip cache, models,
predictions) and `annotations.csv`.

| step | GUI page | command |
|---|---|---|
| choose a project | Project | |
| settings | Configure | edit `config.yaml` |
| detect, track, cut clips | Detect & crop | `trackanno prepare` |
| label clips | Annotate | |
| train, predict | Model | `trackanno suggest --train`, `trackanno predict` |
| inspect bouts | Review | |
| overlay movies | Export | `trackanno render` |
| bout statistics | | `trackanno transitions` |
| re-link orphaned labels | | `trackanno repair-links` |

`trackanno --help` lists the commands and `trackanno <command> --help` their
options. Every command takes `--config PATH`. The GUI and the CLI call the same
functions.

`trackanno prepare --stage detect` needs the model and a GPU; `--stage clips`
needs only the movie. See [split runs](docs/detection.md#detecting-on-one-machine-annotating-on-another).

## Documentation

| page | covers |
|---|---|
| [The workflow window](docs/workflow-window.md) | pages, presets, templates, the Configure page |
| [Configuration](docs/configuration.md) | the keys a user sets |
| [Detection and the clip cache](docs/detection.md) | tiling, trackers, shared models, split runs, pose, ND2 |
| [The annotation screen](docs/annotation.md) | shortcuts, view, modes, clip order, the CSV |
| [The label suggester](docs/suggester.md) | features, embeddings, context, the audit holdout |
| [Measurements](docs/measurements.md) | what each feature measures, and its validation |
| [Prediction, bouts and transitions](docs/prediction.md) | dense bouts, the Review page, the Markov chain |
| [Label distribution and export](docs/export.md) | the distribution tab, overlay movies |
| [Development](docs/development.md) | tests, source layout, conventions |
