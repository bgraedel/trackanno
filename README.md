# trackanno

Behaviour annotation for individual tracked organisms in large grayscale
microscopy movies (`.tif`, `.nd2`), with the labels mapped back onto the original
recording.

The unit of work is a **clip**: `clips.frames` consecutive frames of one track,
cropped around that track. The clip loops, one keypress labels it, and the view
advances.

## Install

```bash
uv tool install trackanno
trackanno
```

`uv tool install` puts `trackanno` on PATH in an isolated environment. Replace
`trackanno` with `.` from a clone, or with
`git+https://github.com/bgraedel/trackanno`.

On Linux the PyPI torch wheel carries the CUDA runtime, and on Apple Silicon it
carries Metal, so the plain install covers both. `detection.device: auto` picks
CUDA, else MPS, else the CPU at run time. Two extras cover the rest:

```bash
uv tool install "trackanno[cuda]"     # Windows: the only route to an NVIDIA GPU
uv tool install "trackanno[cpu]"      # Linux: skip ~2-3 GB of CUDA wheels
```

From a clone, `uv sync` and `uv run trackanno`, with the same `--extra`.

## The pipeline

A **project** is a folder holding `config.yaml`, `work/` (the clip cache) and
`annotations.csv`. `config.yaml` drives everything; *New project* writes a fully
commented one with the paths filled in, starting from one of three presets:
`annotator` (clips and the annotation screen, a box detector), `suggester` (pose
weights and a model trained on your labels) or `dense bouts` (25-frame windows
stepped by 5, decoded into bouts to review).

`trackanno` opens the workflow window, which runs the steps in order and carries
the state of each. Each is also a CLI entry point, for a headless GPU box or a
batch script:

| step | command | what it does |
|---|---|---|
| 1 Project | | pick the folder to work in |
| 2 Configure | | point it at the movies, choose the labels |
| 3 Detect & crop | `trackanno prepare` | find the animals, cut them into clips |
| 4 Annotate | | watch each clip, press one key |
| 5 Model | `trackanno suggest --train`, `trackanno predict` | train on the labels, label every clip, inspect the distribution |
| 6 Review | | check where the model put each boundary |
| 7 Export | `trackanno render` | put the labels back onto the original movies |
| | `trackanno transitions` | fit the Markov chain the bouts imply |
| repair | `trackanno repair-links --config config.yaml` | re-link labels whose clips no longer exist |

Both paths call the same functions (`prepare_all`, `train`, `predict_all`,
`render_all`), so the GUI and the CLI cannot drift apart.

Step 3 splits further. `--stage detect` needs the GPU, `--stage clips` needs only
the movie, so detection can run on a remote box while the clips are cut where
they are annotated. See
[Detecting on one machine, annotating on another](docs/detection.md#detecting-on-one-machine-annotating-on-another).

## Documentation

| page | covers |
|---|---|
| [The workflow window](docs/workflow-window.md) | steps, project state, templates |
| [Detection and the clip cache](docs/detection.md) | tiling, tracking, shared models, split runs, pose, ND2 |
| [The annotation screen](docs/annotation.md) | shortcuts, the kymograph, review, clip order, the CSV |
| [The label suggester](docs/suggester.md) | features, the audit holdout, what a suggestion means |
| [Prediction, bouts and transitions](docs/prediction.md) | dense prediction, boundaries, the Markov chain |
| [Label distribution and export](docs/export.md) | the label distribution, labelled overlay movies |
| [Tests and packaging](docs/development.md) | the suites, dependency notes |
