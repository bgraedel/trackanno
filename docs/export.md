# Distribution and export

One bar chart per category group, with a text summary flagging:

* values with nothing labelled yet;
* values below `suggest.min_per_class`, dropped at training time;
* values appearing in only one movie, unscoreable by grouped cross-validation;
* values present in the CSV but no longer in the config;
* any value exceeding 90% of the labels, where a constant predictor scores well.

---

# 6 · `render.py`: labelled boxes on the original movie

```bash
uv run render.py --list                          # movies with labels
uv run render.py --movie MOVIE_A
uv run render.py --group quality --hide-unlabelled
uv run render.py --source both                   # hand labels and predictions
uv run render.py --movie MOVIE_A --tif --max-frames 200
```

Reads `tracks.parquet` and the annotation CSV, and colours each track by the
category of the clip covering that frame, using the colours from `config.yaml`.
Works on a partially annotated CSV. Unlabelled tracks are drawn grey or hidden.
Output goes to `render.out_dir`.

**Open project folder** on the Export page shows the project in the file
manager; the rendered movies land in `render.out_dir` inside it.

`--source` selects `labels` (default), `predictions` or `both`. The categories
already own the colours, so provenance is carried by line style instead: a model
value is drawn as a dashed outline with its confidence beside it, and the output
file name records the source, so a model-coloured movie cannot overwrite a
hand-labelled one.

---
