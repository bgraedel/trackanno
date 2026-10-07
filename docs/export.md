# Label distribution and export

## Label distribution

The Model page's **Label distribution** tab draws one bar chart per category
group and flags:

* values with nothing labelled yet;
* values below `suggest.min_per_class`, dropped at training time;
* values found in only one movie, which grouped cross-validation cannot score;
* values in the CSV that the config does not list;
* a value holding over 90% of the labels (from 20 labels on), where a constant
  answer scores well.

## Overlay movies

```bash
trackanno render --list                          # movies with labels
trackanno render --movie MOVIE_A
trackanno render --group quality --hide-unlabelled
trackanno render --source both                   # labels and predictions
trackanno render --movie MOVIE_A --max-frames 200
```

Draws each track's box on the original movie, coloured by the category of the
clip covering that frame, with the colours from `config.yaml`. Works on a
partially annotated CSV. Unlabelled tracks are drawn grey, or hidden with
`--hide-unlabelled` (`render.hide_unlabelled`). `render.show_skeleton` draws
pose keypoints. Output goes to `render.out_dir`, or `--out` for a single movie.

`--source` is `labels` (default), `predictions` or `both`. A prediction is drawn
dashed with its confidence, and the file name records the source.

The Export page offers the same options (**category group**, **colour by**,
**draw only annotated tracks**, **frame limit**); **Open project folder** opens
the project in the file manager.
