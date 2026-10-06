"""Entry point 3: burn labelled boxes back onto the original movie.

    trackanno render --movie <movie_id> [--group behaviour] [--source both]

Reads the tracks table (original full-frame coordinates) and the annotation CSV,
colours each track by the category of the clip covering that frame, and writes an
mp4; a partially annotated CSV is fine. Drawing is shared with the GUI.

``--source`` picks what colours the boxes: your labels (default), the model's
``predictions.parquet``, or both. A prediction is ALWAYS drawn dashed and
captioned with its confidence and a ``~``, so an exported movie can never pass
the model's output off as a hand label.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotate.clips import NORM_FILE, TRACKS_FILE
from ..config import Config, load_config
from ..video.movies import MovieSource, Normalizer, open_movie
from .overlay import draw_detections, gray_to_rgb, hex_to_rgb
from ..annotate.store import MULTI_SEP

SOURCES = ("labels", "predictions", "both")


def _pad_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    return img


def default_group(cfg: Config) -> str:
    for group in cfg.categories:
        if not group.is_multi:
            return group.name
    return cfg.categories[0].name


def build_frame_labels(
    annotations: pd.DataFrame, cfg: Config, group_name: str
) -> dict[int, dict[int, tuple[tuple[int, int, int], str]]]:
    """frame -> {track_id: (rgb, label)} from the annotation rows: a view over
    :func:`build_frame_marks`, never a second implementation of "which colour
    does this cell mean"."""
    marks = build_frame_marks(annotations, None, cfg, group_name, "labels")
    return {
        frame: {tid: (mark.color, mark.text) for tid, mark in entries.items()}
        for frame, entries in marks.items()
    }


@dataclass
class Mark:
    """What to draw on one track for one frame."""

    color: tuple[int, int, int]
    text: str
    predicted: bool          # drawn dashed, so it can never read as a hand label


def build_frame_marks(
    annotations: pd.DataFrame,
    predictions: pd.DataFrame | None,
    cfg: Config,
    group_name: str,
    source: str = "labels",
) -> dict[int, dict[int, Mark]]:
    """frame -> {track_id: Mark}, from the labels, the predictions, or both.

    With ``both`` a hand label always wins the box - drawn solid, the model's
    opinion appended to the caption; untouched tracks keep the dashed mark.

    TWO BOUTS MAY NOT COVER THE SAME FRAME OF THE SAME TRACK, and this function
    raises if they do: within one group bouts are disjoint by construction, so a
    collision means the decode or the file is wrong.

    A hand label landing on a frame a prediction already claimed is NOT a
    collision: it is what ``--source both`` is for, and the labels pass below
    overwrites the dashed mark while keeping the model's answer in the caption.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {list(SOURCES)}")
    group = cfg.group(group_name)
    if group is None:
        raise ValueError(
            f"unknown category group {group_name!r}; available: "
            + ", ".join(g.name for g in cfg.categories)
        )

    def colour_of(value: str) -> tuple[int, int, int]:
        option = group.option(value.split(MULTI_SEP)[0])
        return hex_to_rgb(option.color) if option else (200, 200, 200)

    out: dict[int, dict[int, Mark]] = {}
    if source in ("predictions", "both") and predictions is not None:
        column, conf_column = f"{group_name}_pred", f"{group_name}_conf"
        # One entry per painted ROW, not per painted frame, so the guard below can
        # name the row it collided with without carrying a second frame-sized dict.
        claimed: dict[int, list[tuple[int, int, str]]] = {}
        for row in predictions.to_dict("records"):
            value = _clean(row.get(column))
            if not value:
                continue
            conf = row.get(conf_column)
            try:
                caption = f"~{value} {float(conf):.2f}"
            except (TypeError, ValueError):
                caption = f"~{value}"
            mark = Mark(colour_of(value), caption, True)
            track_id = int(row["track_id"])
            f_start, f_end = int(row["f_start"]), int(row["f_end"])
            for frame in range(f_start, f_end):
                marks = out.setdefault(frame, {})
                if track_id in marks:
                    raise ValueError(_overlap_message(claimed, track_id, frame,
                                                      f_start, f_end, value))
                marks[track_id] = mark
            claimed.setdefault(track_id, []).append((f_start, f_end, value))

    if source in ("labels", "both"):
        if group_name not in annotations.columns:
            raise ValueError(
                f"the annotation CSV has no column {group_name!r}; "
                f"available: {[c for c in annotations.columns]}"
            )
        # By column name, not itertuples: itertuples renames any group whose name
        # is not a Python identifier ("cell state" -> "_5").
        for row in annotations.to_dict("records"):
            value = _clean(row.get(group_name))
            if not value:
                continue
            mark = Mark(colour_of(value), value, False)
            track_id = int(row["track_id"])
            # No overlap guard on this pass: a hand label overwriting a
            # prediction is what 'both' means, and two annotation rows
            # overlapping is what `clips.stride` below `clips.frames` asks for.
            for frame in range(int(row["f_start"]), int(row["f_end"])):
                existing = out.get(frame, {}).get(track_id)
                text = value
                if source == "both" and existing is not None and existing.predicted:
                    text = f"{value}  |  {existing.text}"
                out.setdefault(frame, {})[track_id] = Mark(mark.color, text, False)
    return out


def _overlap_message(
    claimed: dict[int, list[tuple[int, int, str]]],
    track_id: int,
    frame: int,
    f_start: int,
    f_end: int,
    value: str,
) -> str:
    """Why two BOUTS meeting on one frame is fatal, in terms of the rows.

    Names the frame, the track and BOTH spans, taken from the rows themselves,
    so a duplicated row and a genuinely overlapping pair - which want different
    fixes - do not read alike. Only reachable for rows carrying ``bout_id``;
    overlapping WINDOW rows are a legitimate geometry and are painted, see
    :func:`build_frame_marks`.
    """
    earlier = next(
        (f"[{s}, {e}) says {v!r}" for s, e, v in claimed.get(track_id, [])
         if s <= frame < e),
        "an earlier row says something else",
    )
    return (
        f"frame {frame}, track {track_id}: two decoded bouts both cover it - "
        f"{earlier} and [{f_start}, {f_end}) says {value!r}. A decode cuts a "
        "track's frames into runs between inferred boundaries, so within one "
        "group its bouts are disjoint by construction and two of them meeting is "
        "not something a setting can ask for. The colour this frame would be "
        "exported with comes from the order the rows happen to sit in the file "
        "rather than from the model, so the movie would look entirely normal and "
        "be wrong. Either predictions.parquet was edited or concatenated from "
        "two runs, or the decode itself is broken - re-run `trackanno predict` and, if it "
        "happens again, report it. Do not render this file as it stands"
    )


def _clean(value) -> str:
    """A category cell as a usable string, or "" if there is nothing there: an
    empty cell reads back from pandas as float NaN, whose str() is a truthy
    "nan"."""
    text = "" if value is None else str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def render_movie(
    cfg: Config,
    movie: MovieSource,
    tracks: pd.DataFrame,
    annotations: pd.DataFrame,
    group_name: str,
    out_path: Path,
    hide_unlabelled: bool = False,
    max_frames: int | None = None,
    reporter=None,
    predictions: pd.DataFrame | None = None,
    source: str = "labels",
) -> Path:
    import imageio.v2 as imageio

    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    n_frames = min(movie.n_frames, max_frames) if max_frames else movie.n_frames

    norm_path = cfg.movie_dir(movie.id) / NORM_FILE
    normalizer = (
        Normalizer.load(norm_path) if norm_path.exists()
        else Normalizer.from_movie(movie, cfg.normalize)
    )
    # A curve built with `trackanno prepare --limit-frames N` covers only N frames;
    # clamping past its end reuses frame N-1's mapping and jumps the brightness.
    if not normalizer.covers(n_frames):
        reporter.log(
            f"intensity curve covers only {normalizer.length} of {n_frames} frames "
            "(prepared with --limit-frames); recomputing it for the full range"
        )
        normalizer = Normalizer.from_movie(
            movie, cfg.normalize,
            progress=lambda it: reporter.track(it, f"{movie.id}  intensity levels"),
            n_frames=n_frames,
        )

    marks_by_frame = build_frame_marks(annotations, predictions, cfg, group_name, source)
    unlabelled_rgb = hex_to_rgb(cfg.render.unlabelled_color)
    tracks_by_frame = {f: df for f, df in tracks.groupby("frame", sort=False)}

    fps = movie.fps if cfg.render.fps == "auto" else float(cfg.render.fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec=cfg.render.codec,
        quality=cfg.render.quality, macro_block_size=1,
    )

    try:
        for frame_idx in reporter.track(range(n_frames), f"{movie.id}  drawing", n_frames):
            rgb = np.ascontiguousarray(
                gray_to_rgb(normalizer.to_u8(movie.frame(frame_idx), frame_idx))
            )
            records = tracks_by_frame.get(frame_idx)
            if records is not None and len(records):
                marked = marks_by_frame.get(frame_idx, {})
                colors: dict[int, tuple[int, int, int]] = {}
                text: dict[int, str] = {}
                dashed: set[int] = set()
                for track_id in records["track_id"].unique():
                    track_id = int(track_id)
                    entry = marked.get(track_id)
                    if entry is None:
                        if hide_unlabelled:
                            continue
                        colors[track_id] = unlabelled_rgb
                        text[track_id] = f"t{track_id}" if cfg.render.show_track_id else ""
                    else:
                        colors[track_id] = entry.color
                        if entry.predicted:
                            dashed.add(track_id)
                        parts = []
                        if cfg.render.show_track_id:
                            parts.append(f"t{track_id}")
                        if cfg.render.show_category:
                            parts.append(entry.text)
                        text[track_id] = " ".join(parts)
                draw_detections(
                    rgb, records, cfg, colors, text,
                    thickness=cfg.render.thickness,
                    font_scale=cfg.render.font_scale,
                    show_skeleton=cfg.render.show_skeleton,
                    dashed=dashed,
                )
            writer.append_data(_pad_even(rgb))
    finally:
        writer.close()
    return out_path


# --------------------------------------------------------------------------- #
# batch driver - shared by the CLI and the GUI
# --------------------------------------------------------------------------- #
def annotated_movies(cfg: Config) -> dict[str, int]:
    """{movie_id: annotated clip count}, straight from the CSV. Never raises: a
    throw on the GUI's project-open path, which startup repeats, would lock the
    window out for good. A broken CSV is reported and treated as empty."""
    if not cfg.paths.csv.exists():
        return {}
    try:
        frame = pd.read_csv(cfg.paths.csv)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.warn(f"could not read {cfg.paths.csv}: {exc}", stacklevel=2)
        return {}
    if frame.empty or "movie_id" not in frame.columns:
        return {}
    counts = frame["movie_id"].astype(str).value_counts()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def render_all(
    cfg: Config,
    movie_ids: list[str] | None = None,
    group_name: str | None = None,
    hide_unlabelled: bool | None = None,
    max_frames: int | None = None,
    out_path: Path | None = None,
    reporter=None,
    source: str = "labels",
) -> list[Path]:
    """Draw labelled boxes onto every requested movie.

    ``source`` is ``labels`` (the CSV, default), ``predictions`` or ``both``; a
    missing predictions file raises rather than rendering an empty overlay,
    which looks exactly like "the model had no opinion". ``hide_unlabelled``
    None means ``cfg.render.hide_unlabelled``."""
    if hide_unlabelled is None:
        hide_unlabelled = cfg.render.hide_unlabelled
    from ..progress import Cancelled, NullReporter

    reporter = reporter or NullReporter()
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {list(SOURCES)}")

    annotations = pd.DataFrame()
    if source != "predictions":
        if not cfg.paths.csv.exists():
            raise FileNotFoundError(f"no annotation CSV at {cfg.paths.csv}")
        annotations = pd.read_csv(cfg.paths.csv)
        if annotations.empty:
            raise RuntimeError(f"{cfg.paths.csv} has no rows yet")

    group_name = group_name or default_group(cfg)
    predictions = None
    if source != "labels":
        from ..infer.predict import load_predictions, staleness

        predictions = load_predictions(cfg, group_name)
        stale = staleness(cfg)
        if stale:
            # Not fatal, but say it out loud: nothing in the rendered movie
            # reveals which model drew the dashed boxes.
            reporter.log(f"the predictions may be out of date - {stale}", "warning")

    if source == "labels":
        available = sorted(annotations["movie_id"].astype(str).unique())
    elif source == "predictions":
        available = sorted(predictions["movie_id"].astype(str).unique())
    else:
        available = sorted(
            set(annotations["movie_id"].astype(str))
            | set(predictions["movie_id"].astype(str))
        )
    wanted = movie_ids or available
    unknown = [m for m in wanted if m not in available]
    if unknown:
        raise ValueError(f"no annotations for {unknown}. Annotated movies: {available}")

    suffix = ".mp4"
    # Provenance in the name: never overwrite or impersonate a hand-coloured one.
    tag = "" if source == "labels" else f"__{source}"
    written: list[Path] = []

    for i, movie_id in enumerate(wanted):
        reporter.overall(i, len(wanted), movie_id)
        tracks_path = cfg.movie_dir(movie_id) / TRACKS_FILE
        if not tracks_path.exists():
            reporter.log(f"{movie_id}: missing {tracks_path}; prepare it first", "warning")
            continue
        tracks = pd.read_parquet(tracks_path)
        subset = (
            annotations[annotations["movie_id"].astype(str) == movie_id]
            if len(annotations) else annotations
        )
        pred_subset = (
            predictions[predictions["movie_id"].astype(str) == movie_id]
            if predictions is not None else None
        )
        target = (
            Path(out_path).resolve() if out_path and len(wanted) == 1
            else cfg.render.out_dir / f"{movie_id}__{group_name}{tag}{suffix}"
        )
        movie = open_movie(cfg, movie_id)
        try:
            reporter.log(f"{movie_id}: {len(subset)} annotated clips -> {target}")
            if pred_subset is not None:
                reporter.log(
                    f"{movie_id}: {len(pred_subset)} predicted bouts"
                )
            render_movie(
                cfg, movie, tracks, subset, group_name, target,
                hide_unlabelled=hide_unlabelled,
                max_frames=max_frames, reporter=reporter,
                predictions=pred_subset, source=source,
            )
            written.append(target)
        except Cancelled:
            raise
        except Exception as exc:
            reporter.log(f"{movie_id}: FAILED - {type(exc).__name__}: {exc}", "error")
        finally:
            movie.close()
    reporter.overall(len(wanted), len(wanted), "done")
    return written


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trackanno render",
        description="Draw annotated bounding boxes onto the original movie.",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--movie", nargs="*", default=None, metavar="ID",
                   help="movie ids to render (default: every movie present in the CSV)")
    p.add_argument("--group", default=None, help="category group to colour by")
    p.add_argument("--source", default="labels", choices=list(SOURCES),
                   help="colour by your labels, the model's predictions, or both "
                        "(a prediction is drawn dashed, with its confidence)")
    p.add_argument("--out", default=None, help="output file (single movie only)")
    p.add_argument("--hide-unlabelled", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="draw only tracks that carry an annotation "
                        "(default: render.hide_unlabelled)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--list", action="store_true", help="list annotated movies and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.list:
        counts = annotated_movies(cfg)
        if not counts:
            print(f"error: no annotations at {cfg.paths.csv}", file=sys.stderr)
            return 2
        for movie_id, n in counts.items():
            print(f"  {movie_id:<45} {n:>5} annotated clips")
        return 0

    from ..progress import ConsoleReporter

    try:
        render_all(
            cfg, args.movie, args.group,
            hide_unlabelled=args.hide_unlabelled,
            max_frames=args.max_frames,
            out_path=Path(args.out) if args.out else None,
            reporter=ConsoleReporter(),
            source=args.source,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\nWrote overlays to {cfg.render.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
