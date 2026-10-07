"""Reading ``predictions.parquet`` into per-track bout lanes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...config import Config

__all__ = ["Segment", "TrackPredictions", "PredictionReader"]


@dataclass(frozen=True)
class Segment:
    """One decoded bout. ``label`` is ``""`` for an unscored stretch, ``conf``
    is NaN where the file has none."""

    f_start: int
    f_end: int
    label: str
    conf: float
    skip_reason: str

    @property
    def n_frames(self) -> int:
        return self.f_end - self.f_start


@dataclass
class TrackPredictions:
    """One (movie, track): its frame span and one sorted lane per group."""

    movie_id: str
    track_id: int
    f_lo: int = 0
    f_hi: int = 0                                   # exclusive
    lanes: dict = field(default_factory=dict)       # group -> [Segment]

    @property
    def empty(self) -> bool:
        return not any(self.lanes.values())

    @property
    def n_segments(self) -> int:
        return sum(len(v) for v in self.lanes.values())


def _labelled(values: pd.Series) -> pd.Series:
    """Cells holding a real class name (not ``""`` or NaN)."""
    text = values.astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


class PredictionReader:
    """``predictions.parquet``, read once and indexed by (movie, track).

    ``problem`` is a one-line reason the file cannot be reviewed, or ``""``.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.problem = ""
        self.frame = pd.DataFrame()
        self.groups: list[str] = []
        self.movies: list[str] = []
        self.stale = ""
        self._index: dict = {}
        self._tracks: dict[str, list[tuple[int, int]]] = {}

        from ...infer.predict import load_predictions, staleness

        try:
            self.frame = load_predictions(cfg)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.problem = str(exc)
            return
        if "bout_id" not in self.frame.columns:
            self.problem = ("these predictions are not decoded bouts; label every "
                            "clip with the model again")
            return
        self.stale = staleness(cfg)
        self.groups = sorted(
            c[: -len("_pred")] for c in self.frame.columns if c.endswith("_pred")
        )
        if not self.groups:
            self.problem = ("the predictions have no <group>_pred column; re-run "
                            "'Label all clips with the model'")
            return
        movies = self.frame["movie_id"].astype(str)
        tracks = pd.to_numeric(self.frame["track_id"], errors="coerce")
        self._index = {
            (str(m), int(t)): rows
            for (m, t), rows in self.frame.groupby([movies, tracks]).indices.items()
        }
        self.movies = sorted(movies.unique())

    def tracks(self, movie_id: str) -> list[tuple[int, int]]:
        """``[(track_id, n_rows)]``, including detected tracks with no rows."""
        movie_id = str(movie_id)
        cached = self._tracks.get(movie_id)
        if cached is not None:
            return cached
        counts = {t: len(rows) for (m, t), rows in self._index.items() if m == movie_id}
        for track_id in self._detected_tracks(movie_id):
            counts.setdefault(track_id, 0)
        out = [(t, counts[t]) for t in sorted(counts)]
        self._tracks[movie_id] = out
        return out

    def _detected_tracks(self, movie_id: str) -> list[int]:
        """Track ids in ``tracks.parquet``, or [] if it cannot be read."""
        from ...annotate.clips import TRACKS_FILE

        path = self.cfg.movie_dir(movie_id) / TRACKS_FILE
        if not path.exists():
            return []
        try:
            table = pd.read_parquet(path, columns=["track_id"])
        except (OSError, ValueError, KeyError):
            return []
        ids = pd.to_numeric(table["track_id"], errors="coerce").dropna()
        return sorted({int(t) for t in ids.unique()})

    def track(self, movie_id: str, track_id: int) -> TrackPredictions:
        out = TrackPredictions(movie_id=str(movie_id), track_id=int(track_id))
        index = self._index.get((str(movie_id), int(track_id)))
        if index is None or not len(index):
            return out
        rows = self.frame.iloc[index]
        starts = pd.to_numeric(rows["f_start"], errors="coerce")
        ends = pd.to_numeric(rows["f_end"], errors="coerce")
        if not (starts.notna().any() and ends.notna().any()):
            return out
        out.f_lo = int(starts.min())
        out.f_hi = int(ends.max())
        for group in self.groups:
            out.lanes[group] = self._segments(rows, group)
        return out

    def _segments(self, rows: pd.DataFrame, group: str) -> list[Segment]:
        """One group's bouts in frame order; rows without a span are dropped."""
        rows = rows[rows["group"].astype(str) == group]
        if rows.empty:
            return []
        blank = pd.Series(np.nan, index=rows.index)
        pred = rows[f"{group}_pred"].astype(str).str.strip()
        known = _labelled(pred).to_numpy()
        conf = pd.to_numeric(
            rows[f"{group}_conf"] if f"{group}_conf" in rows.columns else blank,
            errors="coerce",
        ).to_numpy(dtype=float)
        starts = pd.to_numeric(rows["f_start"], errors="coerce").to_numpy(dtype=float)
        ends = pd.to_numeric(rows["f_end"], errors="coerce").to_numpy(dtype=float)
        reason = (rows["skip_reason"].astype(str) if "skip_reason" in rows.columns
                  else pd.Series("", index=rows.index)).to_numpy()
        names = pred.to_numpy()

        out = []
        for i in range(len(rows)):
            if not (np.isfinite(starts[i]) and np.isfinite(ends[i])):
                continue
            text = str(reason[i])
            out.append(Segment(
                f_start=int(starts[i]), f_end=int(ends[i]),
                label=str(names[i]) if known[i] else "",
                conf=float(conf[i]) if known[i] else float("nan"),
                skip_reason="" if text.lower() == "nan" else text,
            ))
        out.sort(key=lambda s: (s.f_start, s.f_end))
        return out
