"""Cut review frames from the movie (not the clip cache) on a worker thread.

One crop per session, centred on a path smoothed over the whole track.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal

from ...config import FIXED_SETTINGS, Config

__all__ = ["REVIEW_CHUNK", "ReviewCrop", "review_crop", "MovieSource", "TrackCutter",
           "ChunkDecoder", "Cancelled"]

REVIEW_CHUNK = 50                 # frames per decode


class Cancelled(Exception):
    """A cut abandoned because its decoder was stopped or superseded."""


@dataclass(frozen=True)
class ReviewCrop:
    """The project-wide review crop: ``side`` movie pixels, delivered as a ``size`` square."""

    side: int
    size: int

    @property
    def scale(self) -> float:
        """Delivered pixels per movie pixel."""
        return self.size / float(self.side)


def review_crop(cfg: Config) -> ReviewCrop:
    """99th percentile of cached crop widths, rounded up to 16, capped at clips.max_size."""
    from ...annotate.clips import MANIFEST_FILE

    widths: list[np.ndarray] = []
    work = getattr(getattr(cfg, "paths", None), "work_dir", None)
    if work is not None and work.exists():
        for movie_dir in sorted(p for p in work.iterdir() if p.is_dir()):
            path = movie_dir / MANIFEST_FILE
            if not path.exists():
                continue
            try:
                column = pd.read_parquet(path, columns=["crop_w"])["crop_w"]
            except (OSError, ValueError, KeyError):
                continue
            values = pd.to_numeric(column, errors="coerce").dropna().to_numpy(float)
            if values.size:
                widths.append(values)

    ceiling = max(32, int(FIXED_SETTINGS["clips.max_size"]))
    if widths:
        side = int(np.ceil(float(np.percentile(np.concatenate(widths), 99)) / 16.0) * 16)
    else:
        declared = getattr(cfg.clips, "crop_size", "auto")
        side = int(declared) if isinstance(declared, int) else ceiling
    side = max(16, side)
    return ReviewCrop(side=side, size=min(side, ceiling))


class MovieSource:
    """An open movie with its normalizer and ``tracks.parquet``; owned by one thread."""

    NEEDED = ["frame", "track_id", "x1", "y1", "x2", "y2"]

    def __init__(self, cfg: Config, movie_id: str):
        self.cfg = cfg
        self.movie_id = str(movie_id)
        self.movie = None
        self.normalizer = None
        self.table: pd.DataFrame | None = None

    def open(self) -> None:
        """Open the movie and read the tracks once. Raises with a one-line reason."""
        from ...annotate.clips import NORM_FILE, TRACKS_FILE
        from ...video.movies import Normalizer, open_movie

        if self.movie is not None:
            return
        movie = open_movie(self.cfg, self.movie_id)
        try:
            path = self.cfg.movie_dir(self.movie_id) / NORM_FILE
            self.normalizer = (Normalizer.load(path) if path.exists()
                               else Normalizer.from_movie(movie, self.cfg.normalize))
            n_kpts = int(getattr(self.cfg, "n_keypoints", 0) or 0)
            kcols = [f"kp_{a}_{k}" for k in range(n_kpts) for a in ("x", "y")]
            path = self.cfg.movie_dir(self.movie_id) / TRACKS_FILE
            try:
                table = pd.read_parquet(path, columns=self.NEEDED + ["conf"] + kcols)
            except (OSError, ValueError, KeyError):
                table = pd.read_parquet(path)
            missing = [c for c in self.NEEDED if c not in table.columns]
            if missing:
                raise ValueError(f"{TRACKS_FILE} has no {', '.join(missing)} column")
        except BaseException:
            movie.close()
            raise
        self.table = table
        self.movie = movie

    def close(self) -> None:
        movie, self.movie, self.table = self.movie, None, None
        if movie is not None:
            try:
                movie.close()
            except Exception:
                pass


class TrackCutter:
    """Synchronous cutter for one track; opens its own source if none is given."""

    def __init__(self, cfg: Config, movie_id: str, track_id: int, crop: ReviewCrop,
                 chunk: int = REVIEW_CHUNK, source: MovieSource | None = None):
        self.cfg = cfg
        self.movie_id = str(movie_id)
        self.track_id = int(track_id)
        self.crop = crop
        self.chunk = max(1, int(chunk))
        self.skeleton_problem = ""
        self._owns = source is None
        self.source = source if source is not None else MovieSource(cfg, movie_id)
        self._ready = False
        self._boxes: np.ndarray | None = None     # (n, 4), hold-filled
        self._present: np.ndarray | None = None   # (n,) really detected
        self._kpts: np.ndarray | None = None      # (n, K, 2) movie px
        self._context: tuple | None = None        # other tracks, sorted by frame
        self._path: np.ndarray | None = None      # (n, 2) crop centres

    @property
    def _movie(self):
        return self.source.movie

    def open(self) -> None:
        """Open the source if needed and load this track."""
        if self._ready:
            return
        self.source.open()
        self._load_tracks(int(len(self.source.movie)))
        self._ready = True

    def close(self) -> None:
        self._ready = False
        if self._owns:
            self.source.close()

    def _load_tracks(self, n: int) -> None:
        """The track's boxes, keypoints, neighbours and crop path."""
        from ...annotate.clips import TRACKS_FILE, _hold_fill

        n_kpts = int(getattr(self.cfg, "n_keypoints", 0) or 0)
        kcols = [f"kp_{a}_{k}" for k in range(n_kpts) for a in ("x", "y")]
        table = self.source.table
        frames = pd.to_numeric(table["frame"], errors="coerce").to_numpy(np.int64)
        ids = pd.to_numeric(table["track_id"], errors="coerce").to_numpy(np.int64)
        xyxy = table[["x1", "y1", "x2", "y2"]].to_numpy(float)
        inside = (frames >= 0) & (frames < n)
        mine = inside & (ids == self.track_id)
        if not mine.any():
            raise ValueError(f"track {self.track_id} was never detected in "
                             f"{self.movie_id}")

        boxes = np.full((n, 4), np.nan)
        boxes[frames[mine]] = xyxy[mine]
        present = np.zeros(n, bool)
        present[frames[mine]] = True
        self._boxes = _hold_fill(boxes, present)
        self._present = present

        self._kpts = None
        if n_kpts and all(c in table.columns for c in kcols):
            kpts = np.full((n, n_kpts, 2), np.nan, np.float32)
            kpts[frames[mine]] = (
                table[kcols].to_numpy(np.float32).reshape(-1, n_kpts, 2)[mine]
            )
            self._kpts = kpts
        self.skeleton_problem = (
            "" if self._kpts is not None or not n_kpts
            else f"{TRACKS_FILE} has no keypoints, so no skeleton"
        )

        self._context = None
        if bool(FIXED_SETTINGS["clips.context_tracks"]):
            other = inside & (ids != self.track_id)
            conf = (table["conf"].to_numpy(float) if "conf" in table.columns
                    else np.ones(len(table)))
            order = np.argsort(frames[other], kind="stable")
            self._context = (frames[other][order], ids[other][order],
                             xyxy[other][order], conf[other][order])
        self._path = self._crop_path()

    def _crop_path(self) -> np.ndarray:
        """Crop centre per frame: smoothed track centre, nudged to keep the box inside."""
        from ...annotate.clips import _moving_average

        boxes = self._boxes
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                            (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
        window = int(FIXED_SETTINGS["clips.smooth_window"])
        path = _moving_average(centres, window)
        half = self.crop.side / 2.0
        out = np.empty_like(path)
        for axis, (far, near) in enumerate(((2, 0), (3, 1))):
            lo, hi = boxes[:, far] - half, boxes[:, near] + half
            out[:, axis] = np.where(lo > hi, centres[:, axis],
                                    np.clip(path[:, axis], lo, hi))
        return out

    def cut(self, first: int, last: int | None = None,
            cancelled: Callable[[], bool] | None = None,
            ) -> tuple[np.ndarray, pd.DataFrame]:
        """Cropped frames ``first`` to ``last`` (default one chunk) and their marks."""
        import cv2

        from ...annotate.clips import _crop

        self.open()
        movie = self._movie
        normalizer = self.source.normalizer
        first = int(first)
        wanted = self.chunk if last is None else int(last) - first
        n = min(max(1, wanted), max(0, int(len(movie)) - first))
        if n <= 0 or first < 0:
            raise ValueError(f"frame {first} is outside {self.movie_id}")
        side, size = self.crop.side, self.crop.size
        origins = np.round(self._path[first: first + n] - side / 2.0).astype(np.int64)
        out = np.zeros((n, size, size), np.uint8)
        for t in range(n):
            if cancelled is not None and cancelled():
                raise Cancelled()
            frame = first + t
            u8 = normalizer.to_u8(movie.frame(frame), frame)
            crop = _crop(u8, int(origins[t, 0]), int(origins[t, 1]), side, side,
                         movie.height, movie.width)
            if crop.shape[0] != size:
                interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (size, size), interpolation=interp)
            out[t] = crop
        return out, self._marks(first, n, origins)

    def _marks(self, first: int, n: int, origins: np.ndarray) -> pd.DataFrame:
        """Box table for these frames in ``build_box_table`` layout; undetected frames omitted."""
        scale = self.crop.scale
        span = slice(first, first + n)
        present = self._present[span]
        t = np.flatnonzero(present).astype(np.int32)
        shift = np.tile(origins[present], 2)
        marks = (self._boxes[span][present] - shift) * scale
        block = {
            "t": t,
            "track_id": np.full(t.size, self.track_id, np.int32),
            "is_target": np.ones(t.size, bool),
            "x1": marks[:, 0], "y1": marks[:, 1],
            "x2": marks[:, 2], "y2": marks[:, 3],
            "conf": np.ones(t.size, np.float32),
        }
        if self._kpts is not None:
            kpts = (self._kpts[span][present] - origins[present][:, None, :]) * scale
            for k in range(kpts.shape[1]):
                block[f"kp_x_{k}"] = kpts[:, k, 0]
                block[f"kp_y_{k}"] = kpts[:, k, 1]
        target = pd.DataFrame(block)
        if self._context is None:
            return target

        frames, ids, xyxy, conf = self._context
        lo, hi = np.searchsorted(frames, [first, first + n])
        if hi <= lo:
            return target
        rel = (frames[lo:hi] - first).astype(np.int64)
        near = (xyxy[lo:hi] - np.tile(origins[rel], 2)) * scale
        edge = float(self.crop.size)
        seen = ((near[:, 2] > 0) & (near[:, 0] < edge)
                & (near[:, 3] > 0) & (near[:, 1] < edge))
        if not seen.any():
            return target
        other = pd.DataFrame({
            "t": rel[seen].astype(np.int32),
            "track_id": ids[lo:hi][seen].astype(np.int32),
            "is_target": np.zeros(int(seen.sum()), bool),
            "x1": near[seen, 0], "y1": near[seen, 1],
            "x2": near[seen, 2], "y2": near[seen, 3],
            "conf": conf[lo:hi][seen].astype(np.float32),
        })
        return pd.concat([target, other], ignore_index=True)


class ChunkDecoder(QObject):
    """One worker thread cutting chunks for one movie: one urgent job, one read-ahead.

    Jobs are ``(generation, track, first, last)``; stale generations are dropped.
    """

    ready = Signal(int, int, object)           # generation, first, (pixels, marks)
    failed = Signal(int, int, str, bool)       # generation, first, reason, fatal

    def __init__(self, make_cutter: Callable[[int], object],
                 closer: Callable[[], None] | None = None, parent=None):
        super().__init__(parent)
        self.make_cutter = make_cutter
        self.closer = closer
        self.thread: threading.Thread | None = None
        self.done = threading.Event()          # set once everything is closed
        self.cutter = None                     # the current track's cutter
        self._track: int | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest = 0
        self._wanted: tuple | None = None
        self._ahead: tuple | None = None
        self._cutting: tuple | None = None
        self._urgent = False

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def retarget(self, generation: int) -> None:
        """Drop every queued job and abandon the cut in progress."""
        with self._cond:
            self._latest = int(generation)
            self._wanted = self._ahead = None
            self._cond.notify_all()

    def want(self, generation: int, track: int, first: int, last: int) -> None:
        """Cut this chunk next. Replaces any earlier urgent job."""
        job = (int(generation), int(track), int(first), int(last))
        with self._cond:
            self._latest = job[0]
            self._wanted = None if job == self._cutting else job
            if self._ahead == job:
                self._ahead = None
            self._cond.notify_all()
        self._start()

    def read_ahead(self, generation: int, track: int, first: int, last: int) -> None:
        """Cut this chunk when idle. Replaces any earlier read-ahead."""
        job = (int(generation), int(track), int(first), int(last))
        with self._cond:
            if job in (self._cutting, self._wanted):
                return
            self._latest = job[0]
            self._ahead = job
            self._cond.notify_all()
        self._start()

    def stop(self) -> None:
        """Ask the worker to finish; returns at once."""
        self._stop.set()
        with self._cond:
            self._wanted = self._ahead = None
            self._cond.notify_all()
        if self.thread is None:
            self._close()

    def _start(self) -> None:
        if self.thread is None and not self._stop.is_set():
            self.thread = threading.Thread(
                target=self._run, name="trackanno-review-decode", daemon=True)
            self.thread.start()

    def _stale(self, gen: int) -> bool:
        return self._stop.is_set() or gen != self._latest

    def _cancelled(self) -> bool:
        job = self._cutting
        if job is None or self._stale(job[0]):
            return True
        # A read-ahead gives way to an urgent request.
        return not self._urgent and self._wanted is not None

    def _next(self) -> tuple | None:
        with self._cond:
            while not self._stop.is_set() and self._wanted is None and self._ahead is None:
                self._cond.wait()
            if self._stop.is_set():
                return None
            if self._wanted is not None:
                job, self._wanted, self._urgent = self._wanted, None, True
            else:
                job, self._ahead, self._urgent = self._ahead, None, False
            self._cutting = job
            return job

    def _cutter_for(self, gen: int, track: int):
        """The cutter for ``track``, made and opened if the track changed."""
        cutter = self.cutter
        if cutter is not None and self._track == track:
            return cutter
        if cutter is not None:
            cutter.close()
            self.cutter = None
        cutter = self.make_cutter(track)
        try:
            cutter.open()
        except Exception as exc:
            cutter.close()
            if not self._stale(gen):
                self.failed.emit(gen, -1, _one_line(exc), True)
            return None
        self.cutter, self._track = cutter, track
        return cutter

    def _run(self) -> None:
        try:
            while True:
                job = self._next()
                if job is None:
                    return
                gen, track, first, last = job
                try:
                    if self._stale(gen):
                        continue
                    cutter = self._cutter_for(gen, track)
                    if cutter is None:
                        continue
                    payload = cutter.cut(first, last, self._cancelled)
                except Cancelled:
                    continue
                except Exception as exc:
                    if not self._stale(gen):
                        self.failed.emit(gen, first, _one_line(exc), False)
                    continue
                finally:
                    with self._cond:
                        self._cutting = None
                if not self._stale(gen):
                    self.ready.emit(gen, first, payload)
        finally:
            self._close()

    def _close(self) -> None:
        cutter, self.cutter = self.cutter, None
        try:
            if cutter is not None:
                cutter.close()
            if self.closer is not None:
                self.closer()
        finally:
            self.done.set()


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split()) or type(exc).__name__
