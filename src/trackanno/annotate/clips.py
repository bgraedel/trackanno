"""Clip planning, crop geometry and the pre-cropped clip cache.

A *clip* is ``clips.frames`` consecutive frames of one track, cropped to a
square window around it and pre-extracted by ``trackanno prepare`` into an
uncompressed ``np.memmap``: one contiguous read (~1 MB) per clip, no decoding.

Crop geometry rules:

* Centred on the track and **never slid back inside the frame**; ``_crop``
  zero-pads instead. Clamping pins border-swimming animals against their own
  crop edge.
* Sized to hold the animal on every frame plus ``clips.pad`` of margin, which
  is what lets the GUI zoom and re-centre at display time from the one cache.
* Uniform ``S x S``. A clip needing more is downscaled and its ``scale``
  recorded, so every cached pixel maps back to an exact original coordinate.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FIXED_SETTINGS, Config

CROP_FROM = FIXED_SETTINGS["clips.crop_from"]            # bbox: p95 box extent
SMOOTH_WINDOW = FIXED_SETTINGS["clips.smooth_window"]    # frames, follow: smooth
MAX_SIZE = FIXED_SETTINGS["clips.max_size"]              # cap on an auto crop size
CONTEXT_TRACKS = FIXED_SETTINGS["clips.context_tracks"]  # record neighbours in the crop
MAX_CLIPS_PER_TRACK = FIXED_SETTINGS["clips.max_clips_per_track"]  # 0 is unlimited

__all__ = [
    "ClipSpec", "clip_id_for", "plan_clips", "border_flags", "drop_border_rows",
    "too_much_border", "choose_crop_size", "finalise_geometry",
    "build_box_table", "write_clip_cache", "ClipStore", "StagedWrite",
]

CACHE_FILE = "clips.dat"
MANIFEST_FILE = "clips.parquet"
BOXES_FILE = "clip_boxes.parquet"
ORIGINS_FILE = "clip_origins.npy"
FEATURES_FILE = "instance_features.npy"
HEADER_FILE = "cache.json"
TRACKS_FILE = "tracks.parquet"
NORM_FILE = "norm.json"
DONE_FILE = "prepare_done.json"
# The detection stage's marker. Separate from DONE_FILE because `project_status`
# counts that one as "this movie has clips": a movie detected on a GPU box but
# not yet cut must still read as unprepared.
DETECT_FILE = "detect_done.json"


@dataclass
class ClipSpec:
    """One annotatable unit, before the pixels are cut."""

    clip_id: str
    movie_id: str
    track_id: int
    f_start: int
    f_end: int              # exclusive
    crop_x0: int            # original full-frame pixel coords
    crop_y0: int
    crop_w: int
    crop_h: int
    scale: float            # cache px per original px (1.0 unless downscaled)
    n_present: int          # frames in the window where the track was detected
    mean_conf: float


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def clip_id_for(movie_id: str, track_id: int, f_start: int) -> str:
    """The join key for a (movie, track, window).

    Used by the annotation CSV, the manifest and the feature tables, so a grid
    built elsewhere must produce byte-identical ids. It embeds the track id, so
    re-running detection renumbers tracks and changes every id (see
    :mod:`trackanno.annotate.reassociate`).
    """
    return f"{movie_id}__t{int(track_id):04d}__f{int(f_start):06d}"


def _windows(cfg: Config, n_frames: int) -> list[tuple[int, int]]:
    """Global window grid for a movie, aligned to the movie's frame grid rather
    than to each track's start, so clips from different tracks share frame
    ranges."""
    length, stride = cfg.clips.frames, cfg.clips.stride
    if cfg.clips.window_mode == "first":
        return [(0, min(length, n_frames))] if n_frames >= cfg.clips.min_clip_frames else []
    out = []
    start = 0
    while start + cfg.clips.min_clip_frames <= n_frames:
        out.append((start, min(start + length, n_frames)))
        start += stride
    return out


def _hold_fill(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Fill gaps (frames where the track was not detected) from the nearest hit."""
    if present.all():
        return values
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return values
    all_idx = np.arange(values.shape[0])
    nearest = idx[np.clip(np.searchsorted(idx, all_idx), 0, idx.size - 1)]
    prev = idx[np.clip(np.searchsorted(idx, all_idx, side="right") - 1, 0, idx.size - 1)]
    pick = np.where(np.abs(nearest - all_idx) <= np.abs(prev - all_idx), nearest, prev)
    out = values.copy()
    out[~present] = values[pick[~present]]
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average along axis 0, with edge padding."""
    if window <= 1 or values.shape[0] <= 2:
        return values
    half = max(1, window // 2)
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])],
        axis=1,
    )


BORDER_BOX_COLUMNS = ("x1", "y1", "x2", "y2")
# Carried on the planner's own copy of the tracks table, so the border test is
# run once per movie rather than once per window.
_AT_BORDER = "_at_border"


def border_flags(tracks: pd.DataFrame, frame_size, cfg: Config) -> "pd.Series":
    """Per detection, whether its box comes within
    ``tracking.border_margin_px`` of any side of the frame.

    At the edge the box is truncated by the frame rather than by the animal
    ending, so a pose model places its keypoints against a shape that is not all
    there. Such a detection is treated as no detection: it is the frames that
    go, not the track, so an animal that visits the edge and swims back is
    measured over the rest of its life as usual.

    All False unless ``tracking.border_margin_px`` says otherwise, and refuses
    rather than passing everything through when it cannot be measured.
    """
    off = pd.Series(False, index=tracks.index, dtype=bool)
    if int(cfg.tracking.border_margin_px) <= 0 or tracks.empty:
        return off
    if frame_size is None:
        raise ValueError(
            "tracking.border_margin_px is set, so detections at the frame edge "
            "are to be excluded, but this caller did not say how big the frame "
            "is and the rule cannot be applied without it"
        )
    if any(c not in tracks.columns for c in BORDER_BOX_COLUMNS):
        raise ValueError(
            "tracking.border_margin_px is set, but this tracks table carries no "
            f"{', '.join(BORDER_BOX_COLUMNS)} columns, so how near the edge a "
            "detection sits cannot be measured. Re-run detection to rebuild it"
        )
    height, width = int(frame_size[0]), int(frame_size[1])
    m = float(cfg.tracking.border_margin_px)
    x1 = pd.to_numeric(tracks["x1"], errors="coerce").to_numpy(dtype=float)
    y1 = pd.to_numeric(tracks["y1"], errors="coerce").to_numpy(dtype=float)
    x2 = pd.to_numeric(tracks["x2"], errors="coerce").to_numpy(dtype=float)
    y2 = pd.to_numeric(tracks["y2"], errors="coerce").to_numpy(dtype=float)
    at_edge = (x1 <= m) | (y1 <= m) | (x2 >= width - m) | (y2 >= height - m)
    return pd.Series(at_edge, index=tracks.index, dtype=bool)


def drop_border_rows(tracks: pd.DataFrame, frame_size, cfg: Config
                     ) -> tuple[pd.DataFrame, dict[int, float]]:
    """``tracks`` without the detections at the frame edge, and the share of
    each affected track that went.

    The track itself survives: only the frames inside the margin are removed,
    and what comes before and after them is planned as usual.
    """
    flags = border_flags(tracks, frame_size, cfg)
    if not flags.any():
        return tracks, {}
    share = flags.groupby(tracks["track_id"].to_numpy()).mean()
    return tracks[~flags.to_numpy()], {
        int(t): float(v) for t, v in share.items() if v > 0
    }


def too_much_border(n_border: int, n_present: int, cfg: Config) -> bool:
    """Whether a window spent too much of itself at the frame edge to be a unit.

    ``tracking.border_max_frac`` is the share of a window's detections allowed
    inside the margin. 0, the default, refuses a window that reaches the edge at
    all; the same track's windows before and after the excursion are untouched.
    Without this the edge frames would only be missing, and a window mostly at
    the edge would still be cut around the few frames that were not.
    """
    if n_border <= 0:
        return False
    return n_border / (n_border + n_present) > float(cfg.tracking.border_max_frac)


def plan_clips(tracks: pd.DataFrame, movie_id: str, n_frames: int, cfg: Config,
               frame_size=None, at_border: dict | None = None) -> list[ClipSpec]:
    """Turn a movie's tracks table into clip specs. Reads no pixels.

    Crop size comes from the *per-frame* box (``crop_from: bbox``), not the
    track's whole excursion, so the crop has to move. ``clips.follow`` says how:

    ``track``   re-centred every frame - smallest crop, but translation becomes
                invisible, so "swimming" and "immotile" look identical.
    ``smooth``  moving average of the centre - drift removed, local motion kept.
                Default.
    ``fixed``   one crop for the window - all motion visible, biggest crop.

    ``frame_size`` is ``(height, width)`` and is only needed by
    ``tracking.border_margin_px``; without it that setting cannot be applied and
    refuses rather than passing everything through.

    ``at_border``, when given, is filled with ``{track_id: share of its
    detections at the frame edge}`` for every track that reached it. A caller
    wanting to report them takes them from here, since recomputing the number
    means restating which detections were candidates in the first place.
    """
    if tracks.empty:
        return []

    # A detection inside the margin is not one to measure, so it counts neither
    # towards min_track_length nor towards a clip's frames. Marked rather than
    # dropped, because a window still has to see how much of itself was there.
    edge_flags = border_flags(tracks, frame_size, cfg).to_numpy()
    if at_border is not None and edge_flags.any():
        share = pd.Series(edge_flags).groupby(tracks["track_id"].to_numpy()).mean()
        at_border.update({int(t): float(v) for t, v in share.items() if v > 0})

    counts = pd.Series(~edge_flags).groupby(tracks["track_id"].to_numpy()).sum()
    keep = counts[counts >= cfg.tracking.min_track_length].index
    kept_rows = tracks["track_id"].isin(keep).to_numpy()
    tracks, edge_flags = tracks[kept_rows], edge_flags[kept_rows]
    if tracks.empty:
        return []
    tracks = tracks.assign(**{_AT_BORDER: edge_flags})

    windows = _windows(cfg, n_frames)
    length = cfg.clips.frames
    specs: list[ClipSpec] = []

    for track_id, tdf in tracks.groupby("track_id", sort=True):
        tdf = tdf.sort_values("frame")
        frames = tdf["frame"].to_numpy()
        edge = tdf[_AT_BORDER].to_numpy()
        made = 0
        for f_start, f_end in windows:
            inside = (frames >= f_start) & (frames < f_end)
            sel = inside & ~edge
            n_present = int(sel.sum())
            if n_present < cfg.clips.min_clip_frames:
                continue
            if too_much_border(int((inside & edge).sum()), n_present, cfg):
                continue

            sub = tdf.iloc[np.flatnonzero(sel)]
            span = f_end - f_start
            offsets = sub["frame"].to_numpy() - f_start

            # Per-frame boxes, gaps held from the nearest detected frame.
            boxes = np.full((span, 4), np.nan, dtype=np.float64)
            boxes[offsets] = sub[["x1", "y1", "x2", "y2"]].to_numpy()
            present = np.zeros(span, bool)
            present[offsets] = True
            boxes = _hold_fill(boxes, present)

            centres = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1
            )
            extents = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

            if cfg.clips.follow == "track":
                path = centres
            elif cfg.clips.follow == "fixed":
                path = np.repeat(centres.mean(axis=0, keepdims=True), span, axis=0)
            else:
                path = _moving_average(centres, SMOOTH_WINDOW)

            if CROP_FROM == "union":
                base = float(np.nanmax(
                    [boxes[:, 2].max() - boxes[:, 0].min(), boxes[:, 3].max() - boxes[:, 1].min()]
                ))
            else:
                base = float(np.percentile(extents, 95))

            # The crop must still hold the animal on every frame given the
            # (possibly smoothed) centre, so take whichever is larger.
            reach = np.max(np.abs(
                np.concatenate([boxes[:, :2] - path, boxes[:, 2:] - path], axis=1)
            ), axis=1)
            need = int(np.ceil(max(base * cfg.clips.pad, float(reach.max()) * 2.0 * 1.1)))
            need = max(need, 16)

            specs.append(
                ClipSpec(
                    clip_id=clip_id_for(movie_id, int(track_id), int(f_start)),
                    movie_id=movie_id,
                    track_id=int(track_id),
                    f_start=int(f_start),
                    f_end=int(f_end),
                    crop_x0=0, crop_y0=0,           # filled by finalise_geometry
                    crop_w=need, crop_h=need,
                    scale=1.0,
                    n_present=n_present,
                    mean_conf=float(sub["conf"].mean()),
                )
            )
            # Per-frame crop centres, consumed by finalise_geometry.
            padded_path = np.repeat(path[-1:], length - span, axis=0) if span < length else None
            specs[-1].__dict__["_path"] = (
                np.concatenate([path, padded_path], axis=0) if padded_path is not None else path
            )

            made += 1
            if cfg.clips.window_mode == "first_per_track":
                break
            if MAX_CLIPS_PER_TRACK and made >= MAX_CLIPS_PER_TRACK:
                break

    return specs


def choose_crop_size(specs: list[ClipSpec], cfg: Config) -> int:
    """Uniform cache side length S, in cache pixels."""
    if isinstance(cfg.clips.crop_size, int):
        return max(16, int(cfg.clips.crop_size))
    if not specs:
        return 64
    needed = np.array([s.crop_w for s in specs], dtype=float)
    size = int(np.ceil(np.percentile(needed, 99) / 16.0) * 16)
    return int(np.clip(size, 32, MAX_SIZE))


def finalise_geometry(specs: list[ClipSpec], size: int, n_frames: int) -> np.ndarray:
    """Fix each clip's crop rectangle and scale; return per-frame crop origins.

    An ``(n_clips, T, 2)`` int32 array of top-left corners in original pixel
    coordinates: the crop moves with the track, so one origin per clip cannot
    map pixels back. ``ClipSpec.crop_x0/y0`` keeps the ``f_start`` value.
    """
    origins = np.zeros((len(specs), n_frames, 2), dtype=np.int32)
    for row, s in enumerate(specs):
        path = s.__dict__.pop("_path", None)
        if path is None:
            # Never guess: zeros crop every clip to the frame's top-left corner
            # while the manifest and clip_boxes still look well-formed.
            raise ValueError(
                f"{s.clip_id}: no crop path. finalise_geometry() must be called exactly "
                "once, on specs straight from plan_clips() (a spec rebuilt from "
                "clips.parquet has lost its trajectory)."
            )

        # <= S: crop exactly S px and keep 1:1 (free extra context).
        # >  S: crop what is needed and downscale into S.
        side = size if s.crop_w <= size else s.crop_w
        s.crop_w = s.crop_h = int(side)
        s.scale = size / float(side)

        # Deliberately NOT clamped into the frame; `_crop` pads outside with black.
        corners = np.round(path - side / 2.0).astype(np.int64)
        origins[row] = corners.astype(np.int32)

        s.crop_x0 = int(corners[0, 0])
        s.crop_y0 = int(corners[0, 1])
    return origins


# --------------------------------------------------------------------------- #
# overlay geometry (target + neighbours, in cache coordinates)
# --------------------------------------------------------------------------- #
def build_box_table(
    specs: list[ClipSpec], origins: np.ndarray, tracks: pd.DataFrame, cfg: Config,
    n_kpts: int,
) -> pd.DataFrame:
    """Per clip and frame: the target track's box plus any neighbour inside the
    crop, in *cache* pixels so the GUI needs to know nothing about the original
    movie.

    One vectorised pass per clip, over the slice of a frame-sorted copy its
    window covers. Per FRAME it was a DataFrame filter and an ``itertuples``
    row, which is clips x clips.frames pandas calls: minutes of silence in the
    middle of a run on a movie with a few thousand clips.
    """
    if not specs or tracks.empty:
        return pd.DataFrame()

    # Stable, so detections keep their order within a frame and the table comes
    # out in the order the per-frame loop produced.
    order = np.argsort(tracks["frame"].to_numpy(), kind="stable")
    frames = tracks["frame"].to_numpy()[order].astype(np.int64)
    ids = tracks["track_id"].to_numpy()[order].astype(np.int64)
    boxes = tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)[order]
    confs = tracks["conf"].to_numpy(dtype=np.float64)[order]

    kx = [f"kp_x_{k}" for k in range(n_kpts)]
    ky = [f"kp_y_{k}" for k in range(n_kpts)]
    kc = [f"kp_c_{k}" for k in range(n_kpts)]
    has_kpts = n_kpts > 0 and all(c in tracks.columns for c in kx)
    if has_kpts:
        kpx = tracks[kx].to_numpy(dtype=np.float64)[order]
        kpy = tracks[ky].to_numpy(dtype=np.float64)[order]
        kpc = tracks[kc].to_numpy(dtype=np.float64)[order]

    parts: list[dict] = []
    for row, s in enumerate(specs):
        lo = int(np.searchsorted(frames, s.f_start, side="left"))
        hi = int(np.searchsorted(frames, s.f_end, side="left"))
        if hi <= lo:
            continue
        t = frames[lo:hi] - s.f_start
        x0 = origins[row, t, 0].astype(np.float64)
        y0 = origins[row, t, 1].astype(np.float64)
        window, target = boxes[lo:hi], ids[lo:hi]
        if CONTEXT_TRACKS:
            keep = (
                (window[:, 2] > x0) & (window[:, 0] < x0 + s.crop_w)
                & (window[:, 3] > y0) & (window[:, 1] < y0 + s.crop_h)
            )
        else:
            keep = target == s.track_id
        pick = np.flatnonzero(keep)
        if pick.size == 0:
            continue
        sc, x0, y0 = s.scale, x0[pick], y0[pick]
        part = {
            "row": np.full(pick.size, row, dtype=np.int32),
            "t": t[pick].astype(np.int16),
            "track_id": target[pick].astype(np.int32),
            "is_target": target[pick] == s.track_id,
            "x1": (window[pick, 0] - x0) * sc,
            "y1": (window[pick, 1] - y0) * sc,
            "x2": (window[pick, 2] - x0) * sc,
            "y2": (window[pick, 3] - y0) * sc,
            "conf": confs[lo:hi][pick],
        }
        if has_kpts:
            for k in range(n_kpts):
                part[kx[k]] = (kpx[lo:hi][pick, k] - x0) * sc
                part[ky[k]] = (kpy[lo:hi][pick, k] - y0) * sc
                part[kc[k]] = kpc[lo:hi][pick, k]
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(
        {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    )


# --------------------------------------------------------------------------- #
# cache writing
# --------------------------------------------------------------------------- #
def write_clip_cache(
    movie,
    specs: list[ClipSpec],
    origins: np.ndarray,
    size: int,
    out_dir: Path,
    normalizer,
    cfg: Config,
    progress=None,
    staged: "StagedWrite | None" = None,
) -> None:
    """Cut every clip's pixels in ONE sequential pass: each frame is read at most
    once and scattered into every clip that needs it, because random access into
    a large TIFF/ND2 on a share would dominate the runtime."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips, length = len(specs), cfg.clips.frames

    header = {
        "n_clips": n_clips,
        "frames": length,
        "size": size,
        "dtype": "uint8",
        "movie": movie.describe(),
    }

    # Staged beside the real names and renamed in at the very end, so a cancelled
    # or killed run leaves the previous cache intact. The caller may pass its own
    # staging area to land the manifest and box table in the SAME commit.
    owned = staged is None
    staged = staged or StagedWrite(out_dir)
    staged.path(HEADER_FILE).write_text(json.dumps(header, indent=2), encoding="utf-8")

    if n_clips == 0:
        staged.path(CACHE_FILE).write_bytes(b"")
        if owned:
            staged.commit()
        return

    try:
        # Through a file object: np.save appends ".npy" to any path not already
        # ending in it, which would rename the staged "clip_origins.npy.part".
        with open(staged.path(ORIGINS_FILE), "wb") as fh:
            np.save(fh, origins)

        cache = np.memmap(
            staged.path(CACHE_FILE), dtype=np.uint8, mode="w+",
            shape=(n_clips, length, size, size),
        )
    except OSError as exc:
        staged.discard()
        # EINVAL here almost always means "something has this file mapped".
        raise OSError(
            f"could not write the clip cache in {out_dir}: {exc}\n"
            "On Windows a memory-mapped file cannot be overwritten, so this "
            "usually means something still has the cache open - another "
            "trackanno window, a notebook, or a file browser preview. Close it "
            "and run again."
        ) from exc

    # frame -> [(row, spec), ...]
    needed: dict[int, list[tuple[int, ClipSpec]]] = {}
    for row, s in enumerate(specs):
        for f in range(s.f_start, s.f_end):
            needed.setdefault(f, []).append((row, s))

    height, width = movie.height, movie.width
    frames = sorted(needed)
    iterator = progress(frames) if progress else frames

    try:
        for frame_idx in iterator:
            u8 = normalizer.to_u8(movie.frame(frame_idx), frame_idx)

            for row, s in needed[frame_idx]:
                t = frame_idx - s.f_start
                crop = _crop(
                    u8, int(origins[row, t, 0]), int(origins[row, t, 1]),
                    s.crop_w, s.crop_h, height, width,
                )
                if crop.shape[0] != size or crop.shape[1] != size:
                    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
                    crop = cv2.resize(crop, (size, size), interpolation=interp)
                cache[row, t] = crop

        cache.flush()
        del cache
    except BaseException:
        # Cancelled, killed, or the share went away: discard the staged files.
        # Rebinding rather than `del`, which is what this used to do: on the
        # happy path `cache` is deleted before the flushes that can fail, so an
        # unconditional `del` raises NameError over the real error and skips the
        # discard. Windows cannot unlink a mapped file, so the references do
        # have to go before `staged.discard()`.
        cache = None  # noqa: F841 - releases the memmap for discard()
        staged.discard()
        raise

    if owned:
        staged.commit()


class StagedWrite:
    """Write beside the real names, then rename everything into place at once.

    ``os.replace`` is atomic per file, not across files. The live header goes
    first and comes back last, so a crash between renames leaves no cache.json
    and ``ClipStore`` refuses the movie.
    """

    SUFFIX = ".part"

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self._names: list[str] = []
        # A previous crash may have left staged files behind.
        for leftover in self.out_dir.glob(f"*{self.SUFFIX}"):
            leftover.unlink(missing_ok=True)

    def path(self, name: str) -> Path:
        if name not in self._names:
            self._names.append(name)
        return self.out_dir / (name + self.SUFFIX)

    _LAST = (MANIFEST_FILE, HEADER_FILE)

    def commit(self) -> None:
        if (self.out_dir / (HEADER_FILE + self.SUFFIX)).exists():
            (self.out_dir / HEADER_FILE).unlink(missing_ok=True)
        last = [n for n in self._LAST if n in self._names]
        for name in [n for n in self._names if n not in last] + last:
            staged = self.out_dir / (name + self.SUFFIX)
            if staged.exists():
                staged.replace(self.out_dir / name)
        self._names.clear()

    def discard(self) -> None:
        for name in self._names:
            (self.out_dir / (name + self.SUFFIX)).unlink(missing_ok=True)
        self._names.clear()


def _crop(img: np.ndarray, x0: int, y0: int, w: int, h: int, height: int, width: int) -> np.ndarray:
    """Crop with zero padding where the window falls outside the frame."""
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + w), min(height, y0 + h)
    if sx0 == x0 and sy0 == y0 and sx1 == x0 + w and sy1 == y0 + h:
        return img[sy0:sy1, sx0:sx1]
    out = np.zeros((h, w), dtype=img.dtype)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def specs_to_frame(specs: list[ClipSpec]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(s) for s in specs])
    if not df.empty:
        df.insert(0, "row", np.arange(len(df), dtype=np.int32))
    return df


# --------------------------------------------------------------------------- #
# cache reading
# --------------------------------------------------------------------------- #
class ClipStore:
    """Read-only view over every prepared movie's clip cache.

    One flat, deterministically ordered index across all movies. The GUI
    addresses clips by that index; ``clip_id`` stays the stable CSV key, so
    re-preparing a movie never corrupts existing labels.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        if not self.work_dir.exists():
            raise FileNotFoundError(
                f"clip cache not found: {self.work_dir}\nRun `trackanno prepare` first."
            )

        manifests, boxes, self._headers = [], [], {}
        self.incomplete: dict[str, str] = {}
        for movie_dir in sorted(p for p in self.work_dir.iterdir() if p.is_dir()):
            man_path = movie_dir / MANIFEST_FILE
            if not man_path.exists():
                continue
            man = pd.read_parquet(man_path)
            if man.empty:
                continue
            reason = self._inconsistent(movie_dir, man)
            if reason:
                # Refuse rather than mis-associate: a manifest that disagrees with
                # its cache pairs each row with another clip's pixels, which looks
                # completely normal while you label it.
                self.incomplete[movie_dir.name] = reason
                warnings.warn(
                    f"{movie_dir.name}: {reason}. Skipping it; re-run detection "
                    "for this movie with 'redo' ticked.",
                    stacklevel=2,
                )
                continue
            manifests.append(man)
            self._headers[movie_dir.name] = json.loads(
                (movie_dir / HEADER_FILE).read_text(encoding="utf-8")
            )
            box_path = movie_dir / BOXES_FILE
            if box_path.exists():
                bx = pd.read_parquet(box_path)
                if not bx.empty:
                    bx = bx.copy()
                    bx["movie_id"] = movie_dir.name
                    boxes.append(bx)

        if not manifests:
            raise FileNotFoundError(
                f"no prepared clips under {self.work_dir}\nRun `trackanno prepare` first."
            )

        self.manifest = (
            pd.concat(manifests, ignore_index=True)
            .sort_values(["movie_id", "track_id", "f_start"], kind="stable")
            .reset_index(drop=True)
        )
        self.boxes = (
            pd.concat(boxes, ignore_index=True).set_index(["movie_id", "row"]).sort_index()
            if boxes else pd.DataFrame()
        )
        self._closed = False
        self._mm: dict[str, np.memmap] = {}
        self._origins: dict[str, np.ndarray | None] = {}
        self._id_to_index = {cid: i for i, cid in enumerate(self.manifest["clip_id"])}

    @staticmethod
    def _inconsistent(movie_dir: Path, manifest: pd.DataFrame) -> str:
        """Why this movie's cache cannot be trusted, or "" if it can.

        Manifest, cache header and file size record the same shape
        independently, so disagreement means they came from different runs.
        """
        header_path = movie_dir / HEADER_FILE
        if not header_path.exists():
            return "cache.json is missing"
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"cache.json is unreadable ({exc})"

        n_clips, frames = int(header.get("n_clips", -1)), int(header.get("frames", 0))
        size = int(header.get("size", 0))
        if n_clips != len(manifest):
            return (
                f"clips.parquet lists {len(manifest)} clips but cache.json says "
                f"{n_clips}; the cache was rebuilt but the manifest was not"
            )

        cache_path = movie_dir / CACHE_FILE
        if not cache_path.exists():
            return "clips.dat is missing"
        expected = n_clips * frames * size * size
        actual = cache_path.stat().st_size
        if actual != expected:
            return (
                f"clips.dat is {actual} bytes but {n_clips} clips need exactly "
                f"{expected}; it does not belong to this manifest"
            )
        return ""

    # -- basics ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.manifest)

    def meta(self, index: int) -> pd.Series:
        return self.manifest.iloc[index]

    def index_of(self, clip_id: str) -> int | None:
        return self._id_to_index.get(clip_id)

    @property
    def movie_ids(self) -> list[str]:
        return sorted(self.manifest["movie_id"].unique().tolist())

    def header(self, movie_id: str) -> dict:
        return self._headers[movie_id]

    # -- pixels -------------------------------------------------------------- #
    def _memmap(self, movie_id: str) -> np.memmap:
        if self._closed:
            # Otherwise a late prefetch re-creates the mapping the caller just
            # released, re-locking the file.
            raise RuntimeError("this clip cache has been closed")
        mm = self._mm.get(movie_id)
        if mm is None:
            h = self._headers[movie_id]
            mm = np.memmap(
                self.work_dir / movie_id / CACHE_FILE,
                dtype=np.dtype(h["dtype"]),
                mode="r",
                shape=(h["n_clips"], h["frames"], h["size"], h["size"]),
            )
            self._mm[movie_id] = mm
        return mm

    def clip(self, index: int) -> np.ndarray:
        """(T, S, S) uint8 - one contiguous read from the cache."""
        m = self.meta(index)
        return np.asarray(self._memmap(m["movie_id"])[int(m["row"])])

    def touch(self, index: int) -> None:
        """Warm the OS page cache for a clip without materialising it (prefetch)."""
        m = self.meta(index)
        arr = self._memmap(m["movie_id"])[int(m["row"])]
        _ = int(arr[0, 0, 0]) + int(arr[-1, -1, -1])

    def origins(self, index: int) -> np.ndarray | None:
        """(T, 2) top-left corners in ORIGINAL pixel coords, one per frame.

        The crop follows the track: a cached pixel maps back as
        ``origins[t] + cache_xy / scale``. crop_x0/y0 is only the f_start value.
        """
        m = self.meta(index)
        movie_id = str(m["movie_id"])
        if movie_id not in self._origins:
            path = self.work_dir / movie_id / ORIGINS_FILE
            self._origins[movie_id] = np.load(path, mmap_mode="r") if path.exists() else None
        arr = self._origins[movie_id]
        return None if arr is None else np.asarray(arr[int(m["row"])])

    def close(self) -> None:
        """Release every memory map this store holds.

        Windows will not let a mapped file be overwritten, so the GUI must call
        this before running detection; otherwise re-preparing a movie whose cache
        is open fails with ``OSError: [Errno 22] Invalid argument``.
        """
        self._closed = True
        for holder in (self._mm, self._origins):
            for key in list(holder):
                array = holder.pop(key, None)
                handle = getattr(array, "_mmap", None)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

    def clip_boxes(self, index: int) -> pd.DataFrame:
        """Overlay geometry for one clip, in cache pixel coordinates."""
        if self.boxes.empty:
            return pd.DataFrame()
        m = self.meta(index)
        key = (m["movie_id"], int(m["row"]))
        try:
            out = self.boxes.loc[[key]]
        except KeyError:
            return pd.DataFrame()
        return out.reset_index(drop=True)
