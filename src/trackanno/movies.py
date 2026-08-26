"""Format-agnostic movie access.

Everything downstream (detection, clip extraction, the GUI, the renderer) sees
one abstraction: a *MovieSource* is a 2-D grayscale time series with a stable id
whose ``frame(i)`` returns the raw i-th frame.

``TiffMovie``
    Multi-page / stacked TIFF. Uncompressed contiguous files open as a true
    ``np.memmap``; anything else falls back to per-page decoding.

``ND2Movie``
    Nikon ND2 via the ``nd2`` package. ND2 carries P (multipoint), T, C and Z
    axes, so **each P position becomes its own MovieSource**, with channel / z
    from the config and the frame interval read out of the ND2 metadata.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config

__all__ = [
    "MovieSource",
    "TiffMovie",
    "ND2Movie",
    "Normalizer",
    "discover_movies",
    "open_movie",
]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 2:
        return values
    half = max(1, window // 2)
    padded = np.pad(values, half, mode="edge")
    return np.array([
        np.median(padded[i : i + 2 * half + 1]) for i in range(values.size)
    ], dtype=np.float64)


@dataclass
class Normalizer:
    """Percentile -> uint8 mapping for one movie, optionally per frame.

    These acquisitions drift, so one fixed mapping faithfully reproduces a movie
    that visibly dims, while per-frame percentiles remove the drift but make
    brightness breathe as animals enter and leave. ``smooth`` (the default) does
    neither: percentiles measured on a subset of frames, median-filtered along
    time, then interpolated to every frame. The curve is persisted so prepare,
    the GUI and the renderer agree pixel-for-pixel.
    """

    lo: float | list[float]
    hi: float | list[float]
    mode: str = "fixed"

    def __post_init__(self) -> None:
        self._lo = np.atleast_1d(np.asarray(self.lo, dtype=np.float32))
        self._hi = np.atleast_1d(np.asarray(self.hi, dtype=np.float32))
        self._per_frame = self._lo.size > 1

    @property
    def length(self) -> int:
        """Frames this curve covers (1 for a single fixed mapping)."""
        return int(self._lo.size)

    def covers(self, n_frames: int) -> bool:
        return not self._per_frame or self._lo.size >= n_frames

    def _range(self, index: int | None) -> tuple[float, float]:
        if not self._per_frame or index is None:
            return float(self._lo[0]), float(self._hi[0])
        i = int(np.clip(index, 0, self._lo.size - 1))
        return float(self._lo[i]), float(self._hi[i])

    def to_u8(self, frame: np.ndarray, index: int | None = None) -> np.ndarray:
        im = np.asarray(frame)
        lo, hi = self._range(index)
        if im.dtype == np.uint8 and lo == 0.0 and hi == 255.0:
            return np.ascontiguousarray(im)
        span = max(hi - lo, 1e-6)
        out = (im.astype(np.float32) - lo) * (255.0 / span)
        return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))

    def to_rgb_u8(self, frame: np.ndarray, index: int | None = None) -> np.ndarray:
        u8 = self.to_u8(frame, index)
        if u8.ndim == 2:
            u8 = np.repeat(u8[:, :, None], 3, axis=2)
        elif u8.ndim == 3 and u8.shape[2] == 1:
            u8 = np.repeat(u8, 3, axis=2)
        return np.ascontiguousarray(u8)

    # -- persistence -------------------------------------------------------- #
    def save(self, path: Path) -> None:
        payload = {
            "mode": self.mode,
            "lo": self._lo.tolist() if self._per_frame else float(self._lo[0]),
            "hi": self._hi.tolist() if self._per_frame else float(self._hi[0]),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(lo=data["lo"], hi=data["hi"], mode=data.get("mode", "fixed"))

    @classmethod
    def from_movie(
        cls, movie: "MovieSource", options=None, progress=None, n_frames: int | None = None
    ) -> "Normalizer":
        """Measure the intensity mapping for a movie.

        Percentiles come from a strided pixel subsample: ample for a percentile,
        and cheap even on very large frames.
        """
        from .config import Normalize

        opt = options or Normalize()
        n = min(movie.n_frames, n_frames) if n_frames else movie.n_frames
        if opt.mode == "none":
            return cls(0.0, 255.0, mode="none")

        step = max(1, opt.subsample)
        pcts = opt.percentiles

        if opt.mode == "fixed":
            idx = np.unique(np.linspace(0, n - 1, min(20, n)).astype(int))
        elif opt.mode == "per_frame":
            # Every frame gets its own mapping: illumination here swings frame to
            # frame, so anything coarser leaves visible pulsing.
            idx = np.arange(n)
        else:
            every = opt.sample_every
            if not isinstance(every, int):
                every = max(1, n // 100)      # ~100 samples is plenty for slow drift
            idx = np.unique(np.arange(0, n, max(1, int(every))))
            if idx[-1] != n - 1:
                idx = np.append(idx, n - 1)

        iterator = progress(idx) if progress else idx
        los, his = [], []
        for i in iterator:
            frame = np.asarray(movie.frame(int(i)))[::step, ::step]
            if frame.size == 0:
                los.append(0.0)
                his.append(255.0)
                continue
            lo, hi = np.nanpercentile(frame, pcts)
            los.append(float(lo))
            his.append(float(hi))

        if not los:
            return cls(0.0, 255.0, mode=opt.mode)

        los_a, his_a = np.asarray(los, float), np.asarray(his, float)
        if opt.mode == "fixed":
            lo, hi = float(np.median(los_a)), float(np.median(his_a))
            return cls(lo, max(hi, lo + 1.0), mode="fixed")

        if opt.mode == "smooth":
            los_a = _rolling_median(los_a, opt.smooth_window)
            his_a = _rolling_median(his_a, opt.smooth_window)

        frames = np.arange(n)
        lo_curve = np.interp(frames, idx, los_a)
        hi_curve = np.interp(frames, idx, his_a)
        hi_curve = np.maximum(hi_curve, lo_curve + 1.0)
        return cls(lo=lo_curve.tolist(), hi=hi_curve.tolist(), mode=opt.mode)


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class MovieSource:
    """One annotatable 2-D grayscale time series."""

    id: str
    path: Path
    series: int          # ND2 position index; always 0 for TIFF
    channel: int
    n_frames: int
    height: int
    width: int
    dtype: np.dtype
    fps: float
    pixel_size_um: float | None

    def frame(self, index: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass

    def __enter__(self) -> "MovieSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n_frames):
            yield self.frame(i)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def describe(self) -> dict:
        return {
            "movie_id": self.id,
            "source_file": str(self.path),
            "series": self.series,
            "channel": self.channel,
            "n_frames": self.n_frames,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.dtype),
            "fps": self.fps,
            "pixel_size_um": self.pixel_size_um,
        }

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.id!r} {self.n_frames}x{self.height}x{self.width} "
            f"{self.dtype} @ {self.fps:g} fps>"
        )


# --------------------------------------------------------------------------- #
# TIFF
# --------------------------------------------------------------------------- #
class TiffMovie(MovieSource):
    def __init__(self, path: Path, cfg: Config):
        import tifffile

        self.path = Path(path)
        self.id = self.path.stem
        self.series = 0
        self.channel = int(cfg.tiff.channel)
        self.fps = float(cfg.tiff.fps)
        self.pixel_size_um = None

        self._tif = tifffile.TiffFile(str(self.path))
        series = self._tif.series[0]
        axes, shape = series.axes, tuple(series.shape)

        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"{self.path.name}: cannot find Y/X axes in TIFF axes {axes!r}")
        self._axes = axes
        self._shape = shape
        yi, xi = axes.index("Y"), axes.index("X")
        self.height, self.width = int(shape[yi]), int(shape[xi])
        self.dtype = np.dtype(series.dtype)

        self._chan_axis = next((axes.index(a) for a in ("C", "S") if a in axes), None)
        self._n_channels = int(shape[self._chan_axis]) if self._chan_axis is not None else 1
        if self.channel >= self._n_channels:
            raise ValueError(
                f"{self.path.name}: tiff.channel={self.channel} but the file has "
                f"{self._n_channels} channel(s)"
            )

        # Every axis that is not Y/X/C is time (multiple such axes get flattened).
        self._time_axes = [i for i in range(len(axes)) if i not in (yi, xi, self._chan_axis)]
        self._time_shape = [int(shape[i]) for i in self._time_axes]
        self.n_frames = int(np.prod(self._time_shape)) if self._time_shape else 1

        # Pages are laid out by every non-YX axis in the series' own order, so
        # derive the layout rather than assuming channel-fastest
        # (`t * n_channels + c`), which reads the wrong plane for a CTYX stack.
        self._page_axes = [i for i in range(len(axes)) if i not in (yi, xi)]
        self._page_shape = [int(shape[i]) for i in self._page_axes]

        # Fast path: uncompressed + contiguous -> real memmap, zero-copy reads.
        self._mm: np.ndarray | None = None
        self._cache: np.ndarray | None = None
        try:
            mm = tifffile.memmap(str(self.path))
            if tuple(mm.shape) == shape:
                self._mm = mm
        except (ValueError, MemoryError, NotImplementedError):
            self._mm = None

        if self._mm is None and len(self._tif.pages) == 1 and len(shape) >= 3:
            # Whole stack lives inside a single page: decode once, keep it.
            self._cache = series.asarray()

    # -- indexing ---------------------------------------------------------- #
    def _index_tuple(self, t: int) -> tuple:
        coords = np.unravel_index(t, self._time_shape) if self._time_shape else ()
        idx: list = [slice(None)] * len(self._shape)
        for axis, c in zip(self._time_axes, np.atleast_1d(coords)):
            idx[axis] = int(c)
        if self._chan_axis is not None:
            idx[self._chan_axis] = self.channel
        return tuple(idx)

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_frames:
            raise IndexError(f"frame {index} out of range (0..{self.n_frames - 1})")
        src = self._mm if self._mm is not None else self._cache
        if src is not None:
            return np.asarray(src[self._index_tuple(index)])

        # Page-wise fallback (compressed / non-contiguous files). Map the frame
        # and channel onto a page using the series' real axis order.
        if not self._page_shape:
            page = 0
        else:
            coords = np.unravel_index(index, self._time_shape) if self._time_shape else ()
            time_pos = {axis: int(c) for axis, c in zip(self._time_axes, np.atleast_1d(coords))}
            page_coords = [
                self.channel if axis == self._chan_axis else time_pos.get(axis, 0)
                for axis in self._page_axes
            ]
            page = int(np.ravel_multi_index(tuple(page_coords), tuple(self._page_shape)))

        arr = np.asarray(self._tif.pages[page].asarray())
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr[..., min(self.channel, arr.shape[-1] - 1)]
        if arr.ndim != 2:
            raise ValueError(
                f"{self.path.name}: page {page} decoded to shape {arr.shape}, expected a 2-D "
                f"plane (series axes {self._axes!r}). Re-save the file uncompressed, or "
                "report this layout."
            )
        return arr

    def close(self) -> None:
        self._mm = None
        self._cache = None
        try:
            self._tif.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# ND2
# --------------------------------------------------------------------------- #
def _nd2_fps(f) -> float | None:
    """Frame rate from the ND2 acquisition loops, else from event timestamps."""
    try:
        for loop in getattr(f, "experiment", []) or []:
            params = getattr(loop, "parameters", None)
            if params is None:
                continue
            period = getattr(params, "periodMs", None)
            if period:
                return 1000.0 / float(period)
            periods = getattr(params, "periods", None)  # NETimeLoop
            if periods:
                period = getattr(periods[0], "periodMs", None)
                if period:
                    return 1000.0 / float(period)
    except Exception:
        pass
    try:
        events = f.events()
        times = [e.get("Time [s]") for e in events if e.get("Time [s]") is not None]
        if len(times) > 2:
            dt = float(np.median(np.diff(np.asarray(times, dtype=float))))
            if dt > 0:
                return 1.0 / dt
    except Exception:
        pass
    return None


def _nd2_channel_index(f, want: int | str) -> int:
    if isinstance(want, int):
        return want
    names = []
    try:
        for ch in f.metadata.channels:
            names.append(str(ch.channel.name))
    except Exception:
        pass
    if want in names:
        return names.index(want)
    raise ValueError(f"nd2.channel={want!r} not found; available channels: {names or 'unknown'}")


class ND2Movie(MovieSource):
    """One position (P) of an ND2 file, at a fixed channel and z."""

    def __init__(self, path: Path, cfg: Config, position: int = 0, _file=None):
        import nd2

        self.path = Path(path)
        self._own_file = _file is None
        self._f = _file if _file is not None else nd2.ND2File(str(self.path))
        f = self._f

        sizes = dict(f.sizes)
        self._sizes = sizes
        self._axes = list(sizes)  # ordered as stored

        if "Y" not in sizes or "X" not in sizes:
            raise ValueError(f"{self.path.name}: ND2 has no Y/X axes (sizes={sizes})")
        self.height, self.width = int(sizes["Y"]), int(sizes["X"])
        self.dtype = np.dtype(f.dtype)

        self.series = int(position)
        n_pos = int(sizes.get("P", 1))
        if self.series >= n_pos:
            raise ValueError(f"{self.path.name}: position {self.series} but file has {n_pos}")
        self.id = f"{self.path.stem}__P{self.series:02d}" if n_pos > 1 else self.path.stem

        self.channel = _nd2_channel_index(f, cfg.nd2.channel)
        n_chan = int(sizes.get("C", 1))
        if self.channel >= n_chan:
            raise ValueError(f"{self.path.name}: nd2.channel={self.channel} but file has {n_chan}")

        self._n_z = int(sizes.get("Z", 1))
        self._z = cfg.nd2.z
        if isinstance(self._z, int) and self._z >= self._n_z:
            raise ValueError(f"{self.path.name}: nd2.z={self._z} but file has {self._n_z} z-planes")

        self.n_frames = int(sizes.get("T", 1))

        # Axes that index a *sequence* entry (everything except channel and YX);
        # nd2's read_frame() takes a flat index over exactly these, in file order.
        self._seq_axes = [a for a in self._axes if a not in ("C", "Y", "X", "S")]
        self._seq_shape = [int(sizes[a]) for a in self._seq_axes]

        fps = cfg.nd2.fps
        if isinstance(fps, str) and fps == "auto":
            detected = _nd2_fps(f)
            if detected is None:
                warnings.warn(
                    f"{self.path.name}: could not read a frame rate from the ND2 metadata; "
                    "falling back to 100 fps. Set nd2.fps explicitly to silence this.",
                    stacklevel=2,
                )
            self.fps = float(detected or 100.0)
        else:
            self.fps = float(fps)

        try:
            self.pixel_size_um = float(f.voxel_size().x)
        except Exception:
            self.pixel_size_um = None

        self._dask = None  # lazily built fallback

    # -- indexing ---------------------------------------------------------- #
    def _seq_index(self, t: int, z: int) -> int:
        coords = []
        for axis in self._seq_axes:
            if axis == "T":
                coords.append(t)
            elif axis == "Z":
                coords.append(z)
            elif axis == "P":
                coords.append(self.series)
            else:
                coords.append(0)
        return int(np.ravel_multi_index(tuple(coords), tuple(self._seq_shape)))

    def _read_plane(self, t: int, z: int) -> np.ndarray:
        try:
            raw = np.asarray(self._f.read_frame(self._seq_index(t, z)))
        except Exception:
            raw = self._read_plane_dask(t, z)
        # read_frame returns (C, Y, X) when the file is multi-channel, else (Y, X);
        # RGB files come back as (Y, X, S).
        if raw.ndim == 3:
            if raw.shape[-1] in (3, 4) and "S" in self._axes:
                raw = raw[..., min(self.channel, raw.shape[-1] - 1)]
            else:
                raw = raw[self.channel]
        return raw

    def _read_plane_dask(self, t: int, z: int) -> np.ndarray:
        if self._dask is None:
            self._dask = self._f.to_dask()
        idx: list = []
        for axis in self._axes:
            if axis == "T":
                idx.append(t)
            elif axis == "Z":
                idx.append(z)
            elif axis == "P":
                idx.append(self.series)
            elif axis == "C":
                idx.append(self.channel)
            elif axis in ("Y", "X"):
                idx.append(slice(None))
            else:
                idx.append(0)
        return np.asarray(self._dask[tuple(idx)])

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n_frames:
            raise IndexError(f"frame {index} out of range (0..{self.n_frames - 1})")
        if isinstance(self._z, int):
            return self._read_plane(index, self._z)
        stack = np.stack([self._read_plane(index, z) for z in range(self._n_z)], axis=0)
        return {"max": stack.max, "mean": stack.mean, "min": stack.min}[self._z](axis=0)

    def close(self) -> None:
        self._dask = None
        if self._own_file:
            try:
                self._f.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _nd2_positions(path: Path, cfg: Config) -> list[int]:
    import nd2

    with nd2.ND2File(str(path)) as f:
        n_pos = int(dict(f.sizes).get("P", 1))
    wanted = cfg.nd2.positions
    if isinstance(wanted, str):
        if wanted != "all":
            raise ValueError("nd2.positions must be 'all' or a list of integers")
        return list(range(n_pos))
    out = [int(p) for p in wanted]
    bad = [p for p in out if p >= n_pos]
    if bad:
        raise ValueError(f"{path.name}: nd2.positions {bad} out of range (file has {n_pos})")
    return out


def discover_movies(cfg: Config, only: list[str] | None = None) -> list[MovieSource]:
    """Open every movie under ``paths.video_dir``. ND2 positions become separate movies."""
    video_dir = cfg.paths.video_dir
    if not video_dir.exists():
        raise FileNotFoundError(f"paths.video_dir does not exist: {video_dir}")

    files: list[Path] = []
    for pattern in cfg.paths.patterns:
        files.extend(sorted(video_dir.glob(pattern)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no files matching {cfg.paths.patterns} in {video_dir}")

    sources: list[MovieSource] = []
    for path in files:
        try:
            if path.suffix.lower() == ".nd2":
                for pos in _nd2_positions(path, cfg):
                    sources.append(ND2Movie(path, cfg, position=pos))
            else:
                sources.append(TiffMovie(path, cfg))
        except Exception as exc:
            warnings.warn(f"skipping {path.name}: {exc}", stacklevel=2)

    if only:
        wanted = set(only)
        selected = [s for s in sources if s.id in wanted]
        missing = wanted - {s.id for s in selected}
        for s in sources:
            if s.id not in wanted:
                s.close()
        if missing:
            available = ", ".join(sorted(s.id for s in sources))
            raise ValueError(f"unknown movie id(s): {sorted(missing)}. Available: {available}")
        return selected
    return sources


def open_movie(cfg: Config, movie_id: str) -> MovieSource:
    """Reopen a single movie by its id (used by the GUI minimap and the renderer)."""
    sources = discover_movies(cfg, only=[movie_id])
    return sources[0]
