"""Typed configuration loaded from ``config.yaml``.

Plain dataclasses rather than pydantic: the schema is small, and validation
errors here should read like advice, not like a stack trace.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["Config", "load_config", "CategoryGroup", "CategoryOption", "RESERVED_GROUP_NAMES"]

# A category group becomes a CSV column, so a group named like a built-in column
# silently overwrites it. Kept here rather than imported from store.py, which
# imports this module.
RESERVED_GROUP_NAMES = frozenset({
    "clip_id", "movie_id", "source_file", "series", "channel",
    "track_id", "f_start", "f_end",
    "crop_x0", "crop_y0", "crop_w", "crop_h", "scale",
    "uncertain", "notes", "annotator", "timestamp_utc", "dwell_s",
    "suggested", "suggest_conf", "suggest_shown",
})


class ConfigError(ValueError):
    """Raised for a malformed config.yaml, with a message aimed at a human."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _known(section: str, given: dict, allowed: Sequence[str]) -> None:
    unknown = set(given) - set(allowed)
    if unknown:
        warnings.warn(
            f"ignoring unknown key(s) in config section '{section}': {sorted(unknown)}",
            stacklevel=3,
        )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """'#4C9F70' -> (76, 159, 112). Everything downstream works in RGB."""
    s = color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ConfigError(f"colour {color!r} is not a #rrggbb hex string")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    video_dir: Path
    model: Path
    work_dir: Path
    csv: Path
    patterns: list[str] = field(default_factory=lambda: ["*.tif", "*.tiff", "*.nd2"])

    @classmethod
    def parse(cls, d: dict, root: Path) -> "Paths":
        _known("paths", d, ["video_dir", "model", "work_dir", "csv", "patterns"])
        return cls(
            video_dir=_resolve(root, d.get("video_dir", "./videos")),
            model=_resolve(root, d.get("model", "./best.pt")),
            work_dir=_resolve(root, d.get("work_dir", "./work")),
            csv=_resolve(root, d.get("csv", "./annotations.csv")),
            patterns=list(d.get("patterns", ["*.tif", "*.tiff", "*.nd2"])),
        )


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(str(value).replace("\\", "/"))
    return p if p.is_absolute() or str(p).startswith("//") else (root / p).resolve()


@dataclass
class ND2Options:
    channel: int | str = 0
    z: int | str = 0
    positions: str | list[int] = "all"
    fps: float | str = "auto"

    @classmethod
    def parse(cls, d: dict) -> "ND2Options":
        _known("nd2", d, ["channel", "z", "positions", "fps"])
        z = d.get("z", 0)
        if isinstance(z, str) and z not in ("max", "mean", "min"):
            raise ConfigError("nd2.z must be an integer index or one of 'max', 'mean', 'min'")
        return cls(
            channel=d.get("channel", 0),
            z=z,
            positions=d.get("positions", "all"),
            fps=d.get("fps", "auto"),
        )


@dataclass
class Normalize:
    """How raw frames become uint8.

    ``smooth`` tracks slow illumination/focus drift without the frame-to-frame
    breathing of per-frame scaling: percentiles sampled on a subset of frames,
    median-filtered over time, then interpolated to every frame.
    """

    mode: str = "per_frame"
    percentiles: tuple[float, float] = (0.5, 99.5)
    sample_every: int | str = "auto"
    smooth_window: int = 9
    subsample: int = 4

    @classmethod
    def parse(cls, d: dict) -> "Normalize":
        _known("normalize", d,
               ["mode", "percentiles", "sample_every", "smooth_window", "subsample"])
        mode = str(d.get("mode", "per_frame"))
        if mode not in ("smooth", "fixed", "per_frame", "none"):
            raise ConfigError(
                "normalize.mode must be 'per_frame' (default), 'smooth' (slow-drift only), "
                "'fixed' (one mapping for the whole movie - faithful to real intensity "
                "changes, including flicker) or 'none'"
            )
        pcts = d.get("percentiles", [0.5, 99.5])
        if len(pcts) != 2 or not 0 <= float(pcts[0]) < float(pcts[1]) <= 100:
            raise ConfigError("normalize.percentiles must be [low, high] within 0..100")
        return cls(
            mode=mode,
            percentiles=(float(pcts[0]), float(pcts[1])),
            sample_every=d.get("sample_every", "auto"),
            smooth_window=max(1, int(d.get("smooth_window", 9))),
            subsample=max(1, int(d.get("subsample", 4))),
        )


@dataclass
class TiffOptions:
    fps: float = 100.0
    channel: int = 0

    @classmethod
    def parse(cls, d: dict) -> "TiffOptions":
        _known("tiff", d, ["fps", "channel"])
        return cls(fps=float(d.get("fps", 100.0)), channel=int(d.get("channel", 0)))


@dataclass
class Detection:
    task: str = "pose"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    max_det: int = 300
    device: str = "auto"
    batch: int = 8
    half: bool = False

    @classmethod
    def parse(cls, d: dict) -> "Detection":
        _known("detection", d, ["task", "conf", "iou", "imgsz", "max_det", "device", "batch", "half"])
        task = str(d.get("task", "pose"))
        if task not in ("pose", "detect"):
            raise ConfigError("detection.task must be 'pose' or 'detect'")
        return cls(
            task=task,
            conf=float(d.get("conf", 0.25)),
            iou=float(d.get("iou", 0.5)),
            imgsz=int(d.get("imgsz", 640)),
            max_det=int(d.get("max_det", 300)),
            device=str(d.get("device", "auto")),
            batch=max(1, int(d.get("batch", 8))),
            half=bool(d.get("half", False)),
        )


@dataclass
class Tiling:
    enabled: str | bool = "auto"
    tile: int = 640
    overlap: float = 0.25
    merge_iou: float = 0.6
    drop_edge_truncated: bool = True
    edge_margin_px: int = 2

    @classmethod
    def parse(cls, d: dict) -> "Tiling":
        _known(
            "tiling", d,
            ["enabled", "tile", "overlap", "merge_iou", "drop_edge_truncated", "edge_margin_px"],
        )
        overlap = float(d.get("overlap", 0.25))
        if not 0.0 <= overlap < 0.9:
            raise ConfigError("tiling.overlap must be in [0, 0.9)")
        return cls(
            enabled=d.get("enabled", "auto"),
            tile=int(d.get("tile", 640)),
            overlap=overlap,
            merge_iou=float(d.get("merge_iou", 0.6)),
            drop_edge_truncated=bool(d.get("drop_edge_truncated", True)),
            edge_margin_px=int(d.get("edge_margin_px", 2)),
        )


@dataclass
class Tracking:
    backend: str = "tracktrack"
    tracker_cfg: str = ""            # "" -> the backend's bundled yaml
    min_track_length: int = 50
    greedy_max_dist_px: float = 25.0
    overrides: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, d: dict) -> "Tracking":
        from .tracking import TRACKERS

        _known("tracking", d,
               ["backend", "tracker_cfg", "min_track_length", "greedy_max_dist_px", "overrides"])
        backend = str(d.get("backend", "tracktrack"))
        allowed = sorted(TRACKERS) + ["greedy"]
        if backend not in allowed:
            raise ConfigError(f"tracking.backend must be one of {allowed}")
        overrides = d.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ConfigError("tracking.overrides must be a mapping of tracker parameters")
        return cls(
            backend=backend,
            tracker_cfg=str(d.get("tracker_cfg", "") or ""),
            min_track_length=int(d.get("min_track_length", 50)),
            greedy_max_dist_px=float(d.get("greedy_max_dist_px", 25.0)),
            overrides=dict(overrides),
        )


@dataclass
class Features:
    """Per-detection CNN features, pooled from the same forward pass."""

    enabled: bool = True
    levels: list[str] = field(default_factory=lambda: ["P3", "P4", "P5"])
    pool: int = 1
    dtype: str = "float16"

    @classmethod
    def parse(cls, d: dict) -> "Features":
        _known("features", d, ["enabled", "levels", "pool", "dtype"])
        levels = [str(l).upper() for l in d.get("levels", ["P3", "P4", "P5"])]
        bad = [l for l in levels if l not in ("P3", "P4", "P5")]
        if bad:
            raise ConfigError(f"features.levels entries must be P3/P4/P5, got {bad}")
        dtype = str(d.get("dtype", "float16"))
        if dtype not in ("float16", "float32"):
            raise ConfigError("features.dtype must be float16 or float32")
        return cls(
            enabled=bool(d.get("enabled", True)),
            levels=levels,
            pool=max(1, int(d.get("pool", 1))),
            dtype=dtype,
        )


@dataclass
class Suggest:
    enabled: bool = True
    model: str = "catboost"
    pca_components: int = 32
    retrain_every: int = 10
    min_per_class: int = 5
    min_macro_f1: float = 0.45
    audit_fraction: float = 0.10
    audit_visible: bool = False
    seed: int = 0
    audit_seed: int = 12345
    min_confidence: float = 0.0
    # Hyperparameters, shared by both backends so switching model does not
    # silently change the fit as well.
    iterations: int = 400
    depth: int = 5
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    balance_classes: bool = True
    # Which feature families the model may use: the CNN columns re-identify
    # individuals almost perfectly, so a project that wants behaviour rather
    # than re-identification may drop them.
    feature_groups: list[str] = field(
        default_factory=lambda: ["kinematic", "pose", "waveform", "cnn"])

    @classmethod
    def parse(cls, d: dict) -> "Suggest":
        _known(
            "suggest", d,
            ["enabled", "model", "pca_components", "retrain_every", "min_per_class",
             "min_macro_f1", "audit_fraction", "audit_visible", "seed", "audit_seed",
             "min_confidence",
             "iterations", "depth", "learning_rate", "l2_leaf_reg", "balance_classes",
             "feature_groups"],
        )
        model = str(d.get("model", "catboost"))
        if model not in ("hist", "catboost"):
            raise ConfigError("suggest.model must be 'hist' or 'catboost'")
        depth = int(d.get("depth", 5))
        if not 1 <= depth <= 10:
            raise ConfigError(
                "suggest.depth must be 1-10. CatBoost trees are symmetric, so depth "
                "alone sets the width (2**depth leaves) - past 6 it overfits a "
                "hand-labelled set fast."
            )
        from .clipfeatures import FEATURE_GROUPS

        groups = d.get("feature_groups")
        groups = list(FEATURE_GROUPS) if groups is None else [str(g) for g in groups]
        unknown = [g for g in groups if g not in FEATURE_GROUPS]
        if unknown:
            raise ConfigError(
                f"suggest.feature_groups has unknown entries {unknown}; "
                f"choose from {sorted(FEATURE_GROUPS)}"
            )
        if not groups:
            raise ConfigError(
                "suggest.feature_groups is empty, so the model would have no "
                f"inputs at all; choose at least one of {sorted(FEATURE_GROUPS)}"
            )
        audit = float(d.get("audit_fraction", 0.10))
        if not 0.0 <= audit < 1.0:
            raise ConfigError("suggest.audit_fraction must be in [0, 1)")
        return cls(
            enabled=bool(d.get("enabled", True)),
            model=model,
            pca_components=max(2, int(d.get("pca_components", 32))),
            retrain_every=max(1, int(d.get("retrain_every", 10))),
            min_per_class=max(1, int(d.get("min_per_class", 5))),
            min_macro_f1=float(d.get("min_macro_f1", 0.45)),
            audit_fraction=audit,
            audit_visible=bool(d.get("audit_visible", False)),
            seed=int(d.get("seed", 0)),
            # Deliberately independent of `seed`: changing the CV seed must not
            # silently redraw the audit set and invalidate it retroactively.
            audit_seed=int(d.get("audit_seed", 12345)),
            min_confidence=float(d.get("min_confidence", 0.0)),
            iterations=max(10, int(d.get("iterations", 400))),
            depth=depth,
            learning_rate=float(d.get("learning_rate", 0.05)),
            l2_leaf_reg=float(d.get("l2_leaf_reg", 3.0)),
            balance_classes=bool(d.get("balance_classes", True)),
            feature_groups=groups,
        )


@dataclass
class Spectral:
    """The window a BEAT is measured over, which is not the window it is labelled over.

    A clip's length is chosen so a person can judge one behaviour, not for a
    Fourier transform: at 25 frames and 100 fps one FFT bin is 4 Hz and the
    resolvability floor lands at 12 Hz, above most of these cells' beats, so the
    window is longer than the clip and CENTRED on it. Against 170 hand labels:

        frames   direction accuracy   clips with a measurable frequency
            25         0.853                       78 / 170
            50         0.865                       95 / 170
            75         0.853                      112 / 170
           100         0.806                      122 / 170
           150         0.702                      104 / 170

    Past 75 accuracy falls because behaviour changes inside a track (70% of
    multi-clip tracks reverse at least once) and a long window averages over the
    reversal. Raise it if your cells hold a direction longer, lower it if they
    switch faster than ~0.75 s.
    """

    window: int = 75
    # Frames between successive spectral estimates. The series is computed once
    # per track, so a small hop is cheap: 5 is ~4x the work of 25 and localises a
    # reversal to within one hop.
    hop: int = 5

    @classmethod
    def parse(cls, d: dict) -> "Spectral":
        _known("spectral", d, ["window", "hop"])
        window = int(d.get("window", 75))
        hop = int(d.get("hop", 5))
        if window < 8:
            raise ConfigError(
                "spectral.window must be at least 8 frames; below that there is no "
                "band left to search once the resolvability floor is applied"
            )
        if not 1 <= hop <= window:
            raise ConfigError("spectral.hop must be between 1 and spectral.window")
        return cls(window=window, hop=hop)


@dataclass
class Predict:
    """How much of each movie the bulk labelling pass covers.

    ``manifest`` scores exactly the clips cut for annotation, a sampling chosen
    for a human's attention, so everything between two annotation windows comes
    out unlabelled and unpainted in the rendered movie. ``dense`` lays its own
    grid over every frame of every track - see :mod:`trackanno.densepredict`.
    """

    coverage: str = "dense"
    window_frames: int | str = "auto"
    # Detections a dense window needs before its features mean anything. Two is
    # the floor at which ANY motion feature can exist. Deliberately NOT
    # clips.min_clip_frames, which exists so an annotator gets a watchable video
    # and would punch the holes dense coverage removes back into the export.
    # Every row carries n_present and present_frac for a stricter downstream bar.
    min_frames: int = 2

    @classmethod
    def parse(cls, d: dict) -> "Predict":
        _known("predict", d, ["coverage", "window_frames", "min_frames"])
        coverage = str(d.get("coverage", "dense"))
        if coverage not in ("dense", "manifest"):
            raise ConfigError(
                "predict.coverage must be 'dense' (every frame of every track) or "
                "'manifest' (only the clips that were cut for annotation)"
            )
        window = d.get("window_frames", "auto")
        if isinstance(window, str):
            if window != "auto":
                raise ConfigError(
                    "predict.window_frames must be a frame count or 'auto' "
                    "(min of clips.frames and clips.stride)"
                )
        elif int(window) < 1:
            raise ConfigError(
                f"predict.window_frames must be at least 1 (got {int(window)}); it is "
                "both the window length and the stride, so 0 never terminates"
            )
        min_frames = int(d.get("min_frames", 2))
        if min_frames < 1:
            raise ConfigError(
                f"predict.min_frames must be at least 1 (got {min_frames}); a window "
                "with no detections in it has nothing to measure"
            )
        return cls(
            coverage=coverage,
            window_frames=window if isinstance(window, str) else int(window),
            min_frames=min_frames,
        )


@dataclass
class Keypoints:
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def edge_idx(self) -> list[tuple[int, int]]:
        idx = {n: i for i, n in enumerate(self.nodes)}
        return [(idx[a], idx[b]) for a, b in self.edges]

    @classmethod
    def parse(cls, d: dict) -> "Keypoints":
        _known("keypoints", d, ["nodes", "edges"])
        nodes = [str(n) for n in d.get("nodes", [])]
        edges: list[tuple[str, str]] = []
        for e in d.get("edges", []):
            if len(e) != 2:
                raise ConfigError(f"keypoints.edges entry {e!r} must have exactly two node names")
            a, b = str(e[0]), str(e[1])
            for name in (a, b):
                if name not in nodes:
                    raise ConfigError(f"keypoints.edges references unknown node {name!r}")
            edges.append((a, b))
        return cls(nodes=nodes, edges=edges)


@dataclass
class Clips:
    frames: int = 50
    stride: int = 50
    window_mode: str = "all"
    follow: str = "smooth"
    crop_from: str = "bbox"
    smooth_window: int = 25
    pad: float = 2.0
    crop_size: int | str = "auto"
    max_size: int = 256
    min_clip_frames: int = 40
    max_clips_per_track: int = 0
    context_tracks: bool = True
    thumbnails: bool = False
    thumbnail_size: int = 256

    @classmethod
    def parse(cls, d: dict) -> "Clips":
        _known(
            "clips", d,
            ["frames", "stride", "window_mode", "follow", "crop_from", "smooth_window", "pad",
             "crop_size", "max_size", "min_clip_frames", "max_clips_per_track", "context_tracks",
             "thumbnails", "thumbnail_size"],
        )
        follow = str(d.get("follow", "smooth"))
        if follow not in ("smooth", "fixed", "track"):
            raise ConfigError(
                "clips.follow must be 'smooth' (crop follows a smoothed trajectory - small crop, "
                "local motion still visible), 'fixed' (one crop for the window - shows all motion "
                "but needs a big crop) or 'track' (re-centred every frame - smallest crop, but "
                "translation becomes invisible)"
            )
        crop_from = str(d.get("crop_from", "bbox"))
        if crop_from not in ("bbox", "union"):
            raise ConfigError("clips.crop_from must be 'bbox' or 'union'")
        mode = str(d.get("window_mode", "all"))
        if mode not in ("first", "first_per_track", "all"):
            raise ConfigError(
                "clips.window_mode must be 'first' (only the movie's first window), "
                "'first_per_track' (first window of each track) or 'all' (every window)"
            )
        frames = int(d.get("frames", 50))
        min_present = int(d.get("min_clip_frames", 40))
        if min_present > frames:
            raise ConfigError(
                f"clips.min_clip_frames ({min_present}) cannot exceed clips.frames ({frames}); "
                "no window could ever qualify"
            )
        stride = int(d.get("stride", frames))
        if stride < 1:
            raise ConfigError(
                f"clips.stride must be at least 1 (got {stride}); it is the step "
                "the window grid advances by, so 0 or less never terminates"
            )
        return cls(
            frames=frames,
            stride=stride,
            window_mode=mode,
            follow=follow,
            crop_from=crop_from,
            smooth_window=max(1, int(d.get("smooth_window", 25))),
            pad=float(d.get("pad", 2.0)),
            crop_size=d.get("crop_size", "auto"),
            max_size=int(d.get("max_size", 256)),
            min_clip_frames=min_present,
            max_clips_per_track=int(d.get("max_clips_per_track", 0)),
            context_tracks=bool(d.get("context_tracks", True)),
            thumbnails=bool(d.get("thumbnails", False)),
            thumbnail_size=int(d.get("thumbnail_size", 256)),
        )


@dataclass
class CategoryOption:
    name: str
    key: str
    color: str

    @property
    def rgb(self) -> tuple[int, int, int]:
        return _hex_to_rgb(self.color)


@dataclass
class CategoryGroup:
    name: str
    type: str
    required: bool
    options: list[CategoryOption]

    @property
    def is_multi(self) -> bool:
        return self.type == "multi"

    def option(self, name: str) -> CategoryOption | None:
        return next((o for o in self.options if o.name == name), None)


def _parse_categories(raw: Any) -> list[CategoryGroup]:
    if not raw:
        raise ConfigError("config must define at least one entry under 'categories'")
    groups: list[CategoryGroup] = []
    seen_keys: dict[str, str] = {}
    seen_names: set[str] = set()
    for g in raw:
        name = str(g.get("name", "")).strip()
        if not name:
            raise ConfigError("every category group needs a 'name'")
        if name in RESERVED_GROUP_NAMES:
            raise ConfigError(
                f"category group {name!r} collides with a built-in column of the "
                f"annotation CSV, which would overwrite it. Reserved names: "
                f"{', '.join(sorted(RESERVED_GROUP_NAMES))}"
            )
        if name in seen_names:
            raise ConfigError(
                f"two category groups are both named {name!r}; they would share "
                "one CSV column"
            )
        seen_names.add(name)
        _known(f"categories[{g.get('name')}]", g, ["name", "type", "required", "options"])
        gtype = str(g.get("type", "single"))
        if gtype not in ("single", "multi"):
            raise ConfigError(f"category group {g.get('name')!r}: type must be 'single' or 'multi'")
        options: list[CategoryOption] = []
        for o in g.get("options", []):
            _known("category option", o, ["name", "key", "color"])
            key = str(o.get("key", ""))
            if not key:
                raise ConfigError(f"category option {o.get('name')!r} needs a 'key' shortcut")
            label = f"{g.get('name')}.{o.get('name')}"
            if key.lower() in seen_keys:
                raise ConfigError(
                    f"keyboard shortcut {key!r} is used by both {seen_keys[key.lower()]} "
                    f"and {label} - shortcuts must be unique"
                )
            seen_keys[key.lower()] = label
            color = str(o.get("color", "#CCCCCC"))
            _hex_to_rgb(color)  # validate early
            options.append(CategoryOption(name=str(o["name"]), key=key, color=color))
        if not options:
            raise ConfigError(f"category group {g.get('name')!r} has no options")
        groups.append(
            CategoryGroup(
                name=str(g["name"]),
                type=gtype,
                required=bool(g.get("required", False)),
                options=options,
            )
        )
    return groups


def _one_of(value, allowed, where: str) -> str:
    """A choice setting, refused loudly rather than silently falling back.

    A typo that quietly reverts to the default looks applied and is not.
    """
    text = str(value)
    if text not in allowed:
        raise ConfigError(f"{where} must be one of {list(allowed)}, got {text!r}")
    return text


@dataclass
class UI:
    theme: str = "dark"
    fps: int = 30
    autoplay: bool = True
    loop: bool = True
    auto_advance: bool = True
    show_overlay: bool = True
    show_skeleton: bool = False
    show_context_tracks: bool = False
    show_minimap: bool = False
    show_kymograph: bool = True
    kymo_interpolate: bool = True
    kymo_smooth_px: float = 0.6
    kymo_field: str = "curvature"
    kymo_sign: str = "canonical"
    follow: str = "centred"
    view_zoom: float = 1.0
    smooth_window: int = 15
    prefetch: int = 8
    resume: str = "first_unannotated"
    annotator: str = ""
    shuffle: bool = True
    shuffle_seed: int = 0

    @classmethod
    def parse(cls, d: dict) -> "UI":
        _known(
            "ui", d,
            ["theme", "fps", "autoplay", "loop", "auto_advance", "show_overlay",
             "show_skeleton",
             "show_context_tracks", "show_minimap", "show_kymograph",
             "kymo_interpolate", "kymo_smooth_px", "kymo_field", "kymo_sign",
             "follow",
             "view_zoom", "smooth_window",
             "prefetch", "resume", "annotator", "shuffle", "shuffle_seed"],
        )
        resume = str(d.get("resume", "first_unannotated"))
        if resume not in ("first_unannotated", "last_position"):
            raise ConfigError("ui.resume must be 'first_unannotated' or 'last_position'")
        theme = str(d.get("theme", "dark"))
        if theme not in ("dark", "light", "system"):
            raise ConfigError("ui.theme must be 'dark', 'light' or 'system'")

        follow = str(d.get("follow", "centred"))
        if follow not in ("centred", "smooth", "locked"):
            raise ConfigError(
                "ui.follow must be 'centred' (one view on the track's median position - all "
                "motion visible), 'smooth' (view follows a moving average - drift removed, "
                "beat kept) or 'locked' (re-centred every frame - translation invisible)"
            )
        return cls(
            theme=theme,
            fps=int(d.get("fps", 30)),
            autoplay=bool(d.get("autoplay", True)),
            loop=bool(d.get("loop", True)),
            auto_advance=bool(d.get("auto_advance", True)),
            show_overlay=bool(d.get("show_overlay", True)),
            show_skeleton=bool(d.get("show_skeleton", False)),
            show_context_tracks=bool(d.get("show_context_tracks", False)),
            show_minimap=bool(d.get("show_minimap", False)),
            show_kymograph=bool(d.get("show_kymograph", True)),
            kymo_interpolate=bool(d.get("kymo_interpolate", True)),
            kymo_smooth_px=max(0.0, float(d.get("kymo_smooth_px", 0.6))),
            kymo_field=_one_of(d.get("kymo_field", "curvature"),
                               ("curvature", "tangent"), "ui.kymo_field"),
            kymo_sign=_one_of(d.get("kymo_sign", "canonical"),
                              ("raw", "canonical", "absolute"), "ui.kymo_sign"),
            follow=follow,
            view_zoom=max(1.0, float(d.get("view_zoom", 1.0))),
            smooth_window=max(1, int(d.get("smooth_window", 15))),
            prefetch=max(0, int(d.get("prefetch", 8))),
            resume=resume,
            annotator=str(d.get("annotator", "")),
            shuffle=bool(d.get("shuffle", True)),
            shuffle_seed=int(d.get("shuffle_seed", 0)),
        )


_DEFAULT_KEYS = {
    "next": ["Right", "Return"],
    "prev": ["Left"],
    "next_unannotated": ["Tab"],
    "play_pause": ["Space"],
    "overlay": ["o"],
    "stabilise": ["s"],
    "context": ["c"],
    "zoom_in": ["+", "="],
    "zoom_out": ["-"],
    "zoom_reset": ["0"],
    "minimap": ["m"],
    "uncertain": ["u"],
    "clear": ["Backspace"],
    "undo": ["Ctrl+Z"],
    "save": ["Ctrl+S"],
    "first_unannotated": ["Ctrl+U"],
    "notes": ["n"],
}


@dataclass
class Render:
    fps: float | str = "auto"
    out_dir: Path = Path("./renders")
    thickness: int = 2
    font_scale: float = 0.4
    show_track_id: bool = True
    show_category: bool = True
    show_skeleton: bool = False
    hide_unlabelled: bool = False
    unlabelled_color: str = "#555555"
    codec: str = "libx264"
    quality: int = 8

    @classmethod
    def parse(cls, d: dict, root: Path) -> "Render":
        _known(
            "render", d,
            ["fps", "out_dir", "thickness", "font_scale", "show_track_id", "show_category",
             "show_skeleton", "hide_unlabelled", "unlabelled_color", "codec", "quality"],
        )
        return cls(
            fps=d.get("fps", "auto"),
            out_dir=_resolve(root, d.get("out_dir", "./renders")),
            thickness=int(d.get("thickness", 2)),
            font_scale=float(d.get("font_scale", 0.4)),
            show_track_id=bool(d.get("show_track_id", True)),
            show_category=bool(d.get("show_category", True)),
            show_skeleton=bool(d.get("show_skeleton", False)),
            hide_unlabelled=bool(d.get("hide_unlabelled", False)),
            unlabelled_color=str(d.get("unlabelled_color", "#555555")),
            codec=str(d.get("codec", "libx264")),
            quality=int(d.get("quality", 8)),
        )


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    path: Path
    paths: Paths
    nd2: ND2Options
    tiff: TiffOptions
    normalize: Normalize
    detection: Detection
    tiling: Tiling
    tracking: Tracking
    keypoints: Keypoints
    features: Features
    spectral: Spectral
    suggest: Suggest
    predict: Predict
    clips: Clips
    categories: list[CategoryGroup]
    ui: UI
    keys: dict[str, list[str]]
    render: Render

    # -- convenience ------------------------------------------------------- #
    @property
    def is_pose(self) -> bool:
        return self.detection.task == "pose"

    @property
    def n_keypoints(self) -> int:
        return self.keypoints.n if self.is_pose else 0

    @property
    def needs_keypoints(self) -> bool:
        """``task: pose`` but nothing says how many keypoints the model has.

        Resolvable, not fatal: the count lives in the checkpoint's ``kpt_shape``;
        the names do not, so they are generated and can be renamed afterwards.
        """
        return self.is_pose and not self.keypoints.nodes

    def movie_dir(self, movie_id: str) -> Path:
        return self.paths.work_dir / movie_id

    def group(self, name: str) -> CategoryGroup | None:
        return next((g for g in self.categories if g.name == name), None)

    def color_for(self, group: str, value: str) -> str | None:
        g = self.group(group)
        if g is None:
            return None
        opt = g.option(value)
        return opt.color if opt else None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate config.yaml. Relative paths resolve against its folder."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")

    root = path.parent
    keys = dict(_DEFAULT_KEYS)
    for action, binding in _section(data, "keys").items():
        if action not in _DEFAULT_KEYS:
            warnings.warn(f"ignoring unknown key binding action 'keys.{action}'", stacklevel=2)
            continue
        keys[action] = [binding] if isinstance(binding, str) else [str(b) for b in binding]

    cfg = Config(
        path=path,
        paths=Paths.parse(_section(data, "paths"), root),
        nd2=ND2Options.parse(_section(data, "nd2")),
        tiff=TiffOptions.parse(_section(data, "tiff")),
        normalize=Normalize.parse(_section(data, "normalize")),
        detection=Detection.parse(_section(data, "detection")),
        tiling=Tiling.parse(_section(data, "tiling")),
        tracking=Tracking.parse(_section(data, "tracking")),
        keypoints=Keypoints.parse(_section(data, "keypoints")),
        features=Features.parse(_section(data, "features")),
        spectral=Spectral.parse(_section(data, "spectral")),
        suggest=Suggest.parse(_section(data, "suggest")),
        predict=Predict.parse(_section(data, "predict")),
        clips=Clips.parse(_section(data, "clips")),
        categories=_parse_categories(data.get("categories")),
        ui=UI.parse(_section(data, "ui")),
        keys=keys,
        render=Render.parse(_section(data, "render"), root),
    )

    # `task: pose` with no keypoints listed is INCOMPLETE, not invalid, and must
    # not raise here: the count is readable from the checkpoint, and refusing to
    # load would make switching the task to 'pose' from the GUI impossible.
    # `Config.needs_keypoints` carries the state and TiledDetector fills it in.

    # Category keys must not collide with the navigation shortcuts.
    nav = {k.lower() for binds in keys.values() for k in binds}
    for g in cfg.categories:
        for o in g.options:
            if o.key.lower() in nav:
                raise ConfigError(
                    f"category shortcut {o.key!r} ({g.name}.{o.name}) collides with a "
                    f"navigation shortcut in the 'keys' section"
                )
    return cfg
