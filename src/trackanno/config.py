"""Typed configuration loaded from ``config.yaml``.

One dataclass per section. Each field carries its default, a one-line help
string and its allowed values; :func:`_parse_section` reads any section from
that metadata. Hand-written code is kept for rules that span fields.
"""

from __future__ import annotations

import copy
import hashlib
import re
import warnings
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclasses_fields
from pathlib import Path
from typing import Any, Callable, ClassVar

import yaml

__all__ = ["Config", "ConfigError", "load_config", "CategoryGroup", "CategoryOption",
           "ModelSource", "RESERVED_GROUP_NAMES", "FIXED_SETTINGS", "VIEW_SETTINGS",
           "SECTIONS", "section_defaults", "effective_values", "schema_keys", "help_for",
           "meta_for"]

# A category group becomes a CSV column, so a group named like a built-in column
# would overwrite it. Mirrors ``store.IDENTITY_COLUMNS`` and
# ``store.TRAILING_COLUMNS`` (not imported, to avoid a cycle);
# ``tests/test_store_provenance.py`` asserts they are equal.
RESERVED_GROUP_NAMES = frozenset({
    "clip_id", "movie_id", "source_file", "series", "channel",
    "track_id", "f_start", "f_end",
    "crop_x0", "crop_y0", "crop_w", "crop_h", "scale",
    "uncertain", "notes", "annotator", "timestamp_utc", "dwell_s",
    "suggested", "suggest_conf", "suggest_shown",
    "selected_by", "audit_holdout",
})


class ConfigError(ValueError):
    """A malformed config.yaml, with a one-line message."""


# --------------------------------------------------------------------------- #
# migration tables
# --------------------------------------------------------------------------- #
# Settings that are no longer configurable, with the value now used. Modules
# read their constants from here, so the warning and the code cannot disagree.
FIXED_SETTINGS: dict[str, Any] = {
    "normalize.subsample": 4,
    "detection.iou": 0.5,
    "detection.max_det": 300,
    "tiling.drop_edge_truncated": True,
    "tiling.edge_margin_px": 2,
    "tracking.greedy_max_dist_px": 25.0,
    "tracking.tracker_cfg": "",
    "features.levels": ["P3", "P4", "P5"],
    "features.pool": 1,
    "features.dtype": "float16",
    "spectral.hop": 5,
    "spectral.flow_frames": 5,
    "suggest.pca_components": 32,
    "suggest.min_macro_f1": 0.45,
    "suggest.seed": 0,
    "suggest.audit_seed": 12345,
    "suggest.depth": 6,
    "suggest.balance_classes": True,
    "suggest.select_batch_size": 20,
    "suggest.audit_visible": False,
    "clips.crop_from": "bbox",
    "clips.smooth_window": 25,
    "clips.max_size": 256,
    "clips.context_tracks": True,
    "clips.max_clips_per_track": 0,
    "ui.autoplay": True,
    "ui.loop": True,
    "ui.prefetch": 8,
    "ui.resume": "first_unannotated",
    "ui.shuffle_seed": 0,
    "ui.show_kymograph": True,
    "ui.kymo_smooth_px": 0.6,
    "render.thickness": 2,
    "render.font_scale": 0.4,
    "render.show_track_id": True,
    "render.show_category": True,
    "render.unlabelled_color": "#555555",
    "render.codec": "libx264",
    "render.quality": 8,
}

# Extra advice for a fixed setting whose old value had a replacement.
_FIXED_HINTS = {
    "tracking.tracker_cfg": "set tracker parameters under tracking.overrides",
}

# Display toggles that moved to per-user settings in the annotation window,
# with the default each starts at.
VIEW_SETTINGS: dict[str, Any] = {
    "ui.show_overlay": True,
    "ui.show_skeleton": False,
    "ui.show_context_tracks": False,
    "ui.follow": "centred",
    "ui.view_zoom": 1.0,
}

# Keys removed outright, with the one-line reason.
_REMOVED_KEYS = {
    "predict.decode": "Viterbi decoding was removed; predict.min_bout_frames is the hard floor",
    "predict.self_transition": "Viterbi decoding was removed; predict.min_bout_frames is the hard floor",
}

# Removed values of settings that remain, and what an old config now gets.
REMOVED_VALUES: dict[str, dict[str, str]] = {
    "normalize.mode": {"smooth": "per_frame", "none": "fixed"},
    "suggest.select_rule": {"least_confidence": "margin", "entropy": "margin"},
    "ui.kymo_sign": {"absolute": "canonical"},
}
REMOVED_NORMALIZE_MODES = REMOVED_VALUES["normalize.mode"]


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=4)


# Old values that already meant the fixed one.
_FIXED_ALIASES = {"tracking.tracker_cfg": (None,)}


def _same(value, fixed) -> bool:
    """Whether an old config's ``value`` is the fixed one, loosely typed."""
    if isinstance(fixed, list):
        return isinstance(value, (list, tuple)) and             [str(x).upper() for x in value] == [str(x).upper() for x in fixed]
    converter = {bool: _to_bool, int: _to_float, float: _to_float}.get(type(fixed), _to_str)
    try:
        return converter(value) == fixed
    except _Bad:
        return False


def _migrate_key(dotted: str, value) -> bool:
    """Warn about a key the schema no longer has. True when it was handled."""
    if dotted in FIXED_SETTINGS:
        fixed = FIXED_SETTINGS[dotted]
        hint = _FIXED_HINTS.get(dotted)
        if _same(value, fixed) or value in _FIXED_ALIASES.get(dotted, ()):
            _warn(f"{dotted} is no longer configurable; using {fixed!r}")
        else:
            _warn(f"{dotted} is no longer configurable; using {fixed!r}, NOT the "
                  f"{value!r} this config sets" + (f" ({hint})" if hint else ""))
        return True
    if dotted in VIEW_SETTINGS:
        _warn(f"{dotted} is now a per-user view setting in the annotation window; "
              "ignoring it here")
        return True
    if dotted in _REMOVED_KEYS:
        _warn(f"ignoring {dotted}: {_REMOVED_KEYS[dotted]}")
        return True
    return False


# --------------------------------------------------------------------------- #
# field metadata and the generic parse
# --------------------------------------------------------------------------- #
def knob(default, help: str = "", *, choices=None, min=None, max=None, below=None,
         special: tuple = (), parse: Callable | None = None):
    """A setting: its default, one line of help and what it accepts.

    ``choices`` limits string values (a callable is read lazily), ``min``/``max``
    bound numbers inclusively and ``below`` exclusively, ``special`` lists words
    a numeric setting also takes (``auto``), ``parse`` replaces the coercion.
    """
    meta = {"help": help, "choices": choices, "min": min, "max": max, "below": below,
            "special": tuple(special), "parse": parse}
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: copy.deepcopy(default), metadata=meta)
    return field(default=default, metadata=meta)


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


class _Bad(Exception):
    """A value that does not fit one type in a union."""


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    raise _Bad


def _to_int(v):
    if isinstance(v, bool):
        raise _Bad
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            number = float(v)
        except ValueError:
            raise _Bad from None
        if number.is_integer():
            return int(number)
    raise _Bad


def _to_float(v):
    if isinstance(v, bool):
        raise _Bad
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raise _Bad from None
    raise _Bad


def _to_str(v):
    if isinstance(v, (dict, list, tuple)):
        raise _Bad
    return str(v)


def _to_str_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    raise _Bad


def _to_int_list(v):
    items = v if isinstance(v, (list, tuple)) else [v]
    return [_to_int(x) for x in items]


def _to_dict(v):
    if isinstance(v, dict):
        return dict(v)
    raise _Bad


_CONVERT = {
    "bool": (_to_bool, "true or false"), "int": (_to_int, "a whole number"),
    "float": (_to_float, "a number"), "str": (_to_str, "text"),
    "Path": (_to_str, "a path"), "list[str]": (_to_str_list, "a list of names"),
    "list[int]": (_to_int_list, "a list of whole numbers"),
    "dict": (_to_dict, "a mapping"),
}


def _choices(meta: dict) -> list | None:
    c = meta.get("choices")
    if c is None:
        return None
    return list(c() if callable(c) else c)


def _coerce(where: str, value, type_text: str, meta: dict):
    """``value`` as the field's type, checked against its choices and range."""
    special = meta["special"]
    if isinstance(value, str) and value in special:
        return value
    kinds = [k.strip() for k in type_text.split("|")]
    if value is None and "None" in kinds:
        return None
    if special:
        kinds = [k for k in kinds if k != "str"]
    choices = _choices(meta)
    if choices is not None and isinstance(value, bool) and "bool" not in kinds:
        # An unquoted on/off arrives as a YAML boolean.
        word = "on" if value else "off"
        value = word if word in choices else value
    out = _NOTHING = object()
    for kind in kinds:
        convert = _CONVERT.get(kind)
        if convert is None:
            continue
        try:
            out = convert[0](value)
            break
        except _Bad:
            continue
    if out is _NOTHING:
        shapes = [_CONVERT[k][1] for k in kinds if k in _CONVERT]
        shapes += [repr(s) for s in special]
        raise ConfigError(f"{where} must be {' or '.join(shapes)}, got {value!r}")
    if isinstance(out, str) and choices is not None and out not in choices:
        raise ConfigError(f"{where} must be one of {choices}, got {out!r}")
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        lo, hi, below = meta["min"], meta["max"], meta["below"]
        if (lo is not None and out < lo) or (hi is not None and out > hi) \
                or (below is not None and out >= below):
            span = (f"at least {lo}" if hi is None and below is None
                    else f"in [{lo if lo is not None else '-inf'}, "
                         f"{below if below is not None else hi}"
                         f"{')' if below is not None else ']'}")
            raise ConfigError(f"{where} must be {span}, got {out!r}")
    return out


def _parse_section(cls, d: dict, root: Path | None = None, name: str | None = None):
    """Build section ``cls`` from mapping ``d`` using its field metadata."""
    name = name or cls.NAME
    d = dict(d or {})
    known = {f.name for f in dataclasses_fields(cls)}
    for key in [k for k in d if k not in known]:
        if not _migrate_key(f"{name}.{key}", d[key]):
            _warn(f"ignoring unknown key {name}.{key}")
        d.pop(key)
    values = {}
    for f in dataclasses_fields(cls):
        where = f"{name}.{f.name}"
        given = f.name in d and (d[f.name] is not None or "None" in str(f.type))
        if given:
            raw = d[f.name]
            old = REMOVED_VALUES.get(where, {})
            if isinstance(raw, str) and raw in old:
                _warn(f"{where}: {raw!r} was removed; using {old[raw]!r}")
                raw = old[raw]
            parse = f.metadata.get("parse")
            values[f.name] = (parse(raw, where) if parse
                              else _coerce(where, raw, str(f.type), f.metadata))
        if str(f.type) == "Path":
            value = values.get(f.name, _default_of(f))
            values[f.name] = _resolve(root, value) if root is not None else Path(str(value))
    obj = cls(**values)
    obj._check(d)
    return obj


class _Section:
    """Shared parse for every section dataclass."""

    NAME: ClassVar[str] = ""

    @classmethod
    def parse(cls, d: dict | None = None, root: Path | None = None):
        return _parse_section(cls, d or {}, root)

    def _check(self, raw: dict) -> None:
        """Rules that span fields. ``raw`` is the mapping as given."""


def _resolve(root: Path, value) -> Path:
    p = Path(str(value).replace("\\", "/"))
    return p if p.is_absolute() or str(p).startswith("//") else (root / p).resolve()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """'#4C9F70' -> (76, 159, 112)."""
    s = color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ConfigError(f"colour {color!r} is not a #rrggbb hex string")
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        raise ConfigError(f"colour {color!r} is not a #rrggbb hex string") from None


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class Paths(_Section):
    NAME: ClassVar[str] = "paths"

    video_dir: Path = knob("./videos", "Folder holding the movies (.tif, .tiff, .nd2).")
    model: Path = knob("./best.pt", "YOLO weights used for detection.")
    work_dir: Path = knob("./work", "Clip cache and per-movie tables; keep it on a local disk.")
    csv: Path = knob("./annotations.csv", "The annotation log, one row per labelled clip.")
    patterns: list[str] = knob(["*.tif", "*.tiff", "*.nd2"],
                               "Which files in the video folder are movies.")


@dataclass
class ND2Options(_Section):
    NAME: ClassVar[str] = "nd2"

    channel: int | str = knob(0, "Channel index, or its name.")
    z: int | str = knob(0, "Z index, or max / mean / min for a projection.",
                        choices=("max", "mean", "min"))
    positions: str | list[int] = knob("all", "'all', or a list of positions; each is a movie.",
                                      special=("all",))
    fps: float | str = knob("auto", "Frame rate; auto reads it from the file.",
                            special=("auto",), min=0)


@dataclass
class TiffOptions(_Section):
    NAME: ClassVar[str] = "tiff"

    fps: float = knob(100.0, "Acquisition frame rate; TIFF does not record it.", min=0)
    channel: int = knob(0, "Channel index.", min=0)


def _percentiles(value, where: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where} must be [low, high], got {value!r}") from None
    if not 0 <= lo < hi <= 100:
        raise ConfigError(f"{where} must be [low, high] within 0..100, got {value!r}")
    return (lo, hi)


@dataclass
class Normalize(_Section):
    """How raw frames become uint8."""

    NAME: ClassVar[str] = "normalize"

    mode: str = knob("per_frame", "per_frame: a mapping per frame; fixed: one per movie.",
                     choices=("per_frame", "fixed"))
    percentiles: tuple[float, float] = knob((0.5, 99.5), "Percentiles mapped to 0 and 255.",
                                            parse=_percentiles)


@dataclass
class Detection(_Section):
    NAME: ClassVar[str] = "detection"

    task: str = knob("detect", "pose if the model predicts keypoints, detect for boxes only.",
                     choices=("pose", "detect"))
    conf: float = knob(0.25, "Minimum detection confidence.", min=0.0, max=1.0)
    device: str = knob("auto", "auto, cpu, mps, or a CUDA index such as 0.")
    batch: int = knob(8, "Tiles per forward pass; lower it if the GPU runs out of memory.",
                      min=1)
    imgsz: int = knob(640, "Resolution each tile is seen at; match the model's training size.",
                      min=32)
    half: bool = knob(False, "fp16 inference on a GPU.")


def _trackers() -> list[str]:
    from .detect.tracking import TRACKERS

    return sorted(TRACKERS) + ["greedy"]


@dataclass
class Tiling(_Section):
    NAME: ClassVar[str] = "tiling"

    enabled: bool | str = knob("auto", "auto tiles only frames larger than one tile.",
                               special=("auto",))
    tile: int = knob(640, "Tile size cut from the frame, in pixels.", min=32)
    overlap: float = knob(0.25, "Share of a tile shared with its neighbour.", min=0.0, below=1.0)
    merge_iou: float = knob(0.6, "Overlap above which boxes from two tiles are one animal.",
                            min=0.0, max=1.0)


@dataclass
class Tracking(_Section):
    NAME: ClassVar[str] = "tracking"

    backend: str = knob("tracktrack", "Which tracker links detections across frames.",
                        choices=_trackers)
    min_track_length: int = knob(40, "Drop tracks seen in fewer frames than this.", min=1)
    border_margin_px: int = knob(0, "How near the frame edge counts as at it; 0 is off.",
                                 min=0)
    border_max_frac: float = knob(0.0, "Share of one clip's frames allowed at the edge.",
                                  min=0.0, max=1.0)
    overrides: dict = knob({"gmc_method": "none"}, "Parameters passed to the tracker.")


def _edges(value, where: str) -> list[tuple[str, str]]:
    out = []
    for e in value or []:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise ConfigError(f"{where} entry {e!r} must have exactly two node names")
        out.append((str(e[0]), str(e[1])))
    return out


@dataclass
class Keypoints(_Section):
    NAME: ClassVar[str] = "keypoints"

    nodes: list[str] = knob([], "Keypoint names, in the model's order.")
    edges: list[tuple[str, str]] = knob([], "Skeleton links, drawn on screen only.",
                                        parse=_edges)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def edge_idx(self) -> list[tuple[int, int]]:
        idx = {n: i for i, n in enumerate(self.nodes)}
        return [(idx[a], idx[b]) for a, b in self.edges]

    def _check(self, raw: dict) -> None:
        for a, b in self.edges:
            for name in (a, b):
                if name not in self.nodes:
                    raise ConfigError(f"keypoints.edges references unknown node {name!r}")


@dataclass
class Features(_Section):
    """Per-detection detector embeddings, pooled from the same forward pass."""

    NAME: ClassVar[str] = "features"

    enabled: bool = knob(True, "Save detector embeddings during detection.")


@dataclass
class EmbeddingSpec:
    """One per-detection embedding input: ``detector`` or a ``file`` per movie."""

    name: str
    source: str = "detector"
    # File source only: a path with ``{movie_id}``, relative to the config.
    path: str | None = None

    @property
    def group(self) -> str:
        return f"emb_{self.name}"


def _embeddings(value, where: str = "suggest.embeddings") -> list[EmbeddingSpec]:
    """A list of ``{name, source, path}`` mappings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list of {{name, source, path}}")
    out: list[EmbeddingSpec] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} entry {entry!r} is not a mapping")
        for key in set(entry) - {"name", "source", "path"}:
            _warn(f"ignoring unknown key {where}.{key}")
        name = str(entry.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9]+", name):
            raise ConfigError(f"{where} name {name!r} must be letters and digits only")
        source = str(entry.get("source", "detector"))
        if source not in ("detector", "file"):
            raise ConfigError(f"{where} {name}: source must be 'detector' or 'file', "
                              f"got {source!r}")
        path = entry.get("path")
        if source == "file" and (not path or "{movie_id}" not in str(path)):
            raise ConfigError(f"{where} {name}: a file source needs a path containing "
                              "{movie_id}, e.g. embeddings/{movie_id}.parquet")
        if any(e.name == name for e in out):
            raise ConfigError(f"{where} names {name!r} twice")
        out.append(EmbeddingSpec(name, source, str(path) if path else None))
    return out


def _feature_columns(value, where: str = "suggest.feature_columns") -> list[str] | None:
    """The column list as given; checked against the groups in ``Suggest._check``."""
    if value is None:
        return None
    if isinstance(value, str):
        raise ConfigError(f"{where} must be a list of column names, got {value!r}")
    names = [str(c).strip() for c in value]
    if not names:
        raise ConfigError(f"{where} is an empty list; use null for every column the groups select")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ConfigError(f"{where} names {repeated} more than once")
    return names


def _check_columns(names: list[str], groups: list[str]) -> None:
    """Refuse names no feature table has, or from a group that is not selected."""
    import difflib

    from .measure.clipfeatures import group_of, known_columns
    from .measure.embeddings import EMBEDDING_COLUMN

    vocabulary = known_columns()
    unknown = [n for n in names
               if n not in set(vocabulary) and not EMBEDDING_COLUMN.match(n)]
    if unknown:
        near = difflib.get_close_matches(unknown[0], vocabulary, n=4, cutoff=0.6)
        raise ConfigError(
            f"suggest.feature_columns names unknown column(s) {unknown[:6]}"
            + (f"; did you mean {', '.join(near)}?" if near else "")
        )
    outside = sorted({n for n in names if group_of(n) not in groups})
    if outside:
        families = sorted({str(group_of(n)) for n in outside})
        raise ConfigError(
            f"suggest.feature_columns names {outside[:6]} from group(s) {families}, "
            f"which suggest.feature_groups ({groups}) does not include; add the group "
            "or drop those columns"
        )


_OLD_CNN = re.compile(r"^cnn(\d+)_(mean|std|delta)$")


_CLIP_ID = re.compile(r"^(?P<movie>.+)__t(?P<track>\d+)__f\d+$")


def audit_track_key(clip_id: str) -> str:
    """``movie_id::track_id`` for a clip_id, the unit the audit holdout is drawn
    over. An id not in ``clip_id_for`` form is its own key."""
    m = _CLIP_ID.match(str(clip_id))
    if m is None:
        return str(clip_id)
    return f"{m['movie']}::{int(m['track'])}"


@dataclass
class Suggest(_Section):
    NAME: ClassVar[str] = "suggest"

    enabled: bool = knob(True, "Show what a classifier trained on your labels predicts.")
    model: str = knob("catboost", "catboost, or hist (scikit-learn HistGradientBoosting).",
                      choices=("catboost", "hist"))
    device: str = knob("cpu", "Where CatBoost fits: cpu, gpu or auto.",
                       choices=("cpu", "gpu", "auto"))
    feature_groups: list[str] = knob(["kinematic", "pose", "waveform"],
                                     "Feature families the model may use.")
    embeddings: list[EmbeddingSpec] = knob([], "Per-detection embedding inputs.",
                                           parse=_embeddings)
    context_frames: int = knob(0, "Frames of kinematic context around each window; 0 is off.",
                               min=0)
    feature_columns: list[str] | None = knob(None, "Narrow the groups to these columns; "
                                             "null keeps every column.",
                                             parse=_feature_columns)
    retrain_every: int = knob(10, "Refit in the background after this many new labels.",
                              min=1)
    min_per_class: int = knob(5, "A class with fewer labels is not modelled.", min=1)
    audit_fraction: float = knob(0.10, "Share of tracks held out from the suggester.",
                                 min=0.0, below=1.0)
    min_confidence: float = knob(0.0, "Hide suggestions below this confidence.",
                                 min=0.0, max=1.0)
    select_rule: str = knob("off", "off keeps the shuffle; margin asks about the least "
                            "certain clips first.", choices=("off", "margin"))
    iterations: int = knob(1000, "Boosting rounds per fit.", min=1)
    learning_rate: float | str = knob("auto", "Boosting step size; auto lets the estimator "
                                      "choose.", special=("auto",), min=0.0)
    l2_leaf_reg: float = knob(3.0, "L2 regularisation of the leaf values.", min=0.0)

    @property
    def selection_on(self) -> bool:
        return self.select_rule != "off"

    def audit_by_hash(self, clip_id: str) -> bool:
        """Whether ``clip_id``'s track hashes into the audit holdout (fallback).

        Hashed on ``movie_id::track_id`` so every clip of a track lands on the
        same side. A labelled clip's recorded membership wins over this.
        """
        frac = float(self.audit_fraction)
        if frac <= 0:
            return False
        seed = FIXED_SETTINGS["suggest.audit_seed"]
        digest = hashlib.blake2b(
            f"{seed}:{audit_track_key(clip_id)}".encode(), digest_size=8
        ).digest()
        return (int.from_bytes(digest, "big") % 10_000) < frac * 10_000

    def _check(self, raw: dict) -> None:
        from .measure.clipfeatures import FEATURE_GROUPS

        groups = list(self.feature_groups)
        if "feature_groups" not in raw:
            groups = [g for g in FEATURE_GROUPS if g != "context" or self.context_frames > 0]
            groups += [e.group for e in self.embeddings]
        if "cnn" in groups:
            _warn("suggest.feature_groups 'cnn' is now the detector embedding 'det' "
                  "(group emb_det); retrain models fitted on cnn columns")
            if not any(e.name == "det" for e in self.embeddings):
                self.embeddings.append(EmbeddingSpec("det", "detector"))
            groups = list(dict.fromkeys("emb_det" if g == "cnn" else g for g in groups))
            if self.feature_columns is not None:
                self.feature_columns = [_OLD_CNN.sub(r"emb_det_\1_\2", c)
                                        for c in self.feature_columns]
        allowed = list(FEATURE_GROUPS) + [e.group for e in self.embeddings]
        unknown = [g for g in groups if g not in allowed]
        if unknown:
            raise ConfigError(f"suggest.feature_groups has unknown entries {unknown}; "
                              f"choose from {allowed}")
        if not groups:
            raise ConfigError("suggest.feature_groups is empty, so the model has no inputs "
                              f"at all; choose from {allowed}")
        unused = [e.group for e in self.embeddings if e.group not in groups]
        if self.context_frames > 0 and "context" not in groups:
            unused.append("context")
        if "feature_groups" in raw and unused:
            _warn(f"{unused} are configured (suggest.embeddings, suggest.context_frames) "
                  "but suggest.feature_groups does not list them, so the model will "
                  "not use them")
        self.feature_groups = groups
        if self.feature_columns is not None:
            _check_columns(self.feature_columns, groups)
        if self.learning_rate == 0:
            raise ConfigError("suggest.learning_rate must be above 0, or 'auto'")
        if self.device == "gpu" and self.model != "catboost":
            raise ConfigError(f"suggest.device 'gpu' needs suggest.model 'catboost', "
                              f"not {self.model!r}")
        if self.select_rule != "off" and self.audit_fraction <= 0.0:
            raise ConfigError(f"suggest.select_rule {self.select_rule!r} needs "
                              "suggest.audit_fraction above 0 (e.g. 0.10) to keep an "
                              "unbiased holdout, or select_rule 'off'")
        if self.select_rule != "off" and not self.enabled:
            raise ConfigError(f"suggest.select_rule {self.select_rule!r} needs "
                              "suggest.enabled: true")


@dataclass
class Spectral(_Section):
    """The window a beat is measured over, centred on the clip."""

    NAME: ClassVar[str] = "spectral"

    window: int = knob(75, "Frames the beat spectrum is measured over.", min=8)


@dataclass
class Predict(_Section):
    """How ``trackanno predict`` scores every frame and cuts it into bouts."""

    NAME: ClassVar[str] = "predict"

    window_stride: int | str = knob("auto", "Frames between window starts; auto is "
                                    "clips.frames.", special=("auto",), min=1)
    min_bout_frames: int = knob(0, "Shorter bouts join a neighbour; 0 keeps every run.",
                                min=0)
    min_frames: int = knob(2, "Detections a window needs to be featurised.", min=1)

    def resolved_stride(self, clips: "Clips") -> int:
        """``window_stride`` with ``auto`` filled in: ``clips.frames``."""
        if isinstance(self.window_stride, str):
            return max(1, int(clips.frames))
        return max(1, int(self.window_stride))


@dataclass
class ModelSource(_Section):
    """Where the detection weights come from, when not ``paths.model``.

    ``source`` is a registry id or an ``https://`` URL; ``sha256`` is written on
    the first fetch and pins the weights from then on.
    """

    NAME: ClassVar[str] = "model"

    source: str = knob("", "Registry id or https URL; empty uses paths.model.")
    version: str = knob("", "Registry version.")
    sha256: str = knob("", "Pinned digest, filled in on the first fetch.")

    @property
    def is_remote(self) -> bool:
        return bool(self.source.strip())

    @property
    def is_url(self) -> bool:
        return self.source.strip().lower().startswith(("http://", "https://"))

    def _check(self, raw: dict) -> None:
        self.source = self.source.strip()
        self.version = self.version.strip()
        self.sha256 = self.sha256.strip().lower()
        if self.source.lower().startswith("http://"):
            from .video.models import is_trusted_url

            if not is_trusted_url(self.source):
                raise ConfigError("model.source must be https, not http")
        digest = self.sha256
        if digest and (len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest)):
            raise ConfigError("model.sha256 must be a 64-character hex digest, or empty")


@dataclass
class Clips(_Section):
    NAME: ClassVar[str] = "clips"

    frames: int = knob(50, "Frames per clip; also the window the model is fitted at.",
                       min=1)
    stride: int = knob(50, "Frames between clip starts; equal to frames tiles.", min=1)
    window_mode: str = knob("all", "first, first_per_track, or all windows.",
                            choices=("first", "first_per_track", "all"))
    follow: str = knob("smooth", "How the crop follows the animal: smooth, fixed or track.",
                       choices=("smooth", "fixed", "track"))
    pad: float = knob(1.6, "Crop side is the box extent times this.", min=0.0)
    crop_size: int | str = knob("auto", "Cache crop size; auto is the p99 of what is "
                                "needed.", special=("auto",), min=1)
    min_clip_frames: int = knob(40, "Detected frames a window needs to become a clip.",
                                min=1)

    def _check(self, raw: dict) -> None:
        if "stride" not in raw:
            self.stride = self.frames
        if self.min_clip_frames > self.frames:
            raise ConfigError(f"clips.min_clip_frames ({self.min_clip_frames}) exceeds "
                              f"clips.frames ({self.frames}); no window could ever qualify")


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


def _unknown(where: str, given: dict, allowed) -> None:
    for key in sorted(set(given) - set(allowed)):
        _warn(f"ignoring unknown key {where}.{key}")


def _parse_categories(raw: Any) -> list[CategoryGroup]:
    if not raw:
        raise ConfigError("config must define at least one entry under 'categories'")
    if not isinstance(raw, list):
        raise ConfigError("categories must be a list of groups")
    groups: list[CategoryGroup] = []
    seen_keys: dict[str, str] = {}
    seen_names: set[str] = set()
    for g in raw:
        if not isinstance(g, dict):
            raise ConfigError(f"category group {g!r} is not a mapping")
        name = str(g.get("name", "")).strip()
        if not name:
            raise ConfigError("every category group needs a 'name'")
        if name in RESERVED_GROUP_NAMES:
            raise ConfigError(f"category group {name!r} collides with a built-in CSV column")
        if name in seen_names:
            raise ConfigError(f"two category groups are both named {name!r}")
        seen_names.add(name)
        _unknown(f"categories[{name}]", g, ["name", "type", "required", "options"])
        gtype = str(g.get("type", "single"))
        if gtype not in ("single", "multi"):
            raise ConfigError(f"category group {name!r}: type must be 'single' or 'multi'")
        options: list[CategoryOption] = []
        for o in g.get("options", []) or []:
            _unknown(f"categories[{name}] option", o, ["name", "key", "color"])
            key = str(o.get("key", ""))
            if not key:
                raise ConfigError(f"category option {o.get('name')!r} needs a 'key' shortcut")
            label = f"{name}.{o.get('name')}"
            if key.lower() in seen_keys:
                raise ConfigError(f"shortcut {key!r} is used by both "
                                  f"{seen_keys[key.lower()]} and {label}")
            seen_keys[key.lower()] = label
            color = str(o.get("color", "#CCCCCC"))
            _hex_to_rgb(color)
            options.append(CategoryOption(name=str(o["name"]), key=key, color=color))
        if not options:
            raise ConfigError(f"category group {name!r} has no options")
        required = _coerce(f"categories[{name}].required", g.get("required", False), "bool",
                           {"special": (), "choices": None, "min": None, "max": None,
                            "below": None})
        groups.append(CategoryGroup(name=name, type=gtype, required=required,
                                    options=options))
    return groups


@dataclass
class UI(_Section):
    NAME: ClassVar[str] = "ui"

    theme: str = knob("dark", "dark, light, or system. Applies on next start.",
                      choices=("dark", "light", "system"))
    fps: int = knob(20, "Playback rate in the annotation window.", min=1)
    auto_advance: bool = knob(True, "One keypress labels the clip and moves on.")
    shuffle: bool = knob(True, "Annotate in a seeded random order.")
    annotator: str = knob("", "Written into every row of the CSV.")
    kymo_sign: str = knob("canonical", "canonical flips each kymograph to its dominant "
                          "sign; raw keeps it as measured.", choices=("canonical", "raw"))


DEFAULT_KEYS = {
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
    "uncertain": ["u"],
    "notes": ["n"],
    "clear": ["Backspace"],
    "undo": ["Ctrl+Z"],
}
_DEFAULT_KEYS = DEFAULT_KEYS


@dataclass
class Render(_Section):
    NAME: ClassVar[str] = "render"

    fps: float | str = knob("auto", "Frame rate of the movie; auto is the source rate.",
                            special=("auto",), min=0)
    out_dir: Path = knob("./renders", "Where overlay movies are written.")
    hide_unlabelled: bool = knob(False, "Draw only tracks that carry a label.")
    show_skeleton: bool = knob(False, "Draw pose keypoints.")


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
SECTIONS: dict[str, type] = {
    "paths": Paths, "nd2": ND2Options, "tiff": TiffOptions, "normalize": Normalize,
    "detection": Detection, "model": ModelSource, "tiling": Tiling,
    "tracking": Tracking, "keypoints": Keypoints, "features": Features,
    "spectral": Spectral, "suggest": Suggest, "predict": Predict, "clips": Clips,
    "ui": UI, "render": Render,
}
_TOP_LEVEL = set(SECTIONS) | {"categories", "keys"}


@dataclass
class Config:
    path: Path
    paths: Paths
    nd2: ND2Options
    tiff: TiffOptions
    normalize: Normalize
    detection: Detection
    model: ModelSource
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

    @property
    def is_pose(self) -> bool:
        return self.detection.task == "pose"

    @property
    def n_keypoints(self) -> int:
        return self.keypoints.n if self.is_pose else 0

    @property
    def needs_keypoints(self) -> bool:
        """``task: pose`` with no keypoint names; the checkpoint supplies the count."""
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


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping, "
                          f"got {type(value).__name__}")
    return value


def _parse_keys(data: dict) -> dict[str, list[str]]:
    keys = {k: list(v) for k, v in DEFAULT_KEYS.items()}
    for action, binding in _section(data, "keys").items():
        if action not in DEFAULT_KEYS:
            _warn(f"ignoring unknown key binding keys.{action}")
            continue
        keys[action] = [binding] if isinstance(binding, str) else [str(b) for b in binding]
    return keys


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

    for name in sorted(set(data) - _TOP_LEVEL):
        _warn(f"ignoring unknown config section '{name}'")
    root = path.parent
    sections = {name: cls.parse(_section(data, name), root) for name, cls in SECTIONS.items()}
    keys = _parse_keys(data)
    cfg = Config(path=path, categories=_parse_categories(data.get("categories")),
                 keys=keys, **sections)

    # Before embeddings, an unset feature_groups meant the detector's cnn group.
    old = data.get("suggest")
    if (isinstance(old, dict) and "feature_groups" not in old
            and "embeddings" not in old and cfg.features.enabled):
        _warn("detector embeddings are now opt-in; add suggest.embeddings "
              "[{name: det, source: detector}] and emb_det to feature_groups to use them")

    stride = cfg.predict.resolved_stride(cfg.clips)
    if stride > int(cfg.clips.frames):
        raise ConfigError(f"predict.window_stride ({stride}) cannot exceed clips.frames "
                          f"({cfg.clips.frames}); frames would go unscored")
    nav = {k.lower() for binds in keys.values() for k in binds}
    for g in cfg.categories:
        for o in g.options:
            if o.key.lower() in nav:
                raise ConfigError(f"category shortcut {o.key!r} ({g.name}.{o.name}) "
                                  "collides with a navigation shortcut")
    return cfg


# --------------------------------------------------------------------------- #
# schema queries, for the form, the template and the tests
# --------------------------------------------------------------------------- #
def _field(dotted: str):
    name, _, key = dotted.partition(".")
    cls = SECTIONS.get(name)
    if cls is None:
        return None
    return next((f for f in dataclasses_fields(cls) if f.name == key), None)


def _default_of(f):
    if f.default is not MISSING:
        return f.default
    return f.default_factory()


def schema_keys() -> list[str]:
    """Every configurable dotted key, in section order."""
    return [f"{name}.{f.name}" for name, cls in SECTIONS.items()
            for f in dataclasses_fields(cls)]


def help_for(dotted: str) -> str:
    """The one-line help for a key, or ""."""
    f = _field(dotted)
    return f.metadata.get("help", "") if f is not None else ""


def meta_for(dotted: str) -> dict:
    """A key's metadata with ``choices`` resolved: help, choices, min, max, below, special."""
    f = _field(dotted)
    if f is None:
        return {}
    meta = dict(f.metadata)
    meta["choices"] = _choices(meta)
    meta["type"] = str(f.type)
    return meta


def effective_values(data: dict) -> dict[str, Any]:
    """Every schema key as the loader sets it from the raw mapping ``data``.

    Paths keep the file's text. A section that does not load shows its raw
    values over the defaults.
    """
    out: dict[str, Any] = {}
    for name, cls in SECTIONS.items():
        raw = data.get(name) if isinstance(data.get(name), dict) else {}
        fields = dataclasses_fields(cls)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                section = cls.parse(copy.deepcopy(raw))
            values = {f.name: getattr(section, f.name) for f in fields}
        except Exception:                                   # noqa: BLE001
            values = {f.name: raw.get(f.name, _default_of(f)) for f in fields}
        for f in fields:
            value = values[f.name]
            if str(f.type) == "Path":
                given = raw.get(f.name)
                value = str(given if given is not None else _default_of(f))
            out[f"{name}.{f.name}"] = value
    return out


def section_defaults(name: str) -> dict:
    """Field defaults of one section, without path-valued ones."""
    cls = SECTIONS.get(name)
    if cls is None:
        return {}
    return {f.name: _default_of(f) for f in dataclasses_fields(cls)
            if not str(f.type).startswith("Path")}
