"""Typed configuration loaded from ``config.yaml``.

Plain dataclasses rather than pydantic: the schema is small, and validation
errors here should read like advice, not like a stack trace.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from dataclasses import fields as dataclasses_fields
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["Config", "load_config", "CategoryGroup", "CategoryOption",
           "ModelSource", "RESERVED_GROUP_NAMES", "section_defaults"]

# A category group becomes a CSV column, so a group named like a built-in column
# silently overwrites it. Hand-maintained mirror of ``store.IDENTITY_COLUMNS``
# and ``store.TRAILING_COLUMNS``, kept here rather than imported to avoid the
# import cycle. Nothing in the language holds the two together;
# ``tests/test_store_provenance.py`` asserts they are equal. Add to both, or to
# neither.
RESERVED_GROUP_NAMES = frozenset({
    "clip_id", "movie_id", "source_file", "series", "channel",
    "track_id", "f_start", "f_end",
    "crop_x0", "crop_y0", "crop_w", "crop_h", "scale",
    "uncertain", "notes", "annotator", "timestamp_utc", "dwell_s",
    "suggested", "suggest_conf", "suggest_shown",
    "selected_by", "audit_holdout",
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


def _optional_auto(value, where: str, expected: str):
    """A setting that may say ``auto``: ``None`` for auto, the value otherwise.

    ``None`` rather than the resolved number: what ``auto`` stands for is a fact
    about another section (``clips.frames``, ``audit_fraction``) this parse
    cannot see yet, and a config that says ``auto`` should keep saying it when
    the GUI writes the file back.
    """
    if isinstance(value, str):
        if value != "auto":
            raise ConfigError(f"{where} must be {expected}, got {value!r}")
        return None
    return value


def _selection_rule(d: dict) -> str:
    """``select_rule`` as ``off`` or ``margin``, or a refusal."""
    raw = d.get("select_rule", "off")
    if isinstance(raw, bool):
        # YAML reads a bare `off` as a boolean.
        raise ConfigError(
            "suggest.select_rule came through as a YAML boolean, which is what an "
            "unquoted `off`/`on` becomes. Quote it: select_rule: \"off\"."
        )
    rule = str(raw)
    if rule in ("least_confidence", "entropy"):
        warnings.warn(
            f"suggest.select_rule {rule!r} is no longer available; using 'margin'",
            stacklevel=3,
        )
        return "margin"
    if rule not in ("off", "margin"):
        raise ConfigError(
            f"suggest.select_rule must be \"off\" or \"margin\", got {rule!r}"
        )
    return rule


def _feature_columns(value, groups: list[str]) -> list[str] | None:
    """``suggest.feature_columns``, checked against the vocabulary and the groups.

    ``None`` (and an absent key) means "do not cut here"; a list narrows what
    ``feature_groups`` selected. The names are checked against
    :func:`trackanno.measure.clipfeatures.known_columns`, which is the full set a
    built table can hold, so a typo is refused here rather than silently changing
    the feature space the model was fitted in. Only when the key is given: the
    schema pulls the measurement modules in, and the default path must not pay
    for that.

    A name whose family is not in ``feature_groups`` is refused rather than
    turning that family back on, because the two keys would then disagree about
    what the model may see and only one of them would be on screen.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise ConfigError(
            f"suggest.feature_columns must be a list of column names, got the "
            f"string {value!r}. One name is still a list: [{value}]."
        )
    names = [str(c).strip() for c in value]
    if not names:
        raise ConfigError(
            "suggest.feature_columns is an empty list, so the model would have no "
            "inputs at all. 'every column the feature groups selected' is written "
            "as null, not as []."
        )
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ConfigError(
            f"suggest.feature_columns names {repeated} more than once. A column "
            "listed twice is not weighted twice, so the repeat says something the "
            "model cannot do; drop it."
        )

    import difflib

    from .measure.clipfeatures import group_of, known_columns
    from .measure.embeddings import EMBEDDING_COLUMN

    # Embedding names are matched by shape: their width follows the PCA, and
    # training refuses the ones the table actually lacks.
    vocabulary = known_columns()
    unknown = [n for n in names
               if n not in set(vocabulary) and not EMBEDDING_COLUMN.match(n)]
    if unknown:
        near = difflib.get_close_matches(unknown[0], vocabulary, n=4, cutoff=0.6)
        raise ConfigError(
            f"suggest.feature_columns names {len(unknown)} column(s) no feature "
            f"table has: {unknown[:6]}"
            + (f". Did you mean {', '.join(near)}?" if near else ".")
            + " Pick them on the Configure page, which lists what this project can "
            "actually compute, or set the key to null to use every column the "
            "feature groups selected."
        )
    outside = sorted({n for n in names if group_of(n) not in groups})
    if outside:
        families = sorted({str(group_of(n)) for n in outside})
        raise ConfigError(
            f"suggest.feature_columns names {outside[:6]} from the feature "
            f"group(s) {families}, which suggest.feature_groups does not include "
            f"({groups}). The two keys would then disagree about what the model may "
            "see. Either add the group to suggest.feature_groups, or drop those "
            "columns from suggest.feature_columns."
        )
    return names


_CLIP_ID = re.compile(r"^(?P<movie>.+)__t(?P<track>\d+)__f\d+$")


def audit_track_key(clip_id: str) -> str:
    """``movie_id::track_id`` for a clip_id, the unit the audit holdout is drawn
    over. An id not in ``clip_id_for`` form is its own key."""
    m = _CLIP_ID.match(str(clip_id))
    if m is None:
        return str(clip_id)
    return f"{m['movie']}::{int(m['track'])}"


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


# Removed normalize.mode values and the mode an old config now gets instead.
REMOVED_NORMALIZE_MODES = {"smooth": "per_frame", "none": "fixed"}


@dataclass
class Normalize:
    """How raw frames become uint8: ``per_frame`` or ``fixed`` percentiles."""

    mode: str = "per_frame"
    percentiles: tuple[float, float] = (0.5, 99.5)
    subsample: int = 4

    @classmethod
    def parse(cls, d: dict) -> "Normalize":
        _known("normalize", d, ["mode", "percentiles", "subsample"])
        mode = str(d.get("mode", "per_frame"))
        if mode in REMOVED_NORMALIZE_MODES:
            fallback = REMOVED_NORMALIZE_MODES[mode]
            warnings.warn(f"normalize.mode: {mode!r} was removed; using {fallback!r}",
                          stacklevel=3)
            mode = fallback
        if mode not in ("fixed", "per_frame"):
            raise ConfigError(
                "normalize.mode must be 'per_frame' (default) or 'fixed' (one mapping "
                "for the whole movie - faithful to real intensity changes, including "
                "flicker)"
            )
        pcts = d.get("percentiles", [0.5, 99.5])
        if len(pcts) != 2 or not 0 <= float(pcts[0]) < float(pcts[1]) <= 100:
            raise ConfigError("normalize.percentiles must be [low, high] within 0..100")
        return cls(
            mode=mode,
            percentiles=(float(pcts[0]), float(pcts[1])),
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
    # How close to the frame edge counts as being at it, and how much of one
    # clip may be spent there before that clip goes. 0 px is off.
    border_margin_px: int = 0
    border_max_frac: float = 0.0
    overrides: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, d: dict) -> "Tracking":
        from .detect.tracking import TRACKERS

        _known("tracking", d,
               ["backend", "tracker_cfg", "min_track_length", "greedy_max_dist_px",
                "border_margin_px", "border_max_frac", "overrides"])
        backend = str(d.get("backend", "tracktrack"))
        allowed = sorted(TRACKERS) + ["greedy"]
        if backend not in allowed:
            raise ConfigError(f"tracking.backend must be one of {allowed}")
        overrides = d.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ConfigError("tracking.overrides must be a mapping of tracker parameters")
        margin = int(d.get("border_margin_px", 0))
        if margin < 0:
            raise ConfigError(
                f"tracking.border_margin_px must be 0 or more (got {margin}); it is "
                "how far from the frame edge still counts as being at it, and "
                "'do not exclude border detections at all' is written as 0"
            )
        border_frac = float(d.get("border_max_frac", 0.0))
        if not 0.0 <= border_frac <= 1.0:
            raise ConfigError(
                f"tracking.border_max_frac must be in [0, 1] (got {border_frac}); it "
                "is the share of ONE CLIP's detections allowed within "
                "tracking.border_margin_px of the edge, so 0 discards a clip that "
                "reaches the edge at all and 1 discards none. The track carries on "
                "either way: it is clips that go, not tracks"
            )
        return cls(
            backend=backend,
            tracker_cfg=str(d.get("tracker_cfg", "") or ""),
            min_track_length=int(d.get("min_track_length", 50)),
            greedy_max_dist_px=float(d.get("greedy_max_dist_px", 25.0)),
            border_margin_px=margin,
            border_max_frac=border_frac,
            overrides=dict(overrides),
        )


@dataclass
class Features:
    """Per-detection detector embeddings, pooled from the same forward pass."""

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
class EmbeddingSpec:
    """One per-detection embedding input: ``detector`` or a ``file`` per movie."""

    name: str
    source: str = "detector"
    # File source only: a path with ``{movie_id}``, relative to the config.
    path: str | None = None

    @property
    def group(self) -> str:
        return f"emb_{self.name}"


def _embeddings(value) -> list[EmbeddingSpec]:
    """``suggest.embeddings``: a list of ``{name, source, path}`` mappings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("suggest.embeddings must be a list of {name, source, path}")
    out: list[EmbeddingSpec] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ConfigError(f"suggest.embeddings entry {entry!r} is not a mapping")
        _known("suggest.embeddings", entry, ["name", "source", "path"])
        name = str(entry.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9]+", name):
            raise ConfigError(
                f"suggest.embeddings name {name!r} must be letters and digits only"
            )
        source = str(entry.get("source", "detector"))
        if source not in ("detector", "file"):
            raise ConfigError(
                f"suggest.embeddings {name}: source must be 'detector' or 'file', "
                f"got {source!r}"
            )
        path = entry.get("path")
        if source == "file" and (not path or "{movie_id}" not in str(path)):
            raise ConfigError(
                f"suggest.embeddings {name}: a file source needs a path containing "
                "{movie_id}, e.g. embeddings/{movie_id}.parquet"
            )
        if any(e.name == name for e in out):
            raise ConfigError(f"suggest.embeddings names {name!r} twice")
        out.append(EmbeddingSpec(name, source, str(path) if path else None))
    return out


_OLD_CNN = re.compile(r"^cnn(\d+)_(mean|std|delta)$")


def _migrate_cnn(groups: list[str], embeddings: list[EmbeddingSpec],
                 columns) -> tuple[list[str], object]:
    """Map the retired ``cnn`` group onto a ``detector`` embedding named ``det``."""
    if "cnn" not in groups:
        return groups, columns
    warnings.warn(
        "suggest.feature_groups 'cnn' is now the detector embedding 'det'; "
        "reading it as suggest.embeddings [{name: det, source: detector}] and "
        "group emb_det. Retrain models fitted on cnn columns.",
        stacklevel=4,
    )
    if not any(e.name == "det" for e in embeddings):
        embeddings.append(EmbeddingSpec("det", "detector"))
    groups = list(dict.fromkeys("emb_det" if g == "cnn" else g for g in groups))
    if columns is not None and not isinstance(columns, str):
        columns = [_OLD_CNN.sub(r"emb_det_\1_\2", str(c)) for c in columns]
    return groups, columns


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
    # silently change the fit as well. The values are CatBoost's own defaults,
    # which are tuned against far more datasets than this project has, and
    # `learning_rate: auto` is the one that cannot be written as a number:
    # CatBoost derives it from the row count and the iteration count, and the
    # right rate for 200 labelled clips is not the right rate for 5000.
    iterations: int = 1000
    depth: int = 6
    learning_rate: float | str = "auto"
    l2_leaf_reg: float = 3.0
    # NOT a CatBoost default, and deliberately so: it describes this data
    # rather than the estimator. One behaviour dominates a behaviour log, and
    # an unweighted model scores well by answering with it.
    balance_classes: bool = True
    # Which processor fits the trees: cpu, gpu or auto. CatBoost only.
    # CatBoost's GPU uses 128 feature borders against the CPU's 254, so the two
    # fit different models. It is also slower below roughly 50 000 rows.
    device: str = "cpu"
    # Which feature families the model may use. Absent, every static family
    # plus every declared embedding.
    feature_groups: list[str] = field(
        default_factory=lambda: ["kinematic", "pose", "waveform"])
    # Per-detection embedding inputs, each a feature group ``emb_<name>``.
    embeddings: list[EmbeddingSpec] = field(default_factory=list)
    # Frames of kinematic context centred on each window; 0 is off.
    context_frames: int = 0
    # The finest of the three cuts: the columns themselves, by name. None is not
    # "no columns" but "do not cut here", which is what keeps the default the
    # behaviour every project on disk was fitted under. It NARROWS what the
    # groups selected rather than replacing it, so a name whose family is not in
    # `feature_groups` is a contradiction and is refused rather than quietly
    # turning that family back on.
    feature_columns: list[str] | None = None
    # -- which clips the annotator is handed next (:mod:`trackanno.learn.select`) --- #
    # 'off' keeps the GUI's seeded shuffle; 'margin' puts the clips the model is
    # least sure about first, in batches re-ranked after each refit.
    select_rule: str = "off"
    select_batch_size: int = 20

    @property
    def selection_on(self) -> bool:
        return self.select_rule != "off"

    def audit_by_hash(self, clip_id: str) -> bool:
        """Whether ``clip_id``'s track hashes into the audit holdout (fallback).

        Hashed on ``movie_id::track_id`` so every clip of a track lands on the
        same side. Track ids change on re-detection, so a labelled clip's
        recorded membership (``store.AUDIT_COLUMN``) wins over this. Uses its
        own seed so varying ``seed`` does not redraw the holdout.
        """
        frac = float(self.audit_fraction)
        if frac <= 0:
            return False
        digest = hashlib.blake2b(
            f"{self.audit_seed}:{audit_track_key(clip_id)}".encode(), digest_size=8
        ).digest()
        return (int.from_bytes(digest, "big") % 10_000) < frac * 10_000

    @classmethod
    def parse(cls, d: dict) -> "Suggest":
        _known(
            "suggest", d,
            ["enabled", "model", "pca_components", "retrain_every", "min_per_class",
             "min_macro_f1", "audit_fraction", "audit_visible", "seed", "audit_seed",
             "min_confidence",
             "iterations", "depth", "learning_rate", "l2_leaf_reg", "balance_classes",
             "device",
             "feature_groups", "feature_columns", "embeddings", "context_frames",
             "select_rule", "select_batch_size"],
        )
        model = str(d.get("model", "catboost"))
        if model not in ("hist", "catboost"):
            raise ConfigError("suggest.model must be 'hist' or 'catboost'")
        depth = int(d.get("depth", 6))
        if not 1 <= depth <= 10:
            raise ConfigError(
                "suggest.depth must be 1-10. CatBoost trees are symmetric, so depth "
                "alone sets the width: 2**depth leaves, 64 at the default 6. Past "
                "that a few hundred hand-labelled clips are memorised rather than "
                "learned, which the reliability gate then refuses to suggest from."
            )
        from .measure.clipfeatures import FEATURE_GROUPS

        embeddings = _embeddings(d.get("embeddings"))
        context_frames = int(d.get("context_frames", 0) or 0)
        if context_frames < 0:
            raise ConfigError("suggest.context_frames must be 0 (off) or a frame count")
        groups = d.get("feature_groups")
        explicit = groups is not None
        if groups is None:
            groups = [g for g in FEATURE_GROUPS if g != "context" or context_frames > 0]
            groups += [e.group for e in embeddings]
        else:
            groups = [str(g) for g in groups]
        groups, columns = _migrate_cnn(groups, embeddings, d.get("feature_columns"))
        allowed = list(FEATURE_GROUPS) + [e.group for e in embeddings]
        unknown = [g for g in groups if g not in allowed]
        if unknown:
            raise ConfigError(
                f"suggest.feature_groups has unknown entries {unknown}; "
                f"choose from {allowed} (emb_<name> needs a suggest.embeddings entry)"
            )
        if not groups:
            raise ConfigError(
                "suggest.feature_groups is empty, so the model would have no "
                f"inputs at all; choose at least one of {allowed}"
            )
        unused = [e.group for e in embeddings if e.group not in groups]
        if context_frames > 0 and "context" not in groups:
            unused.append("context")
        if explicit and unused:
            warnings.warn(
                f"{unused} are configured (suggest.embeddings, suggest.context_frames) "
                "but suggest.feature_groups does not list them, so the model will "
                "not use them",
                stacklevel=3,
            )
        chosen = _feature_columns(columns, groups)
        device = str(d.get("device", "cpu")).lower()
        if device not in ("cpu", "gpu", "auto"):
            raise ConfigError(
                f"suggest.device must be 'cpu', 'gpu' or 'auto', got {device!r}. "
                "'auto' uses a GPU where CatBoost sees one, so the same project "
                "can be fitted two ways on two machines; 'gpu' raises instead."
            )
        if device == "gpu" and model != "catboost":
            raise ConfigError(
                f"suggest.device is 'gpu' but suggest.model is {model!r}. "
                "scikit-learn's HistGradientBoosting is CPU-only, so the setting "
                "would do nothing. Use suggest.model 'catboost', or device 'cpu'."
            )
        audit = float(d.get("audit_fraction", 0.10))
        if not 0.0 <= audit < 1.0:
            raise ConfigError("suggest.audit_fraction must be in [0, 1)")
        enabled = bool(d.get("enabled", True))
        rule = _selection_rule(d)
        batch = int(d.get("select_batch_size", 20))
        if batch < 1:
            raise ConfigError(
                f"suggest.select_batch_size must be at least 1 (got {batch}); it is "
                "how many clips one batch hands the annotator, and a batch of none "
                "is spelled by setting suggest.select_rule to 'off'"
            )
        rate = _optional_auto(
            d.get("learning_rate", "auto"), "suggest.learning_rate",
            "a rate or 'auto' (the estimator's own, which CatBoost derives from "
            "the number of rows and iterations rather than fixing in advance)",
        )
        if rate is not None and not 0.0 < float(rate) <= 1.0:
            raise ConfigError(
                f"suggest.learning_rate must be in (0, 1], got {float(rate)}; it is "
                "the share of each tree's correction that is kept, and 'let the "
                "estimator choose' is written as 'auto'"
            )
        if rule != "off" and audit <= 0.0:
            raise ConfigError(
                f"suggest.select_rule is {rule!r} but suggest.audit_fraction is "
                f"{audit:g}, so this project has no audit holdout. Selection hands "
                "the annotator whichever clips the model is least sure about, which "
                "makes the labelled set a sample assembled to be hard: every "
                "accuracy computed over it afterwards is lower than the truth by an "
                "amount nobody can estimate without knowing how the set was "
                "assembled. The holdout is what keeps one unbiased estimate alive - "
                "clips drawn by a coin, never ranked - and at audit_fraction 0 there "
                "is no such clip to draw. Set suggest.audit_fraction above 0 (0.10 "
                "holds out a tenth), or set suggest.select_rule to 'off'."
            )
        if rule != "off" and not enabled:
            raise ConfigError(
                f"suggest.select_rule is {rule!r} but suggest.enabled is false, so "
                "no model is ever loaded and there is nothing to rank the clips "
                "with. The order would silently stay the GUI's shuffle, which is "
                "the setting quietly not applying. Set suggest.enabled to true, or "
                "suggest.select_rule to 'off'."
            )
        return cls(
            enabled=enabled,
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
            iterations=max(10, int(d.get("iterations", 1000))),
            depth=depth,
            learning_rate="auto" if rate is None else float(rate),
            l2_leaf_reg=float(d.get("l2_leaf_reg", 3.0)),
            balance_classes=bool(d.get("balance_classes", True)),
            device=device,
            feature_groups=groups,
            embeddings=embeddings,
            context_frames=context_frames,
            feature_columns=chosen,
            select_rule=rule,
            select_batch_size=batch,
        )


@dataclass
class Spectral:
    """The window a BEAT is measured over, which is not the window it is labelled over.

    A clip's length is chosen so a person can judge one behaviour, not for a
    Fourier transform: at 25 frames and 100 fps one FFT bin is 4 Hz and the
    resolvability floor lands at 12 Hz, above most of these cells' beats, so the
    window is longer than the clip and CENTRED on it. Past about 75 frames
    accuracy falls again, because behaviour changes inside a track and a long
    window averages over the reversal. Raise it if your cells hold a direction
    longer, lower it if they switch faster than ~0.75 s.
    """

    window: int = 75
    # Frames between successive spectral estimates. The series is computed once
    # per track, so a small hop is cheap and localises a reversal to within one
    # hop.
    hop: int = 5
    # Support of the wave_flow columns, which read the direction from the phase
    # GRADIENT and so need no cycles - the window above is what the FREQUENCY
    # needs. Keep it well under a beat period: it is the shortest stretch over
    # which a direction is claimed, so it sets how sharply a reversal lands.
    flow_frames: int = 5

    @classmethod
    def parse(cls, d: dict) -> "Spectral":
        _known("spectral", d, ["window", "hop", "flow_frames"])
        window = int(d.get("window", 75))
        hop = int(d.get("hop", 5))
        flow_frames = int(d.get("flow_frames", 5))
        if window < 8:
            raise ConfigError(
                "spectral.window must be at least 8 frames; below that there is no "
                "band left to search once the resolvability floor is applied"
            )
        if not 1 <= hop <= window:
            raise ConfigError("spectral.hop must be between 1 and spectral.window")
        if flow_frames < 3:
            raise ConfigError(
                f"spectral.flow_frames must be at least 3 (got {flow_frames}); it is "
                "the stretch a direction is read over, and a centred difference in "
                "time needs three frames to exist"
            )
        return cls(window=window, hop=hop, flow_frames=flow_frames)


@dataclass
class Predict:
    """How the labelling pass scores every frame and cuts it into bouts.

    Windows of ``clips.frames`` start every ``window_stride`` frames over every
    track; their class probabilities are averaged per frame and cut into bouts.
    See :mod:`trackanno.infer.densepredict`.
    """

    # Frames between one window's start and the next. 'auto' is clips.frames,
    # an exact tiling; smaller overlaps the windows and pins a transition to
    # within one stride.
    window_stride: int | str = "auto"
    # Detections a dense window needs before its features mean anything. Two is
    # the floor at which ANY motion feature can exist. Deliberately NOT
    # clips.min_clip_frames, which exists so an annotator gets a watchable video.
    min_frames: int = 2
    # The shortest bout worth believing, in frames: shorter runs are absorbed
    # into their neighbours. 0 keeps the plain per-frame argmax runs.
    min_bout_frames: int = 0

    def resolved_stride(self, clips: "Clips") -> int:
        """``window_stride`` with ``auto`` filled in: ``clips.frames``."""
        if isinstance(self.window_stride, str):
            return max(1, int(clips.frames))
        return max(1, int(self.window_stride))

    @classmethod
    def parse(cls, d: dict) -> "Predict":
        removed = sorted(k for k in ("decode", "self_transition") if k in d)
        if removed:
            warnings.warn(
                f"ignoring predict.{', predict.'.join(removed)}: Viterbi decoding was "
                "removed, and predict.min_bout_frames is now a hard floor on bout length",
                stacklevel=3,
            )
            d = {k: v for k, v in d.items() if k not in removed}
        _known("predict", d, ["window_stride", "min_frames", "min_bout_frames"])
        stride = d.get("window_stride", "auto")
        if isinstance(stride, str):
            if stride != "auto":
                raise ConfigError(
                    "predict.window_stride must be a frame count or 'auto' "
                    "(clips.frames, so the windows tile exactly)"
                )
        elif int(stride) < 1:
            raise ConfigError(
                f"predict.window_stride must be at least 1 (got {int(stride)})"
            )
        min_frames = int(d.get("min_frames", 2))
        if min_frames < 1:
            raise ConfigError(
                f"predict.min_frames must be at least 1 (got {min_frames}); a window "
                "with no detections in it has nothing to measure"
            )
        min_bout = int(d.get("min_bout_frames", 0))
        if min_bout < 0:
            raise ConfigError(
                f"predict.min_bout_frames must be 0 or more (got {min_bout})"
            )
        return cls(
            window_stride=stride if isinstance(stride, str) else int(stride),
            min_frames=min_frames,
            min_bout_frames=min_bout,
        )


@dataclass
class ModelSource:
    """Where the detection weights come from, when not a file on this disk.

    ``source`` is a model id from the registry (see :mod:`trackanno.video.models`) or
    an ``https://`` URL to a ``.pt``. Empty means ``paths.model``, which is what
    a project made from a local file uses.

    ``sha256`` is written back on the first successful fetch and pins the
    weights from then on. A registry that later moves a tag to different weights
    is then a loud mismatch rather than a silent change of model under an
    unchanged name.
    """

    source: str = ""
    version: str = ""
    sha256: str = ""

    @property
    def is_remote(self) -> bool:
        return bool(self.source.strip())

    @property
    def is_url(self) -> bool:
        return self.source.strip().lower().startswith(("http://", "https://"))

    @classmethod
    def parse(cls, d: dict) -> "ModelSource":
        _known("model", d, ["source", "version", "sha256"])
        source = str(d.get("source", "") or "").strip()
        if source.lower().startswith("http://"):
            from .video.models import is_trusted_url

            if not is_trusted_url(source):
                raise ConfigError(
                    "model.source must be https, not http: weights are executed "
                    "when they are loaded, so they may not be fetched over a "
                    "channel that can be rewritten in transit"
                )
        digest = str(d.get("sha256", "") or "").strip().lower()
        if digest and (len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest)):
            raise ConfigError("model.sha256 must be a 64-character hex digest, or empty")
        return cls(
            source=source,
            version=str(d.get("version", "") or "").strip(),
            sha256=digest,
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

    @classmethod
    def parse(cls, d: dict) -> "Clips":
        _known(
            "clips", d,
            ["frames", "stride", "window_mode", "follow", "crop_from", "smooth_window", "pad",
             "crop_size", "max_size", "min_clip_frames", "max_clips_per_track", "context_tracks"],
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
    show_kymograph: bool = True
    kymo_smooth_px: float = 0.6
    kymo_sign: str = "canonical"
    follow: str = "centred"
    view_zoom: float = 1.0
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
             "show_context_tracks", "show_kymograph",
             "kymo_smooth_px", "kymo_sign",
             "follow",
             "view_zoom",
             "prefetch", "resume", "annotator", "shuffle", "shuffle_seed"],
        )
        resume = str(d.get("resume", "first_unannotated"))
        if resume not in ("first_unannotated", "last_position"):
            raise ConfigError("ui.resume must be 'first_unannotated' or 'last_position'")
        theme = str(d.get("theme", "dark"))
        if theme not in ("dark", "light", "system"):
            raise ConfigError("ui.theme must be 'dark', 'light' or 'system'")

        kymo_sign = str(d.get("kymo_sign", "canonical"))
        if kymo_sign == "absolute":
            warnings.warn("ui.kymo_sign: 'absolute' was removed; using 'canonical'",
                          stacklevel=3)
            kymo_sign = "canonical"
        follow = str(d.get("follow", "centred"))
        if follow == "smooth":
            warnings.warn("ui.follow: 'smooth' was removed; using 'centred'", stacklevel=3)
            follow = "centred"
        if follow not in ("centred", "locked"):
            raise ConfigError(
                "ui.follow must be 'centred' (one view on the track's median position - all "
                "motion visible) or 'locked' (re-centred every frame - translation invisible)"
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
            show_kymograph=bool(d.get("show_kymograph", True)),
            kymo_smooth_px=max(0.0, float(d.get("kymo_smooth_px", 0.6))),
            kymo_sign=_one_of(kymo_sign, ("raw", "canonical"), "ui.kymo_sign"),
            follow=follow,
            view_zoom=max(1.0, float(d.get("view_zoom", 1.0))),
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
    "uncertain": ["u"],
    "clear": ["Backspace"],
    "undo": ["Ctrl+Z"],
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
        model=ModelSource.parse(_section(data, "model")),
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

    # Before embeddings, an unset feature_groups meant the detector's cnn group.
    old = data.get("suggest")
    if (isinstance(old, dict) and "feature_groups" not in old
            and "embeddings" not in old and cfg.features.enabled):
        warnings.warn(
            "detector embeddings are now opt-in; add suggest.embeddings "
            "[{name: det, source: detector}] and emb_det to feature_groups to use them",
            stacklevel=2,
        )

    # `task: pose` with no keypoints listed is INCOMPLETE, not invalid, and must
    # not raise here: the count is readable from the checkpoint.
    # `Config.needs_keypoints` carries the state and TiledDetector fills it in.

    # The dense window is clips.frames, so the stride check spans two sections.
    stride = cfg.predict.resolved_stride(cfg.clips)
    if stride > int(cfg.clips.frames):
        raise ConfigError(
            f"predict.window_stride ({stride}) is longer than the window "
            f"(clips.frames, {cfg.clips.frames}), which would leave frames no "
            "window scores. Leave it at 'auto' to tile, or go below to overlap"
        )

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


# --------------------------------------------------------------------------- #
# defaults, for a form that has to show a key the file does not mention
# --------------------------------------------------------------------------- #
_SECTION_TYPES = {
    "nd2": "ND2Options", "tiff": "TiffOptions", "normalize": "Normalize",
    "detection": "Detection", "model": "ModelSource", "tiling": "Tiling",
    "tracking": "Tracking", "keypoints": "Keypoints", "features": "Features",
    "spectral": "Spectral", "suggest": "Suggest", "predict": "Predict",
    "clips": "Clips", "ui": "UI", "render": "Render", "paths": "Paths",
}

_DEFAULTS_CACHE: dict[str, dict] = {}


def section_defaults(name: str) -> dict:
    """What the loader uses for the keys ``config.yaml`` does not mention.

    Read by running the section's own ``parse`` over an empty mapping, so it
    cannot drift from what a project actually loads with. That matters because
    the generated file documents most settings and leaves the rest out entirely:
    ``suggest.iterations`` is not in it, and 1000 is still what gets fitted.

    Path-valued defaults are dropped. A path only means something relative to a
    project folder, and this has no project.
    """
    if name in _DEFAULTS_CACHE:
        return _DEFAULTS_CACHE[name]
    cls = globals().get(_SECTION_TYPES.get(name, ""))
    values: dict[str, Any] = {}
    if cls is not None:
        try:
            parsed = cls.parse({}, Path(".")) if name in ("paths", "render") else cls.parse({})
        except Exception:
            parsed = None
        if parsed is not None:
            values = {
                f.name: getattr(parsed, f.name)
                for f in dataclasses_fields(parsed)
                if not isinstance(getattr(parsed, f.name), Path)
            }
    _DEFAULTS_CACHE[name] = values
    return values
