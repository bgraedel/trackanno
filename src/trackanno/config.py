"""Typed configuration loaded from ``config.yaml``.

Plain dataclasses rather than pydantic: the schema is small, and validation
errors here should read like advice, not like a stack trace.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["Config", "load_config", "CategoryGroup", "CategoryOption",
           "ModelSource", "RESERVED_GROUP_NAMES"]

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


def _spelled_out(written: int | str, value: int) -> str:
    """``50``, or ``50 (from 'auto')`` when the user never typed that number.

    A complaint about two settings that disagree is unreadable if one of them
    was written as ``auto`` and the message quotes only what it resolved to.
    """
    return f"{value}, from 'auto'" if isinstance(written, str) else str(value)


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


def _window_scales(value) -> list[int]:
    """``predict.window_scales`` as frame counts, or a refusal naming the choices.

    Only the checks a single section can make: that the entries are frame counts
    at all, and that none is repeated. Whether the list CONTAINS
    ``predict.window_frames`` needs two sections and so belongs to
    :func:`load_config`. A repeat is refused rather than deduplicated: the scales
    are averaged into the per-frame posterior with equal weight each, so listing
    one twice would silently give it double weight.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            "predict.window_scales must be a list of window lengths in frames, "
            f"like [25, 50, 100] (got {value!r}). Leave it empty to score at "
            "predict.window_frames alone, which is what every earlier version did"
        )
    out: list[int] = []
    for entry in value:
        try:
            frames = int(entry)
        except (TypeError, ValueError):
            raise ConfigError(
                f"predict.window_scales entry {entry!r} is not a frame count; "
                "each entry is how many frames one window of that scale spans"
            ) from None
        if frames < 1:
            raise ConfigError(
                f"predict.window_scales entry {frames} must be at least 1; it is "
                "how many frames one window spans, and a window spanning none of "
                "them has nothing to measure"
            )
        if frames in out:
            raise ConfigError(
                f"predict.window_scales lists {frames} twice. Every scale is "
                "averaged into the per-frame posterior with the same weight, so a "
                "repeated scale would count twice and quietly outvote the others - "
                "which is the arbitrary weighting multi-scale scoring is meant to "
                "remove, not to introduce. List each length once"
            )
        out.append(frames)
    return out


def _selection_rules(d: dict) -> tuple[str, str]:
    """``(select_rule, select_target_rule)``, or a refusal naming the choices.

    Both names are validated against :mod:`trackanno.select` itself rather than
    against a list copied into this file, which would rot silently in either
    direction.
    """
    from .select import TARGET_RULES, UNCERTAINTY_RULES

    raw = d.get("select_rule", "off")
    if isinstance(raw, bool):
        # YAML reads a bare `off` (and `no`, and `false`) as a boolean, so the
        # value this key is most likely to be given arrives as the wrong type.
        raise ConfigError(
            "suggest.select_rule came through as a YAML boolean, which is what an "
            "unquoted `off`/`on` becomes. Quote it - select_rule: \"off\" - so it "
            "stays the word it looks like on the page."
        )
    rule = str(raw)
    if rule != "off" and rule not in UNCERTAINTY_RULES:
        raise ConfigError(
            f"suggest.select_rule must be \"off\" (clips reach the annotator in the "
            f"GUI's own shuffled order, as they always have) or one of "
            f"{sorted(UNCERTAINTY_RULES)}, got {rule!r}. They rank differently and "
            "the difference is not cosmetic: least_confidence reads only the top "
            "probability, margin the gap between the top two, entropy the whole "
            "vector and so mostly its tail."
        )
    target_rule = str(d.get("select_target_rule", "target_runner_up"))
    if target_rule not in TARGET_RULES:
        raise ConfigError(
            f"suggest.select_target_rule must be one of {sorted(TARGET_RULES)}, got "
            f"{target_rule!r}. 'target_confident' fills a starved class fastest and "
            "cannot move its boundary; 'target_runner_up' picks the clips sitting on "
            "that boundary, which fills the count more slowly and is the only one of "
            "the two whose labels move the decision surface."
        )
    return rule, target_rule


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
    # A second, ORTHOGONAL cut through the same columns. `feature_groups` asks
    # which instrument a column came from; this asks how its value behaves when
    # the window length changes, which differs inside every group - `beat_*` is
    # length-invariant and its groupmate `body_path_length` is not. 'all' is
    # every column the groups selected; 'length_invariant' keeps only the ones
    # that do not scale with the window, which is what makes ONE model applicable
    # at several window lengths (`predict.window_scales`). That costs real
    # signal, so it is opted into, never a default.
    feature_scope: str = "all"

    # -- which clips the annotator is handed next (:mod:`trackanno.select`) --- #
    # Every key below is prefixed `select_` rather than living in a section of
    # its own, because none of them mean anything without the model this section
    # fits. 'off' is the default: no ranking runs and the GUI keeps its seeded
    # shuffle. Selection stops the labelled set being a random sample of the
    # clips, which changes what every statistic computed over it means, so it is
    # something a project opts into.
    select_rule: str = "off"
    # Clips per batch. Twice `retrain_every` so a batch outlives the refit that
    # produced it, and large enough that the reserved audit share is a whole
    # slot: `int(share * n + 0.5)` rounds a batch of 4 at the 0.10 default down
    # to no draw at all.
    select_batch_size: int = 20
    # Share of each batch drawn uniformly at random from the audit holdout.
    # 'auto' is `audit_fraction`: draw audit clips at any other rate and the
    # labelled audit subset stops being proportional to the audit set, so the one
    # unbiased accuracy in the tool describes a differently-composed sample.
    select_audit_share: float | str = "auto"
    # Frames two ranked clips of one track must be apart. 'auto' is
    # `clips.frames`: closer than one clip length and two windows of a track
    # share footage, so the second charges the annotator again for what the first
    # already showed. Pairs exactly that far apart are kept.
    select_frame_gap: int | str = "auto"
    # Labels a class needs before slots are spent hunting for it. 0 (off) even
    # when selection is on: targeting skews the label marginals, and both
    # `train_group`'s gates - Cohen's kappa and the majority baseline - are
    # computed against those marginals, so it can turn the suggester silent on a
    # model that has not changed.
    select_target_quota: int = 0
    select_target_rule: str = "target_runner_up"

    # -- what the selector actually runs with, 'auto' spelled out ----------- #
    @property
    def selection_on(self) -> bool:
        return self.select_rule != "off"

    def resolved_audit_share(self) -> float:
        """``select_audit_share`` with ``auto`` filled in: ``audit_fraction``.

        One number, two places it has to hold: `audit_fraction` decides which
        clips are in the holdout, and this decides how often the batch draws from
        it. Let them differ and the labelled audit subset fills at the wrong
        rate, either stalling `audit_accuracy` or eating slots the ranking was
        supposed to spend.
        """
        if isinstance(self.select_audit_share, str):
            return float(self.audit_fraction)
        return float(self.select_audit_share)

    def audit_by_hash(self, clip_id: str) -> bool:
        """Whether ``clip_id`` hashes into the audit holdout - the FALLBACK answer.

        Fallback, because this answer is not stable: ``clip_id`` embeds the track
        id and re-running detection renumbers every track, so every id changes
        and this redraws the holdout from scratch. A LABELLED clip's membership
        is therefore written down the first time it is asked
        (``store.AUDIT_COLUMN``) and this is consulted only where no record
        exists, on an unlabelled clip whose membership is still free to move.

        It lives on the config because two modules need it and neither can own
        it: the store stamps membership into the CSV at label time and
        :mod:`trackanno.suggest` reads the stamp back.

        Its own seed, not ``seed``: ``seed`` exists to be varied to check CV
        stability, and doing that must not redraw the holdout.
        """
        frac = float(self.audit_fraction)
        if frac <= 0:
            return False
        digest = hashlib.blake2b(
            f"{self.audit_seed}:{clip_id}".encode(), digest_size=8
        ).digest()
        return (int.from_bytes(digest, "big") % 10_000) < frac * 10_000

    def resolved_frame_gap(self, clips: "Clips") -> int:
        """``select_frame_gap`` with ``auto`` filled in: ``clips.frames``.

        Two windows of one track less than a clip length apart overlap, so a
        human judging both judges some frames twice. Anything wider is a claim
        about how fast behaviour changes, which is per-project and left to be
        typed.
        """
        if isinstance(self.select_frame_gap, str):
            return max(0, int(clips.frames))
        return max(0, int(self.select_frame_gap))

    @classmethod
    def parse(cls, d: dict) -> "Suggest":
        _known(
            "suggest", d,
            ["enabled", "model", "pca_components", "retrain_every", "min_per_class",
             "min_macro_f1", "audit_fraction", "audit_visible", "seed", "audit_seed",
             "min_confidence",
             "iterations", "depth", "learning_rate", "l2_leaf_reg", "balance_classes",
             "feature_groups", "feature_scope",
             "select_rule", "select_batch_size", "select_audit_share",
             "select_frame_gap", "select_target_quota", "select_target_rule"],
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
        scope = str(d.get("feature_scope", "all"))
        if scope not in ("all", "length_invariant"):
            raise ConfigError(
                "suggest.feature_scope must be 'all' (every column the feature "
                "groups selected) or 'length_invariant' (only the columns whose "
                "value does not scale with the window length, which is what lets "
                "one model be applied at several predict.window_scales)"
            )
        audit = float(d.get("audit_fraction", 0.10))
        if not 0.0 <= audit < 1.0:
            raise ConfigError("suggest.audit_fraction must be in [0, 1)")
        enabled = bool(d.get("enabled", True))
        rule, target_rule = _selection_rules(d)
        batch = int(d.get("select_batch_size", 20))
        if batch < 1:
            raise ConfigError(
                f"suggest.select_batch_size must be at least 1 (got {batch}); it is "
                "how many clips one batch hands the annotator, and a batch of none "
                "is spelled by setting suggest.select_rule to 'off'"
            )
        share = _optional_auto(
            d.get("select_audit_share", "auto"), "suggest.select_audit_share",
            "a share of the batch or 'auto' (suggest.audit_fraction, so the batch "
            "draws audit clips at the rate the project contains them)",
        )
        if share is not None and not 0.0 <= float(share) <= 1.0:
            raise ConfigError(
                f"suggest.select_audit_share must be in [0, 1], got {float(share)}; "
                "it is the share of each batch drawn at random from the audit "
                "holdout, and 1.0 - every slot a random draw - is already the "
                "degenerate end of it"
            )
        gap = _optional_auto(
            d.get("select_frame_gap", "auto"), "suggest.select_frame_gap",
            "a number of frames or 'auto' (clips.frames, one clip length, which "
            "is exactly the separation below which two windows of one track "
            "share footage)",
        )
        if gap is not None and int(gap) < 0:
            raise ConfigError(
                f"suggest.select_frame_gap must be 0 or more (got {int(gap)}); 0 "
                "turns the diversity rule off, and a negative separation is not a "
                "thing to require"
            )
        quota = int(d.get("select_target_quota", 0))
        if quota < 0:
            raise ConfigError(
                f"suggest.select_target_quota must be 0 or more (got {quota}); it is "
                "the label count a class stops being starved at, and 'do not target "
                "at all' is written as 0"
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
            iterations=max(10, int(d.get("iterations", 400))),
            depth=depth,
            learning_rate=float(d.get("learning_rate", 0.05)),
            l2_leaf_reg=float(d.get("l2_leaf_reg", 3.0)),
            balance_classes=bool(d.get("balance_classes", True)),
            feature_groups=groups,
            feature_scope=scope,
            select_rule=rule,
            select_batch_size=batch,
            select_audit_share="auto" if share is None else float(share),
            select_frame_gap="auto" if gap is None else int(gap),
            select_target_quota=quota,
            select_target_rule=target_rule,
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
    """How much of each movie the bulk labelling pass covers, and how finely.

    ``manifest`` scores exactly the clips cut for annotation, a sampling chosen
    for a human's attention, so everything between two annotation windows comes
    out unlabelled and unpainted in the rendered movie. ``dense`` lays its own
    grid over every frame of every track - see :mod:`trackanno.densepredict`.

    By default that grid TILES, so a change of behaviour can only be reported at
    a window boundary and the error is up to a window long. ``window_stride``
    below the window length overlaps the windows and pins a transition to within
    one stride; ``window_scales`` scores at several window LENGTHS instead, so a
    behaviour shorter than one window is still visible and a long one is not cut
    into verdicts that can disagree. Both make several windows speak for one
    frame, which is what ``decode`` resolves, so both are only valid together
    with it. Several scales need one condition more: windows of different lengths
    carry different feature values under identical column names, so the model
    must have been fitted on columns that do not scale with the window
    (``suggest.feature_scope``).
    """

    coverage: str = "dense"
    window_frames: int | str = "auto"
    # Frames between one window's start and the next. 'auto' is window_frames,
    # an exact tiling: the only stride whose rows can be painted straight out of
    # predictions.parquet without a decode.
    window_stride: int | str = "auto"
    # Every window LENGTH the dense pass scores at, `window_frames` included.
    # Empty (the default) is the single length `window_frames`. One window length
    # is one guess about the timescale of behaviour: a transition shorter than it
    # is invisible, and a longer bout is chopped into independent verdicts that
    # can disagree. The cost is a feature build per scale, and the constraint is
    # severe - a model fitted at one length reads another length's features on
    # the wrong scale, so several scales are only coherent for a model fitted on
    # length-invariant columns (`suggest.feature_scope`), which `load_config`
    # enforces.
    window_scales: list[int] = field(default_factory=list)
    # Detections a dense window needs before its features mean anything. Two is
    # the floor at which ANY motion feature can exist. Deliberately NOT
    # clips.min_clip_frames, which exists so an annotator gets a watchable video
    # and would punch the holes dense coverage removes back into the export.
    # Every row carries n_present and present_frac for a stricter downstream bar.
    min_frames: int = 2
    # How the per-frame posterior becomes labels. 'none' keeps the one label a
    # window already carries, which is all a tiling grid needs. 'min_bout' cuts
    # the posterior into bouts and absorbs any bout shorter than min_bout_frames
    # into its neighbours. 'viterbi' asks for the most likely label SEQUENCE
    # under a sticky transition prior, so a short bout survives when its evidence
    # outweighs the cost of the two transitions it needs.
    decode: str = "none"
    # The shortest bout worth believing, in frames. A statement about the ANIMAL,
    # not about the model: the shortest episode of a behaviour you would accept
    # as real. 0 means no floor, and so does 1 - a decode's shortest possible run
    # is already one frame - which is why `decode: min_bout` refuses both.
    min_bout_frames: int = 0
    # viterbi only: the probability that a frame keeps the previous frame's
    # label. 'auto' is 1 - 1/min_bout_frames, the stickiness whose average run
    # is min_bout_frames long.
    self_transition: float | str = "auto"
    # Write the per-frame posterior beside the predictions. Off by default: it
    # is one row per frame per track rather than one per window, so it dwarfs
    # predictions.parquet, and it is mostly worth keeping while tuning a decode.
    save_posteriors: bool = False

    # -- 'auto' spelled out ------------------------------------------------- #
    def resolved_window(self, clips: "Clips") -> int:
        """``window_frames`` with ``auto`` filled in: min(frames, stride) of the clips.

        The same number :func:`trackanno.densepredict.window_length` computes,
        repeated here because the stride check in :func:`load_config` compares
        two numbers the user may never have typed, and loading a config must not
        drag in the dense pipeline (and pandas behind it) to learn one integer.

        The ``min`` is only meaningful while the annotation clips do NOT overlap:
        ``auto`` is trying to be both the length the model was fitted at
        (``clips.frames``) and a start that lands on the annotation grid (a
        multiple of ``clips.stride``), and ``clips.stride < clips.frames`` pulls
        those apart. :func:`load_config` refuses ``auto`` there for a dense
        project rather than resolving it; a ``coverage: manifest`` project still
        resolves, and ``predict.py --coverage dense`` on one of those is caught
        by :func:`trackanno.predict._check_trained_window`.
        """
        if isinstance(self.window_frames, str):
            return max(1, min(int(clips.frames), int(clips.stride)))
        return max(1, int(self.window_frames))

    def resolved_stride(self, clips: "Clips") -> int:
        """``window_stride`` with ``auto`` filled in: the window length itself."""
        if isinstance(self.window_stride, str):
            return self.resolved_window(clips)
        return max(1, int(self.window_stride))

    def resolved_scales(self, clips: "Clips") -> list[int]:
        """Every window length to score at, the base one first.

        Empty ``window_scales`` is ONE scale, ``resolved_window``, so no caller
        downstream has to branch on "is multi-scale on".

        The base length leads because it is what the rest of the pipeline is
        planned at; the rest follow in ASCENDING order rather than as typed, so
        two configs listing the same scales differently lay the dense table out
        in the same row order. ``load_config`` has already refused a non-empty
        list that does not contain the base length, so the deduplication below
        cannot silently drop a scale the user asked for.
        """
        base = self.resolved_window(clips)
        if not self.window_scales:
            return [base]
        return [base] + sorted({int(s) for s in self.window_scales} - {base})

    def resolved_self_transition(self) -> float:
        """``self_transition`` with ``auto`` filled in: 1 - 1/min_bout_frames.

        A chain that stays put with probability p has an expected run of
        1/(1 - p) frames, so the identity reads 'the prior whose average bout
        lasts min_bout_frames'. Only defined under ``decode: viterbi``, which is
        the only place ``parse`` guarantees ``min_bout_frames >= 2``.
        """
        if isinstance(self.self_transition, str):
            return 1.0 - 1.0 / float(self.min_bout_frames)
        return float(self.self_transition)

    @classmethod
    def parse(cls, d: dict) -> "Predict":
        _known(
            "predict", d,
            ["coverage", "window_frames", "window_stride", "window_scales",
             "min_frames", "decode", "min_bout_frames", "self_transition",
             "save_posteriors"],
        )
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
                "how many frames one window spans, and a window spanning none of "
                "them has nothing to measure"
            )
        stride = d.get("window_stride", "auto")
        if isinstance(stride, str):
            if stride != "auto":
                raise ConfigError(
                    "predict.window_stride must be a frame count or 'auto' (the same "
                    "number as predict.window_frames, so the windows tile exactly)"
                )
        elif int(stride) < 1:
            raise ConfigError(
                f"predict.window_stride must be at least 1 (got {int(stride)}); it is "
                "the step the window grid advances by, so 0 or less never terminates"
            )
        scales = _window_scales(d.get("window_scales"))
        min_frames = int(d.get("min_frames", 2))
        if min_frames < 1:
            raise ConfigError(
                f"predict.min_frames must be at least 1 (got {min_frames}); a window "
                "with no detections in it has nothing to measure"
            )
        decode = str(d.get("decode", "none"))
        if decode not in ("none", "min_bout", "viterbi"):
            raise ConfigError(
                "predict.decode must be 'none' (one label per window, exactly as it "
                "was predicted), 'min_bout' (cut the per-frame posterior into bouts, "
                "then absorb any bout shorter than predict.min_bout_frames into its "
                "neighbours) or 'viterbi' (the most likely label sequence under a "
                "sticky transition prior, which keeps a short bout the evidence "
                "really supports)"
            )
        min_bout = int(d.get("min_bout_frames", 0))
        if min_bout < 0:
            raise ConfigError(
                f"predict.min_bout_frames must be 0 or more (got {min_bout}); it is a "
                "length in frames, and 'no shortest bout at all' is written as 0, not "
                "as a negative length"
            )
        # 1 is refused for the same reason 0 is: `bouts.decode_min_bout` returns
        # the plain argmax whenever the floor is `<= 1`, since the shortest run a
        # decode can produce is already one frame long.
        if decode == "min_bout" and min_bout <= 1:
            raise ConfigError(
                f"predict.decode is 'min_bout' but predict.min_bout_frames is "
                f"{min_bout}, so the decode has no shortest bout to enforce and "
                "would hand back the labels it was given, unchanged. 0 says there "
                "is no floor at all; 1 sets the floor at the shortest bout that "
                "can exist, which excludes nothing - both come back out of the "
                "decode as the plain per-frame argmax. Set predict.min_bout_frames "
                "to the shortest episode you would accept as real (in frames, so 2 "
                "or more), or set predict.decode to 'none' and mean it"
            )
        stay = d.get("self_transition", "auto")
        if isinstance(stay, str):
            if stay != "auto":
                raise ConfigError(
                    "predict.self_transition must be a probability or 'auto' "
                    "(1 - 1/predict.min_bout_frames)"
                )
        elif not 0.0 < float(stay) < 1.0:
            raise ConfigError(
                f"predict.self_transition must be strictly between 0 and 1 (got "
                f"{float(stay)}); it is the probability that a frame keeps the "
                "previous frame's label, so 1.0 never leaves the state it starts in "
                "and labels the whole track with it, and 0.0 forbids staying, which "
                "makes every bout one frame long"
            )
        if decode == "viterbi" and isinstance(stay, str) and min_bout < 2:
            raise ConfigError(
                "predict.decode is 'viterbi' with predict.self_transition: auto, "
                "which stands for 1 - 1/predict.min_bout_frames and so needs "
                f"predict.min_bout_frames to be at least 2 (got {min_bout}): at 0 the "
                "identity divides by zero, and at 1 it gives 0.0, a prior that forbids "
                "two consecutive frames ever sharing a label. Set "
                "predict.min_bout_frames to the shortest bout you would believe, or "
                "give predict.self_transition a probability of its own"
            )
        if decode != "none" and coverage != "dense":
            raise ConfigError(
                f"predict.decode is {decode!r}, which needs predict.coverage: dense "
                f"(this config asks for {coverage!r}). A decode reads a posterior for "
                "every frame and re-cuts it into bouts; 'manifest' coverage scores "
                "only the clips that were cut for annotation, so most frames have no "
                "posterior at all and the bouts would be drawn across the gaps "
                "between clips. Set predict.coverage to 'dense', or predict.decode "
                "to 'none'"
            )
        return cls(
            coverage=coverage,
            window_frames=window if isinstance(window, str) else int(window),
            window_stride=stride if isinstance(stride, str) else int(stride),
            window_scales=scales,
            min_frames=min_frames,
            decode=decode,
            min_bout_frames=min_bout,
            self_transition=stay if isinstance(stay, str) else float(stay),
            save_posteriors=bool(d.get("save_posteriors", False)),
        )


@dataclass
class ModelSource:
    """Where the detection weights come from, when not a file on this disk.

    ``source`` is a model id from the registry (see :mod:`trackanno.models`) or
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
            from .models import is_trusted_url

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


def _refuse_unresolvable_window(cfg: Config) -> None:
    """Refuse ``predict.window_frames: auto`` over an OVERLAPPING annotation grid.

    ``auto`` promises two things at once and they are the same number only while
    the annotation clips tile:

    * the length the model was fitted at, which is ``clips.frames`` - that is
      what ``clip_features.parquet`` is measured over and what ``train`` stamps
      into the payload as ``trained_window``;
    * a start that lands on the annotation grid, i.e. a multiple of
      ``clips.stride``, so a dense window coinciding with an annotated one is
      handed the same ``clip_id`` and hand labels join to it.

    ``clips.stride < clips.frames`` pulls them apart, and ``min(frames, stride)``
    would then pick landing and drop the trained length, scoring every row at a
    scale the model was not fitted at and at normal confidence. So ``auto`` is
    refused and the number is asked for; the message names the two coherent
    answers, both of which leave every existing label alone. A project whose
    clips tile is untouched.

    Only under ``coverage: dense``, the one coverage this key governs.
    ``predict.py --coverage dense`` can still push a manifest config down the
    dense path, where :func:`trackanno.predict._check_trained_window` refuses the
    mismatch against the model's own recorded length.
    """
    if not isinstance(cfg.predict.window_frames, str):
        return
    if cfg.predict.coverage != "dense":
        return
    frames, stride = int(cfg.clips.frames), int(cfg.clips.stride)
    if stride >= frames:
        return
    raise ConfigError(
        f"predict.window_frames is 'auto', but this project's annotation clips "
        f"overlap - clips.stride is {stride} and clips.frames is {frames} - so "
        "'auto' has nothing to resolve to. It stands for "
        f"min(clips.frames, clips.stride), which is the longest dense window "
        "that is BOTH the length the model is fitted at (clips.frames, "
        f"{frames} - clip_features.parquet is measured over the annotation clip "
        "manifest) AND aligned with that manifest's own starts (every "
        f"clips.stride, {stride} frames). Those are the same number only while "
        f"the clips tile. Here 'auto' would give {min(frames, stride)}, and a "
        f"model fitted at {frames} frames would read every dense row on the "
        "wrong scale - path_length, straightness, every *_std and present_frac "
        "all scale with the window - at unchanged confidence. Write the length "
        "you want; neither answer touches a single existing label, because this "
        "decides what the PREDICTOR measures and not what anyone annotated:\n"
        f"  predict.window_frames: {frames}\n"
        "      Windows of the trained length, still tiling. Their starts stop "
        f"coinciding with the annotation clips' (every {frames} frames rather "
        f"than every {stride}), so dense rows no longer join to hand labels by "
        "clip_id: 'human_labelled' comes back false on all of them and "
        "--include-labelled has nothing to withhold.\n"
        f"  predict.window_frames: {frames} with predict.window_stride: {stride} "
        "and a predict.decode\n"
        "      The trained length AND the annotation grid's starts. The windows "
        "then overlap, which is what the decode is for - it collapses them into "
        "disjoint bouts, whose provenance is the frame-based "
        "human_overlap_frac rather than a clip_id join, so nothing is lost there "
        "either.\n"
        "Raising clips.stride back to clips.frames would also silence this, and "
        "is the one answer that does cost you something: it re-cuts the "
        "annotation grid, so every clip_id moves and the labels already made "
        "against the old ones stop matching anything."
    )


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

    # `task: pose` with no keypoints listed is INCOMPLETE, not invalid, and must
    # not raise here: the count is readable from the checkpoint.
    # `Config.needs_keypoints` carries the state and TiledDetector fills it in.

    # The dense window grid spans two sections: `auto` for the window length
    # means min(clips.frames, clips.stride), so the two numbers can only be
    # compared once `clips` has been parsed as well.
    _refuse_unresolvable_window(cfg)
    length = cfg.predict.resolved_window(cfg.clips)
    stride = cfg.predict.resolved_stride(cfg.clips)
    length_txt = _spelled_out(cfg.predict.window_frames, length)
    stride_txt = _spelled_out(cfg.predict.window_stride, stride)
    if stride > length:
        raise ConfigError(
            f"predict.window_stride ({stride_txt}) is longer than "
            f"predict.window_frames ({length_txt}), so the grid would step over "
            f"{stride - length} frame(s) between the end of one window and the "
            "start of the next. Those frames get no prediction and no colour in "
            "the render - the very hole predict.coverage: dense exists to close. "
            "Leave predict.window_stride at 'auto' to tile exactly, or take it "
            "below predict.window_frames to overlap the windows, which needs a "
            "predict.decode as well"
        )
    if stride < length and cfg.predict.decode == "none":
        raise ConfigError(
            f"predict.window_stride ({stride_txt}) is shorter than "
            f"predict.window_frames ({length_txt}), so dense windows overlap, but "
            "predict.decode is 'none' - these two keys conflict. Overlapping "
            "windows written straight into predictions.parquet give one track "
            "several rows painting the same range(f_start, f_end), and render.py "
            "resolves that by letting whichever row came last win: the exported "
            "colour of an overlapped frame would be decided by parquet row order "
            "rather than by the model. A decode is what makes overlap safe - it "
            "collapses the overlapping windows into one disjoint bout per stretch "
            "of frames. Set predict.decode to 'min_bout' or 'viterbi', or put "
            "predict.window_stride back to 'auto'"
        )

    # Multi-scale scoring spans three sections, and each refusal below is silent
    # skew if it is not made here: nothing downstream can tell a posterior
    # averaged over incommensurable scales from a good one.
    written_scales = list(cfg.predict.window_scales)
    if written_scales and length not in written_scales:
        raise ConfigError(
            f"predict.window_scales is {written_scales} but does not include "
            f"predict.window_frames ({length_txt}), which is the length the rest "
            "of the pipeline is planned at - densepredict.window_length, the "
            "trained-window check and the dense clip ids all use it. Two keys "
            "would then name the window length and disagree. Add "
            f"{length} to predict.window_scales, or set predict.window_frames to "
            "one of the scales you listed"
        )
    scales = cfg.predict.resolved_scales(cfg.clips)
    if len(scales) > 1:
        listed = ", ".join(str(s) for s in scales)
        if cfg.predict.coverage != "dense":
            raise ConfigError(
                f"predict.window_scales asks for {len(scales)} scales ({listed}) "
                f"but predict.coverage is {cfg.predict.coverage!r}. Several scales "
                "exist to be averaged onto the frames they cover, and only dense "
                "coverage lays a grid over every frame; 'manifest' scores the clips "
                "cut for annotation, whose length is clips.frames and nothing else. "
                "Set predict.coverage to 'dense', or empty predict.window_scales"
            )
        if cfg.predict.decode == "none":
            raise ConfigError(
                f"predict.window_scales asks for {len(scales)} scales ({listed}) "
                "but predict.decode is 'none'. Windows of several lengths cover "
                "the same frames, so this is the overlap problem again and worse: "
                "the renderer paints range(f_start, f_end) per track and lets "
                "whichever row came last win, so a frame's exported colour would be "
                "decided by parquet row order - and here the competing rows are not "
                "even the same length. The decode is what averages the scales into "
                "one per-frame posterior and cuts it into disjoint bouts. Set "
                "predict.decode to 'min_bout' or 'viterbi', or empty "
                "predict.window_scales"
            )
        if cfg.suggest.feature_scope != "length_invariant":
            raise ConfigError(
                f"predict.window_scales asks for {len(scales)} scales ({listed}) "
                "but suggest.feature_scope is "
                f"{cfg.suggest.feature_scope!r}, so the model is fitted on columns "
                "whose VALUES scale with the window length - path_length over 100 "
                "frames is simply twice the path_length over 50, under the same "
                "column name. Scoring one such model at several lengths feeds it "
                "rows on scales it was never fitted at and averages the results "
                "together, at unchanged confidence and with nothing in the output "
                "to show for it. Set suggest.feature_scope to 'length_invariant' "
                "and retrain, or empty predict.window_scales. (This is the config "
                "saying what the NEXT model will be fitted on; the model already on "
                "disk is checked against its own recorded columns when a labelling "
                "run starts.)"
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
