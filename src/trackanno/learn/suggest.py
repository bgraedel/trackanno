"""``trackanno suggest``: train the suggestion classifier (leave-movie-out CV) and serve suggestions.

A group's model stays silent unless its grouped-CV kappa and macro-F1 clear the bar.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..measure.clipfeatures import (
    CLIP_FEATURES_FILE, FEATURE_VERSION, build_clip_features, cached_clip_features,
    feature_columns,
)
from ..config import FIXED_SETTINGS, Config, load_config
from .select import select_batch
from ..annotate.store import AUDIT_COLUMN, decode_audit, read_audit_records

MODEL_FILE = "suggest_model.joblib"
REPORT_FILE = "suggest_report.json"
# Per labelled clip, the out-of-fold prediction of a model that never saw its movie.
CV_PREDICTIONS_FILE = "cv_predictions.parquet"

# Boosting hyperparameters shared by both backends; iterations, learning rate
# and l2_leaf_reg come from ``suggest.*``.
DEPTH = FIXED_SETTINGS["suggest.depth"]
BALANCE_CLASSES = FIXED_SETTINGS["suggest.balance_classes"]
SEED = FIXED_SETTINGS["suggest.seed"]
# The model stays silent below this cross-validated macro-F1.
MIN_MACRO_F1 = FIXED_SETTINGS["suggest.min_macro_f1"]
# Up to this many movies each is held out alone; above it they share this many folds.
MAX_FOLDS = 5
# Seeds tried for a stratified split before falling back to leave-one-movie-out.
_FOLD_SEEDS = 20
# Resamples behind kappa's lower bound, and the fewest usable ones it needs.
_KAPPA_RESAMPLES = 400
_KAPPA_MIN_RESAMPLES = 50

# Window-length-dependent columns, named in the mismatch message.
_LENGTH_DEPENDENT_EXAMPLES = ("path_length", "net_displacement", "speed_std")
# cnn* columns, which no current feature table holds.
_RETIRED_CNN = re.compile(r"^cnn\d+_(mean|std|delta)$")


@dataclass(frozen=True)
class AuditHoldout:
    """The audit clips, never suggested on nor trained on.

    Recorded membership (``store.AUDIT_COLUMN``) wins; unrecorded clips use the track hash.
    """

    cfg: Config
    recorded: Mapping[str, bool]

    @classmethod
    def from_labels(cls, cfg: Config, labels: pd.DataFrame | None) -> "AuditHoldout":
        """From an annotation table already in memory."""
        recorded: dict[str, bool] = {}
        if labels is not None and AUDIT_COLUMN in getattr(labels, "columns", ()):
            for clip_id, cell in zip(labels["clip_id"].astype(str),
                                     labels[AUDIT_COLUMN]):
                member = decode_audit(cell)
                if member is not None:
                    recorded[str(clip_id)] = member
        return cls(cfg, recorded)

    @classmethod
    def from_csv(cls, cfg: Config, path: str | Path) -> "AuditHoldout":
        """From the annotation CSV."""
        return cls(cfg, read_audit_records(path))

    @classmethod
    def by_hash(cls, cfg: Config) -> "AuditHoldout":
        """With nothing recorded: every clip answered by its track hash."""
        return cls(cfg, {})

    def __contains__(self, clip_id: object) -> bool:
        member = self.recorded.get(str(clip_id))
        if member is not None:
            return bool(member)
        return self.cfg.suggest.audit_by_hash(str(clip_id))

    def count(self, clip_ids) -> int:
        """How many of ``clip_ids`` are held out."""
        return int(sum(1 for clip_id in clip_ids if clip_id in self))

    @property
    def n_recorded(self) -> int:
        """Clips carrying a record (the labelled ones)."""
        return len(self.recorded)

    def n_moved(self) -> int:
        """Records that differ from what their track hashes to."""
        return int(sum(1 for clip_id, member in self.recorded.items()
                       if member != self.cfg.suggest.audit_by_hash(clip_id)))


def labelled_rows(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Rows of ``frame`` with a non-blank label in ``group``, the label stripped."""
    out = frame.dropna(subset=[group]).copy()
    out[group] = out[group].astype(str).str.strip()
    return out[out[group].str.len() > 0]


__all__ = ["Suggester", "train", "main", "AuditHoldout", "labelled_rows",
           "MODEL_FILE", "REPORT_FILE", "CV_PREDICTIONS_FILE"]


# --------------------------------------------------------------------------- #
@dataclass
class GroupReport:
    group: str
    n_labels: int
    # Labels whose clip_id is not in the feature table.
    n_orphaned: int
    orphaned_by_class: dict
    n_classes: int
    classes: list[str]
    per_class_counts: dict
    dropped_classes: dict
    macro_f1: float
    accuracy: float
    baseline_accuracy: float          # always-predict-the-majority-class
    kappa: float
    kappa_lower: float
    n_groups: int                     # movies the folds were split on
    scope: str
    audit_n: int                      # audit clips actually SCORED below
    audit_population: int             # audit clips this project HAS, labelled or not
    audit_labelled: int               # of those, how many carry a label for this group
    audit_accuracy: float
    audit_kappa: float
    confusion: list
    top_features: list
    reliable: bool
    reason: str
    # [[clip_id, predicted, confidence], ...]; `train` writes it to its own file.
    cv_predictions: list = field(default_factory=list)


def _iterations(cfg: Config) -> int:
    """Boosting rounds per fit."""
    return int(cfg.suggest.iterations)


def _learning_rate(cfg: Config) -> dict:
    """``{"learning_rate": x}``, or empty for ``auto``."""
    rate = cfg.suggest.learning_rate
    return {} if isinstance(rate, str) else {"learning_rate": float(rate)}


def _build_model(cfg: Config, n_classes: int):
    """The classifier: CatBoost, or sklearn HistGradientBoosting as fallback."""
    if cfg.suggest.model == "catboost":
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                **_task_type(cfg),
                iterations=_iterations(cfg), depth=DEPTH,
                l2_leaf_reg=float(cfg.suggest.l2_leaf_reg), **_learning_rate(cfg),
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                auto_class_weights="Balanced" if BALANCE_CLASSES else None,
                verbose=False, random_seed=SEED, allow_writing_files=False,
            )
        except ImportError:
            print(
                "catboost will not import; using sklearn instead (reinstall to fix)",
                file=sys.stderr,
            )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=_iterations(cfg), max_depth=DEPTH,
        l2_regularization=float(cfg.suggest.l2_leaf_reg), **_learning_rate(cfg),
        class_weight="balanced" if BALANCE_CLASSES else None,
        early_stopping=False, random_state=SEED,
    )


def gpu_devices() -> int:
    """GPUs CatBoost can use. 0 when there are none or the build has no CUDA."""
    try:
        from catboost.utils import get_gpu_device_count

        return int(get_gpu_device_count())
    except Exception:                                     # noqa: BLE001
        return 0


def resolved_device(cfg: Config) -> str:
    """``suggest.device`` resolved to ``"cpu"`` or ``"gpu"``; ``gpu`` without a GPU raises."""
    device = str(cfg.suggest.device)
    if device == "cpu":
        return "cpu"
    if gpu_devices():
        return "gpu"
    if device == "auto":
        return "cpu"
    raise RuntimeError("suggest.device is 'gpu' but CatBoost sees no GPU; use 'auto' or 'cpu'")


def _task_type(cfg: Config) -> dict:
    """CatBoost's device arguments. Empty for the CPU, which is its default."""
    return {} if resolved_device(cfg) == "cpu" else {"task_type": "GPU", "devices": "0"}


def describe_model(cfg: Config, n_classes: int, n_rows: int, n_cols: int) -> str:
    """One line naming the estimator and its hyperparameters, for the log."""
    weights = "balanced" if BALANCE_CLASSES else "none"
    iterations, l2 = _iterations(cfg), cfg.suggest.l2_leaf_reg
    lr = cfg.suggest.learning_rate
    if cfg.suggest.model == "catboost":
        try:
            import catboost
            backend = f"CatBoost {catboost.__version__}"
            shape = (f"depth {DEPTH} ({2 ** DEPTH} leaves), iterations {iterations}, "
                     f"lr {lr}, l2_leaf_reg {l2}")
            loss = "MultiClass" if n_classes > 2 else "Logloss"
            extra = f", loss {loss}, class weights {weights}"
        except ImportError:
            backend = "sklearn HistGradientBoosting (catboost not installed)"
            shape = f"max_depth {DEPTH}, max_iter {iterations}, lr {lr}, l2 {l2}"
            extra = f", class weights {weights}"
    else:
        backend = "sklearn HistGradientBoosting"
        shape = f"max_depth {DEPTH}, max_iter {iterations}, lr {lr}, l2 {l2}"
        extra = f", class weights {weights}"
    return (f"{backend}: {shape}{extra}, seed {SEED} | "
            f"{n_rows} clips x {n_cols} features, {n_classes} classes")


class _IterationTicker:
    """CatBoost callback that advances the progress bar and stops the fit on cancel."""

    def __init__(self, reporter, every: int = 10):
        self.reporter, self.every, self._last = reporter, max(1, every), 0

    def after_iteration(self, info) -> bool:
        done = int(getattr(info, "iteration", 0)) + 1
        if done - self._last >= self.every:
            self.reporter.advance(done - self._last)
            self._last = done
        return not self.reporter.cancelled()


def _fit(model, X, y, cfg: Config, reporter, label: str):
    """Fit with a live progress stage where the backend supports one."""
    reporter.raise_if_cancelled()
    reporter.stage(label, _iterations(cfg))
    if cfg.suggest.model == "catboost" and hasattr(model, "get_feature_importance"):
        try:
            model.fit(X, y, callbacks=[_IterationTicker(reporter)])
            reporter.raise_if_cancelled()
            return model
        except TypeError:
            # The GPU backend rejects `callbacks`; fit without the ticker.
            pass
    model.fit(X, y)
    reporter.advance(_iterations(cfg))
    reporter.raise_if_cancelled()
    return model


def _movie_folds(y, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds of whole movies whose training side holds every class; [] if there are none.

    Each movie is held out alone up to ``MAX_FOLDS`` movies. Above that the movies share
    ``MAX_FOLDS`` stratified folds, or are held out alone if no such split keeps every class.
    """
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

    y, groups = np.asarray(y), np.asarray(groups)
    n_classes, n_movies = len(np.unique(y)), len(np.unique(groups))
    if n_movies < 2:
        return []
    rows = np.zeros(len(y))

    def complete(folds) -> bool:
        return all(len(test) and len(np.unique(y[train])) == n_classes
                   for train, test in folds)

    if n_movies > MAX_FOLDS:
        for seed in range(SEED, SEED + _FOLD_SEEDS):
            splitter = StratifiedGroupKFold(n_splits=MAX_FOLDS, shuffle=True, random_state=seed)
            with warnings.catch_warnings():
                # A class smaller than the fold count only stratifies less evenly.
                warnings.simplefilter("ignore", UserWarning)
                folds = list(splitter.split(rows, y, groups=groups))
            if complete(folds):
                return folds
    folds = list(LeaveOneGroupOut().split(rows, y, groups=groups))
    return folds if complete(folds) else []


def _kappa_lower(y, pred, movies, tracks, classes) -> float:
    """5th percentile of kappa over tracks resampled within each movie.

    ``-inf`` when too few resamples hold two classes.
    """
    from sklearn.metrics import cohen_kappa_score

    y, pred = np.asarray(y), np.asarray(pred)
    keys = pd.DataFrame({"movie": np.asarray(movies), "track": np.asarray(tracks)})
    by_movie: dict = {}
    for (movie, _), rows in sorted(keys.groupby(["movie", "track"]).indices.items()):
        by_movie.setdefault(movie, []).append(rows)

    rng = np.random.default_rng(SEED)
    kappas = []
    for _ in range(_KAPPA_RESAMPLES):
        idx = np.concatenate([
            members[i] for members in by_movie.values()
            for i in rng.integers(len(members), size=len(members))
        ])
        if len(np.unique(y[idx])) < 2:
            continue
        kappas.append(cohen_kappa_score(y[idx], pred[idx], labels=classes))
    if len(kappas) < _KAPPA_MIN_RESAMPLES:
        return float("-inf")
    return float(np.percentile(kappas, 5))


def _grouped_cv_predictions(X, y, groups, cfg: Config, folds,
                            reporter=None, label: str = ""):
    """Out-of-fold predictions and confidences over ``folds`` of whole movies."""
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    pred = np.empty(len(y), dtype=object)
    conf = np.full(len(y), np.nan, dtype=float)
    n_splits = len(folds)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        held_out = sorted(set(np.asarray(groups)[test_idx]))
        reporter.log(
            f"{label}fold {fold}/{n_splits}: holding out {', '.join(map(str, held_out))} "
            f"({len(test_idx)} clips), fitting on {len(train_idx)}"
        )
        x_train, x_test = X[train_idx], X[test_idx]
        model = _build_model(cfg, len(np.unique(y)))
        _fit(model, x_train, y[train_idx], cfg, reporter,
             f"{label}fold {fold}/{n_splits}")
        fold_pred = model.predict(x_test).ravel()
        pred[test_idx] = fold_pred
        # Probability of the predicted class, not the maximum.
        try:
            proba = np.asarray(model.predict_proba(x_test), dtype=float)
            where = {c: i for i, c in enumerate(np.asarray(model.classes_))}
            columns = np.array([where[p] for p in fold_pred])
            conf[test_idx] = proba[np.arange(len(columns)), columns]
        except Exception:
            pass          # a model without a posterior still gives predictions
    return pred, conf


def _feature_importance(model, X, y, columns, seed, n_repeats=5, top=15):
    """Top feature importances: CatBoost's own, else permutation importance."""
    native = getattr(model, "get_feature_importance", None)
    if native is not None:
        try:
            values = np.asarray(native(), dtype=float)
            order = np.argsort(-values)[:top]
            return [
                {"feature": columns[i], "importance": round(float(values[i]), 4)}
                for i in order if values[i] > 0
            ]
        except Exception:
            pass

    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=seed, scoring="accuracy"
        )
        order = np.argsort(-result.importances_mean)[:top]
        return [
            {"feature": columns[i], "importance": round(float(result.importances_mean[i]), 4)}
            for i in order if result.importances_mean[i] > 0
        ]
    except Exception:
        return []


def train_group(
    features: pd.DataFrame, labels: pd.DataFrame, group_name: str, cfg: Config,
    columns: list[str] | None = None, verbose: bool = True, reporter=None,
    holdout: AuditHoldout | None = None,
    have_features: set[str] | None = None,
    audit_population: int | None = None,
) -> tuple[object | None, GroupReport]:
    """Fit one category group. Returns (model, report); model is None if unreliable.

    The project-wide ``holdout``, ``have_features`` and ``audit_population`` are computed if absent.
    """
    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    if holdout is None:
        holdout = AuditHoldout.from_labels(cfg, labels)
    if have_features is None or audit_population is None:
        feature_ids = features["clip_id"].astype(str)
        if have_features is None:
            have_features = set(feature_ids)
        if audit_population is None:
            # Audit clips in the feature table, labelled or not.
            audit_population = holdout.count(feature_ids)

    wanted = labelled_rows(labels[["clip_id", group_name]], group_name)
    merged = features.merge(wanted, on="clip_id", how="inner")

    # Labels lost to re-detection (track ids changed); repair-links recovers them.
    orphaned = wanted[~wanted["clip_id"].astype(str).isin(have_features)]
    orphaned_by_class = {
        str(k): int(v)
        for k, v in orphaned[group_name].astype(str).value_counts().sort_index().items()
    }

    columns = columns or feature_columns(features)

    # Classes are chosen before the audit split, so the holdout cannot remove one.
    all_counts = merged[group_name].astype(str).value_counts().to_dict()

    # Rare classes are dropped and reported rather than blocking the group.
    dropped = {k: v for k, v in all_counts.items() if v < cfg.suggest.min_per_class}

    # The audit holdout is excluded from training too.
    audit_mask = merged["clip_id"].astype(str).map(lambda c: c in holdout)
    audit_set = merged[audit_mask].copy()
    merged = merged[~audit_mask]

    if dropped:
        merged = merged[~merged[group_name].astype(str).isin(dropped)]

    # What reaches the fit, after both filters.
    counts = merged[group_name].astype(str).value_counts().to_dict()
    classes = sorted(counts)

    def bail(reason: str) -> tuple[None, GroupReport]:
        return None, GroupReport(
            group=group_name, n_labels=len(merged),
            n_orphaned=len(orphaned), orphaned_by_class=orphaned_by_class,
            n_classes=len(classes), classes=classes,
            per_class_counts=counts, dropped_classes=dropped, macro_f1=0.0, accuracy=0.0,
            baseline_accuracy=0.0, kappa=0.0, kappa_lower=0.0, n_groups=0,
            scope="not validated",
            audit_n=0,
            audit_population=audit_population, audit_labelled=len(audit_set),
            audit_accuracy=float("nan"),
            audit_kappa=float("nan"), confusion=[], top_features=[], reliable=False,
            reason=reason,
        )

    if len(classes) < 2:
        return bail(
            f"only {len(classes)} class has >= {cfg.suggest.min_per_class} labels "
            f"(counts: {all_counts})"
        )

    X = merged[columns].to_numpy(dtype=np.float32)
    y = merged[group_name].astype(str).to_numpy()

    # Grouped by movie: clips of one movie share imaging conditions.
    groups = merged["movie_id"].astype(str).to_numpy()
    n_movies = int(len(np.unique(groups)))
    tracks = merged["track_id"].to_numpy()
    n_tracks = int(len(merged.groupby(["movie_id", "track_id"])))

    # A fold model can only predict a class it was fitted on.
    folds = _movie_folds(y, groups)
    n_splits = len(folds)
    if n_splits < 2:
        if n_movies < 2:
            return bail(
                f"all {len(merged)} training labels come from one movie "
                f"({groups[0]!r}), so leave-movie-out CV is impossible; "
                "annotate some clips in a second movie"
            )
        per_class = merged.groupby(group_name)["movie_id"].nunique()
        confined = sorted(str(c) for c in per_class[per_class < 2].index)
        noun = "class" if len(confined) == 1 else "classes"
        verb = "appears" if len(confined) == 1 else "appear"
        return bail(
            f"{noun} {', '.join(map(repr, confined))} {verb} in only one of "
            f"the {n_movies} annotated movies; annotate the {noun} in another movie"
        )

    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    reporter.log(describe_model(cfg, len(classes), len(merged), X.shape[1]))
    reporter.log(
        f"{group_name}: {n_splits}-fold leave-movie-out over {n_movies} movie(s), "
        f"classes {', '.join(classes)}"
        + (f"; {len(audit_set)} of this project's {audit_population} audit clips are "
           "labelled and held out entirely" if len(audit_set) else "")
    )
    pred, pred_conf = _grouped_cv_predictions(
        X, y, groups, cfg, folds, reporter=reporter, label=f"{group_name} ")
    macro_f1 = float(f1_score(y, pred, average="macro", labels=classes, zero_division=0))
    accuracy = float(accuracy_score(y, pred))
    baseline = float(max(counts.values()) / len(y))
    kappa = float(cohen_kappa_score(y, pred, labels=classes))

    # Clips of one track are not independent, so the bound resamples tracks.
    kappa_lo = _kappa_lower(y, pred, groups, tracks, classes)

    # Gate on chance-corrected kappa (bootstrap lower bound), not accuracy.
    reliable = kappa_lo > 0.0 and macro_f1 >= MIN_MACRO_F1
    if kappa_lo <= 0.0:
        reason = (
            f"agreement is not distinguishable from chance "
            f"(kappa {kappa:.2f}, 5th-percentile bound {kappa_lo:.2f} <= 0 "
            f"over {n_tracks} track(s))"
        )
    elif macro_f1 < MIN_MACRO_F1:
        reason = (
            f"grouped-CV macro-F1 {macro_f1:.2f} below {MIN_MACRO_F1:.2f}"
        )
    else:
        reason = "ok"

    model = _build_model(cfg, len(classes))
    _fit(model, X, y, cfg, reporter, f"{group_name}: final fit on all {len(X)} clips")

    # Score the audit clips with the final model, which never saw them.
    audit_n, audit_acc, audit_kappa = 0, float("nan"), float("nan")
    audit_eval = audit_set[audit_set[group_name].astype(str).isin(classes)]
    if len(audit_eval) >= 3:
        ay = audit_eval[group_name].astype(str).to_numpy()
        apred = model.predict(audit_eval[columns].to_numpy(dtype=np.float32)).ravel()
        audit_n = int(len(ay))
        audit_acc = float(accuracy_score(ay, apred))
        if len(np.unique(ay)) >= 2:
            audit_kappa = float(cohen_kappa_score(ay, apred, labels=classes))

    report = GroupReport(
        group=group_name,
        n_labels=len(merged),
        n_orphaned=len(orphaned),
        orphaned_by_class=orphaned_by_class,
        n_classes=len(classes),
        classes=classes,
        per_class_counts=counts,
        dropped_classes=dropped,
        macro_f1=round(macro_f1, 4),
        accuracy=round(accuracy, 4),
        baseline_accuracy=round(baseline, 4),
        kappa=round(kappa, 4),
        kappa_lower=round(kappa_lo, 4) if np.isfinite(kappa_lo) else -1.0,
        n_groups=n_movies,
        scope=f"leave-movie-out over {n_movies} movies ({n_splits}-fold)",
        audit_n=audit_n,
        audit_population=audit_population,
        audit_labelled=len(audit_set),
        audit_accuracy=round(audit_acc, 4) if np.isfinite(audit_acc) else float("nan"),
        audit_kappa=round(audit_kappa, 4) if np.isfinite(audit_kappa) else float("nan"),
        confusion=confusion_matrix(y, pred, labels=classes).tolist(),
        cv_predictions=[
            [str(clip_id), str(answer), None if not np.isfinite(c) else round(float(c), 4)]
            for clip_id, answer, c in zip(
                merged["clip_id"].astype(str).to_numpy(), pred, pred_conf)
        ],
        top_features=_feature_importance(model, X, y, columns, SEED),
        reliable=reliable,
        reason=reason,
    )
    if verbose:
        flag = "OK " if report.reliable else "-- "
        print(
            f"  {flag}{group_name:<12} n={len(merged):>4}  {n_splits}-fold leave-movie-out "
            f"({n_movies} movies): kappa {kappa:+.3f} [lo {kappa_lo:+.3f}]  "
            f"macro-F1 {macro_f1:.3f}  acc {accuracy:.3f} vs {baseline:.3f} majority"
        )
        if orphaned_by_class:
            print(
                f"       -> IGNORED {len(orphaned)} of {len(wanted)} labels not in the "
                "feature table ("
                + ", ".join(f"{k} {v}" for k, v in orphaned_by_class.items())
                + "); run the label repair on the Model page"
            )
        if dropped:
            print(
                f"       -> not modelled (fewer than {cfg.suggest.min_per_class} labels): "
                + ", ".join(f"{k} ({v})" for k, v in sorted(dropped.items()))
            )
        if not report.reliable:
            print(f"       -> will not suggest: {report.reason}")
    return (model if report.reliable else None), report


def train(cfg: Config, rebuild_features: bool = False, verbose: bool = True,
          reporter=None) -> dict:
    """Fit every category group and save the models, report and CV predictions."""
    import joblib

    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    features = None if rebuild_features else cached_clip_features(cfg)
    if features is None:
        features = build_clip_features(cfg, verbose=verbose, reporter=reporter)
    else:
        reporter.log(f"clip features unchanged, reusing {CLIP_FEATURES_FILE}")
    if not cfg.paths.csv.exists():
        raise FileNotFoundError(f"no annotations yet at {cfg.paths.csv}")
    labels = pd.read_csv(cfg.paths.csv)
    if labels.empty:
        raise RuntimeError(f"{cfg.paths.csv} has no rows")

    holdout = AuditHoldout.from_labels(cfg, labels)
    moved = holdout.n_moved()
    if moved:
        reporter.log(
            f"{moved} of {holdout.n_recorded} labelled clip(s) keep their recorded audit "
            "membership, which differs from their track hash"
        )
    elif not holdout.n_recorded and len(labels):
        reporter.log(
            f"no labelled clip in {cfg.paths.csv.name} records its audit membership; "
            "open the project in the GUI to record it",
            "warning",
        )

    from ..measure.clipfeatures import available_groups

    blocked = {g: why for g, why in available_groups(cfg).items()
               if why and g in cfg.suggest.feature_groups}
    usable = [g for g in cfg.suggest.feature_groups if not blocked.get(g)]
    for group, why in blocked.items():
        reporter.log(f"feature group {group!r} skipped: {why}", "warning")
    if not usable:
        raise RuntimeError(
            f"none of the selected feature groups can be computed: {blocked}; "
            "enable one on the Configure page"
        )

    columns = feature_columns(features, usable)
    if not columns:
        raise RuntimeError(
            f"the feature groups {usable} produced no columns; rebuild the clip "
            "features from the Model page"
        )
    named = cfg.suggest.feature_columns
    if named:
        wanted = set(named)
        absent = [c for c in named if c not in set(columns)]
        if absent:
            raise RuntimeError(
                f"suggest.feature_columns names {len(absent)} column(s) not in "
                f"{CLIP_FEATURES_FILE} ({', '.join(absent[:4])}); rebuild the features "
                "or drop them"
            )
        dropped = len(columns) - len(wanted)
        columns = [c for c in columns if c in wanted]
        reporter.log(
            f"suggest.feature_columns keeps {len(columns)} column(s); {dropped} from "
            f"{', '.join(usable)} are left out"
        )
    reporter.log(f"features: {len(columns)} columns from {', '.join(usable)}")

    # Project-wide, computed once for every group.
    feature_ids = features["clip_id"].astype(str)
    have_features = set(feature_ids)
    audit_population = holdout.count(feature_ids)

    n_orphaned = int((~labels["clip_id"].astype(str).isin(have_features)).sum())
    if n_orphaned:
        reporter.log(
            f"{n_orphaned} of {len(labels)} labels refer to clips not in the feature "
            "table and were ignored; run the label repair on the Model page",
            "warning",
        )

    single = [g for g in cfg.categories if not g.is_multi]
    reporter.stage("fitting category groups", len(single))
    models, reports = {}, {}
    cv_rows: list[dict] = []
    for group in single:
        # multi-select groups need a different (multi-label) formulation
        reporter.log(f"fitting {group.name}")
        reporter.raise_if_cancelled()
        model, report = train_group(
            features, labels, group.name, cfg, columns, verbose, reporter=reporter, holdout=holdout, have_features=have_features,
            audit_population=audit_population,
        )
        reporter.advance()
        entry = asdict(report)
        for clip_id, answer, confidence in entry.pop("cv_predictions", None) or []:
            cv_rows.append({"clip_id": clip_id, "group": group.name,
                            "predicted": answer, "confidence": confidence})
        reports[group.name] = entry
        reporter.log(
            f"{group.name}: n={report.n_labels}  kappa {report.kappa:+.3f} "
            f"[lo {report.kappa_lower:+.3f}]  macro-F1 {report.macro_f1:.3f}"
            + ("" if report.reliable else f"   -> silent: {report.reason}")
        )
        if report.n_orphaned:
            reporter.log(
                f"{group.name}: {report.n_orphaned} of its labels were ignored, "
                "not in the feature table ("
                + ", ".join(f"{k} {v}" for k, v in report.orphaned_by_class.items())
                + ")",
                "warning",
            )
        if model is not None:
            models[group.name] = model

    # The feature version, window length, context and embedding bases travel with the model.
    trained_window = int(cfg.clips.frames)
    from ..measure import embeddings as emb

    on_disk = emb.load_bases(work / emb.BASES_FILE)
    payload = {"columns": columns, "models": models,
               "embedding_bases": {n: on_disk[n] for n in emb.names_in(columns)
                                   if n in on_disk},
               "embedding_specs": {e.name: {"source": e.source, "path": e.path}
                                   for e in cfg.suggest.embeddings
                                   if e.name in emb.names_in(columns)},
               "feature_version": FEATURE_VERSION,
               "trained_window": trained_window,
               "context_frames": int(cfg.suggest.context_frames)}
    joblib.dump(payload, work / MODEL_FILE)
    reporter.log(
        f"model saved: fitted on {trained_window}-frame windows "
        f"(clips.frames), feature version {FEATURE_VERSION}"
    )
    (work / REPORT_FILE).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    # Always written, so a previous run's file is never left behind.
    pd.DataFrame(cv_rows, columns=["clip_id", "group", "predicted", "confidence"]
                 ).to_parquet(work / CV_PREDICTIONS_FILE, index=False)
    reporter.log(f"{len(cv_rows)} out-of-fold prediction(s) saved to {CV_PREDICTIONS_FILE}")
    return reports


# --------------------------------------------------------------------------- #
class Suggester:
    """Loads the trained models and produces per-clip suggestions for the GUI."""

    # Resolved from the annotation CSV by `load`.
    audit: AuditHoldout

    def __init__(self, cfg: Config, labels_csv: str | Path | None = None):
        """``labels_csv`` is read for audit membership instead of the live CSV."""
        self.cfg = cfg
        self.labels_csv = Path(labels_csv) if labels_csv is not None else None
        self.columns: list[str] = []
        self.models: dict = {}
        self.reports: dict = {}
        self.features: pd.DataFrame | None = None
        self.problem: str = ""
        # Frames per training window; None when unrecorded.
        self.trained_window: int | None = None
        # suggest.context_frames the model was fitted with; 0 when unrecorded.
        self.context_frames: int = 0
        # {name: PCA} the model's emb_* columns were fitted in.
        self.embedding_bases: dict = {}
        # {name: EmbeddingSpec} the model's embeddings were read from.
        self.embedding_specs: dict = {}
        self._by_clip: dict[str, np.ndarray] = {}
        # Feature row per clip id, and cached per-group probabilities over every row.
        self._row_of: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._proba: dict[str, np.ndarray] = {}
        # Bumped by every `load`; a batch from an older epoch is stale.
        self.model_epoch: int = 0
        self.load()

    def load(self) -> bool:
        import joblib

        # Bumped first: any load, failed or not, makes old batches stale.
        self.model_epoch += 1
        self.problem = ""
        self.columns, self.models, self.reports = [], {}, {}
        self.features, self._by_clip = None, {}
        self._row_of, self._matrix, self._proba = {}, None, {}
        self.trained_window = None
        self.context_frames = 0
        self.embedding_bases = {}
        self.embedding_specs = {}

        # Before any early return: membership is needed with or without a model.
        self.audit = AuditHoldout.from_csv(self.cfg, self.labels_csv or self.cfg.paths.csv)

        work = self.cfg.paths.work_dir
        model_path, feat_path = work / MODEL_FILE, work / CLIP_FEATURES_FILE
        if not model_path.exists():
            self.problem = "no model has been trained yet"
            return False
        if not feat_path.exists():
            self.problem = f"{CLIP_FEATURES_FILE} is missing; retrain to rebuild it"
            return False
        try:
            payload = joblib.load(model_path)
            columns = list(payload["columns"])
            models = payload["models"]
            features = pd.read_parquet(feat_path)
        except Exception as exc:
            self.problem = f"could not read the model: {exc}"
            return False

        # A missing or unparsable length loads as None; `window_problem` reports it.
        try:
            recorded = payload.get("trained_window")
            self.trained_window = None if recorded is None else int(recorded)
        except (TypeError, ValueError):
            self.trained_window = None

        retired = [c for c in columns if _RETIRED_CNN.match(str(c))]
        if retired:
            self.problem = "this model uses cnn* columns; retrain it on the Model page"
            return False
        self.context_frames = int(payload.get("context_frames") or 0)
        from ..measure import embeddings as emb

        from ..config import EmbeddingSpec

        bases = dict(payload.get("embedding_bases") or {})
        specs = dict(payload.get("embedding_specs") or {})
        used = emb.names_in(columns)
        unbased = [n for n in used if n not in bases or n not in specs]
        if unbased:
            self.problem = (
                f"this model lacks the PCA basis or source of embedding(s) {unbased}; "
                "retrain it on the Model page"
            )
            return False
        stale = self._stamp_problem(features, {n: bases[n] for n in used})
        if stale:
            self.problem = stale
            return False
        self.embedding_bases = bases
        self.embedding_specs = {
            n: EmbeddingSpec(n, str(s.get("source")), s.get("path"))
            for n, s in specs.items()
        }

        fitted_on = payload.get("feature_version")
        built_with = None
        if "_feature_version" in features.columns and len(features):
            built_with = int(features["_feature_version"].iloc[0])
        if fitted_on is None or built_with is None or int(fitted_on) != built_with:
            self.problem = (
                f"this model was fitted on feature version {fitted_on!r} but "
                f"{CLIP_FEATURES_FILE} has version {built_with!r}; retrain on the Model page"
            )
            return False

        missing = [c for c in columns if c not in features.columns]
        if missing:
            shown = ", ".join(missing[:4]) + ("..." if len(missing) > 4 else "")
            self.problem = (
                f"{len(missing)} of the model's {len(columns)} columns are not in the "
                f"feature table ({shown}); retrain on the Model page"
            )
            return False

        self.columns, self.models, self.features = columns, models, features
        matrix = features[columns].to_numpy(dtype=np.float32, copy=False)
        ids = features["clip_id"].astype(str).to_numpy()
        self._by_clip = {clip_id: matrix[i] for i, clip_id in enumerate(ids)}
        self._row_of = {clip_id: i for i, clip_id in enumerate(ids)}
        self._matrix = matrix

        report_path = work / REPORT_FILE
        if report_path.exists():
            try:
                self.reports = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass          # the report is commentary; the model still works
        if not self.models:
            self.problem = "the model file holds no usable classifier"
        return bool(self.models)

    @property
    def ready(self) -> bool:
        return bool(self.models) and bool(self._by_clip)

    def _group_proba(self, group: str) -> np.ndarray:
        """Class probabilities of every feature row under ``group``'s model, cached."""
        proba = self._proba.get(group)
        if proba is None:
            proba = np.asarray(self.models[group].predict_proba(self._matrix), dtype=float)
            self._proba[group] = proba
        return proba

    def warm(self) -> None:
        """Score the pool for selection now, so the first batch is a lookup."""
        cfg = self.cfg
        if not (cfg.suggest.enabled and cfg.suggest.selection_on and self.ready):
            return
        group = self.selection_group()
        if group:
            self._group_proba(group)

    def window_problem(self, length: int) -> str:
        """'' if features built over `length`-frame windows suit this model, else why not."""
        examples = ", ".join(_LENGTH_DEPENDENT_EXAMPLES)
        try:
            length = int(length)
        except (TypeError, ValueError):
            return f"window length {length!r} is not a frame count and could not be checked"
        if self.trained_window is None:
            return (
                f"this model does not record its training window length, so the "
                f"{length}-frame windows could not be checked; retrain to record it"
            )
        if length == self.trained_window:
            return ""
        return (
            f"this model was trained on {self.trained_window}-frame windows but these "
            f"features use {length}; window length changes the values ({examples}). "
            f"Retrain at {length} frames or restore {self.trained_window}."
        )

    def refuse_window_mismatch(self, length: int, context: str, reporter=None) -> None:
        """Raise on a window-length mismatch (appending ``context``); warn if unrecorded."""
        problem = self.window_problem(length)
        if not problem:
            return
        if self.trained_window is None:
            if reporter is None:
                warnings.warn(problem, stacklevel=3)
            else:
                reporter.log(problem, "warning")
            return
        raise RuntimeError(f"{problem} {context}")

    def _stamp_problem(self, features: pd.DataFrame, bases: dict) -> str:
        """Why the feature table was built differently from the model, or ''."""
        from ..measure import embeddings as emb

        built_context = 0
        built_bases: dict = {}
        if len(features):
            if "_context_frames" in features.columns:
                built_context = int(features["_context_frames"].iloc[0])
            if "_embedding_basis" in features.columns:
                built_bases = json.loads(str(features["_embedding_basis"].iloc[0]))
        if built_context != self.context_frames:
            return (
                f"{CLIP_FEATURES_FILE} was built with suggest.context_frames "
                f"{built_context} but the model with {self.context_frames}; "
                "retrain on the Model page"
            )
        moved = [n for n, p in bases.items() if built_bases.get(n) != emb.basis_hash(p)]
        if moved:
            return (
                f"{CLIP_FEATURES_FILE} has a different PCA basis for embedding(s) "
                f"{moved} than the model; Retrain on the Model page"
            )
        return ""

    def embedding_inputs(self) -> dict:
        """``{name: (spec, pca)}`` for every embedding the model uses."""
        return {n: (self.embedding_specs[n], self.embedding_bases[n])
                for n in self.embedding_specs}

    def refuse_context_mismatch(self, context_frames: int) -> None:
        """Raise if the model was fitted with another ``suggest.context_frames``."""
        if int(context_frames) == self.context_frames:
            return
        raise RuntimeError(
            f"this model was fitted with suggest.context_frames {self.context_frames} "
            f"but the config says {int(context_frames)}; retrain or set it back"
        )

    def is_audit_clip(self, clip_id: str) -> bool:
        return clip_id in self.audit

    # ---------------------------------------------------------- selection -- #
    def selection_group(self) -> str:
        """The first single-choice group the model covers, or ``""``."""
        for group in self.cfg.categories:
            if not group.is_multi and group.name in self.models:
                return group.name
        return ""

    def selection_batch(
        self, manifest: pd.DataFrame, done_ids: set[str], n: int
    ) -> list[dict]:
        """The next `n` unannotated, non-audit clips: select_batch records plus ``group``.

        Call once per model (:attr:`model_epoch`), not per label, so the queue stays stable.
        """
        cfg = self.cfg
        if not cfg.suggest.enabled or not cfg.suggest.selection_on:
            return []
        if not self.ready or manifest is None or not len(manifest):
            return []
        group = self.selection_group()
        if not group or int(n) <= 0:
            return []

        self.refuse_window_mismatch(
            int(cfg.clips.frames),
            f"(The {int(cfg.clips.frames)} frames are clips.frames.)",
        )

        ids = manifest["clip_id"].astype(str).to_numpy()

        # Clips without a feature row are left out of the pool.
        rows = [self._row_of.get(clip_id) for clip_id in ids]
        keep = np.array([row is not None for row in rows], dtype=bool)
        if not keep.any():
            return []
        pool = np.flatnonzero(keep)
        pool_ids = ids[pool]

        done = {str(c) for c in (done_ids or ())}
        candidates = np.array(
            [c not in done and c not in self.audit for c in pool_ids], dtype=bool
        )
        if not candidates.any():
            return []

        proba = self._group_proba(group)[[rows[i] for i in pool]]
        classes = [str(c) for c in self.models[group].classes_]

        records = select_batch(proba, pool_ids, classes, int(n),
                               candidates=candidates)
        for record in records:
            record["group"] = group
        return records

    def suggest(self, clip_id: str) -> dict[str, tuple[str, float]]:
        """{group: (predicted_value, confidence)} for one clip; {} if unavailable."""
        if not self.ready or self.is_audit_clip(clip_id):
            return {}
        vector = self._by_clip.get(str(clip_id))
        if vector is None:
            return {}
        row = vector.reshape(1, -1)
        out: dict[str, tuple[str, float]] = {}
        for group, model in self.models.items():
            try:
                proba = np.asarray(model.predict_proba(row)).ravel()
                idx = int(np.argmax(proba))
                confidence = float(proba[idx])
                if confidence < self.cfg.suggest.min_confidence:
                    continue
                out[group] = (str(model.classes_[idx]), confidence)
            except Exception as exc:
                print(f"[trackanno] suggestion failed for {group}: {exc}")
                continue
        return out

    def status_line(self) -> str:
        if self.problem:
            return self.problem
        if not self.reports:
            return (
                f"model loaded ({len(self.models)} group(s), {len(self.columns)} "
                "columns) but its report is missing"
                if self.models else "no model yet; train one on the Model page"
            )
        parts = []
        for group, rep in self.reports.items():
            parts.append(
                f"{group} F1={rep['macro_f1']:.2f}"
                + ("" if rep["reliable"] else " (silent)")
            )
        return "suggestions: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
def _print_report(reports: dict) -> None:
    for group, rep in reports.items():
        print(f"\n=== {group} ===")
        print(f"  labels: {rep['n_labels']}  classes: {rep['n_classes']}")
        print(f"  per class: {rep['per_class_counts']}")
        if rep.get("n_orphaned"):
            print(
                f"  IGNORED (clip not in the feature table): {rep['n_orphaned']} "
                f"-> {rep.get('orphaned_by_class', {})}"
            )
            print("  run the label repair on the Model page to link them back")
        if rep.get("dropped_classes"):
            print(f"  NOT modelled (too few labels): {rep['dropped_classes']}")
        print(f"  scope: {rep.get('scope', '?')}")
        # Missing population (unrecorded) and zero population (no holdout) differ.
        population = rep.get("audit_population")
        labelled = rep.get("audit_labelled", rep.get("audit_n", 0))
        scored = rep.get("audit_n") or 0
        ak, aa = rep.get("audit_kappa"), rep.get("audit_accuracy")
        ak = "n/a" if ak is None or ak != ak else f"{ak:+.3f}"
        aa = "n/a" if aa is None or aa != aa else f"{aa:.3f}"
        if population is None:
            if scored:
                print(
                    f"  audit holdout ({scored} scored, never suggested nor "
                    f"trained on): kappa {ak}  accuracy {aa}"
                )
                print("    this report predates the audit population count; retrain to record it")
        elif population:
            of_population = f"{labelled} of {population} audit clips labelled"
            if scored:
                print(
                    f"  audit holdout ({of_population}, {scored} scored, "
                    f"never suggested nor trained on): kappa {ak}  accuracy {aa}"
                )
            else:
                print(f"  audit holdout ({of_population}): too few to score, "
                      "no unbiased estimate yet")
        else:
            print(
                "  audit holdout: none; every score below is from CV on clips the model "
                "was fitted on; set suggest.audit_fraction for a holdout"
            )
        print(
            f"  kappa {rep.get('kappa', 0):+.3f} (5th pct {rep.get('kappa_lower', 0):+.3f})   "
            f"macro-F1 {rep['macro_f1']:.3f}   accuracy {rep['accuracy']:.3f}   "
            f"majority baseline {rep['baseline_accuracy']:.3f}"
        )
        if not rep["reliable"]:
            print(f"  NOT USED: {rep['reason']}")
        if rep["confusion"]:
            width = max(len(c) for c in rep["classes"]) + 1
            print("  confusion (rows = truth):")
            print(" " * (width + 4) + "  ".join(f"{c[:6]:>6}" for c in rep["classes"]))
            for name, row in zip(rep["classes"], rep["confusion"]):
                print(f"    {name:<{width}}" + "  ".join(f"{v:>6}" for v in row))
        if rep["top_features"]:
            print("  top features (% of the fitted model's total importance):")
            for f in rep["top_features"][:10]:
                print(f"    {f['importance']:>7.4f}  {f['feature']}")


def ablate(cfg: Config) -> None:
    """Print the CV score of each feature group alone; saved files are untouched."""
    features = build_clip_features(cfg, verbose=False, write=False)
    labels = pd.read_csv(cfg.paths.csv)
    from ..measure.clipfeatures import available_groups

    usable = [g for g, why in available_groups(cfg).items() if not why]
    subsets = {g: feature_columns(features, [g]) for g in usable}
    subsets["everything"] = feature_columns(features, usable)
    embedded = [g for g in usable if g.startswith("emb_")]
    if embedded:
        subsets["no embeddings"] = feature_columns(
            features, [g for g in usable if g not in embedded])
    if cfg.suggest.feature_columns:
        print("\n(suggest.feature_columns is ignored here: subsets are whole feature groups)")
    for group in cfg.categories:
        if group.is_multi:
            continue
        print(f"\n=== {group.name} ===")
        for name, cols in subsets.items():
            if not cols:
                continue
            _, rep = train_group(features, labels, group.name, cfg, cols, verbose=False)
            print(f"  {name:<16} {len(cols):>4} cols  macro-F1 {rep.macro_f1:.3f}  "
                  f"acc {rep.accuracy:.3f}  kappa {rep.kappa:+.3f} "
                  f"[5th pct {rep.kappa_lower:+.3f}]"
                  + ("" if rep.reliable else f"   ({rep.reason})"))


def list_features(cfg: Config) -> str:
    """Feature columns by group, with the saved report's importance per category group."""
    from ..measure.clipfeatures import (
        FEATURE_GROUPS, available_groups, group_of, known_columns,
    )

    report_path = cfg.paths.work_dir / REPORT_FILE
    importance: dict[str, dict[str, float]] = {}
    if report_path.exists():
        for name, rep in json.loads(report_path.read_text(encoding="utf-8")).items():
            importance[name] = {f["feature"]: float(f["importance"])
                                for f in rep.get("top_features") or []}
    columns = known_columns(cfg)
    blocked = available_groups(cfg)
    named = set(cfg.suggest.feature_columns or ())
    width = max(len(c) for c in columns)
    widths = {g: max(len(g), 8) for g in importance}

    lines = ["* = used by the current config"]
    if importance:
        lines.append(f"importance: top features per category group from {REPORT_FILE}")
    else:
        lines.append("no trained model; run 'trackanno suggest --train' for importance")
    sources = {e.group: e.source for e in cfg.suggest.embeddings}
    for group in blocked:
        cols = [c for c in columns if group_of(c) == group]
        if not cols:
            continue
        label = (FEATURE_GROUPS[group]["label"] if group in FEATURE_GROUPS
                 else f"{sources.get(group, '')} embedding")
        selected = group in cfg.suggest.feature_groups
        status = blocked[group] or ("selected" if selected else "not in suggest.feature_groups")
        lines += ["", f"{group} ({label}): {status}",
                  f"    {'column':<{width}}"
                  + "".join(f"  {g:>{w}}" for g, w in widths.items())]
        for c in cols:
            used = selected and not blocked[group] and (not named or c in named)
            values = "".join(
                f"  {importance[g][c]:>{w}.4f}" if c in importance[g] else "  " + " " * w
                for g, w in widths.items())
            lines.append(f"  {'*' if used else ' '} {c:<{width}}{values}".rstrip())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="trackanno suggest", description="Train the annotation-suggestion model."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--train", action="store_true", help="build features, fit, cross-validate")
    p.add_argument("--rebuild-features", action="store_true",
                   help="rebuild clip features even if their inputs are unchanged")
    p.add_argument("--report", action="store_true", help="print the last saved report")
    p.add_argument("--ablate", action="store_true",
                   help="compare what each feature group is worth")
    p.add_argument("--features-only", action="store_true", help="build clip features and stop")
    p.add_argument("--list-features", action="store_true",
                   help="list feature columns by group, with importance if trained")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.list_features:
        print(list_features(cfg))
        return 0
    if args.features_only:
        build_clip_features(cfg)
        return 0
    if args.ablate:
        ablate(cfg)
        return 0
    if args.report:
        path = cfg.paths.work_dir / REPORT_FILE
        if not path.exists():
            print(f"no report at {path}; run --train first", file=sys.stderr)
            return 2
        _print_report(json.loads(path.read_text(encoding="utf-8")))
        return 0

    reports = train(cfg, rebuild_features=args.rebuild_features)
    _print_report(reports)
    usable = [g for g, r in reports.items() if r["reliable"]]
    print(
        f"\nModel -> {cfg.paths.work_dir / MODEL_FILE}"
        f"\nGroups that will suggest: {usable or 'none yet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
