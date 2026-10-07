"""Train a classifier on existing labels and suggest categories for new clips.

    trackanno suggest --train      # build features, fit, cross-validate, save
    trackanno suggest --report     # print the last report
    trackanno suggest --ablate     # what each feature group is worth

Design constraints that matter more than the model choice:

* **Cross-validation is grouped by movie.** Embedding features can identify
  individuals, so ungrouped folds would put two windows of the same animal in
  train and test and score mostly re-identification. The folds use exactly the
  columns the saved model is fitted on.
* **The reported number is always the grouped-CV score, never the training score.**
* **The model stays silent unless it clears a bar** (`suggest.min_per_class` and
  MIN_MACRO_F1): a confidently wrong suggester is worse than none,
  because the annotator's corrections train the next one.
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
# Per labelled clip, what a model fitted WITHOUT that clip's movie said about
# it. The confusion matrix in the report is a tally of exactly these, and they
# are the only honest thing to hold a label up against: the saved model was
# fitted on every label, so its opinion of one is an opinion of a row it has
# memorised.
CV_PREDICTIONS_FILE = "cv_predictions.parquet"

# Boosting hyperparameters shared by both backends; iterations, learning rate
# and l2_leaf_reg come from ``suggest.*``.
DEPTH = FIXED_SETTINGS["suggest.depth"]
BALANCE_CLASSES = FIXED_SETTINGS["suggest.balance_classes"]
SEED = FIXED_SETTINGS["suggest.seed"]
# The model stays silent below this cross-validated macro-F1.
MIN_MACRO_F1 = FIXED_SETTINGS["suggest.min_macro_f1"]

# Columns whose VALUE depends on how many frames the window spans, named in the
# mismatch message because "the features mean something else now" is not
# actionable and "path_length is longer because the window is longer" is.
_LENGTH_DEPENDENT_EXAMPLES = ("path_length", "net_displacement", "speed_std")
# Columns of models fitted before embeddings replaced the cnn group.
_RETIRED_CNN = re.compile(r"^cnn\d+_(mean|std|delta)$")


@dataclass(frozen=True)
class AuditHoldout:
    """The clips held out of the suggestion loop: never suggested on, never
    trained on, so agreement measured there is unbiased.

    THE RECORD WINS, and the hash (per ``movie_id::track_id``) is only
    consulted where there is none: re-running detection renumbers the tracks
    and re-hashing would redraw the holdout under a model that has already
    seen part of it. Membership is written into the CSV when a clip is
    LABELLED (``store.AUDIT_COLUMN``) and read back here. A recorded value that
    disagrees with the hash is the normal state of a re-detected project and is
    not reported per clip.

    An object rather than a function taking an optional mapping, because the
    mapping IS the answer and a call site that omits it reads exactly like one
    that cannot: every way in names its source, and reaching the hash where a
    record exists takes writing :meth:`by_hash` down. ``recorded`` holds only
    the clips that HAVE a record; the hash answers for the rest, under its own
    seed - see :meth:`trackanno.config.Suggest.audit_by_hash`.

    Resolve it once per project and pass it down: the constructors read a file
    or walk a frame, and membership itself is a dict lookup.
    """

    cfg: Config
    recorded: Mapping[str, bool]

    @classmethod
    def from_labels(cls, cfg: Config, labels: pd.DataFrame | None) -> "AuditHoldout":
        """From an annotation table already in memory.

        The same records :meth:`from_csv` reads off the file, for the callers
        that have the frame instead. Both share one codec, so the two readings
        cannot disagree about a cell and put a clip in the training set for one
        group and the holdout for another.
        """
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
        """From the annotation CSV, for the callers holding no frame."""
        return cls(cfg, read_audit_records(path))

    @classmethod
    def by_hash(cls, cfg: Config) -> "AuditHoldout":
        """With NOTHING recorded: every clip answered by its track's hash.

        Right only where no record can exist - a project with no annotations, or
        a question asked about clips nobody has labelled. Spelled out because on
        a re-detected project this draws a different holdout from the one the
        model was trained against.
        """
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
        """Clips carrying a record, which is the clips that are labelled."""
        return len(self.recorded)

    def n_moved(self) -> int:
        """Records that are no longer what their track hashes to."""
        return int(sum(1 for clip_id, member in self.recorded.items()
                       if member != self.cfg.suggest.audit_by_hash(clip_id)))


def labelled_rows(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """``frame`` cut down to the rows carrying a label in ``group``, the cell
    normalised to a trimmed string.

    An unlabelled clip is an empty cell or a missing one depending on how the CSV
    was written, and neither blank space nor ``NaN`` is a behaviour. Surrounding
    whitespace is not part of a class name either, so ``" swimming "`` and
    ``"swimming"`` are one class.
    """
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
    # Labels whose clip_id is absent from the feature table, so they never
    # reached the fit. Per class as well as total: a loss spread evenly is a
    # smaller model, a loss concentrated in one class is a class that is gone.
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
    # The audit population and the labelled part of it: `audit_n` alone cannot
    # say how much of the holdout has been spent.
    audit_population: int             # audit clips this project HAS, labelled or not
    audit_labelled: int               # of those, how many carry a label for this group
    audit_accuracy: float
    audit_kappa: float
    confusion: list
    top_features: list
    reliable: bool
    reason: str
    # [[clip_id, predicted, confidence], ...] from the folds above. Lifted out
    # by `train` into its own file rather than written into the report: it is
    # one row per label, where everything else here is one number per group.
    cv_predictions: list = field(default_factory=list)


def _iterations(cfg: Config) -> int:
    """Boosting rounds per fit."""
    return int(cfg.suggest.iterations)


def _learning_rate(cfg: Config) -> dict:
    """``{"learning_rate": x}``, or empty for ``auto``."""
    rate = cfg.suggest.learning_rate
    return {} if isinstance(rate, str) else {"learning_rate": float(rate)}


def _build_model(cfg: Config, n_classes: int):
    """The classifier behind the suggestions.

    CatBoost by default: balanced class weights handle the lopsided labels, and
    it takes NaN natively. The learning rate is left to the estimator.
    """
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
                "suggest.model is 'catboost' but catboost will not import; "
                "using sklearn instead. Reinstall to get the default estimator.",
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
    """``suggest.device`` with ``auto`` decided: ``"cpu"`` or ``"gpu"``.

    ``auto`` falls back to the CPU where there is no GPU. ``gpu`` raises, since
    the two devices fit different models and a silent fallback would return the
    other one.
    """
    device = str(cfg.suggest.device)
    if device == "cpu":
        return "cpu"
    if gpu_devices():
        return "gpu"
    if device == "auto":
        return "cpu"
    raise RuntimeError(
        "suggest.device is 'gpu' and CatBoost sees no GPU: no CUDA device, or a "
        "build without GPU support. Use 'auto' or 'cpu'."
    )


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
    """Drives the progress bar from inside a boosting fit, and can stop it.

    Returning False from ``after_iteration`` stops CatBoost mid-fit, so Stop
    works during the slow part, not only between folds. Ticks every ``every``
    iterations; finer is more UI traffic than anyone can see.
    """

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


def _grouped_cv_predictions(X, y, groups, cfg: Config, n_splits: int,
                            reporter=None, label: str = ""):
    """Out-of-fold predictions with folds split on whole movies.

    ``X`` holds the same columns the saved model is fitted on.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from ..progress import NullReporter

    reporter = reporter or NullReporter()
    pred = np.empty(len(y), dtype=object)
    conf = np.full(len(y), np.nan, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
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
        # The probability OF THE PREDICTED CLASS, not the maximum: taken as the
        # maximum they could name different classes wherever `predict` and an
        # argmax over the posterior disagree, and the number would then be the
        # confidence in an answer nobody was given.
        try:
            proba = np.asarray(model.predict_proba(x_test), dtype=float)
            where = {c: i for i, c in enumerate(np.asarray(model.classes_))}
            columns = np.array([where[p] for p in fold_pred])
            conf[test_idx] = proba[np.arange(len(columns)), columns]
        except Exception:
            pass          # a model without a posterior still gives predictions
    return pred, conf


def _feature_importance(model, X, y, columns, seed, n_repeats=5, top=15):
    """Which features the model actually leans on.

    CatBoost carries its own importances from the fit; permutation importance is
    the sklearn fallback, at a full re-scoring per repeat per column.
    """
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
    """Fit one category group. Returns (model, report); model is None if unusable.

    ``holdout``, ``have_features`` and ``audit_population`` are project-wide and
    identical for every group, so :func:`train` builds them once and passes them
    down. Each is computed here when absent, which is what a caller fitting a
    single group gets: the holdout then comes from ``labels``, the table this
    call is already fitting.
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
            # The DENOMINATOR for `audit_n`: audit clips this project has at
            # all, not only the ones a human has reached. Counted over the
            # feature table, since a clip with no features cannot be scored
            # whether it is held out or not.
            audit_population = holdout.count(feature_ids)

    wanted = labelled_rows(labels[["clip_id", group_name]], group_name)
    merged = features.merge(wanted, on="clip_id", how="inner")

    # The join above is where labels go to die: clip_id embeds the track id, so
    # re-running detection renumbers the tracks and every earlier label stops
    # resolving, dropped silently. reassociate.py is what gets them back.
    orphaned = wanted[~wanted["clip_id"].astype(str).isin(have_features)]
    orphaned_by_class = {
        str(k): int(v)
        for k, v in orphaned[group_name].astype(str).value_counts().sort_index().items()
    }

    columns = columns or feature_columns(features)

    # WHICH CLASSES GET MODELLED is decided on the full label set, BEFORE the
    # audit split, or the holdout deletes a class: two classes sitting on exactly
    # `min_per_class` each lose one to the audit and vanish. The audit measures;
    # it must not change what it measures.
    all_counts = merged[group_name].astype(str).value_counts().to_dict()

    # Rare classes are dropped rather than blocking the whole group: label
    # distributions are long-tailed. Dropped classes are reported so the GUI can
    # say the model does not cover them.
    dropped = {k: v for k, v in all_counts.items() if v < cfg.suggest.min_per_class}

    # The audit holdout is excluded from TRAINING as well as from suggestions;
    # training on it makes the "unbiased" estimate circular. Read off the CSV
    # and not recomputed: after a re-detection a labelled clip's recorded
    # membership is no longer what its track hashes to.
    audit_mask = merged["clip_id"].astype(str).map(lambda c: c in holdout)
    audit_set = merged[audit_mask].copy()
    merged = merged[~audit_mask]

    if dropped:
        merged = merged[~merged[group_name].astype(str).isin(dropped)]

    # What actually reaches the fit, after both filters. Kept apart from
    # `all_counts` so "labels that exist" and "labels trained on" stay
    # distinguishable; they differ by the audit share alone.
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
            # 0 SCORED, which is what audit_n counts: nothing was fitted, so
            # nothing was scored. The labelled count has its own field, or a
            # report that never reached a fit would claim an audit result.
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
    smallest = min(counts.values())

    X = merged[columns].to_numpy(dtype=np.float32)
    y = merged[group_name].astype(str).to_numpy()

    # Group by MOVIE, not by track: cells in one movie share illumination,
    # focus, density and treatment, which conf_mean/area_mean/present_frac proxy
    # and embeddings fingerprint, so only leave-one-movie-out answers
    # "will this work on my next movie".
    groups = merged["movie_id"].astype(str).to_numpy()
    n_movies = int(len(np.unique(groups)))
    tracks_per_clip = merged.groupby(["movie_id", "track_id"]).size()

    # n_splits must count GROUPS per class, not samples per class: a class with
    # 6 clips from 2 movies cannot support 5 folds.
    groups_per_class = int(merged.groupby(group_name)["movie_id"].nunique().min())
    n_splits = int(min(5, groups_per_class, n_movies))
    if n_splits < 2:
        if n_movies < 2:
            # Every class trips the movie count here, so naming a "scarcest"
            # one reads as a claim about its label count, which may be ample.
            return bail(
                f"all {len(merged)} training labels come from one movie "
                f"({groups[0]!r}), and the score is leave-movie-out: with a "
                "single movie nothing can be held out, so performance on an "
                "unseen movie is not estimable however many labels each class "
                "has. Annotate some clips in a second movie."
            )
        # Name the constraint that actually binds: reporting `n_movies` sends a
        # user with four annotated movies off to annotate a fifth when the real
        # problem is a class confined to one movie.
        per_class = merged.groupby(group_name)["movie_id"].nunique()
        confined = sorted(str(c) for c in per_class[per_class < 2].index)
        noun = "class" if len(confined) == 1 else "classes"
        verb = "appears" if len(confined) == 1 else "appear"
        return bail(
            f"{noun} {', '.join(map(repr, confined))} {verb} in only one of "
            f"the {n_movies} annotated movies and so cannot be held out; "
            "performance on an unseen movie is not estimable. Annotate the "
            f"{noun} in another movie, or accept that any score here is "
            "within-movie only."
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
        X, y, groups, cfg, n_splits, reporter=reporter, label=f"{group_name} ")
    macro_f1 = float(f1_score(y, pred, average="macro", labels=classes, zero_division=0))
    accuracy = float(accuracy_score(y, pred))
    baseline = float(max(counts.values()) / len(y))
    kappa = float(cohen_kappa_score(y, pred, labels=classes))

    # Bootstrap the lower bound over GROUPS (movies), the unit the folds were
    # split on; resampling clips would understate the uncertainty that matters,
    # and a single point estimate opens the gate on many pure-noise runs.
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(groups)
    kappas = []
    for _ in range(400):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        kappas.append(cohen_kappa_score(y[idx], pred[idx], labels=classes))
    kappa_lo = float(np.percentile(kappas, 5)) if len(kappas) >= 50 else float("-inf")

    # Gate on Cohen's kappa, not accuracy-vs-majority, which fails both ways: it
    # rejects an informative minority-sensitive model whose accuracy sits below
    # the majority rate, and accepts a near-degenerate one that clears it by
    # luck. Kappa is chance-corrected against the marginals; the bootstrap lower
    # bound stops one lucky split.
    reliable = kappa_lo > 0.0 and macro_f1 >= MIN_MACRO_F1
    if kappa_lo <= 0.0:
        reason = (
            f"agreement is not distinguishable from chance "
            f"(kappa {kappa:.2f}, 5th-percentile bound {kappa_lo:.2f} <= 0 "
            f"over {n_movies} movie(s))"
        )
    elif macro_f1 < MIN_MACRO_F1:
        reason = (
            f"grouped-CV macro-F1 {macro_f1:.2f} below {MIN_MACRO_F1:.2f}"
        )
    else:
        reason = "ok"

    model = _build_model(cfg, len(classes))
    _fit(model, X, y, cfg, reporter, f"{group_name}: final fit on all {len(X)} clips")

    # The audit clips, scored by a model that never saw them: the only number
    # here free of both training label leakage and suggestion feedback.
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
                f"       -> IGNORED {len(orphaned)} of {len(wanted)} labels: their clips "
                "are not in the feature table ("
                + ", ".join(f"{k} {v}" for k, v in orphaned_by_class.items())
                + "). Run the label repair on the Model page."
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
    """Fit every category group and persist models + report.

    Clip features are reused while their cache key holds; ``rebuild_features``
    forces a rebuild.
    """
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

    # Said ONCE per run, next to the scores it qualifies, and only when it is
    # non-zero: on a project detection has only ever run once the count is zero,
    # so a non-zero one is how someone finds out theirs was re-run, and that the
    # holdout below is the original rather than a fresh draw.
    holdout = AuditHoldout.from_labels(cfg, labels)
    moved = holdout.n_moved()
    if moved:
        reporter.log(
            f"{moved} of {holdout.n_recorded} labelled clip(s) are held to the audit "
            "membership recorded when they were labelled, which is no longer what "
            "their track hashes to (detection was re-run, or the stamps predate "
            "per-track hashing). The recorded value keeps the holdout from being "
            "redrawn under the model."
        )
    elif not holdout.n_recorded and len(labels):
        reporter.log(
            f"no labelled clip in {cfg.paths.csv.name} records its audit membership, "
            "so the holdout below is derived from the clip_ids alone and will be "
            "redrawn if detection is re-run. Opening the project in the GUI writes "
            "the memberships down as they stand.",
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
            "none of the selected feature groups can be computed for this "
            f"project: {blocked}. Enable one on the Configure page."
        )

    columns = feature_columns(features, usable)
    if not columns:
        raise RuntimeError(
            f"the selected feature groups {usable} produced no columns; the "
            "clip features may predate them - rebuild them from the Model page."
        )
    # The finer cut, by name, before `columns` becomes the payload: a cut made
    # after it is written would leave the file claiming inputs the fit never saw.
    named = cfg.suggest.feature_columns
    if named:
        wanted = set(named)
        absent = [c for c in named if c not in set(columns)]
        if absent:
            raise RuntimeError(
                f"suggest.feature_columns names {len(absent)} column(s) with no "
                f"column in {CLIP_FEATURES_FILE} ({', '.join(absent[:4])}); the "
                "table may predate them, in which case rebuild it from the Model "
                "page, or drop them from suggest.feature_columns."
            )
        dropped = len(columns) - len(wanted)
        columns = [c for c in columns if c in wanted]
        reporter.log(
            f"suggest.feature_columns names {len(columns)} column(s) by hand, so "
            f"{dropped} that {', '.join(usable)} would have supplied are not in "
            "this fit"
        )
    reporter.log(f"features: {len(columns)} columns from {', '.join(usable)}")

    # Project-wide and the same for every group, so they are built here and
    # handed to `train_group`: the audit denominator hashes every clip id in the
    # feature table, which is not work to repeat per category.
    feature_ids = features["clip_id"].astype(str)
    have_features = set(feature_ids)
    audit_population = holdout.count(feature_ids)

    # Said before any group is fitted: every score below is computed on whatever
    # survived this join, and a group whose rare class was mostly orphaned
    # reports a near-zero F1 for it, which reads as "hard" rather than "gone".
    n_orphaned = int((~labels["clip_id"].astype(str).isin(have_features)).sum())
    if n_orphaned:
        reporter.log(
            f"{n_orphaned} of {len(labels)} labels refer to clips that are not in the "
            "feature table; they were ignored. This is what re-running detection does "
            "to labels written before it - the track ids change, so the clip_id they "
            "were saved under no longer exists. Run the label repair on the Model page.",
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
        # Out of the report and into its own file: a row per label would swell
        # a document that is otherwise a page of numbers per group.
        for clip_id, answer, confidence in entry.pop("cv_predictions", None) or []:
            cv_rows.append({"clip_id": clip_id, "group": group.name,
                            "predicted": answer, "confidence": confidence})
        reports[group.name] = entry
        reporter.log(
            f"{group.name}: n={report.n_labels}  kappa {report.kappa:+.3f} "
            f"[lo {report.kappa_lower:+.3f}]  macro-F1 {report.macro_f1:.3f}"
            + ("" if report.reliable else f"   -> silent: {report.reason}")
        )
        # Per class, because a loss spread evenly is a smaller model and a loss
        # concentrated in one class is a class the model can no longer see.
        if report.n_orphaned:
            reporter.log(
                f"{group.name}: {report.n_orphaned} of its labels were ignored - "
                "their clips are not in the feature table ("
                + ", ".join(f"{k} {v}" for k, v in report.orphaned_by_class.items())
                + ")",
                "warning",
            )
        if model is not None:
            models[group.name] = model

    # The feature-schema version travels with the model, because a model is
    # meaningful only against features that still mean what they meant when it
    # was fitted and the column names cannot attest to that. So does the WINDOW
    # LENGTH, which can change with the schema untouched: same version, same
    # column names, values on a different scale. `Suggester.window_problem`
    # reads it back.
    #
    # `cfg.clips.frames` and not another length because these rows come from
    # clip_features.parquet, built over the ANNOTATION clip manifest, which
    # `clips._windows` lays out at `cfg.clips.frames`. It is the NOMINAL length:
    # a truncated last window is a per-row shortfall within one grid rather than
    # a different grid, which is what `present_frac` is for.
    trained_window = int(cfg.clips.frames)
    # The embedding bases travel with the model, so its emb_* columns are
    # always read in the basis they were fitted in.
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
    # Written even when empty, and always: left behind, the previous run's file
    # would be read as this model's opinion of labels this model never saw.
    pd.DataFrame(cv_rows, columns=["clip_id", "group", "predicted", "confidence"]
                 ).to_parquet(work / CV_PREDICTIONS_FILE, index=False)
    reporter.log(
        f"{len(cv_rows)} out-of-fold prediction(s) saved to {CV_PREDICTIONS_FILE}; "
        "these are what a label can honestly be checked against"
    )
    return reports


# --------------------------------------------------------------------------- #
class Suggester:
    """Loads the trained models and produces per-clip suggestions for the GUI."""

    # Resolved from the annotation CSV by `load`, which runs before anything can
    # ask. Kept here because this object is what the GUI asks "is this clip held
    # out", and re-reading the file per question would read it on every page
    # turn.
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
        # Frames per window the training rows were measured over. None for a
        # model saved before this was recorded - those still load; see
        # `window_problem`.
        self.trained_window: int | None = None
        # suggest.context_frames the model was fitted with; 0 when unrecorded.
        self.context_frames: int = 0
        # {name: PCA} the model's emb_* columns were fitted in.
        self.embedding_bases: dict = {}
        # {name: EmbeddingSpec} the model's embeddings were read from.
        self.embedding_specs: dict = {}
        self._by_clip: dict[str, np.ndarray] = {}
        # Feature row per clip id, and per-group class probabilities over every
        # row, computed once per model so selection does not rescore the pool.
        self._row_of: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._proba: dict[str, np.ndarray] = {}
        # Bumped by every `load`, and all a caller needs to know that a batch
        # from `selection_batch` is out of date.
        self.model_epoch: int = 0
        self.load()

    def load(self) -> bool:
        import joblib

        # Before anything can fail: a load replaces the model whether it
        # succeeds or not, so a batch computed against the previous one is stale
        # either way and must not look current.
        self.model_epoch += 1
        self.problem = ""
        self.columns, self.models, self.reports = [], {}, {}
        self.features, self._by_clip = None, {}
        self._row_of, self._matrix, self._proba = {}, None, {}
        self.trained_window = None
        self.context_frames = 0
        self.embedding_bases = {}
        self.embedding_specs = {}

        # Before every early return below, because membership is asked about
        # clips whether or not a model exists. Resolved HERE and not per
        # question, so turning a page does not read the CSV. An unreadable flag
        # is allowed to raise: nobody could then say which clips are held out.
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

        # Read straight off the payload, and MISSING IS NOT AN ERROR: a model
        # trained before this was recorded has to keep loading, or an upgrade
        # invalidates every model anyone already has. A length that will not
        # parse is treated as absent, and `window_problem` reports it.
        try:
            recorded = payload.get("trained_window")
            self.trained_window = None if recorded is None else int(recorded)
        except (TypeError, ValueError):
            self.trained_window = None

        # The model's input space is fixed at fit time, so a table missing a
        # column it was fitted on makes predicting meaningless rather than
        # degraded. The name check below catches a renamed or dropped feature
        # but not a REDEFINED one, hence the version stamp: bump
        # `FEATURE_VERSION` whenever a feature's MEANING changes. Both artefacts
        # must carry it, and an unstamped one is refused rather than assumed to
        # match.
        retired = [c for c in columns if _RETIRED_CNN.match(str(c))]
        if retired:
            self.problem = (
                "this model was fitted on the retired cnn* columns, which are now "
                "emb_<name>_* embedding columns. Retrain it on the Model page."
            )
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
                f"this model uses embedding(s) {unbased} but does not carry their "
                "PCA basis and source, so its emb_* columns cannot be reproduced. "
                "Retrain it on the Model page."
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
                f"{CLIP_FEATURES_FILE} was built with version {built_with!r}. The "
                "columns may still line up by name while meaning something "
                "different - rebuild the features and retrain on the Model page."
            )
            return False

        missing = [c for c in columns if c not in features.columns]
        if missing:
            shown = ", ".join(missing[:4]) + ("..." if len(missing) > 4 else "")
            self.problem = (
                f"the model was fitted on {len(columns)} columns but "
                f"{len(missing)} of them are no longer computed ({shown}). "
                "The feature groups or the detection task changed since it was "
                "trained - retrain on the Model page."
            )
            return False

        self.columns, self.models, self.features = columns, models, features
        # Vectorised, not itertuples: that is O(clips x columns) attribute
        # lookups, and names that are not valid identifiers come back as _1, _2,
        # which silently mispairs values.
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
        """'' if a feature table built at `length` is safe for this model, else why not.

        Window length changes what a column's VALUE is worth and not what it is
        called, so a 50-frame model handed 100-frame rows passes every check
        `load` makes and predicts confidently on a scale it has never seen.

        Returns a sentence rather than raising, and one for the unrecorded case
        too, so the caller can refuse a mismatch and merely warn about "cannot
        be checked".
        """
        examples = ", ".join(_LENGTH_DEPENDENT_EXAMPLES)
        try:
            length = int(length)
        except (TypeError, ValueError):
            return (
                f"the window length to check against is {length!r}, which is not "
                "a frame count, so this model's training window could not be "
                "compared with it."
            )
        if self.trained_window is None:
            return (
                f"this model does not record the window length it was trained "
                f"at, so the {length}-frame windows these features were built "
                "over could not be checked against it. It was saved before that "
                "length was recorded. Window length changes the feature VALUES "
                f"and not their names ({examples} all scale with it), so if the "
                "two differ nothing else here will notice. Retrain to record it."
            )
        if length == self.trained_window:
            return ""
        return (
            f"this model was trained on {self.trained_window}-frame windows but "
            f"these features were built over {length}-frame windows. Window "
            "length changes the feature VALUES, not just how many rows there "
            f"are: {examples} are all measured over the window, so they scale "
            f"with it, and a model fitted at {self.trained_window} frames reads "
            f"a {length}-frame row on the wrong scale while reporting normal "
            f"confidence. Retrain the model at {length} frames, or restore the "
            f"window length it was trained at ({self.trained_window} frames)."
        )

    def refuse_window_mismatch(self, length: int, context: str, reporter=None) -> None:
        """Act on :meth:`window_problem`: refuse a mismatch, warn about silence.

        ONE decision for every caller, because this is a train/serve-skew gate:
        a second copy tightened on its own would let bulk prediction and
        selection disagree about the same model file, and the looser path would
        score on the wrong scale. A MISMATCH scores every row on a scale the
        model has never seen and nothing downstream shows it, so it raises. An
        UNRECORDED length is a model saved before the stamp existed, and
        refusing those would invalidate every model anyone already has, so it
        warns and the caller carries on.

        ``context`` is the caller's trailing sentence, which is the actionable
        half: this method is on the model and cannot know WHICH config key the
        length was read from. ``reporter`` takes the warning where there is one,
        :mod:`warnings` where there is not.
        """
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
                f"{built_context} but the model with {self.context_frames}. "
                "Retrain on the Model page."
            )
        moved = [n for n, p in bases.items() if built_bases.get(n) != emb.basis_hash(p)]
        if moved:
            return (
                f"{CLIP_FEATURES_FILE} was built with a different PCA basis for "
                f"embedding(s) {moved} than the model was fitted in. Retrain on "
                "the Model page."
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
            f"this model was fitted with suggest.context_frames "
            f"{self.context_frames} but the config says {int(context_frames)}; "
            "the ctx_* features would differ. Retrain, or set it back."
        )

    def is_audit_clip(self, clip_id: str) -> bool:
        return clip_id in self.audit

    # ---------------------------------------------------------- selection -- #
    def selection_group(self) -> str:
        """Which group's model ranks the clips, or ``""`` if none can.

        The first single-choice group in config order that the model covers,
        which is the group the GUI already records as the primary suggestion, so
        the ranking and the suggestion on screen come from one model.

        Multi-select groups are not candidates: their label is a SET, and every
        rule in :mod:`trackanno.learn.select` reads a posterior over mutually
        exclusive classes.
        """
        for group in self.cfg.categories:
            if not group.is_multi and group.name in self.models:
                return group.name
        return ""

    def selection_batch(
        self, manifest: pd.DataFrame, done_ids: set[str], n: int
    ) -> list[dict]:
        """The next `n` clips to annotate, in order, each carrying its provenance.

        Returns [] when selection is off, the model is not ready, or nothing is
        left to choose. Each record is a
        :func:`trackanno.learn.select.select_batch` record plus the ``group``
        whose model scored it; ``clip_id`` is the join key.

        Call this when the model changes (:attr:`model_epoch`), not once per
        label, so the queue does not re-order under the annotator.

        ``manifest`` is the pool (any subset with a ``clip_id`` column),
        ``done_ids`` the clips already annotated. Audit clips are never ranked.
        """
        cfg = self.cfg
        if not cfg.suggest.enabled or not cfg.suggest.selection_on:
            return []
        if not self.ready or manifest is None or not len(manifest):
            return []
        group = self.selection_group()
        if not group or int(n) <= 0:
            return []

        # Before any scoring: the features every candidate is scored from were
        # built over the annotation clip grid, whose windows are clips.frames
        # long, and a model fitted at another length ranks them on the wrong
        # scale while reporting normal confidence. An ORDER carries no
        # confidence number, so there is nothing to notice afterwards.
        self.refuse_window_mismatch(
            int(cfg.clips.frames),
            f"The {int(cfg.clips.frames)} frames are clips.frames, the length "
            f"{CLIP_FEATURES_FILE} - the table every candidate in this batch is "
            "scored from - was built over.",
        )

        ids = manifest["clip_id"].astype(str).to_numpy()

        # A clip with no feature row is left OUT of the pool rather than
        # imputed: an imputed row still ranks, on numbers the clip does not
        # have. The ordinary state after re-running detection rebuilds the
        # manifest and not the feature table, so it is a filter, not an error.
        rows = [self._row_of.get(clip_id) for clip_id in ids]
        keep = np.array([row is not None for row in rows], dtype=bool)
        if not keep.any():
            return []
        pool = np.flatnonzero(keep)
        pool_ids = ids[pool]

        done = {str(c) for c in (done_ids or ())}
        # Audit clips stay out of the ranking so the holdout remains a random
        # sample of the project.
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
                # A near-uniform posterior is noise with a number attached, and
                # it still anchors the annotator.
                if confidence < self.cfg.suggest.min_confidence:
                    continue
                out[group] = (str(model.classes_[idx]), confidence)
            except Exception as exc:
                print(f"[trackanno] suggestion failed for {group}: {exc}")
                continue
        return out

    def status_line(self) -> str:
        # The problem comes first: a stale model is not the same as no model, and
        # telling someone to train one they already trained fixes nothing.
        if self.problem:
            return self.problem
        if not self.reports:
            # A loaded model with no report beside it: train() writes both, so
            # the report was deleted, not the model. "No model yet" here would
            # send people to retrain something that works.
            return (
                f"model loaded ({len(self.models)} group(s), {len(self.columns)} "
                "columns) but its report is missing"
                if self.models else "no model yet - train one on the Model page"
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
        # Against the POPULATION where the report carries one, so a result on
        # 12 audit clips is read next to how much of the holdout that is.
        # Printed even when nothing was scored: "the holdout exists and is
        # empty" is the more actionable answer.
        #
        # THREE cases, kept apart. A report written before `audit_population`
        # existed has no such key, and a `.get` default of 0 would mean both
        # "this report cannot say" and "this project holds nothing out" - two
        # facts with different answers (retrain / raise suggest.audit_fraction).
        # So the missing key is read as missing.
        population = rep.get("audit_population")
        labelled = rep.get("audit_labelled", rep.get("audit_n", 0))
        scored = rep.get("audit_n") or 0
        ak, aa = rep.get("audit_kappa"), rep.get("audit_accuracy")
        ak = "n/a" if ak is None or ak != ak else f"{ak:+.3f}"
        aa = "n/a" if aa is None or aa != aa else f"{aa:.3f}"
        if population is None:
            # An older report, gated on audit_n as the audit line was before the
            # population was recorded, with the missing denominator named rather
            # than guessed at.
            if scored:
                print(
                    f"  audit holdout ({scored} scored, never suggested nor "
                    f"trained on): kappa {ak}  accuracy {aa}"
                )
                print(
                    "    this report predates the audit population count, so what "
                    "share of the holdout those clips are is not recorded in it; "
                    "retrain to get the denominator back"
                )
        elif population:
            of_population = f"{labelled} of {population} audit clips labelled"
            if scored:
                print(
                    f"  audit holdout ({of_population}, {scored} scored, "
                    f"never suggested nor trained on): kappa {ak}  accuracy {aa}"
                )
            else:
                print(
                    f"  audit holdout ({of_population}): too few to score, so there "
                    "is no unbiased estimate for this group yet"
                )
        else:
            # The population was counted and came out 0: this project holds
            # nothing out. Said rather than skipped, because a report with no
            # audit line reads as one that forgot to compute it.
            print(
                "  audit holdout: none - no clip in this project is held out, so "
                "every number below was computed on clips the model was fitted on "
                "some fold of. suggest.audit_fraction is what creates a holdout"
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
    """What each feature group is worth, on this project's own labels.

    The table is built in memory, so the saved model's files are untouched.
    """
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
    # Whole groups this project CAN compute, not the configured ones, so the
    # rows are comparable.
    if cfg.suggest.feature_columns:
        print(f"\n(suggest.feature_columns names {len(cfg.suggest.feature_columns)} "
              "column(s) by hand; the subsets below are whole feature groups and "
              "ignore it)")
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
