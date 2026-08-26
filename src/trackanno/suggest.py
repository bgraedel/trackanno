"""Train a classifier on existing labels and suggest categories for new clips.

    uv run suggest.py --train      # build features, fit, cross-validate, save
    uv run suggest.py --report     # print the last report
    uv run suggest.py --ablate     # kinematic vs CNN vs both

Design constraints that matter more than the model choice:

* **Cross-validation is grouped by (movie, track).** The CNN features identify
  individuals almost perfectly, so ungrouped folds would put two windows of the
  same animal in train and test and score mostly re-identification.
* **The reported number is always the grouped-CV score, never the training score.**
* **The model stays silent unless it clears a bar** (`suggest.min_per_class` and
  `suggest.min_macro_f1`): a confidently wrong suggester is worse than none,
  because the annotator's corrections train the next one.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .clipfeatures import (
    CLIP_CNN_RAW_FILE, CLIP_FEATURES_FILE, FEATURE_VERSION, build_clip_features,
    feature_columns,
)
from .config import Config, load_config
from .select import select_batch
from .store import AUDIT_COLUMN, decode_audit, read_audit_records

MODEL_FILE = "suggest_model.joblib"
REPORT_FILE = "suggest_report.json"

# Columns whose VALUE depends on how many frames the window spans, named in the
# mismatch message because "the features mean something else now" is not
# actionable and "path_length is longer because the window is longer" is.
#
# Verified against trackseries/clipfeatures rather than guessed. Length-dependent
# are the extensive quantities (`path_length`, `net_displacement`,
# `radius_gyration`, the `body_*` equivalents), everything derived from them
# (`straightness`, `body_straightness`, `body_displacement_per_len`), the
# lag-based turn statistics through `turn_lag_frames`/`turn_n_valid`, every
# dispersion (`*_std`, `flagellum_len_cv`, `tip_angle_std`), the order
# statistics (`speed_max`, `speed_p90`) and `present_frac`.
#
# Length-INVARIANT, and so no reason to refuse on their own: every `beat_*` and
# `wave_*` column, which are measured on a window pinned to `cfg.spectral.window`
# and not on the requested range at all, plus the intensive means
# (`speed_mean`, `area_mean`, `body_len_mean`, `immotile_frac`, ...).
_LENGTH_DEPENDENT_EXAMPLES = ("path_length", "net_displacement", "speed_std")

# --------------------------------------------------------------------------- #
# which columns survive a change of window length
# --------------------------------------------------------------------------- #
# The paragraph above says which families move with the window. This turns it
# into a predicate, because two things now have to agree about it and a prose
# comment cannot hold them together: `train` uses it to FIT a model that may
# legitimately be scored at several window lengths, and
# :meth:`Suggester.scale_problem` uses it to REFUSE a model that may not. Two
# copies of the rule would agree until one was edited, after which a project
# would fit on one subset and be checked against another.
#
# An ALLOWLIST, deliberately, and this is the whole safety argument. Under a
# denylist a column added to the schema later matches no "length-dependent"
# pattern, is treated as invariant, and is scored at three window lengths with
# nothing raising - which is precisely the silent skew this machinery exists to
# stop. Under an allowlist the same new column is treated as dependent, costs a
# little signal, and says so. Wrong in the recoverable direction.
#
# Two ways in:
#
# * ``beat_*`` and ``wave_*`` are invariant BY CONSTRUCTION, not by measurement.
#   Since ``FEATURE_VERSION`` 2 they are measured over a window of
#   ``cfg.spectral.window`` frames CENTRED on the requested range rather than
#   over the range itself (:func:`trackanno.clipfeatures._beat_on_window`,
#   :func:`trackanno.wavespectra.window_values`), so the requested length does
#   not enter their definition at all.
# * An INTENSIVE reduction of a per-frame series - a mean, a median, or the
#   mean of an indicator, which is what a ``_frac`` is. Averaging more frames
#   does not scale the average. The reducer is legible in the suffix because
#   :data:`trackanno.trackseries.SERIES_SPEC` names features
#   ``column + reducer``, so ``_std``, ``_p90`` and ``_max`` - the dispersions
#   and order statistics, which do grow with the number of samples - are
#   excluded by not being on this list rather than by being on another one.
#   ``cnn{i}_mean`` comes through the same door: it is the mean of the window's
#   per-detection embeddings, with no length term.
_LENGTH_INVARIANT_PREFIXES = ("beat_", "wave_")
_INTENSIVE_SUFFIXES = ("_mean", "_median", "_frac")

# Named because the suffix rule would otherwise wave them through, and each is a
# mean of something the window length itself changes:
#
# * ``present_frac``'s denominator IS the window. A longer window at a track's
#   end reaches further past the last detection, so the fraction falls with
#   length even though the animal did nothing different.
# * ``turn_abs_mean`` averages turns measured over ``turn_lag_frames``, and
#   :func:`trackanno.clipfeatures._heading_lag` sets that lag to
#   ``min(n_frames // 4, 25)`` - an explicit function of the window length. The
#   mean is intensive; the quantity being averaged is not the same quantity.
_LENGTH_DEPENDENT_DESPITE_SUFFIX = frozenset({"present_frac", "turn_abs_mean"})

# `shape_elong` reduced by `mean`, renamed in `trackseries._FEATURE_NAMES`
# because renaming it back would change the input schema of every model already
# trained. An intensive mean wearing no suffix, so it has to be named here.
_LENGTH_INVARIANT_NAMES = frozenset({"shape_elongation"})

# Deliberately NOT on the list, so that their absence reads as a decision rather
# than an oversight: `body_rotation_speed`, `tip_angular_speed` and
# `cnn*_delta` are all per-step means and so are intensive by construction, but
# none of them has been measured across window lengths in this repository, and
# the allowlist's rule is that construction alone is enough only where the
# window length is absent from the definition outright (the two spectral
# families above). Measure one and it can move.


def is_length_invariant(column: str) -> bool:
    """Whether ``column``'s VALUE survives a change of window length.

    Not whether it is a good feature, and not whether it is stable: a column can
    be invariant and useless. It answers the one question that decides whether a
    model fitted at 50 frames may be shown a 100-frame row.
    """
    name = str(column)
    if name.startswith(_LENGTH_INVARIANT_PREFIXES):
        return True
    if name in _LENGTH_INVARIANT_NAMES:
        return True
    if name in _LENGTH_DEPENDENT_DESPITE_SUFFIX:
        return False
    return name.endswith(_INTENSIVE_SUFFIXES)


def length_invariant_columns(columns) -> list[str]:
    """``columns`` in their original order, keeping only the invariant ones."""
    return [c for c in columns if is_length_invariant(c)]


def length_dependent_columns(columns) -> list[str]:
    """``columns`` in their original order, keeping only the ones that scale.

    The complement of :func:`length_invariant_columns`, spelled separately
    because the refusal has to NAME the offenders: "your features scale with the
    window" is not actionable and "path_length, speed_std and 71 others do" is.
    """
    return [c for c in columns if not is_length_invariant(c)]


def scoped_columns(columns, scope: str) -> list[str]:
    """``columns`` narrowed by ``suggest.feature_scope``.

    The second axis of feature selection, applied after
    :func:`trackanno.clipfeatures.feature_columns` has applied the first. They
    compose rather than overlap: the groups say which instrument a column came
    from, this says how it behaves when the window length changes, and a project
    that wants "the spectral columns only, at three scales" gets there by
    setting both rather than by typing out 34 column names.

    An unknown scope RAISES rather than falling through to "all". Falling
    through would fit on more columns than were asked for and report success,
    which is the failure mode a typo in a config key already has too much of.
    """
    if scope == "all":
        return list(columns)
    if scope == "length_invariant":
        return length_invariant_columns(columns)
    raise ValueError(
        f"unknown suggest.feature_scope {scope!r}; expected 'all' or "
        "'length_invariant'"
    )


def is_audit_clip(clip_id: str, cfg: Config,
                  recorded: Mapping[str, bool] | None = None) -> bool:
    """Hold a share of clips out of the suggestion loop: never suggested on,
    never trained on, so agreement measured there is unbiased.

    THE RECORD WINS, and the hash is only consulted where there is none.
    ``clip_id`` embeds the track id, so re-running detection renumbers the
    tracks, every id changes, and a hash of the id redraws the holdout from
    scratch - roughly ``2*f*(1-f)`` of the clips change side, about a fifth at
    the default fraction. ``reassociate`` then rewrites the id in place to put
    each label back on its clip, so a label earned on a held-out clip can arrive
    under an id that hashes into the training set: the next fit trains on the
    holdout and reports ``audit_accuracy`` over a different, partly overlapping
    set. Membership is therefore written into the CSV when a clip is LABELLED
    (``store.AUDIT_COLUMN``) and read back here.

    A recorded value that disagrees with the hash is the NORMAL state of a
    re-detected project, not an error, and is not reported per clip; the count
    is surfaced once, by :func:`train` and by the store on open, because it is
    the only way anyone learns their project was re-detected.

    ``recorded`` maps ``clip_id`` to its recorded membership, holding only the
    clips that HAVE one. Absent (the default) means no records were available,
    which is the pre-existing behaviour exactly - so ``predict`` and any caller
    that only ever asks about unlabelled clips need no records and no change.

    Uses its OWN seed - see :meth:`trackanno.config.Suggest.audit_by_hash`.
    """
    if recorded:
        member = recorded.get(str(clip_id))
        if member is not None:
            return bool(member)
    return cfg.suggest.audit_by_hash(str(clip_id))


def audit_records(labels: pd.DataFrame | None) -> dict[str, bool]:
    """``{clip_id: was_held_out}`` from an annotation table already in memory.

    The same mapping :func:`trackanno.store.read_audit_records` builds from the
    file, for the callers that have the frame instead (``train`` reads the CSV
    with pandas once and hands it to every group). Sharing the codec rather than
    re-deriving the flag is what keeps the two readings from disagreeing about a
    cell - a disagreement that would put a clip in the training set for one
    group and the holdout for another.
    """
    if labels is None or AUDIT_COLUMN not in getattr(labels, "columns", ()):
        return {}
    out: dict[str, bool] = {}
    for clip_id, cell in zip(labels["clip_id"].astype(str), labels[AUDIT_COLUMN]):
        member = decode_audit(cell)
        if member is not None:
            out[clip_id] = member
    return out


__all__ = ["Suggester", "train", "main", "is_audit_clip", "audit_records",
           "is_length_invariant", "length_invariant_columns",
           "length_dependent_columns", "scoped_columns",
           "MODEL_FILE", "REPORT_FILE"]


def _selection_slots(n: int, share: float, ranked: int, audit: int) -> int:
    """The largest batch <= ``n`` that the two candidate pools can actually fill.

    :func:`trackanno.select.select_batch` refuses a batch bigger than its pools
    on the grounds that the caller built the mask and so knows the pool size
    before it asks. This is that caller doing the knowing. A project simply runs
    out of clips near the end, and an annotator who has labelled all but three
    should be handed three, not an exception.

    Both pools bind, and they bind separately: the ranked slots need ranked
    candidates and the reserved slots need unannotated AUDIT candidates, and
    topping one up from the other is exactly what the stratum exists to prevent.
    The reserve is not linear in the batch size - it is ``int(share * n + 0.5)``,
    a rounding - so this searches downward from ``n`` rather than solving for it,
    over a batch size that is a screenful of clips.

    Returns ``0`` when nothing fits, which is the "nothing left to choose" the
    caller reports as an empty batch.
    """
    for size in range(int(n), 0, -1):
        reserved = int(float(share) * size + 0.5) if audit else 0
        if reserved <= audit and size - reserved <= ranked:
            return size
    return 0


# --------------------------------------------------------------------------- #
@dataclass
class GroupReport:
    group: str
    n_labels: int
    # Labels whose clip_id is absent from the feature table, so they never
    # reached the fit. Per class as well as total: 31% missing overall hurts,
    # 77% of ONE class destroys the model.
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
    # How the CNN basis was fitted for the CROSS-VALIDATED score: a per-fold PCA,
    # while the saved model uses the global one already in clip_features.parquet.
    # Both deliberate, but different fits - `top_features` is the saved model's.
    cv_pca: str
    # How the CNN basis was fitted for the AUDIT score, which is a different
    # answer and an honest admission rather than a description. The audit clips
    # are scored on `columns`, whose `cnn*` block comes from a PCA fitted in
    # `clipfeatures` over the detections of every prepared movie - the audit
    # clips' own included. The CV path refits that basis inside each fold and so
    # keeps the held-out movie out of it; the audit path has no folds and does
    # not. The labels never leak, so this is unsupervised leakage only and is
    # probably small - but "probably small" is not measured here, and the figure
    # it inflates is the one number this tool asks people to trust. Stated in
    # the report so a reader is not left to infer it from the source. NOT fixed:
    # a per-audit-clip refit is a different change with its own cost.
    audit_pca: str
    audit_n: int                      # audit clips actually SCORED below
    # The audit population and the labelled part of it, because `audit_n` alone
    # cannot be read: "10 audit clips, all correct" and "10 of 200 audit clips
    # have been labelled" are the same 10 and completely different claims about
    # how much of the holdout has been spent.
    audit_population: int             # audit clips this project HAS, labelled or not
    audit_labelled: int               # of those, how many carry a label for this group
    audit_accuracy: float
    audit_kappa: float
    confusion: list
    top_features: list
    reliable: bool
    reason: str


def _build_model(cfg: Config, n_classes: int):
    """The classifier behind the suggestions.

    CatBoost is the default because **the label distribution is lopsided**: one
    state dominates and an unweighted model answers by predicting it, while
    ``auto_class_weights="Balanced"`` reweights the loss by inverse class
    frequency. It also takes NaN natively, which matters because the kinematic
    features use NaN to mean "undefined" (no beat detected, straightness of a
    stationary track), and 0.0 would disguise those as real extremes.
    """
    if cfg.suggest.model == "catboost":
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=cfg.suggest.iterations,
                # CatBoost grows SYMMETRIC trees, so depth alone sets the width:
                # 2**depth leaves, no separate knob, and depth 6 is already 64 -
                # which overfits fast on a few hundred hand-labelled clips.
                depth=cfg.suggest.depth,
                learning_rate=cfg.suggest.learning_rate,
                l2_leaf_reg=cfg.suggest.l2_leaf_reg,
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                auto_class_weights="Balanced" if cfg.suggest.balance_classes else None,
                verbose=False, random_seed=cfg.suggest.seed, allow_writing_files=False,
            )
        except ImportError:
            # catboost is an ordinary dependency, so this means a broken
            # environment rather than a missing extra. Say so and carry on with
            # sklearn instead of failing the whole run.
            print(
                "suggest.model is 'catboost' but catboost will not import; "
                "using sklearn instead. Reinstall to get the default estimator.",
                file=sys.stderr,
            )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=cfg.suggest.iterations,
        max_depth=cfg.suggest.depth,
        learning_rate=cfg.suggest.learning_rate,
        l2_regularization=cfg.suggest.l2_leaf_reg,
        # sklearn's equivalent of auto_class_weights; it has no "Balanced"
        # string for this estimator until 1.4, hence the guard.
        class_weight="balanced" if cfg.suggest.balance_classes else None,
        early_stopping=False, random_state=cfg.suggest.seed,
    )


def describe_model(cfg: Config, n_classes: int, n_rows: int, n_cols: int) -> str:
    """One line naming the estimator and every hyperparameter that shapes the fit.

    A score is only interpretable next to what produced it. Written to the
    reporter, so it lands in the GUI's log.
    """
    s = cfg.suggest
    if s.model == "catboost":
        try:
            import catboost  # noqa: F401
            backend = f"CatBoost {catboost.__version__}"
            shape = (f"depth {s.depth} (symmetric, {2 ** s.depth} leaves), "
                     f"iterations {s.iterations}, lr {s.learning_rate}, "
                     f"l2_leaf_reg {s.l2_leaf_reg}")
            loss = "MultiClass" if n_classes > 2 else "Logloss"
            weights = "Balanced" if s.balance_classes else "none"
            extra = f", loss {loss}, class weights {weights}"
        except ImportError:
            backend = "sklearn HistGradientBoosting (catboost not installed)"
            shape = (f"max_depth {s.depth}, max_iter {s.iterations}, "
                     f"lr {s.learning_rate}, l2 {s.l2_leaf_reg}")
            extra = f", class weights {'balanced' if s.balance_classes else 'none'}"
    else:
        backend = "sklearn HistGradientBoosting"
        shape = (f"max_depth {s.depth}, max_iter {s.iterations}, "
                 f"lr {s.learning_rate}, l2 {s.l2_leaf_reg}")
        extra = f", class weights {'balanced' if s.balance_classes else 'none'}"
    return (f"{backend}: {shape}{extra}, seed {s.seed} | "
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
    reporter.stage(label, cfg.suggest.iterations)
    if cfg.suggest.model == "catboost" and hasattr(model, "get_feature_importance"):
        try:
            model.fit(X, y, callbacks=[_IterationTicker(reporter)])
            reporter.raise_if_cancelled()
            return model
        except TypeError:
            # The GPU backend rejects `callbacks`; fit without the ticker.
            pass
    model.fit(X, y)
    reporter.advance(cfg.suggest.iterations)
    reporter.raise_if_cancelled()
    return model


def _grouped_cv_predictions(X, y, groups, cfg: Config, n_splits: int, raw_cnn=None,
                            reporter=None, label: str = ""):
    """Out-of-fold predictions with folds split on whole movies.

    When ``raw_cnn`` is supplied the CNN PCA is refitted **inside each fold** on
    training rows only, and ``X`` carries the kinematic features alone. One PCA
    over everything lets the held-out movie into the basis and the global mean,
    where between-movie variance dominates - the leak that flatters this score.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedGroupKFold

    from .progress import NullReporter

    reporter = reporter or NullReporter()
    pred = np.empty(len(y), dtype=object)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.suggest.seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        held_out = sorted(set(np.asarray(groups)[test_idx]))
        reporter.log(
            f"{label}fold {fold}/{n_splits}: holding out {', '.join(map(str, held_out))} "
            f"({len(test_idx)} clips), fitting on {len(train_idx)}"
        )
        x_train, x_test = X[train_idx], X[test_idx]
        if raw_cnn is not None:
            n_comp = int(min(cfg.suggest.pca_components, len(train_idx), raw_cnn.shape[1]))
            if n_comp >= 2:
                fold_pca = PCA(n_components=n_comp, random_state=cfg.suggest.seed)
                fold_pca.fit(raw_cnn[train_idx])
                x_train = np.hstack([x_train, fold_pca.transform(raw_cnn[train_idx])])
                x_test = np.hstack([x_test, fold_pca.transform(raw_cnn[test_idx])])
        model = _build_model(cfg, len(np.unique(y)))
        _fit(model, x_train, y[train_idx], cfg, reporter,
             f"{label}fold {fold}/{n_splits}")
        pred[test_idx] = model.predict(x_test).ravel()
    return pred


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
    columns: list[str] | None = None, verbose: bool = True,
    raw_cnn: np.ndarray | None = None, reporter=None,
) -> tuple[object | None, GroupReport]:
    """Fit one category group. Returns (model, report); model is None if unusable."""
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    wanted = labels[["clip_id", group_name]].dropna(subset=[group_name])
    wanted = wanted[wanted[group_name].astype(str).str.len() > 0]
    merged = features.merge(wanted, on="clip_id", how="inner")

    # The join above is where labels go to die: clip_id embeds the track id, so
    # re-running detection renumbers the tracks and every earlier label stops
    # resolving, dropped silently. reassociate.py is what gets them back.
    have_features = set(features["clip_id"].astype(str))
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

    # Hold the audit clips out of TRAINING as well as out of the display;
    # training on them makes the "unbiased" estimate circular.
    #
    # Read off the CSV, not recomputed: a labelled clip's membership is whatever
    # was recorded when it was labelled, and after a re-detection that is no
    # longer what its clip_id hashes to. Recomputing here is what would let a
    # clip the model was deliberately kept away from walk into this fit.
    recorded = audit_records(labels)
    audit_mask = merged["clip_id"].astype(str).map(lambda c: is_audit_clip(c, cfg, recorded))
    audit_set = merged[audit_mask].copy()
    merged = merged[~audit_mask]

    # The DENOMINATOR for `audit_n`: audit clips this project has at all, not
    # only the ones a human has reached. Counted over the feature table because
    # that is the set of clips that could be scored; a clip with no features
    # cannot be audited whether it is held out or not.
    audit_population = int(sum(
        is_audit_clip(c, cfg, recorded) for c in features["clip_id"].astype(str)
    ))
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
            scope="not validated", cv_pca="not validated",
            audit_pca="not scored",
            # 0 SCORED, which is what audit_n counts: nothing was fitted, so
            # nothing was scored. The labelled count goes in its own field
            # rather than standing in here, or a report that never reached a fit
            # would claim an audit result it does not have.
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

    # Drop the globally-PCA'd CNN columns; the CV gets the un-reduced aggregate
    # instead, so each fold fits its own PCA.
    fold_cnn, X_cv = None, X
    if raw_cnn is not None and "_row" in merged.columns:
        cv_columns = [c for c in columns if not c.startswith("cnn")]
        X_cv = merged[cv_columns].to_numpy(dtype=np.float32)
        fold_cnn = raw_cnn[merged["_row"].to_numpy()]

    # Group by MOVIE, not by track: with clips.window_mode: first there is
    # exactly one clip per track, making track groups identical to ungrouped CV,
    # and cells in one movie share illumination, focus, density and treatment,
    # which conf_mean/area_mean/present_frac proxy and the CNN embeddings
    # fingerprint. Only leave-one-movie-out answers "will this work on my next
    # movie".
    groups = merged["movie_id"].astype(str).to_numpy()
    n_movies = int(len(np.unique(groups)))
    tracks_per_clip = merged.groupby(["movie_id", "track_id"]).size()

    # n_splits must count GROUPS per class, not samples per class: a class with
    # 6 clips from 2 movies cannot support 5 folds.
    groups_per_class = int(merged.groupby(group_name)["movie_id"].nunique().min())
    n_splits = int(min(5, groups_per_class, n_movies))
    if n_splits < 2:
        # Name the constraint that actually binds: reporting `n_movies` sends a
        # user with four annotated movies off to annotate a fifth when the real
        # problem is a class confined to one movie.
        scarcest = merged.groupby(group_name)["movie_id"].nunique().idxmin()
        return bail(
            f"class {scarcest!r} appears in only {groups_per_class} of the "
            f"{n_movies} annotated movie(s), so it cannot be held out; performance "
            "on an unseen movie is not estimable. Annotate that class in another "
            "movie, or accept that any score here is within-movie only."
        )

    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    # Said before the fit takes any time. The width reported is the one the model
    # is fitted on, not the raw CNN aggregate handed to the fold PCA: 5760 raw
    # dimensions reduce to `pca_components`, and the larger figure describes a
    # model nobody fits.
    fold_width = X_cv.shape[1]
    if fold_cnn is not None:
        fold_width += int(min(cfg.suggest.pca_components, len(merged), fold_cnn.shape[1]))
    reporter.log(describe_model(cfg, len(classes), len(merged), fold_width))
    if fold_cnn is not None:
        reporter.log(
            f"CNN basis refitted inside each fold ({fold_cnn.shape[1]}-d aggregate "
            f"-> {fold_width - X_cv.shape[1]} components); the SAVED model instead uses "
            f"the global PCA already in the feature table, {len(columns)} columns in total"
        )
    reporter.log(
        f"{group_name}: {n_splits}-fold leave-movie-out over {n_movies} movie(s), "
        f"classes {', '.join(classes)}"
        + (f"; {len(audit_set)} of this project's {audit_population} audit clips are "
           "labelled and held out entirely" if len(audit_set) else "")
    )
    pred = _grouped_cv_predictions(X_cv, y, groups, cfg, n_splits, raw_cnn=fold_cnn,
                                   reporter=reporter, label=f"{group_name} ")
    macro_f1 = float(f1_score(y, pred, average="macro", labels=classes, zero_division=0))
    accuracy = float(accuracy_score(y, pred))
    baseline = float(max(counts.values()) / len(y))
    kappa = float(cohen_kappa_score(y, pred, labels=classes))

    # Bootstrap the lower bound over GROUPS (movies), the unit the folds were
    # split on; resampling clips would understate the uncertainty that matters.
    # A single point estimate opens the gate on ~27% of pure-noise runs here.
    rng = np.random.default_rng(cfg.suggest.seed)
    unique_groups = np.unique(groups)
    kappas = []
    for _ in range(400):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        kappas.append(cohen_kappa_score(y[idx], pred[idx], labels=classes))
    kappa_lo = float(np.percentile(kappas, 5)) if len(kappas) >= 50 else float("-inf")

    # Gate on Cohen's kappa, not accuracy-vs-majority, which fails BOTH ways: it
    # rejects an informative minority-sensitive model (kappa 0.41, balanced
    # accuracy 0.75, accuracy below the majority rate) and accepts a
    # near-degenerate one that clears it by luck. Kappa is chance-corrected
    # against the marginals; the bootstrap lower bound stops one lucky split.
    reliable = kappa_lo > 0.0 and macro_f1 >= cfg.suggest.min_macro_f1
    if kappa_lo <= 0.0:
        reason = (
            f"agreement is not distinguishable from chance "
            f"(kappa {kappa:.2f}, 5th-percentile bound {kappa_lo:.2f} <= 0 "
            f"over {n_movies} movie(s))"
        )
    elif macro_f1 < cfg.suggest.min_macro_f1:
        reason = (
            f"grouped-CV macro-F1 {macro_f1:.2f} below suggest.min_macro_f1 "
            f"{cfg.suggest.min_macro_f1:.2f}"
        )
    else:
        reason = "ok"

    model = _build_model(cfg, len(classes))
    _fit(model, X, y, cfg, reporter, f"{group_name}: final fit on all {len(X)} clips")

    # The audit clips, scored by a model that never saw them: the only number
    # here free of both training LABEL leakage and suggestion feedback. Not free
    # of the unsupervised kind - `columns` carries the globally-fitted CNN basis
    # and these rows were in the fit of it. See `GroupReport.audit_pca`, which is
    # where that is written down rather than left to be discovered.
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
        cv_pca=(
            "per-fold CNN PCA (saved model uses the global one)"
            if fold_cnn is not None else
            "no CNN columns" if not any(c.startswith("cnn") for c in columns)
            # Either clip_cnn_raw.npy is missing or the caller withheld it (see
            # suggest.feature_scope in `train`); the run log says which, and
            # what matters here is the same either way - the held-out movie's
            # detections were in the basis these columns came from.
            else "global CNN PCA (no fold-wise basis in use)"
        ),
        audit_pca=(
            "global CNN PCA, fitted over every prepared movie INCLUDING these "
            "audit clips' own detections - unsupervised leakage into this score, "
            "unmeasured. The labels never enter it. Not refitted per audit clip "
            "the way the CV refits per fold."
            if any(c.startswith("cnn") for c in columns) else "no CNN columns"
        ),
        audit_n=audit_n,
        audit_population=audit_population,
        audit_labelled=len(audit_set),
        audit_accuracy=round(audit_acc, 4) if np.isfinite(audit_acc) else float("nan"),
        audit_kappa=round(audit_kappa, 4) if np.isfinite(audit_kappa) else float("nan"),
        confusion=confusion_matrix(y, pred, labels=classes).tolist(),
        top_features=_feature_importance(model, X, y, columns, cfg.suggest.seed),
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


def train(cfg: Config, rebuild_features: bool = True, verbose: bool = True,
          reporter=None) -> dict:
    """Build features, fit every category group, persist models + report."""
    import joblib

    from .progress import NullReporter

    reporter = reporter or NullReporter()
    work = cfg.paths.work_dir
    path = work / CLIP_FEATURES_FILE
    features = (
        build_clip_features(cfg, verbose=verbose, reporter=reporter)
        if rebuild_features or not path.exists()
        else pd.read_parquet(path)
    )
    if not cfg.paths.csv.exists():
        raise FileNotFoundError(f"no annotations yet at {cfg.paths.csv}")
    labels = pd.read_csv(cfg.paths.csv)
    if labels.empty:
        raise RuntimeError(f"{cfg.paths.csv} has no rows")

    # Said ONCE per run, next to the scores it qualifies, and only when it is
    # non-zero. A labelled clip is held to the membership recorded when it was
    # labelled; disagreeing with the hash is what a re-detected project looks
    # like and is not an error, so there is no per-row warning. The count is
    # still news: on a project detection has only ever run once it is zero, so a
    # non-zero number is how someone finds out theirs was re-run - and that the
    # holdout below is the original one rather than a fresh draw.
    recorded = audit_records(labels)
    moved = sum(1 for clip_id, member in recorded.items()
                if member != cfg.suggest.audit_by_hash(clip_id))
    if moved:
        reporter.log(
            f"{moved} of {len(recorded)} labelled clip(s) are held to the audit "
            "membership recorded when they were labelled, which is no longer what "
            "their clip_id hashes to - detection has been re-run since. The recorded "
            "value is what keeps the holdout from being redrawn under the model."
        )
    elif not recorded and len(labels):
        reporter.log(
            f"no labelled clip in {cfg.paths.csv.name} records its audit membership, "
            "so the holdout below is derived from the clip_ids alone and will be "
            "redrawn if detection is re-run. Opening the project in the GUI writes "
            "the memberships down as they stand.",
            "warning",
        )

    # Positional tag, so the un-reduced CNN aggregate can be re-indexed after the
    # label merge reorders rows.
    features = features.reset_index(drop=True)
    features["_row"] = np.arange(len(features))
    raw_path = work / CLIP_CNN_RAW_FILE
    raw_cnn = None
    if raw_path.exists():
        candidate = np.load(raw_path)
        if len(candidate) == len(features):
            raw_cnn = candidate
        else:
            print(
                f"  {CLIP_CNN_RAW_FILE} has {len(candidate)} rows but there are "
                f"{len(features)} clips; falling back to the global PCA for CV",
                file=sys.stderr,
            )

    from .clipfeatures import available_groups

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

    # The fold-wise CNN basis is only wanted when the CNN columns are in the
    # model: `_grouped_cv_predictions` appends it on `raw_cnn is not None` alone,
    # so gating on file existence would fit a CNN-free model and cross-validate a
    # CNN-carrying one. The CV has to score the model that is saved.
    if raw_cnn is not None and "cnn" not in usable:
        raw_cnn = None
        reporter.log(
            "feature_groups excludes 'cnn', so the fold-wise CNN basis is not "
            "used either - the reported score is for the CNN-free model",
        )

    columns = feature_columns(features, usable)
    if not columns:
        raise RuntimeError(
            f"the selected feature groups {usable} produced no columns; the "
            "clip features may predate them - rebuild them from the Model page."
        )
    # The second cut, and it has to happen HERE rather than per group: `columns`
    # is what gets written into the model payload, and that payload is what
    # `Suggester.scale_problem` reads back to decide whether the model may be
    # scored at several window lengths. Narrowing it later, or per group, would
    # leave the file claiming a wider input space than the fit used.
    scope = str(cfg.suggest.feature_scope)
    if scope != "all":
        dropped = length_dependent_columns(columns)
        columns = scoped_columns(columns, scope)
        if not columns:
            raise RuntimeError(
                f"suggest.feature_scope is {scope!r} and it removed every one of "
                f"the {len(dropped)} columns the feature groups {usable} "
                "selected, so the model would have no inputs at all. The "
                "length-invariant columns are the beat_* and wave_* families "
                "(both need detection.task: pose) and the intensive means; add a "
                "feature group that has some, or set suggest.feature_scope back "
                "to 'all'."
            )
        shown = ", ".join(dropped[:4]) + ("..." if len(dropped) > 4 else "")
        reporter.log(
            f"suggest.feature_scope is {scope!r}: {len(dropped)} column(s) whose "
            f"value scales with the window length were dropped ({shown}). That is "
            "a real loss of signal, bought to make one model applicable at "
            "several predict.window_scales."
        )
        if raw_cnn is not None:
            # The fold-wise basis is fitted on `clip_cnn_raw.npy`, which is the
            # mean, the std AND the per-step delta of the embeddings
            # concatenated - so its components mix the length-invariant half
            # with the two halves this scope just removed, and no subset of the
            # components is the invariant part. Left in, the CROSS-VALIDATED
            # score would describe a model that is not the one being saved: it
            # would be scored with length-dependent inputs the saved model does
            # not have. Dropping it costs the fold-PCA leak fix on this path and
            # says so, which is the lesser of two wrong numbers.
            raw_cnn = None
            reporter.log(
                f"suggest.feature_scope is {scope!r}, so the fold-wise CNN basis "
                f"is not used: {CLIP_CNN_RAW_FILE} holds the mean, std and delta "
                "aggregate together and its components cannot be narrowed to the "
                "length-invariant half. The cnn*_mean columns are cross-validated "
                "on the GLOBAL PCA instead, which the held-out movie's detections "
                "were in - that leak flatters the score below.",
                "warning",
            )
    reporter.log(f"features: {len(columns)} columns from {', '.join(usable)}")

    # Said before any group is fitted: every score below is computed on whatever
    # survived this join, and a group whose rare class was mostly orphaned
    # reports 0.000 F1 for it, which reads as "hard" rather than "gone".
    n_orphaned = int((~labels["clip_id"].astype(str)
                      .isin(set(features["clip_id"].astype(str)))).sum())
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
    for group in single:
        # multi-select groups need a different (multi-label) formulation
        reporter.log(f"fitting {group.name}")
        reporter.raise_if_cancelled()
        model, report = train_group(
            features, labels, group.name, cfg, columns, verbose, raw_cnn=raw_cnn,
            reporter=reporter,
        )
        reporter.advance()
        reports[group.name] = asdict(report)
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

    # The feature-schema version travels with the model: a model is meaningful
    # only against features that still mean what they meant when it was fitted,
    # which the column names cannot attest to.
    #
    # So does the WINDOW LENGTH, for the same reason and a second one: the
    # version stamp catches a schema that changed, but the window length can
    # change with the schema untouched - same version, same column names, values
    # on a different scale. `Suggester.window_problem` is what reads it back.
    #
    # Why `cfg.clips.frames` and not something else: these rows come from
    # clip_features.parquet, which is built over the ANNOTATION clip manifest
    # (`build_clip_features` iterates the manifest's (f_start, f_end) pairs), and
    # that manifest is laid out by `clips._windows`, whose window length is
    # `cfg.clips.frames`. `cfg.clips.stride` only sets how far the grid advances,
    # and `cfg.predict.resolved_window(cfg.clips)` belongs to the dense export
    # grid, a different table this model was not fitted on.
    #
    # It is the NOMINAL length: `_windows` truncates the last window of a track
    # against the movie end, and `clips.min_clip_frames` lets a partly-present
    # window through. Those are per-row shortfalls within one grid, not a
    # different grid, and they are exactly what `present_frac` is for.
    trained_window = int(cfg.clips.frames)
    payload = {"columns": columns, "models": models,
               "feature_version": FEATURE_VERSION,
               "trained_window": trained_window}
    joblib.dump(payload, work / MODEL_FILE)
    reporter.log(
        f"model saved: fitted on {trained_window}-frame windows "
        f"(clips.frames), feature version {FEATURE_VERSION}"
    )
    (work / REPORT_FILE).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


# --------------------------------------------------------------------------- #
class Suggester:
    """Loads the trained models and produces per-clip suggestions for the GUI."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.columns: list[str] = []
        self.models: dict = {}
        self.reports: dict = {}
        self.features: pd.DataFrame | None = None
        self.problem: str = ""
        # Audit membership as recorded in the annotation CSV, for the clips that
        # record one. Kept here because this object is what the GUI asks "is
        # this clip held out", and it is the only reader of that file on this
        # path; see `load` for why it is refreshed there and nowhere else.
        self._audit_records: dict[str, bool] = {}
        # Frames per window the training rows were measured over. None for a
        # model saved before this was recorded - those still load; see
        # `window_problem`.
        self.trained_window: int | None = None
        self._by_clip: dict[str, np.ndarray] = {}
        # Bumped by every `load`, and the only thing a caller needs in order to
        # know that a batch from `selection_batch` is out of date. See that
        # method on why the answer must NOT be recomputed per label.
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
        self.trained_window = None

        # Before every early return below, because `is_audit_clip` is asked on
        # clips whether or not a model exists - the GUI asks it to decide
        # whether to explain a missing suggestion. Refreshed HERE and not per
        # question: a clip labelled since the last refit is a clip the annotator
        # has finished with, and re-reading the CSV on every clip change would
        # put a file read in the path of turning a page. An unreadable flag is
        # allowed to raise: it means nobody can say which clips are held out,
        # and carrying on would train on the holdout silently.
        self._audit_records = read_audit_records(self.cfg.paths.csv)

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
        # trained before this was recorded has to keep loading, or upgrading
        # trackanno silently invalidates every model anyone already has. A
        # length that will not parse is treated as absent for the same reason -
        # `window_problem` reports "unrecorded" and the caller warns.
        try:
            recorded = payload.get("trained_window")
            self.trained_window = None if recorded is None else int(recorded)
        except (TypeError, ValueError):
            self.trained_window = None

        # The model's input space is fixed at fit time, so a table missing a
        # column it was fitted on makes predicting meaningless, not degraded.
        # Refuse, and SAY SO rather than failing silently.
        #
        # The name check below catches a renamed or dropped feature but not a
        # REDEFINED one, which is the failure that matters. Hence the version
        # stamp: bump ``FEATURE_VERSION`` whenever a feature's MEANING changes,
        # not only when one is added. Both artefacts must carry it; an unstamped
        # one is refused rather than assumed to match, since "unknown" is exactly
        # the case the stamp exists to catch.
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

    def window_problem(self, length: int) -> str:
        """'' if a feature table built at `length` is safe for this model, else why not.

        The skew this closes is silent by construction: window length changes
        what the features are worth, not what they are called, so a 50-frame
        model handed 100-frame rows passes every check `load` makes - same
        version, same columns - and predicts confidently on inputs whose scale
        it has never seen.

        Returns a sentence rather than raising, and returns one for the
        unrecorded case too, so the caller can choose: a mismatch is worth
        refusing over, "cannot be checked" only worth a warning.
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

    def scale_problem(self, scales) -> str:
        """'' if this model may be scored at all of ``scales``, else why not.

        The authoritative half of the multi-scale check, and the reason there
        are two. :func:`trackanno.config.load_config` refuses
        ``predict.window_scales`` against ``suggest.feature_scope``, which is a
        statement about the model the project will fit NEXT; this asks the model
        actually on disk what it was fitted on, by reading the column names in
        its own payload. A project whose config was changed after the last fit
        passes the first check and fails this one, which is the case that
        matters: the config is a plan and the payload is a fact.

        ONE scale is always fine - that is the run this codebase has always
        done, and :meth:`window_problem` is the check that belongs to it. Beyond
        one, a length-dependent column is scored at a length the model has never
        seen; the columns line up, the feature version matches, the confidences
        look normal, and the answer is averaged together with the scales that
        were right. Nothing downstream can see it, so it is refused here.

        Returns a sentence rather than raising, like :meth:`window_problem`, so
        the caller decides between refusing and warning.
        """
        lengths = [int(s) for s in scales]
        if len(set(lengths)) <= 1:
            return ""
        offenders = length_dependent_columns(self.columns)
        if not offenders:
            return ""
        shown = ", ".join(offenders[:6]) + ("..." if len(offenders) > 6 else "")
        listed = ", ".join(str(s) for s in sorted(set(lengths)))
        return (
            f"this model was fitted on {len(offenders)} of its {len(self.columns)} "
            f"columns whose VALUE scales with the window length ({shown}), so it "
            f"cannot be scored at {len(set(lengths))} window lengths at once "
            f"({listed} frames). path_length over 100 frames is simply twice the "
            "path_length over 50 under the same column name, so the rows from "
            "every scale but the one it was trained at reach it on a scale it has "
            "never seen - at normal confidence, and then averaged into the same "
            "per-frame posterior as the rows that were right. Retrain with "
            "suggest.feature_scope: length_invariant, or empty "
            "predict.window_scales."
        )

    def is_audit_clip(self, clip_id: str) -> bool:
        return is_audit_clip(clip_id, self.cfg, self._audit_records)

    # ---------------------------------------------------------- selection -- #
    def selection_group(self) -> str:
        """Which group's model ranks the clips, or ``""`` if none can.

        The first single-choice group in config order that the model covers,
        which is the same group the GUI already records as the primary
        suggestion - so the ranking and the suggestion on screen come from one
        model rather than two, and a project with several groups does not
        silently rank on whichever one a dict happened to yield first.

        Multi-select groups are not candidates: their label is a SET, and every
        rule in :mod:`trackanno.select` reads a posterior over mutually
        exclusive classes.
        """
        for group in self.cfg.categories:
            if not group.is_multi and group.name in self.models:
                return group.name
        return ""

    def _refuse_window_mismatch(self, length: int) -> None:
        """Refuse a model fitted at another window length; only warn if it never said.

        The same split :func:`trackanno.predict._check_trained_window` makes, for
        the same reason. A MISMATCH means every clip in the batch is scored on a
        scale the model has never seen, and the batch that comes back looks
        exactly like a good one - the ordering is the whole output, so there is
        nothing to notice afterwards. An UNRECORDED length is a model saved
        before the stamp existed, and refusing those would mean upgrading
        trackanno stops selection working on every project that already has a
        model.
        """
        problem = self.window_problem(length)
        if not problem:
            return
        if self.trained_window is None:
            warnings.warn(problem, stacklevel=3)
            return
        raise RuntimeError(
            f"{problem} The {length} frames are clips.frames, the length "
            f"{CLIP_FEATURES_FILE} - the table every candidate in this batch is "
            "scored from - was built over."
        )

    def selection_batch(
        self, manifest: pd.DataFrame, done_ids: set[str], n: int
    ) -> list[dict]:
        """The next `n` clips to annotate, in order, each carrying its provenance.

        Returns [] when selection is off, when the model is not ready, or when
        there is nothing left to choose - the caller then keeps its existing
        order untouched.

        Each record is a :func:`trackanno.select.select_batch` record, passed
        through unchanged and with the ``group`` whose model produced it added:
        the store's provenance codec and ``select`` were designed against each
        other, and a second shape here would be a translation nobody asked for.
        ``index`` is a row of the probability matrix built below, NOT a position
        in ``manifest``; ``clip_id`` is the join key.

        WHEN TO CALL THIS, which matters as much as what it returns. Recompute
        when the model changes - :attr:`model_epoch` counts that, and the GUI
        refits every ``suggest.retrain_every`` labels - and NOT once per label.
        Re-ranking after every label would re-order the queue under the
        annotator mid-session: the cursor only moves forward, so a clip that
        dropped a place is not revisited, and which clips get stranded depends
        on the order the labels happened to arrive in. The staleness that buys:
        within one batch every score is the model's opinion as of the last
        refit, so the last clip of a batch is ranked by a model up to
        ``retrain_every`` labels out of date. That is a bounded, stated error;
        a queue that reshuffles under the person reading it is not.

        The batch is a function of the pool's contents, the config and
        ``suggest.audit_seed`` alone - never of the row order of ``manifest`` -
        because :mod:`trackanno.select` breaks every tie on ``clip_id``.

        Parameters
        ----------
        manifest
            The clip manifest: ``clip_id``, plus ``movie_id``, ``track_id`` and
            ``f_start`` when the diversity rule is on. Any subset of the project
            (one movie, say) is a valid pool - it is the pool that is ranked.
        done_ids
            Clips already annotated. Excluded from the candidate mask, so a
            batch never hands back work that is finished.
        n
            How many clips to ask for. The answer may be SHORTER: the diversity
            rule can forbid the rest, and the request is sized down to what the
            pools actually hold rather than refused, because this method built
            those pools and so is the caller that ought to know.
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
        # scale while reporting normal confidence.
        self._refuse_window_mismatch(int(cfg.clips.frames))

        gap = cfg.suggest.resolved_frame_gap(cfg.clips)
        ids = manifest["clip_id"].astype(str).to_numpy()

        # A clip with no feature row cannot be scored, so it is left OUT of the
        # pool rather than imputed: an imputed row still ranks, and it would
        # rank on numbers the clip does not have. This is the ordinary state
        # after re-running detection - the manifest is rebuilt, the feature
        # table is not - so it is a filter, not an error.
        rows = [self._by_clip.get(clip_id) for clip_id in ids]
        keep = np.array([row is not None for row in rows], dtype=bool)
        if not keep.any():
            return []
        pool = np.flatnonzero(keep)
        pool_ids = ids[pool]

        done = {str(c) for c in (done_ids or ())}
        candidates = np.array([c not in done for c in pool_ids], dtype=bool)
        # Audit clips stay IN the pool and are flagged, never filtered out:
        # `select_batch` removes them from the ranked pool itself and draws the
        # reserved slots from them. Filtering them here would leave it with
        # nothing to draw and silently turn the random stratum off.
        audit = np.array([is_audit_clip(c, cfg, self._audit_records) for c in pool_ids],
                         dtype=bool)
        n_audit = int((candidates & audit).sum())
        n_ranked = int((candidates & ~audit).sum())

        share = cfg.suggest.resolved_audit_share()
        size = _selection_slots(int(n), share, n_ranked, n_audit)
        if size <= 0:
            return []

        model = self.models[group]
        matrix = np.vstack([rows[i] for i in pool])
        proba = np.asarray(model.predict_proba(matrix), dtype=float)
        classes = [str(c) for c in model.classes_]

        records = select_batch(
            proba, pool_ids, classes, size,
            rule=cfg.suggest.select_rule,
            candidates=candidates,
            # An audit pool with nothing unannotated left in it is passed as
            # None. That is not the "this project has no holdout" lie the
            # parameter warns about: every audit clip is already labelled, so
            # the two masks select the same rows, and the only difference is
            # whether slots are reserved for a draw that has nothing to draw.
            audit=audit if n_audit else None,
            audit_share=share,
            # The audit set's membership and its sampling are one mechanism, so
            # they share one seed. `suggest.seed` is deliberately not used: it
            # exists to be varied to check CV stability, and doing that must not
            # change which clips a human is handed.
            seed=cfg.suggest.audit_seed,
            class_counts=self._label_counts(group, classes),
            target_quota=cfg.suggest.select_target_quota,
            target_rule=cfg.suggest.select_target_rule,
            track_ids=self._track_keys(manifest, pool) if gap else None,
            frames=manifest["f_start"].to_numpy()[pool] if gap else None,
            min_frame_gap=gap,
        )
        for record in records:
            # Which model scored the clip, which `predicted_class` is meaningless
            # without on a project with more than one category group. The store's
            # codec takes any extra key, precisely so a rule that needs to record
            # one more thing does not force a schema migration.
            record["group"] = group
            # A record that cannot be written down is not provenance. `select`
            # scores a random draw NaN - it has no score, and inventing 0.0 would
            # read as "the model was certain" - but `store.encode_provenance`
            # refuses NaN outright, on the grounds that JSON has no such value
            # and it does not read back the same everywhere. So the one float
            # that can be non-finite is translated here, at the seam, to the JSON
            # value that means the same thing and survives the round trip. None
            # is still not 0.0, which is the distinction that mattered.
            for key, value in list(record.items()):
                if isinstance(value, float) and not np.isfinite(value):
                    record[key] = None
        return records

    def _label_counts(self, group: str, classes: list[str]) -> dict:
        """Labels per class as of the last fit, for the starved-class quota.

        Read off the training report rather than recounted from the CSV: these
        are the counts the model was actually FITTED on, which is the right
        denominator for "does it have enough examples of this class", and they
        move only when the model does - so the quota cannot re-plan the batch
        halfway through it.

        Classes with no column in the probability matrix are dropped rather than
        passed: `select_batch` refuses a count it cannot target, and a report
        left behind by an earlier fit can legitimately name a class the current
        model does not have. The consequence worth stating is that a class too
        rare to be MODELLED at all (below ``suggest.min_per_class``) has no
        column either, so targeting cannot reach the one class that is scarcest
        of all - only more labels of it can, which is what min_per_class is
        already saying.
        """
        if int(self.cfg.suggest.select_target_quota) <= 0:
            return {}
        report = self.reports.get(group) or {}
        counts = report.get("per_class_counts") or {}
        known = set(classes)
        return {str(k): int(v) for k, v in counts.items() if str(k) in known}

    @staticmethod
    def _track_keys(manifest: pd.DataFrame, pool: np.ndarray) -> np.ndarray:
        """``movie::track`` per pool row - the identity the diversity rule needs.

        Track ids are numbered per MOVIE and so are ``f_start`` frames, so two
        movies routinely hold a track 3 whose clips start at frame 250. Handed a
        bare ``track_id`` the diversity rule would read those as the same animal
        at the same moment and suppress one of them: a genuinely informative
        clip dropped from the batch for a collision between two movies that
        share nothing. The pair is unique because ``clip_id`` is built from it.
        """
        missing = [c for c in ("movie_id", "track_id", "f_start")
                   if c not in manifest.columns]
        if missing:
            raise ValueError(
                f"suggest.select_frame_gap is on but the manifest has no {missing} "
                "column(s); the diversity rule needs to know which clips are "
                "successive windows of one animal. Set suggest.select_frame_gap to "
                "0 to turn it off deliberately."
            )
        movies = manifest["movie_id"].astype(str).to_numpy()[pool]
        tracks = manifest["track_id"].astype(str).to_numpy()[pool]
        return np.array([f"{m}::{t}" for m, t in zip(movies, tracks)], dtype=str)

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
        # Against the POPULATION where the report carries one. "12 audit clips,
        # all correct" and "12 of 340 audit clips have been labelled" are the
        # same 12 and different claims: the first reads as a result, the second
        # as a result resting on 3.5% of the holdout. Printed even when nothing
        # was scored, because "the holdout exists and is empty" is the more
        # actionable of the two.
        #
        # THREE cases, and they are kept apart on purpose. `audit_population` is
        # newer than `suggest_report.json` itself, so a report written before it
        # existed has no such key - and gating the whole block on it deleted a
        # kappa that had printed the day before, silently, from a file whose
        # numbers were still perfectly real. A `.get` default cannot fix that
        # either: 0 would then mean both "this report cannot say" and "this
        # project holds nothing out", which are different facts with different
        # answers (retrain / raise suggest.audit_fraction). So the missing key is
        # read as missing.
        population = rep.get("audit_population")
        labelled = rep.get("audit_labelled", rep.get("audit_n", 0))
        scored = rep.get("audit_n") or 0
        ak, aa = rep.get("audit_kappa"), rep.get("audit_accuracy")
        ak = "n/a" if ak is None or ak != ak else f"{ak:+.3f}"
        aa = "n/a" if aa is None or aa != aa else f"{aa:.3f}"
        if population is None:
            # An older report. It carries audit_n, which is what the audit line
            # was gated on before the population was recorded, so that is what it
            # is gated on here - the same line it printed then, with the missing
            # denominator named rather than guessed at.
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
                if rep.get("audit_pca"):
                    print(f"    caveat: {rep['audit_pca']}")
            else:
                print(
                    f"  audit holdout ({of_population}): too few to score, so there "
                    "is no unbiased estimate for this group yet"
                )
        else:
            # The population was counted and came out 0: this project holds
            # nothing out at all. Said rather than skipped, because a report with
            # no audit line reads as one that forgot to compute it.
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
            print("  top features:")
            for f in rep["top_features"][:8]:
                print(f"    {f['importance']:>7.4f}  {f['feature']}")


def ablate(cfg: Config) -> None:
    """What each subset of the features is worth, on this project's own labels.

    Kinematic vs CNN vs both - does the 190 MB of embeddings earn its place? -
    plus the length-invariant subset, which is priced here because it is the
    only one of these that buys something other than accuracy: it is the
    condition on scoring at several ``predict.window_scales``, and what it costs
    is a per-project number nothing else reports.
    """
    features = build_clip_features(cfg, verbose=False)
    labels = pd.read_csv(cfg.paths.csv)
    from .clipfeatures import FEATURE_GROUPS, available_groups

    # The same positional tag and fold-wise PCA that `train` sets up: without
    # `_row` the guard inside train_group never fires and every row is
    # cross-validated on the GLOBALLY fitted PCA - the leak this comparison
    # exists to quantify (0.047 / 0.237 leaked, 0.101 / 0.269 with the fold PCA).
    features = features.reset_index(drop=True)
    features["_row"] = np.arange(len(features))
    raw_cnn = None
    raw_path = cfg.paths.work_dir / CLIP_CNN_RAW_FILE
    if raw_path.exists():
        candidate = np.load(raw_path)
        if len(candidate) == len(features):
            raw_cnn = candidate

    usable = [g for g in FEATURE_GROUPS if not available_groups(cfg)[g]]
    all_cols = feature_columns(features, usable)
    subsets = {g: feature_columns(features, [g]) for g in usable}
    subsets["everything"] = all_cols
    if "cnn" in usable:
        subsets["no CNN"] = feature_columns(features, [g for g in usable if g != "cnn"])
    # The subset `suggest.feature_scope: length_invariant` would fit on, priced
    # the same way as the groups. It is the only one of these that buys
    # something other than accuracy - it is what makes `predict.window_scales`
    # legitimate - so the number worth reading here is what dropping the
    # length-dependent columns COSTS, which nothing else in this tool reports.
    invariant = length_invariant_columns(all_cols)
    if invariant and len(invariant) < len(all_cols):
        subsets["length-invariant"] = invariant
    for group in cfg.categories:
        if group.is_multi:
            continue
        print(f"\n=== {group.name} ===")
        for name, cols in subsets.items():
            if not cols:
                continue
            # Fold basis only when this subset carries CNN columns, for the same
            # reason `train` gates it.
            fold_cnn = raw_cnn if any(c.startswith("cnn") for c in cols) else None
            _, rep = train_group(features, labels, group.name, cfg, cols,
                                 verbose=False, raw_cnn=fold_cnn)
            # kappa's 5th percentile is in the line because it is the column
            # that answers the question a subset comparison is usually asked:
            # macro-F1 differences of a few hundredths over a few hundred labels
            # are not distinguishable from each other, while the bootstrap lower
            # bound over MOVIES is about whether the subset holds up on the next
            # one. Reading the point estimates alone is how a difference that is
            # not there gets acted on.
            print(f"  {name:<16} {len(cols):>4} cols  macro-F1 {rep.macro_f1:.3f}  "
                  f"acc {rep.accuracy:.3f}  kappa {rep.kappa:+.3f} "
                  f"[5th pct {rep.kappa_lower:+.3f}]"
                  + ("" if rep.reliable else f"   ({rep.reason})"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="suggest", description="Train the annotation-suggestion model."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--train", action="store_true", help="build features, fit, cross-validate")
    p.add_argument("--report", action="store_true", help="print the last saved report")
    p.add_argument("--ablate", action="store_true",
                   help="compare kinematic-only vs CNN-only vs both, and the "
                        "length-invariant subset predict.window_scales needs")
    p.add_argument("--features-only", action="store_true", help="build clip features and stop")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

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

    reports = train(cfg)
    _print_report(reports)
    usable = [g for g, r in reports.items() if r["reliable"]]
    print(
        f"\nModel -> {cfg.paths.work_dir / MODEL_FILE}"
        f"\nGroups that will suggest: {usable or 'none yet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
