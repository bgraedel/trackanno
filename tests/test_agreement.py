"""Annotation windows whose label differs from the model's suggestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trackanno.infer.agreement import differs, find


@pytest.fixture(autouse=True)
def _short_fits(monkeypatch):
    """Short boosting fits keep the suite fast."""
    monkeypatch.setattr("trackanno.learn.suggest._iterations", lambda cfg: 120)


GROUP = "behaviour"
SECOND = "posture"
TOL = 1e-9


def labels(*spec, movie: str = "m", track: int = 0, group: str = GROUP) -> pd.DataFrame:
    """``(label, f_start)`` pairs into annotation rows 25 frames long."""
    return pd.DataFrame([
        {"clip_id": f"{movie}__t{track:04d}__f{a:06d}", "movie_id": movie,
         "track_id": track, "f_start": a, "f_end": a + 25, group: label}
        for label, a in spec
    ])


def said(*spec, movie: str = "m", track: int = 0, group: str = GROUP) -> dict:
    """``(f_start, value, confidence)`` triples into a suggest_many answer."""
    return {
        f"{movie}__t{track:04d}__f{a:06d}": {group: (value, conf)}
        for a, value, conf in spec
    }


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #
def test_a_window_disagrees_when_the_suggestion_is_not_the_label() -> None:
    """Each annotation window is compared with the suggestion for that window."""
    hand = labels(("swim", 0), ("swim", 25), ("turn", 50))
    answers = said((0, "swim", 0.9), (25, "turn", 0.8), (50, "turn", 0.7))
    found = find(hand, answers)

    assert [d.f_start for d in found] == [25], (
        [(d.f_start, d.label, d.suggestion) for d in found])
    d = found[0]
    assert d.label == "swim" and d.suggestion == "turn"
    assert abs(d.confidence - 0.8) < TOL
    assert d.group == GROUP
    assert (d.clip_id == "m__t0000__f000025" and d.movie_id == "m"
            and d.track_id == 0 and d.f_end == 50), d
    assert d.frame == 37, "the frame to look at is the middle of the window"


def test_the_most_confident_disagreement_comes_first() -> None:
    hand = labels(("swim", 0), ("swim", 25), ("swim", 50))
    answers = said((0, "turn", 0.41), (25, "turn", 0.95), (50, "turn", 0.62))
    found = find(hand, answers)

    assert len(found) == 3, found
    assert [round(d.confidence, 2) for d in found] == [0.95, 0.62, 0.41], (
        [d.confidence for d in found])


def test_a_window_with_no_answer_is_left_alone() -> None:
    hand = labels(("swim", 0), ("swim", 25))
    assert [d.f_start for d in find(hand, said((0, "turn", 0.9)))] == [0]


@pytest.mark.parametrize("hand, answers", [
    pytest.param(labels(("swim", 0), ("swim", 25)), {"m__t0000__f000000": {}},
                 id="empty-answer"),
    pytest.param(labels(("swim", 0), ("swim", 25)), said((0, "", 0.9)),
                 id="suggestion-of-nothing"),
    pytest.param(labels(("", 0)), said((0, "turn", 0.9)), id="unlabelled-window"),
    pytest.param(labels(("swim", 0), ("swim", 25)), {}, id="no-suggestions"),
])
def test_no_claim_on_either_side_is_not_a_disagreement(hand, answers) -> None:
    assert find(hand, answers) == []


def test_a_confidence_nobody_measured_is_not_a_confidence_of_zero() -> None:
    """An unmeasured confidence stays None and sorts after every measured one."""
    hand = labels(("swim", 0), ("swim", 25), ("swim", 50))
    answers = {
        "m__t0000__f000000": {GROUP: ("turn", None)},
        "m__t0000__f000025": {GROUP: ("turn", 0.9)},
        "m__t0000__f000050": {GROUP: ("turn", 0.1)},
    }
    found = find(hand, answers)

    assert len(found) == 3, found
    assert found[-1].confidence is None, [d.confidence for d in found]
    assert [d.f_start for d in found] == [25, 50, 0], (
        [(d.f_start, d.confidence) for d in found])


def test_every_group_is_compared_not_only_the_first() -> None:
    hand = labels(("swim", 0), ("swim", 25))
    hand[SECOND] = ["curled", "straight"]
    answers = {
        "m__t0000__f000000": {GROUP: ("swim", 0.9), SECOND: ("straight", 0.85)},
        "m__t0000__f000025": {GROUP: ("swim", 0.9), SECOND: ("straight", 0.7)},
    }
    found = find(hand, answers, [GROUP, SECOND])

    assert [(d.group, d.f_start) for d in found] == [(SECOND, 0)], (
        [(d.group, d.f_start) for d in found])


def test_differs_names_each_disagreeing_group() -> None:
    """``differs`` is the one definition both the annotation screen and review use."""
    row = {GROUP: "swim", SECOND: "curled", "notes": "", "uncertain": "0"}
    suggestion = {GROUP: ("turn", 0.8), SECOND: ("curled", 0.9)}
    assert differs(row, suggestion) == [f"{GROUP}: turn"], differs(row, suggestion)
    assert differs(row, {SECOND: ("curled", 0.9)}) == [], "agreement is silence"
    assert differs({GROUP: ""}, suggestion) == [], "an unlabelled group is not a disagreement"
    assert differs(row, suggestion, groups=[SECOND]) == [], "it can be held to one group"


# --------------------------------------------------------------------------- #
# the estimator this is allowed to use
# --------------------------------------------------------------------------- #
def _noisy_project(root, name: str, n_per_movie: int = 24):
    """Labels over three movies and a feature that predicts them only most of the time."""
    from fixtures import make_project

    from trackanno.config import load_config

    cfg = load_config(make_project(root))
    cfg.suggest.min_per_class = 3
    rng = np.random.default_rng(0)
    rows, feats = [], []
    for movie in ("mov_a", "mov_b", "mov_c"):
        for i in range(n_per_movie):
            clip_id = f"{movie}__t0000__f{i:06d}"
            truth = "swim" if i % 2 else "turn"
            rows.append({
                "clip_id": clip_id, "movie_id": movie, "track_id": 0,
                "f_start": i * 25, "f_end": i * 25 + 25, "behaviour": truth,
                "audit_holdout": "0",
            })
            feats.append({
                "clip_id": clip_id, "movie_id": movie, "track_id": i % 4,
                "speed_mean": (1.0 if truth == "swim" else -1.0) + rng.normal(0, 1.1),
            })
    return cfg, pd.DataFrame(rows), pd.DataFrame(feats)


def test_the_shortlist_is_the_confusion_matrix_s_off_diagonal(root) -> None:
    """Disagreements come from the CV folds, which match the report, not the saved model."""
    from trackanno.learn.suggest import _build_model, train_group

    cfg, labels, features = _noisy_project(root, "off-diagonal")
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    assert len(report.cv_predictions) == len(labels), (
        f"{len(report.cv_predictions)} of {len(labels)} labels scored")

    answers = {
        clip_id: {"behaviour": (predicted, confidence or 0.0)}
        for clip_id, predicted, confidence in report.cv_predictions
    }
    found = find(labels, answers, ["behaviour"])
    off_diagonal = sum(
        value for i, row in enumerate(report.confusion)
        for j, value in enumerate(row) if i != j
    )
    assert off_diagonal > 0, f"the folds never disagreed: {report.confusion}"
    assert len(found) == off_diagonal, f"{len(found)} found vs {off_diagonal} off-diagonal"

    # The in-sample model reproduces its training labels, so it would hide disagreements.
    model = _build_model(cfg, 2)
    model.fit(features[["speed_mean"]].to_numpy(dtype=np.float32),
              labels["behaviour"].to_numpy())
    in_sample = model.predict(
        features[["speed_mean"]].to_numpy(dtype=np.float32)
    ).ravel()
    disagreed = int((in_sample != labels["behaviour"].to_numpy()).sum())
    assert disagreed < off_diagonal, f"in-sample {disagreed} vs out-of-fold {off_diagonal}"


def test_training_leaves_the_fold_predictions_on_disk(root) -> None:
    """The fold predictions file is required, and reads back keyed by clip and group."""
    from trackanno.infer.agreement import load_cv_predictions
    from trackanno.learn.suggest import CV_PREDICTIONS_FILE, train_group

    cfg, labels, features = _noisy_project(root, "written")
    path = cfg.paths.work_dir / CV_PREDICTIONS_FILE
    path.unlink(missing_ok=True)

    answers, problem = load_cv_predictions(cfg)
    assert answers == {} and "Train the model again" in problem, problem

    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    pd.DataFrame(
        [{"clip_id": c, "group": "behaviour", "predicted": p, "confidence": k}
         for c, p, k in report.cv_predictions],
        columns=["clip_id", "group", "predicted", "confidence"],
    ).to_parquet(path, index=False)

    answers, problem = load_cv_predictions(cfg)
    assert problem == "" and len(answers) == len(labels), problem
    one = next(iter(answers.values()))
    assert list(one) == ["behaviour"] and len(one["behaviour"]) == 2, one
    assert all(0.0 <= v[1] <= 1.0 for g in answers.values() for v in g.values()), (
        "confidences are probabilities")
    path.unlink(missing_ok=True)


def test_the_single_movie_refusal_does_not_name_an_arbitrary_class(root) -> None:
    """On a one-movie project the refusal blames the movie count, not a class."""
    from trackanno.learn.suggest import train_group

    cfg, labels_, features = _noisy_project(root, "single-movie")
    one_movie = labels_["movie_id"] == "mov_a"
    model, report = train_group(
        features[features["movie_id"] == "mov_a"], labels_[one_movie],
        "behaviour", cfg, columns=["speed_mean"], verbose=False)
    assert model is None
    assert "one movie" in report.reason and "'mov_a'" in report.reason, report.reason
    assert "'swim'" not in report.reason and "'turn'" not in report.reason, report.reason
    assert "second movie" in report.reason, report.reason


def test_the_confined_class_refusal_names_every_confined_class(root) -> None:
    from trackanno.learn.suggest import train_group

    cfg, labels_, features = _noisy_project(root, "confined")
    elsewhere = labels_["movie_id"] != "mov_a"
    labels_.loc[elsewhere, "behaviour"] = "drift"    # swim and turn: mov_a only
    model, report = train_group(features, labels_, "behaviour", cfg,
                                columns=["speed_mean"], verbose=False)
    assert model is None
    assert "'swim'" in report.reason and "'turn'" in report.reason, report.reason
    assert "'drift'" not in report.reason, report.reason
