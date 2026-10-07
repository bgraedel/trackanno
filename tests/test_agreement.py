"""Annotation windows whose label is not what the model suggests.

Two halves. The first plants a label table and a set of suggestions by hand, so
the expected answer is available by reading: which windows disagree and in what
order. The second checks the fold predictions they are measured against.

The suggestions are planted rather than fitted. Whether a real classifier would
disagree with a label is a statement about a classifier; what this suite is
about is what the screen does once one does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harness import check

from trackanno.infer.agreement import differs, find


@pytest.fixture(autouse=True)
def _short_fits(monkeypatch):
    """A test, not a fit worth keeping."""
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
    """The unit is the annotation window, and the model's claim is its suggestion.

    No frame arithmetic and no decode: a window is compared with the answer for
    that window, which is what makes this the same comparison the annotation
    screen draws as a dashed outline.
    """
    hand = labels(("swim", 0), ("swim", 25), ("turn", 50))
    answers = said((0, "swim", 0.9), (25, "turn", 0.8), (50, "turn", 0.7))
    found = find(hand, answers)

    check("only the window the suggestion contradicts is reported",
          [d.f_start for d in found] == [25],
          str([(d.f_start, d.label, d.suggestion) for d in found]))
    d = found[0]
    check("the report names both claims",
          d.label == "swim" and d.suggestion == "turn")
    check("...how sure the model is of its half", abs(d.confidence - 0.8) < TOL)
    check("...which group they are about", d.group == GROUP)
    check("...and which clip to open",
          d.clip_id == "m__t0000__f000025" and d.movie_id == "m"
          and d.track_id == 0 and d.f_end == 50, str(d))
    check("the frame to look at is the middle of the window", d.frame == 37)


def test_the_most_confident_disagreement_comes_first() -> None:
    """A confident one is a label to fix or a model failure worth understanding.

    One at 0.35 is the model shrugging, and ordering by anything that puts those
    first would bury the rest of the walk under them.
    """
    hand = labels(("swim", 0), ("swim", 25), ("swim", 50))
    answers = said((0, "turn", 0.41), (25, "turn", 0.95), (50, "turn", 0.62))
    found = find(hand, answers)

    check("all three are reported", len(found) == 3, str(found))
    check("and the one it is surest about leads",
          [round(d.confidence, 2) for d in found] == [0.95, 0.62, 0.41],
          str([d.confidence for d in found]))


def test_a_window_the_model_will_not_answer_about_is_not_a_disagreement() -> None:
    """No answer and the wrong answer are not the same fact about a model.

    The suggester refuses a clip in the audit holdout and one with no feature
    row, so those clips simply do not appear in what it returns.
    """
    hand = labels(("swim", 0), ("swim", 25))
    check("a clip with no answer at all is left alone",
          find(hand, said((0, "turn", 0.9))) and
          [d.f_start for d in find(hand, said((0, "turn", 0.9)))] == [0])
    check("an empty answer is not an answer",
          find(hand, {"m__t0000__f000000": {}}) == [])
    check("nor is a suggestion of nothing",
          find(hand, said((0, "", 0.9))) == [])
    check("an unlabelled window is not a claim to disagree with",
          find(labels(("", 0)), said((0, "turn", 0.9))) == [])
    check("and no suggestions at all is not a wall of disagreements",
          find(hand, {}) == [])


def test_a_confidence_nobody_measured_is_not_a_confidence_of_zero() -> None:
    """An estimator with no posterior said nothing about how sure it was.

    Read as 0.0 it would sort dead last in the surest-first walk, which is the
    whole ordering, and print as "0% sure" - a claim the model never made.
    """
    hand = labels(("swim", 0), ("swim", 25), ("swim", 50))
    answers = {
        "m__t0000__f000000": {GROUP: ("turn", None)},
        "m__t0000__f000025": {GROUP: ("turn", 0.9)},
        "m__t0000__f000050": {GROUP: ("turn", 0.1)},
    }
    found = find(hand, answers)

    check("all three are reported", len(found) == 3, str(found))
    check("the unmeasured one keeps no number rather than gaining a zero",
          found[-1].confidence is None, str([d.confidence for d in found]))
    check("...and sorts after the one the model was barely sure of",
          [d.f_start for d in found] == [25, 50, 0],
          str([(d.f_start, d.confidence) for d in found]))


def test_every_group_is_compared_not_only_the_first() -> None:
    """A suggestion is per group, and so is a disagreement about one."""
    hand = labels(("swim", 0), ("swim", 25))
    hand[SECOND] = ["curled", "straight"]
    answers = {
        "m__t0000__f000000": {GROUP: ("swim", 0.9), SECOND: ("straight", 0.85)},
        "m__t0000__f000025": {GROUP: ("swim", 0.9), SECOND: ("straight", 0.7)},
    }
    found = find(hand, answers, [GROUP, SECOND])

    check("the second group's disagreement is found",
          [(d.group, d.f_start) for d in found] == [(SECOND, 0)],
          str([(d.group, d.f_start) for d in found]))


def test_one_definition_of_a_disagreement_serves_both_screens() -> None:
    """``differs`` is what the annotation screen and the review page both use.

    Two ideas of what counts would sooner or later disagree with each other, and
    the walk would then stop at clips the other screen called agreed.
    """
    row = {GROUP: "swim", SECOND: "curled", "notes": "", "uncertain": "0"}
    suggestion = {GROUP: ("turn", 0.8), SECOND: ("curled", 0.9)}
    check("it names the group and the class suggested instead",
          differs(row, suggestion) == [f"{GROUP}: turn"],
          str(differs(row, suggestion)))
    check("agreement is silence", differs(row, {SECOND: ("curled", 0.9)}) == [])
    check("an unlabelled group is not a disagreement",
          differs({GROUP: ""}, suggestion) == [])
    check("and it can be held to one group",
          differs(row, suggestion, groups=[SECOND]) == [])


# --------------------------------------------------------------------------- #
# the estimator this is allowed to use
# --------------------------------------------------------------------------- #
def _noisy_project(root, name: str, n_per_movie: int = 24):
    """A label table and a feature that predicts it only most of the time.

    A separable one would have nothing to disagree about, and a real
    hand-labelled set is not separable: that is the whole reason for looking.
    """
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
            # Overlapping distributions, so a held-out model gets a good share
            # of them and not all of them.
            feats.append({
                "clip_id": clip_id, "movie_id": movie, "track_id": i % 4,
                "speed_mean": (1.0 if truth == "swim" else -1.0) + rng.normal(0, 1.1),
            })
    return cfg, pd.DataFrame(rows), pd.DataFrame(feats)


def test_the_shortlist_is_the_confusion_matrix_s_off_diagonal(root) -> None:
    """The number a reviewer is shown has to be the number in the report.

    This is the bug the whole module turns on. Asked about a labelled clip, the
    SAVED model is being asked about a row it was fitted on: it reproduces
    essentially every label however wrong, so a shortlist built on it comes back
    empty while the report's confusion matrix is full of off-diagonal counts.
    The two are one tally now, because both are read off the same folds.
    """
    from trackanno.learn.suggest import _build_model, train_group

    cfg, labels, features = _noisy_project(root, "off-diagonal")
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    check("the folds scored every label", len(report.cv_predictions) == len(labels),
          f"{len(report.cv_predictions)} of {len(labels)}")

    answers = {
        clip_id: {"behaviour": (predicted, confidence or 0.0)}
        for clip_id, predicted, confidence in report.cv_predictions
    }
    found = find(labels, answers, ["behaviour"])
    off_diagonal = sum(
        value for i, row in enumerate(report.confusion)
        for j, value in enumerate(row) if i != j
    )
    check("the folds did disagree with some of them, or this proves nothing",
          off_diagonal > 0, str(report.confusion))
    check("and the shortlist is exactly that off-diagonal",
          len(found) == off_diagonal,
          f"{len(found)} found vs {off_diagonal} off-diagonal")

    # The estimator this must NOT use, measured rather than asserted.
    model = _build_model(cfg, 2)
    model.fit(features[["speed_mean"]].to_numpy(dtype=np.float32),
              labels["behaviour"].to_numpy())
    in_sample = model.predict(
        features[["speed_mean"]].to_numpy(dtype=np.float32)
    ).ravel()
    disagreed = int((in_sample != labels["behaviour"].to_numpy()).sum())
    check("the saved model reproduces far more of its own training labels than "
          "a held-out one does, which is why it cannot be the one asked",
          disagreed < off_diagonal,
          f"in-sample {disagreed} vs out-of-fold {off_diagonal}")


def test_training_leaves_the_fold_predictions_on_disk(root) -> None:
    """The review reads a file, so a fit has to write one - and has to write it
    every time, or the previous model's opinions are read as this one's."""
    import pandas as pd_

    from trackanno.infer.agreement import load_cv_predictions
    from trackanno.learn.suggest import CV_PREDICTIONS_FILE, train_group

    cfg, labels, features = _noisy_project(root, "written")
    path = cfg.paths.work_dir / CV_PREDICTIONS_FILE
    path.unlink(missing_ok=True)

    answers, problem = load_cv_predictions(cfg)
    check("with no file the reader refuses rather than returning nothing",
          answers == {} and "Train the model again" in problem, problem)

    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    pd_.DataFrame(
        [{"clip_id": c, "group": "behaviour", "predicted": p, "confidence": k}
         for c, p, k in report.cv_predictions],
        columns=["clip_id", "group", "predicted", "confidence"],
    ).to_parquet(path, index=False)

    answers, problem = load_cv_predictions(cfg)
    check("and reads them back once they are there",
          problem == "" and len(answers) == len(labels), problem)
    one = next(iter(answers.values()))
    check("...keyed by clip, one entry per group, value and confidence",
          list(one) == ["behaviour"] and len(one["behaviour"]) == 2, str(one))
    check("...with a confidence that is a probability",
          all(0.0 <= v[1] <= 1.0 for g in answers.values() for v in g.values()))
    path.unlink(missing_ok=True)


def test_the_single_movie_refusal_does_not_name_an_arbitrary_class(root) -> None:
    """On a one-movie project the old refusal read "class 'base-to-tip' appears
    in only 1 of the 1 annotated movie(s)", naming whichever class sorted first
    however many labels it had; a user with 72 of them was told the count was
    the problem. The constraint is the movie count, so the message must say so
    and name no class at all."""
    from trackanno.learn.suggest import train_group

    cfg, labels_, features = _noisy_project(root, "single-movie")
    one_movie = labels_["movie_id"] == "mov_a"
    model, report = train_group(
        features[features["movie_id"] == "mov_a"], labels_[one_movie],
        "behaviour", cfg, columns=["speed_mean"], verbose=False)
    check("no model comes back", model is None)
    check("the reason names the one movie, not a class",
          "one movie" in report.reason and "'mov_a'" in report.reason,
          report.reason)
    check("no class is blamed",
          "'swim'" not in report.reason and "'turn'" not in report.reason,
          report.reason)
    check("and it says what to do", "second movie" in report.reason, report.reason)


def test_the_confined_class_refusal_names_every_confined_class(root) -> None:
    """With several classes each stuck in one movie, the old message named only
    the alphabetically first; fixing that class's spread would just surface the
    next one. All binding classes belong in one refusal."""
    from trackanno.learn.suggest import train_group

    cfg, labels_, features = _noisy_project(root, "confined")
    elsewhere = labels_["movie_id"] != "mov_a"
    labels_.loc[elsewhere, "behaviour"] = "drift"    # swim and turn: mov_a only
    model, report = train_group(features, labels_, "behaviour", cfg,
                                columns=["speed_mean"], verbose=False)
    check("no model comes back", model is None)
    check("both confined classes are named",
          "'swim'" in report.reason and "'turn'" in report.reason,
          report.reason)
    check("the spread-out class is not",
          "'drift'" not in report.reason, report.reason)
