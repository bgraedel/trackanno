"""Bout-to-bout statistics, measured against planted bout sequences."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from trackanno.infer.transitions import (
    TRANSITIONS_FILE, analyse, compute, exponential_fit, kaplan_meier,
    movie_fps, summarise, write_report,
)

GROUP = "behaviour"
TOL = 1e-9


# --------------------------------------------------------------------------- #
# planting bouts
# --------------------------------------------------------------------------- #
def plant(spec, *, movie: str = "movie_a", track: int = 0, start: int = 0,
          group: str = GROUP, human=()) -> list[dict]:
    """``(label, n_frames)`` pairs into gapless bout rows; ``""`` is an unlabelled run.

    ``human`` names positions whose bouts overlap hand-labelled frames.
    """
    rows, f = [], int(start)
    for i, (label, n) in enumerate(spec):
        rows.append({
            "bout_id": f"{movie}:{track}:{f}-{f + n}:{group}",
            "movie_id": movie,
            "track_id": int(track),
            "f_start": f,
            "f_end": f + n,
            "n_frames": int(n),
            "group": group,
            f"{group}_pred": label,
            f"{group}_conf": np.nan if not label else 0.9,
            "human_overlap_frac": 1.0 if i in set(human) else 0.0,
            "n_covering_mean": 3.0,
            "skip_reason": "" if label else "no window could be featurised",
        })
        f += n
    return rows


def table(*blocks) -> pd.DataFrame:
    return pd.DataFrame([row for block in blocks for row in block])


def pairs(report: dict, group: str = GROUP) -> dict:
    """{(from, to): count} for one group, empty rows dropped."""
    counts = report["groups"][group]["transitions"]["counts"]
    return {(a, b): int(n) for a, row in counts.items() for b, n in row.items()}


def dwell(report: dict, name: str, group: str = GROUP) -> dict:
    return report["groups"][group]["dwell"]["per_class"][name]


def markov(report: dict, group: str = GROUP) -> dict:
    return report["groups"][group]["markov"]


# --------------------------------------------------------------------------- #
# the transition counts
# --------------------------------------------------------------------------- #
def test_a_planted_sequence_yields_exactly_the_transitions_planted() -> None:
    """Six bouts, five changes, and the matrix says which five."""
    frame = table(plant([
        ("swim", 20), ("turn", 30), ("swim", 40), ("turn", 50),
        ("stop", 60), ("swim", 70),
    ]))
    report = analyse(frame, fps=100.0, min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    assert pairs(report) == {("swim", "turn"): 2, ("turn", "swim"): 1,
                             ("turn", "stop"): 1, ("stop", "swim"): 1}, pairs(report)
    assert t["n_pairs_considered"] == 5, t["n_pairs_considered"]
    assert t["n_counted"] == 5, t["n_counted"]
    assert t["skipped"] == {"gap": 0, "not_adjacent": 0, "human_overlap": 0}, t["skipped"]
    assert t["row_totals"] == {"stop": 1, "swim": 2, "turn": 2}, t["row_totals"]
    assert abs(t["probabilities"]["turn"]["swim"] - 0.5) < TOL
    assert abs(t["probabilities"]["turn"]["stop"] - 0.5) < TOL
    assert t["probabilities"]["swim"] == {"turn": 1.0}, t["probabilities"]["swim"]
    assert all(abs(sum(row.values()) - 1.0) < TOL for row in t["probabilities"].values())
    assert all(a != b for a, b in pairs(report)), "the diagonal is empty for maximal runs"
    assert "diagonal is empty by construction" in t["note"], t["note"]


def test_a_gap_breaks_the_transition_rather_than_joining_across_it() -> None:
    frame = table(plant([
        ("swim", 20), ("turn", 30), ("", 40), ("stop", 50), ("swim", 60),
    ]))
    report = analyse(frame, fps=100.0, min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    assert pairs(report) == {("swim", "turn"): 1, ("stop", "swim"): 1}, pairs(report)
    assert t["skipped"]["gap"] == 2, t["skipped"]
    assert t["n_pairs_considered"] == 4 and t["n_counted"] == 2, (
        f"{t['n_pairs_considered']} considered, {t['n_counted']} counted")
    assert report["groups"][GROUP]["n_bouts_gap"] == 1, "the gap bout is reported"
    assert "" not in report["groups"][GROUP]["classes"], "the gap is not a class"


def test_a_track_boundary_and_a_movie_boundary_are_not_transitions() -> None:
    """Contiguous frames across two tracks, and across two movies, join nothing."""
    frame = table(
        plant([("turn", 10), ("swim", 20)], track=0, start=0),
        plant([("turn", 30), ("stop", 40)], track=1, start=30),
        plant([("turn", 50), ("stop", 60)], movie="movie_b", track=1, start=100),
    )
    report = analyse(frame, fps=100.0, min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    assert pairs(report) == {("turn", "swim"): 1, ("turn", "stop"): 2}, pairs(report)
    assert t["n_pairs_considered"] == 3, t["n_pairs_considered"]
    assert t["n_track_ends"] == 3 and t["n_movie_ends"] == 2, (
        f"{t['n_track_ends']} track ends over {t['n_movie_ends']} movies")


def test_excluding_hand_labelled_bouts_breaks_the_pair_it_sits_in() -> None:
    """Dropping a bout must not silently join the two around it."""
    frame = table(plant(
        [("swim", 20), ("turn", 30), ("stop", 40)], human=(1,)
    ))
    kept = analyse(frame, fps=100.0, min_bout_frames=12)
    dropped = analyse(frame, fps=100.0, min_bout_frames=12,
                      exclude_human_overlap=True)

    assert pairs(kept) == {("swim", "turn"): 1, ("turn", "stop"): 1}, pairs(kept)
    assert pairs(dropped) == {}, pairs(dropped)
    assert dropped["groups"][GROUP]["transitions"]["skipped"]["human_overlap"] == 2, (
        dropped["groups"][GROUP]["transitions"]["skipped"])
    assert kept["groups"][GROUP]["n_bouts_human_overlap"] == 1
    assert kept["groups"][GROUP]["n_bouts_used"] == 3
    assert dropped["groups"][GROUP]["n_bouts_human_overlap"] == 1
    assert dropped["groups"][GROUP]["n_bouts_used"] == 2
    assert dropped["options"]["exclude_human_overlap"] is True
    assert kept["options"]["exclude_human_overlap"] is False


# --------------------------------------------------------------------------- #
# dwell times and censoring
# --------------------------------------------------------------------------- #
def test_a_track_final_bout_does_not_enter_the_dwell_estimate_as_if_complete() -> None:
    """``turn`` has complete bouts of 20 and 300 and a censored final 200.

    Complete-case mean 160, naive mean 173.33, Kaplan-Meier 206.67.
    """
    frame = table(plant([
        ("swim", 10), ("turn", 20), ("swim", 30),
        ("turn", 300), ("swim", 40), ("turn", 200),
    ]))
    report = analyse(frame, fps=100.0, min_bout_frames=12)
    turn = dwell(report, "turn")

    assert turn["n_complete"] == 2 and turn["n_right_censored"] == 1, (
        f"{turn['n_complete']} complete, {turn['n_right_censored']} censored")
    assert abs(turn["frames"]["mean"] - 160.0) < TOL, turn["frames"]["mean"]
    assert abs(turn["km_frames"]["restricted_mean"] - 620.0 / 3.0) < 1e-6, (
        turn["km_frames"]["restricted_mean"])
    assert (turn["km_frames"]["n"] == 3 and turn["km_frames"]["n_events"] == 2
            and turn["km_frames"]["n_censored"] == 1), turn["km_frames"]
    assert abs(turn["seconds"]["mean"] - 1.6) < TOL, turn["seconds"]["mean"]


def test_a_bout_whose_start_was_never_seen_is_dropped() -> None:
    frame = table(plant([
        ("swim", 10), ("turn", 20), ("swim", 30),
        ("turn", 300), ("swim", 40), ("turn", 200),
    ]))
    swim = dwell(analyse(frame, fps=100.0, min_bout_frames=12), "swim")
    assert swim["n_left_open_dropped"] == 1 and swim["n_complete"] == 2, (
        f"{swim['n_left_open_dropped']} dropped, {swim['n_complete']} complete")
    assert abs(swim["frames"]["mean"] - 35.0) < TOL, swim["frames"]["mean"]


def test_a_gap_censors_the_bouts_on_either_side_of_it() -> None:
    """An unlabelled run hides an end on one side and a start on the other."""
    frame = table(plant([
        ("swim", 10), ("turn", 20), ("", 30), ("stop", 40), ("swim", 50),
    ]))
    report = analyse(frame, fps=100.0, min_bout_frames=12)

    assert dwell(report, "turn")["n_right_censored"] == 1
    assert dwell(report, "turn")["n_complete"] == 0
    assert dwell(report, "stop")["n_left_open_dropped"] == 1
    assert dwell(report, "stop")["n_complete"] == 0
    assert all(dwell(report, name)["n_complete"] == 0
               for name in report["groups"][GROUP]["classes"])


def test_the_floor_the_decode_imposes_is_in_the_output() -> None:
    frame = table(plant([("swim", 20), ("turn", 30), ("swim", 40)]))
    floored = analyse(frame, fps=100.0, min_bout_frames=12)

    assert floored["groups"][GROUP]["dwell"]["floor_frames"] == 12
    assert ("predict.min_bout_frames: 12"
            in floored["groups"][GROUP]["dwell"]["floor_source"])
    assert any("min_bout_frames: 12" in c for c in floored["caveats"]), floored["caveats"]
    assert "shortest possible bout 12 frame(s)" in summarise(floored)
    assert dwell(floored, "swim")["observed_min_frames"] == 20.0, (
        "the observed shortest bout is reported beside the floor")


def test_no_floor_is_declared_without_one() -> None:
    frame = table(plant([("swim", 20), ("turn", 30), ("swim", 40)]))
    free = analyse(frame, fps=100.0, min_bout_frames=0)
    assert free["groups"][GROUP]["dwell"]["floor_frames"] == 1
    assert "No hard floor" in free["groups"][GROUP]["dwell"]["floor_source"]


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def test_kaplan_meier_without_censoring_is_the_sample_mean() -> None:
    plain = kaplan_meier([1.0, 2.0, 3.0], [True, True, True])
    assert abs(plain["restricted_mean"] - 2.0) < TOL, plain["restricted_mean"]
    assert plain["reaches_zero"] and plain["restricted_to"] == 3.0
    assert plain["median"] == 2.0, plain["median"]


def test_kaplan_meier_keeps_a_censored_observation_at_risk() -> None:
    """S drops to 2/3 at t=20 and to 0 at t=300; the 200 is censored in between."""
    censored = kaplan_meier([20.0, 200.0, 300.0], [True, False, True])
    assert abs(censored["restricted_mean"] - (20.0 + (2.0 / 3.0) * 280.0)) < 1e-9, (
        censored["restricted_mean"])
    assert (censored["n"] == 3 and censored["n_events"] == 2
            and censored["n_censored"] == 1), censored


def test_kaplan_meier_that_never_reaches_zero_restricts_its_mean() -> None:
    tail = kaplan_meier([10.0, 50.0, 100.0], [True, False, False])
    assert not tail["reaches_zero"]
    assert tail["restricted_to"] == 100.0
    assert tail["median"] is None


def test_kaplan_meier_of_nothing_is_none() -> None:
    assert kaplan_meier([], [])["restricted_mean"] is None


def test_the_exponential_fit_is_the_textbook_estimate() -> None:
    """Two ends over ten units at risk is a mean of five, censoring included."""
    fit = exponential_fit([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
    assert abs(fit["mean"] - 5.0) < TOL and fit["n_ends"] == 2, fit
    assert abs(fit["rate"] - 0.2) < TOL, fit["rate"]
    assert exponential_fit([1.0, 2.0], [False, False])["mean"] is None


# --------------------------------------------------------------------------- #
# the Markov summary
# --------------------------------------------------------------------------- #
ALTERNATING = [
    ("swim", 20), ("turn", 50), ("swim", 20), ("turn", 50),
    ("swim", 20), ("turn", 50), ("swim", 20),
]


@pytest.fixture(scope="module")
def alternating() -> pd.DataFrame:
    """Seven alternating bouts: the first has no seen start, the last no seen end."""
    return table(plant(ALTERNATING))


def test_a_lifetime_is_the_time_at_risk_over_the_ends_seen(alternating) -> None:
    """swim offers 3 x 0.2 s against 2 ends, turn 3 x 0.5 s against 3."""
    m = markov(analyse(alternating, fps=100.0, min_bout_frames=0))

    assert m["unit"] == "seconds"
    assert m["lifetime"]["swim"]["n_ends"] == 2
    assert abs(m["lifetime"]["swim"]["at_risk"] - 0.6) < TOL
    assert abs(m["lifetime"]["swim"]["mean"] - 0.3) < TOL
    assert abs(m["lifetime"]["turn"]["mean"] - 0.5) < TOL
    assert m["lifetime"]["swim"]["mean"] > m["lifetime"]["swim"]["km_restricted_mean"], (
        "a censored bout lifts the fit above the restricted mean")
    assert len(m["survival"]["turn"]["t"]) == len(m["survival"]["turn"]["fit"])
    assert m["survival"]["turn"]["p"][0] == 1.0
    assert m["lifetime"]["turn"]["fit_max_deviation"] is not None


def test_a_rate_is_a_jump_probability_over_a_lifetime(alternating) -> None:
    """q(i->j) = P(next=j | in i) / lifetime(i)."""
    report = analyse(alternating, fps=100.0, min_bout_frames=0)
    m = markov(report)
    probabilities = report["groups"][GROUP]["transitions"]["probabilities"]

    assert abs(m["jump"]["swim"]["turn"]["p"] - 1.0) < TOL
    assert abs(m["jump"]["turn"]["swim"]["p"] - 1.0) < TOL
    assert m["jump"]["swim"]["turn"]["p"] == probabilities["swim"]["turn"], (
        "the jump matrix is the transition matrix")
    assert abs(m["rates"]["swim"]["turn"]["q"] - 1.0 / 0.3) < TOL
    assert abs(m["rates"]["turn"]["swim"]["q"] - 2.0) < TOL
    assert abs(m["exit_rate"]["swim"] - 1.0 / 0.3) < TOL
    assert abs(m["exit_rate"]["turn"] - 2.0) < TOL


@pytest.mark.parametrize("floor, lower", [(0, False), (12, True)])
def test_rates_are_lower_bounds_only_under_a_floor(alternating, floor, lower) -> None:
    caveat = markov(analyse(alternating, fps=100.0, min_bout_frames=floor))["caveat"]
    assert ("LOWER BOUNDS" in caveat) is lower, caveat


def test_the_steady_state_weights_a_state_by_how_long_it_is_held(alternating) -> None:
    """Entered equally often, held 0.3 s against 0.5 s: 3/8 and 5/8."""
    m = markov(analyse(alternating, fps=100.0, min_bout_frames=0))
    steady = m["steady_state"]

    assert steady is not None and m["steady_state_refusal"] is None
    assert abs(steady["swim"]["p"] - 0.375) < TOL, steady
    assert abs(steady["turn"]["p"] - 0.625) < TOL, steady
    assert abs(sum(v["p"] for v in steady.values()) - 1.0) < TOL
    assert abs(m["occupancy_observed"]["swim"] - 80.0 / 230.0) < TOL
    assert abs(m["occupancy_observed"]["turn"] - 150.0 / 230.0) < TOL


def test_a_state_with_no_successor_has_no_steady_state_and_says_so() -> None:
    frame = table(plant([("a", 10), ("b", 10), ("c", 10)]))
    m = markov(analyse(frame, fps=100.0, min_bout_frames=0))

    assert m["steady_state"] is None
    assert "no observed successor" in m["steady_state_refusal"], m["steady_state_refusal"]
    assert abs(m["jump"]["a"]["b"]["p"] - 1.0) < TOL, "seen transitions are still reported"


def test_one_track_cannot_support_an_error_bar(alternating) -> None:
    one = markov(analyse(alternating, fps=100.0, min_bout_frames=0))
    assert one["lifetime"]["turn"]["se"] is None
    assert one["bootstrap"]["n_tracks"] == 1


def test_the_error_bars_come_from_resampling_whole_tracks() -> None:
    spread = table(*[
        plant([("swim", 20), ("turn", 30 + 10 * t), ("swim", 20),
               ("turn", 30 + 10 * t), ("swim", 20)], track=t)
        for t in range(8)
    ])
    m = markov(analyse(spread, fps=100.0, min_bout_frames=0))
    assert m["bootstrap"]["n_tracks"] == 8
    assert m["bootstrap"]["unit"] == "track"
    assert m["lifetime"]["turn"]["se"] > 0.0
    assert m["steady_state"]["swim"]["se"] > 0.0

    again = markov(analyse(spread, fps=100.0, min_bout_frames=0))
    assert again["lifetime"]["turn"]["se"] == m["lifetime"]["turn"]["se"], (
        "the same bouts and seed give the same error bar")


@pytest.fixture(scope="module")
def simulated():
    """A memoryless 3-state chain with known lifetimes, cut into 400 tracks."""
    names = ("A", "B", "C")
    tau = np.array([20.0, 60.0, 40.0])                 # mean dwell, frames
    rho = np.array([[0.0, 0.7, 0.3],
                    [0.5, 0.0, 0.5],
                    [0.2, 0.8, 0.0]])
    values, vectors = np.linalg.eig(rho.T)
    nu = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    nu = nu / nu.sum()
    pi = nu * tau / float((nu * tau).sum())

    rng = np.random.default_rng(7)
    blocks = []
    for track in range(400):
        spec, f, s = [], 0, int(rng.choice(3, p=pi))
        while f < 300:
            n = min(int(rng.geometric(1.0 / tau[s])), 300 - f)
            spec.append((names[s], n))
            f += n
            s = int(rng.choice(3, p=rho[s]))
        blocks.append(plant(spec, track=track))
    report = analyse(table(*blocks), n_bootstrap=50, bootstrap_seed=0)
    return names, tau, rho, pi, report


def test_a_simulated_chain_returns_its_lifetimes(simulated) -> None:
    names, tau, _, _, report = simulated
    m = markov(report)
    est = np.array([m["lifetime"][name]["mean"] for name in names])
    se = np.array([m["lifetime"][name]["se"] for name in names])
    assert np.all(np.abs(est - tau) < 3.0 * se), f"{est.round(1)} vs {tau} +- {se.round(1)}"

    naive = np.array([dwell(report, name)["frames"]["mean"] for name in names])
    assert np.all(naive < est) and naive[1] < 0.9 * tau[1], (
        f"complete-only {naive.round(1)} is not below fitted {est.round(1)}")

    worst = max(m["lifetime"][name]["fit_max_deviation"] for name in names)
    assert worst < 0.05, f"a memoryless chain fits its exponential badly: {worst:.3f}"


def test_a_simulated_chain_returns_its_jump_matrix_and_occupancy(simulated) -> None:
    names, _, rho, pi, report = simulated
    m = markov(report)
    jump = np.array([[(m["jump"].get(a, {}).get(b) or {}).get("p") or 0.0
                      for b in names] for a in names])
    assert float(np.abs(jump - rho).max()) < 0.03, float(np.abs(jump - rho).max())

    steady = np.array([m["steady_state"][name]["p"] for name in names])
    occupancy = np.array([m["occupancy_observed"][name] for name in names])
    assert float(np.abs(steady - pi).max()) < 0.02, f"{steady.round(3)} vs {pi.round(3)}"
    assert float(np.abs(occupancy - pi).max()) < 0.02, f"{occupancy.round(3)} vs {pi.round(3)}"


# --------------------------------------------------------------------------- #
# frame rates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_rates() -> pd.DataFrame:
    """The same bouts in a 100 fps movie and a 10 fps one."""
    return table(
        plant([("swim", 10), ("turn", 100), ("swim", 10)], movie="fast"),
        plant([("swim", 10), ("turn", 100), ("swim", 10)], movie="slow", track=1),
    )


def test_seconds_come_from_each_movies_own_frame_rate(two_rates) -> None:
    report = analyse(two_rates, fps={"fast": 100.0, "slow": 10.0})
    seconds = dwell(report, "turn")["seconds"]

    assert abs(seconds["mean"] - 5.5) < TOL, seconds
    assert seconds["min"] == 1.0 and seconds["max"] == 10.0, seconds
    assert abs(dwell(report, "turn")["frames"]["mean"] - 100.0) < TOL
    assert report["fps"]["uniform"] is None
    assert report["fps"]["per_movie"] == {"fast": 100.0, "slow": 10.0}


def test_a_movie_with_no_frame_rate_is_named_and_left_out_of_seconds(two_rates) -> None:
    partial = analyse(two_rates, fps={"fast": 100.0})
    assert partial["fps"]["missing"] == ["slow"]
    assert dwell(partial, "turn")["n_without_fps"] == 1
    assert abs(dwell(partial, "turn")["seconds"]["mean"] - 1.0) < TOL
    assert "no frame rate for slow" in summarise(partial)


# --------------------------------------------------------------------------- #
# what the file has to be
# --------------------------------------------------------------------------- #
_WINDOWS = pd.DataFrame({
    "clip_id": ["a", "b"],
    "movie_id": ["movie_a", "movie_a"],
    "track_id": [0, 0],
    "f_start": [0, 50],
    "f_end": [50, 100],
    "skip_reason": ["", ""],
    f"{GROUP}_pred": ["swim", "turn"],
    f"{GROUP}_conf": [0.9, 0.8],
})


@pytest.mark.parametrize("frame", [
    pytest.param(_WINDOWS, id="dense-windows"),
    pytest.param(_WINDOWS.drop(columns=["skip_reason"]), id="clip-manifest"),
])
def test_a_table_that_is_not_bouts_is_refused(frame) -> None:
    with pytest.raises(ValueError, match="not decoded bouts"):
        analyse(frame, fps=100.0)


def test_a_bout_table_missing_a_column_is_refused_by_name() -> None:
    thin = table(plant([("swim", 20), ("turn", 30)])).drop(columns=["n_covering_mean"])
    with pytest.raises(ValueError, match="n_covering_mean"):
        analyse(thin, fps=100.0)


def test_the_report_says_these_are_predictions_and_points_at_the_confusion() -> None:
    frame = table(plant([
        ("swim", 20), ("turn", 30), ("swim", 40), ("turn", 50),
    ]))
    model = {GROUP: {
        "classes": ["swim", "turn"],
        "confusion": [[95, 5], [30, 70]],      # 30 of 100 true turns called swim
        "kappa": 0.61,
        "accuracy": 0.83,
        "reliable": True,
    }}
    report = analyse(frame, fps=100.0, min_bout_frames=12,
                     model_report=model, model_report_path="work/suggest_report.json")
    text = summarise(report)
    worst = report["groups"][GROUP]["model"]["most_confused"]

    assert any("DECODED PREDICTIONS, not ground truth" in c for c in report["caveats"]), (
        report["caveats"])
    assert any("suggest_report.json" in c for c in report["caveats"]), report["caveats"]
    assert "suggest_report.json" in text
    assert worst["from"] == "turn" and worst["to"] == "swim", worst
    assert abs(worst["frac"] - 0.3) < TOL, worst
    assert "mistakes turn for swim" in text, text
    assert abs(report["groups"][GROUP]["model"]["kappa"] - 0.61) < TOL


def test_a_missing_model_report_is_stated() -> None:
    frame = table(plant([("swim", 20), ("turn", 30)]))
    silent = analyse(frame, fps=100.0, min_bout_frames=12)
    assert any("there is no suggest_report.json" in c for c in silent["caveats"]), (
        silent["caveats"])


def test_the_printed_summary_answers_the_four_questions() -> None:
    """What follows what, how long, how much of the record, and on what evidence."""
    frame = table(
        plant([("swim", 20), ("turn", 30), ("swim", 40), ("turn", 50),
               ("swim", 60)], track=0),
        plant([("turn", 25), ("stop", 35), ("turn", 45)], track=1, start=200),
    )
    text = summarise(analyse(frame, fps=100.0, min_bout_frames=12))

    for needle in ("turn 1.00 (2)", "n=2", "compl", "cens", "drop",
                   "durations in seconds", "pairs counted", "steady", "measured",
                   "track resamples"):
        assert needle in text, f"{needle!r} missing from: {text}"


def test_the_report_is_json_with_no_nan_in_it() -> None:
    frame = table(
        plant([("swim", 20), ("turn", 30)]),
        plant([("stop", 40)], track=9, start=500),     # every statistic undefined
    )
    report = analyse(frame, fps=None)
    text = json.dumps(report, indent=2, allow_nan=False)

    assert "NaN" not in text and "Infinity" not in text
    assert (json.loads(text)["groups"][GROUP]["dwell"]["per_class"]["stop"]
            ["frames"]["mean"] is None), "an undefined statistic is null rather than absent"
    assert report["groups"][GROUP]["transitions"]["row_totals"]["stop"] == 0
    assert report["fps"]["per_movie"] == {}
    assert dwell(report, "swim")["seconds"]["n"] == 0
    assert "transitions and dwell times" in summarise(report)


# --------------------------------------------------------------------------- #
# against a real project
# --------------------------------------------------------------------------- #
def test_a_project_is_measured_from_its_own_files(root) -> None:
    """``compute`` reads the config, the frame rate and the model report itself."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    project = root / "measured"
    config_path = create_project(project)
    write_config(config_path, {"predict.min_bout_frames": 12})
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    (work / "movie_a").mkdir(parents=True, exist_ok=True)
    (work / "movie_a" / "cache.json").write_text(
        json.dumps({"movie": {"fps": 50.0, "n_frames": 400}}), encoding="utf-8")
    (work / "suggest_report.json").write_text(
        json.dumps({GROUP: {"classes": ["swimming", "immotile"],
                            "confusion": [[90, 10], [20, 80]],
                            "kappa": 0.7, "accuracy": 0.85, "reliable": True}}),
        encoding="utf-8")
    table(plant([
        ("swimming", 20), ("immotile", 30), ("swimming", 40),
        ("immotile", 50), ("swimming", 60),
    ])).to_parquet(work / "predictions.parquet", index=False)

    assert movie_fps(cfg) == {"movie_a": 50.0}, movie_fps(cfg)

    report = compute(cfg)
    assert report["source"]["min_bout_frames"] == 12
    assert report["groups"][GROUP]["dwell"]["floor_frames"] == 12
    assert pairs(report) == {("swimming", "immotile"): 2,
                             ("immotile", "swimming"): 2}, pairs(report)
    assert abs(dwell(report, "immotile")["seconds"]["mean"] - 0.8) < TOL, (
        dwell(report, "immotile")["seconds"]["mean"])
    assert report["source"]["suggest_report"].endswith("suggest_report.json")

    path = write_report(cfg, report)
    assert path == work / TRANSITIONS_FILE and path.exists()
    assert (json.loads(path.read_text(encoding="utf-8"))["groups"][GROUP]
            ["transitions"]["counts"] == report["groups"][GROUP]["transitions"]["counts"])
    assert not list(work.glob("*.part")), "nothing staged is left behind"
