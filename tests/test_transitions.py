"""Checks for :mod:`trackanno.infer.transitions`, the bout-to-bout statistics.

Every number below is measured against a PLANTED sequence of bouts, so the
expected answer is available by counting on your fingers and a disagreement is
an error in the module rather than a disagreement between two implementations.

Five things are worth testing separately:

* the transition counts are exactly the ones planted, with a denominator beside
  every probability;
* a gap, a track boundary and a movie boundary each BREAK a pair rather than
  joining across it, and each is counted as skipped;
* a bout that runs to a track end or into a gap does not enter the dwell
  estimate as if it were complete, and a bout whose start was never seen is
  dropped;
* the Kaplan-Meier estimate is the textbook one, and collapses to the sample
  mean when nothing is censored;
* a predictions file that holds windows or clips is refused with a message
  naming the settings that turn a decode on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from harness import check
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
    """``(label, n_frames)`` pairs into gapless bout rows.

    ``""`` is a run the decode could not label, which is what a real decode
    emits for frames no window reached. ``human`` names positions whose bouts
    overlap hand-labelled frames.
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
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    check("the planted transitions are counted exactly once each",
          pairs(report) == {("swim", "turn"): 2, ("turn", "swim"): 1,
                            ("turn", "stop"): 1, ("stop", "swim"): 1},
          f"got {pairs(report)}")
    check("every consecutive pair of the one track is considered",
          t["n_pairs_considered"] == 5, f"got {t['n_pairs_considered']}")
    check("all five are counted, none skipped",
          t["n_counted"] == 5 and t["skipped"] == {
              "gap": 0, "not_adjacent": 0, "human_overlap": 0},
          f"got {t['n_counted']} counted, skipped {t['skipped']}")
    check("a row total is the denominator of that row's probabilities",
          t["row_totals"] == {"stop": 1, "swim": 2, "turn": 2},
          f"got {t['row_totals']}")
    check("turn splits evenly between the two states that followed it",
          abs(t["probabilities"]["turn"]["swim"] - 0.5) < TOL
          and abs(t["probabilities"]["turn"]["stop"] - 0.5) < TOL)
    check("swim only ever went to turn, at probability 1",
          t["probabilities"]["swim"] == {"turn": 1.0},
          f"got {t['probabilities']['swim']}")
    check("every row of probabilities sums to 1",
          all(abs(sum(row.values()) - 1.0) < TOL
              for row in t["probabilities"].values()))
    check("the diagonal is empty, because a decode emits maximal runs",
          all(a != b for a, b in pairs(report)))
    check("the summary states that the diagonal is empty by construction",
          "diagonal is empty by construction" in t["note"])


def test_a_gap_breaks_the_transition_rather_than_joining_across_it() -> None:
    """A run the decode could not label is not a transition between its neighbours."""
    frame = table(plant([
        ("swim", 20), ("turn", 30), ("", 40), ("stop", 50), ("swim", 60),
    ]))
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    check("only the two pairs that touch a labelled bout on both sides count",
          pairs(report) == {("swim", "turn"): 1, ("stop", "swim"): 1},
          f"got {pairs(report)}")
    check("turn -> stop is never invented across the gap",
          ("turn", "stop") not in pairs(report))
    check("the two pairs the gap sits in are counted as skipped",
          t["skipped"]["gap"] == 2, f"got {t['skipped']}")
    check("the skipped pairs are still counted as considered",
          t["n_pairs_considered"] == 4 and t["n_counted"] == 2,
          f"got {t['n_pairs_considered']} / {t['n_counted']}")
    check("the gap bout is reported rather than silently dropped",
          report["groups"][GROUP]["n_bouts_gap"] == 1)
    check("the gap is not a class of its own",
          "" not in report["groups"][GROUP]["classes"])


def test_a_track_boundary_and_a_movie_boundary_are_not_transitions() -> None:
    """Contiguous frames across two tracks, and across two movies, join nothing."""
    frame = table(
        # Track 0 ends at frame 30 and track 1 starts there, so only the track
        # id separates the last bout of one from the first of the next.
        plant([("turn", 10), ("swim", 20)], track=0, start=0),
        plant([("turn", 30), ("stop", 40)], track=1, start=30),
        # Same track id, other movie, again picking up where the last left off.
        plant([("turn", 50), ("stop", 60)], movie="movie_b", track=1, start=100),
    )
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    t = report["groups"][GROUP]["transitions"]

    check("the three within-track changes are the only ones counted",
          pairs(report) == {("turn", "swim"): 1, ("turn", "stop"): 2},
          f"got {pairs(report)}")
    check("swim -> turn is never formed across the track boundary",
          ("swim", "turn") not in pairs(report))
    check("stop -> turn is never formed across the movie boundary",
          ("stop", "turn") not in pairs(report))
    check("a pair is never even considered across a boundary",
          t["n_pairs_considered"] == 3, f"got {t['n_pairs_considered']}")
    check("the boundaries that produced no successor are reported",
          t["n_track_ends"] == 3 and t["n_movie_ends"] == 2,
          f"got {t['n_track_ends']} track ends over {t['n_movie_ends']} movies")


def test_excluding_hand_labelled_bouts_breaks_the_pair_it_sits_in() -> None:
    """Dropping a bout must not silently join the two around it."""
    frame = table(plant(
        [("swim", 20), ("turn", 30), ("stop", 40)], human=(1,)
    ))
    kept = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    dropped = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12,
                      exclude_human_overlap=True)

    check("with the hand-labelled bout kept, both changes are counted",
          pairs(kept) == {("swim", "turn"): 1, ("turn", "stop"): 1},
          f"got {pairs(kept)}")
    check("excluding it counts no transitions at all",
          pairs(dropped) == {}, f"got {pairs(dropped)}")
    check("swim -> stop is never invented over the excluded bout",
          ("swim", "stop") not in pairs(dropped))
    check("the excluded pairs are counted under their own reason",
          dropped["groups"][GROUP]["transitions"]["skipped"]["human_overlap"] == 2,
          f"got {dropped['groups'][GROUP]['transitions']['skipped']}")
    check("both counts are reported either way",
          kept["groups"][GROUP]["n_bouts_human_overlap"] == 1
          and kept["groups"][GROUP]["n_bouts_used"] == 3
          and dropped["groups"][GROUP]["n_bouts_human_overlap"] == 1
          and dropped["groups"][GROUP]["n_bouts_used"] == 2)
    check("the option taken is recorded in the report",
          dropped["options"]["exclude_human_overlap"] is True
          and kept["options"]["exclude_human_overlap"] is False)


# --------------------------------------------------------------------------- #
# dwell times and censoring
# --------------------------------------------------------------------------- #
def test_a_track_final_bout_does_not_enter_the_dwell_estimate_as_if_complete() -> None:
    """Six bouts: one starts unseen, one ends unseen, four are whole.

    ``turn`` has complete bouts of 20 and 300 frames and a track-final one of
    200. Three means separate the treatments: 160 counts only the complete
    ones, 173.33 would count the censored one as if it had ended, and the
    Kaplan-Meier estimate is 206.67.
    """
    frame = table(plant([
        ("swim", 10), ("turn", 20), ("swim", 30),
        ("turn", 300), ("swim", 40), ("turn", 200),
    ]))
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    turn, swim = dwell(report, "turn"), dwell(report, "swim")

    check("the track-final bout is counted as censored, not as complete",
          turn["n_complete"] == 2 and turn["n_right_censored"] == 1,
          f"got {turn['n_complete']} complete, {turn['n_right_censored']} censored")
    check("the complete-case mean is over the complete bouts alone",
          abs(turn["frames"]["mean"] - 160.0) < TOL, f"got {turn['frames']['mean']}")
    check("the censored bout is not averaged in as if it had ended",
          abs(turn["frames"]["mean"] - 500.0 / 3.0) > 1.0)
    check("Kaplan-Meier folds the censored bout in as a lower bound",
          abs(turn["km_frames"]["restricted_mean"] - 620.0 / 3.0) < 1e-6,
          f"got {turn['km_frames']['restricted_mean']}")
    check("the censoring-aware mean exceeds the complete-case one here",
          turn["km_frames"]["restricted_mean"] > turn["frames"]["mean"])
    check("the Kaplan-Meier population is the complete plus the censored",
          turn["km_frames"]["n"] == 3 and turn["km_frames"]["n_events"] == 2
          and turn["km_frames"]["n_censored"] == 1)
    check("a bout whose start was never seen is dropped and counted",
          swim["n_left_open_dropped"] == 1 and swim["n_complete"] == 2,
          f"got {swim['n_left_open_dropped']} dropped, {swim['n_complete']} complete")
    check("the track-start bout is not in the complete-case mean either",
          abs(swim["frames"]["mean"] - 35.0) < TOL, f"got {swim['frames']['mean']}")
    check("seconds come from the frame rate beside the frames",
          abs(turn["seconds"]["mean"] - 1.6) < TOL, f"got {turn['seconds']['mean']}")


def test_a_gap_censors_the_bouts_on_either_side_of_it() -> None:
    """An unlabelled run hides an end on one side and a start on the other."""
    frame = table(plant([
        ("swim", 10), ("turn", 20), ("", 30), ("stop", 40), ("swim", 50),
    ]))
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)

    check("the bout that runs into the gap is right-censored, not complete",
          dwell(report, "turn")["n_right_censored"] == 1
          and dwell(report, "turn")["n_complete"] == 0)
    check("the bout that comes out of the gap has an unseen start and is dropped",
          dwell(report, "stop")["n_left_open_dropped"] == 1
          and dwell(report, "stop")["n_complete"] == 0)
    check("no class claims a complete bout in this track",
          all(dwell(report, name)["n_complete"] == 0
              for name in report["groups"][GROUP]["classes"]))


def test_the_floor_the_decode_imposes_is_in_the_output() -> None:
    """The shortest possible bout is a setting, and the report says so."""
    frame = table(plant([("swim", 20), ("turn", 30), ("swim", 40)]))
    floored = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    free = analyse(frame, fps=100.0, decode="none", min_bout_frames=0)
    soft = analyse(frame, fps=100.0, decode="viterbi", min_bout_frames=25)

    check("min_bout reports its floor as a number, not only as prose",
          floored["groups"][GROUP]["dwell"]["floor_frames"] == 12)
    check("the setting that set the floor is named in the report",
          "predict.min_bout_frames: 12"
          in floored["groups"][GROUP]["dwell"]["floor_source"])
    check("the floor is a caveat on the whole report, not only on one group",
          any("min_bout_frames: 12" in c for c in floored["caveats"]),
          f"got {floored['caveats']}")
    check("the floor reaches the printed summary",
          "shortest bout the decode can emit: 12 frame(s)" in summarise(floored))
    check("without a decode there is no floor to declare",
          free["groups"][GROUP]["dwell"]["floor_frames"] == 1
          and "No hard floor" in free["groups"][GROUP]["dwell"]["floor_source"])
    check("viterbi's floor is named as a prior rather than a hard limit",
          soft["groups"][GROUP]["dwell"]["floor_frames"] == 1
          and "rather than a hard floor" in soft["groups"][GROUP]["dwell"]["floor_source"])
    check("the observed shortest bout is reported beside the settable floor",
          dwell(floored, "swim")["observed_min_frames"] == 20.0)


def test_kaplan_meier_is_the_textbook_estimate() -> None:
    """Hand-computed curves, and the uncensored case that must be the mean."""
    plain = kaplan_meier([1.0, 2.0, 3.0], [True, True, True])
    check("with nothing censored the restricted mean is the sample mean",
          abs(plain["restricted_mean"] - 2.0) < TOL,
          f"got {plain['restricted_mean']}")
    check("with nothing censored the curve reaches zero",
          plain["reaches_zero"] and plain["restricted_to"] == 3.0)
    check("the median is the first time survival drops to a half",
          plain["median"] == 2.0, f"got {plain['median']}")

    # S drops to 2/3 at t=20 and to 0 at t=300; the 200 is censored in between.
    censored = kaplan_meier([20.0, 200.0, 300.0], [True, False, True])
    check("a censored observation stays at risk until its own time",
          abs(censored["restricted_mean"] - (20.0 + (2.0 / 3.0) * 280.0)) < 1e-9,
          f"got {censored['restricted_mean']}")
    check("the censored observation is counted but is not an event",
          censored["n"] == 3 and censored["n_events"] == 2
          and censored["n_censored"] == 1)

    tail = kaplan_meier([10.0, 50.0, 100.0], [True, False, False])
    check("a curve that never reaches zero says so, and restricts its mean",
          not tail["reaches_zero"] and tail["restricted_to"] == 100.0
          and tail["median"] is None)
    check("an empty sample is answered with None rather than a number",
          kaplan_meier([], [])["restricted_mean"] is None)


ALTERNATING = [
    ("swim", 20), ("turn", 50), ("swim", 20), ("turn", 50),
    ("swim", 20), ("turn", 50), ("swim", 20),
]


def test_a_lifetime_is_the_time_at_risk_over_the_ends_seen() -> None:
    """The exponential fit of Fig. 1h, planted so it is exact.

    Seven alternating bouts on one track. The first has no seen start and is
    dropped; the last runs off the end of the track and contributes its time
    without an end. So swim offers 3 x 0.2 s against 2 ends and turn 3 x 0.5 s
    against 3, and the two means come apart by exactly that censoring.
    """
    frame = table(plant(ALTERNATING))
    m = markov(analyse(frame, fps=100.0, decode="none", min_bout_frames=0))

    check("the lifetimes are in seconds where a frame rate is known",
          m["unit"] == "seconds")
    check("swim was seen to end twice over 0.6 s at risk",
          m["lifetime"]["swim"]["n_ends"] == 2
          and abs(m["lifetime"]["swim"]["at_risk"] - 0.6) < TOL)
    check("swim's fitted lifetime is 0.6 s over 2 ends",
          abs(m["lifetime"]["swim"]["mean"] - 0.3) < TOL)
    check("turn's fitted lifetime is 1.5 s over 3 ends",
          abs(m["lifetime"]["turn"]["mean"] - 0.5) < TOL)
    check("a censored bout lifts the fit above the mean it was censored out of",
          m["lifetime"]["swim"]["mean"] > m["lifetime"]["swim"]["km_restricted_mean"])
    check("the fit is carried beside the measured curve it is judged against",
          len(m["survival"]["turn"]["t"]) == len(m["survival"]["turn"]["fit"])
          and m["survival"]["turn"]["p"][0] == 1.0
          and m["lifetime"]["turn"]["fit_max_deviation"] is not None)


def test_the_exponential_fit_is_the_textbook_estimate() -> None:
    """Two ends over ten units at risk is a mean of five, censoring included."""
    fit = exponential_fit([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
    check("the mean is the total time at risk over the number of ends",
          abs(fit["mean"] - 5.0) < TOL and fit["n_ends"] == 2)
    check("the rate is one over that mean", abs(fit["rate"] - 0.2) < TOL)
    check("nothing having ended leaves the mean unstated",
          exponential_fit([1.0, 2.0], [False, False])["mean"] is None)


def test_a_rate_is_a_jump_probability_over_a_lifetime() -> None:
    """q(i->j) = P(next=j | in i) / lifetime(i), on the same planted bouts."""
    frame = table(plant(ALTERNATING))
    report = analyse(frame, fps=100.0, decode="none", min_bout_frames=0)
    m = markov(report)
    probabilities = report["groups"][GROUP]["transitions"]["probabilities"]

    check("turn always follows swim here, and swim always follows turn",
          abs(m["jump"]["swim"]["turn"]["p"] - 1.0) < TOL
          and abs(m["jump"]["turn"]["swim"]["p"] - 1.0) < TOL)
    check("the jump matrix is the transition matrix, counted once",
          m["jump"]["swim"]["turn"]["p"] == probabilities["swim"]["turn"])
    check("a directed rate is that probability over the lifetime",
          abs(m["rates"]["swim"]["turn"]["q"] - 1.0 / 0.3) < TOL
          and abs(m["rates"]["turn"]["swim"]["q"] - 2.0) < TOL)
    check("the exit rate is one over the lifetime",
          abs(m["exit_rate"]["swim"] - 1.0 / 0.3) < TOL
          and abs(m["exit_rate"]["turn"] - 2.0) < TOL)
    check("with no floor imposed the rates are not called lower bounds",
          "LOWER BOUNDS" not in m["caveat"])

    floored = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    check("with a floor imposed the rates are called lower bounds",
          "LOWER BOUNDS" in markov(floored)["caveat"])


def test_the_steady_state_weights_a_state_by_how_long_it_is_held() -> None:
    """The occupancies of Fig. 1i: time spent, not bouts counted.

    Swim and turn alternate, so each is entered as often as the other and the
    jump chain alone puts them at a half each. Weighted by a lifetime of 0.3 s
    against 0.5 s they come to 3/8 and 5/8.
    """
    frame = table(plant(ALTERNATING))
    m = markov(analyse(frame, fps=100.0, decode="none", min_bout_frames=0))
    steady = m["steady_state"]

    check("the steady state is stated",
          steady is not None and m["steady_state_refusal"] is None)
    check("it is time-weighted and not a count of bouts",
          abs(steady["swim"]["p"] - 0.375) < TOL
          and abs(steady["turn"]["p"] - 0.625) < TOL)
    check("it sums to one", abs(sum(v["p"] for v in steady.values()) - 1.0) < TOL)
    check("the share measured off the record is reported beside it",
          abs(m["occupancy_observed"]["swim"] - 80.0 / 230.0) < TOL
          and abs(m["occupancy_observed"]["turn"] - 150.0 / 230.0) < TOL)


def test_a_state_with_no_successor_has_no_steady_state_and_says_so() -> None:
    """Three classes in a row that never returns has no stationary law."""
    frame = table(plant([("a", 10), ("b", 10), ("c", 10)]))
    m = markov(analyse(frame, fps=100.0, decode="none", min_bout_frames=0))

    check("no steady state is invented", m["steady_state"] is None)
    check("the refusal names the reason",
          "no observed successor" in m["steady_state_refusal"])
    check("the transitions that were seen are still reported",
          abs(m["jump"]["a"]["b"]["p"] - 1.0) < TOL)


def test_the_error_bars_come_from_resampling_whole_tracks() -> None:
    """One track cannot support an error bar; eight unequal ones can."""
    one = markov(analyse(table(plant(ALTERNATING)), fps=100.0, decode="none",
                         min_bout_frames=0))
    check("a single track leaves every error unstated",
          one["lifetime"]["turn"]["se"] is None
          and one["bootstrap"]["n_tracks"] == 1)

    spread = table(*[
        plant([("swim", 20), ("turn", 30 + 10 * t), ("swim", 20),
               ("turn", 30 + 10 * t), ("swim", 20)], track=t)
        for t in range(8)
    ])
    m = markov(analyse(spread, fps=100.0, decode="none", min_bout_frames=0))
    check("eight tracks are eight bootstrap units", m["bootstrap"]["n_tracks"] == 8)
    check("the resampling unit is named", m["bootstrap"]["unit"] == "track")
    check("a lifetime that differs between tracks carries an error bar",
          m["lifetime"]["turn"]["se"] > 0.0)
    check("so does a steady state probability",
          m["steady_state"]["swim"]["se"] > 0.0)

    again = markov(analyse(spread, fps=100.0, decode="none", min_bout_frames=0))
    check("the same bouts and the same seed give the same error bar",
          again["lifetime"]["turn"]["se"] == m["lifetime"]["turn"]["se"])


def test_a_simulated_chain_comes_back_with_its_own_parameters() -> None:
    """Ground truth by simulation rather than by planting: a memoryless chain
    with known lifetimes and jump matrix, cut into 400 short tracks, must come
    back with those parameters. Geometric dwells are the discrete memoryless
    case, so the censored fit (time at risk over ends) is consistent for the
    true mean, while the complete-bouts-only mean shows the truncation bias
    the censoring exists to remove.
    """
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
    m = markov(report)

    est = np.array([m["lifetime"][name]["mean"] for name in names])
    se = np.array([m["lifetime"][name]["se"] for name in names])
    check("every lifetime is within three bootstrap errors of the truth",
          bool(np.all(np.abs(est - tau) < 3.0 * se)),
          f"{est.round(1)} vs {tau} +- {se.round(1)}")

    naive = np.array([dwell(report, name)["frames"]["mean"] for name in names])
    check("the complete-bouts-only mean sits below the censored fit",
          bool(np.all(naive < est)) and naive[1] < 0.9 * tau[1],
          f"complete-only {naive.round(1)} vs fitted {est.round(1)}")

    jump = np.array([[(m["jump"].get(a, {}).get(b) or {}).get("p") or 0.0
                      for b in names] for a in names])
    check("the jump matrix is the one simulated",
          float(np.abs(jump - rho).max()) < 0.03,
          f"max error {float(np.abs(jump - rho).max()):.4f}")

    steady = np.array([m["steady_state"][name]["p"] for name in names])
    occupancy = np.array([m["occupancy_observed"][name] for name in names])
    check("the steady state and the observed occupancy both recover pi",
          float(np.abs(steady - pi).max()) < 0.02
          and float(np.abs(occupancy - pi).max()) < 0.02,
          f"steady {steady.round(3)} vs true {pi.round(3)}")

    worst = max(m["lifetime"][name]["fit_max_deviation"] for name in names)
    check("a genuinely memoryless chain fits its exponential closely",
          worst < 0.05, f"worst gap {worst:.3f}")


def test_seconds_come_from_each_movies_own_frame_rate() -> None:
    """Two movies at different rates, so one conversion cannot serve both."""
    frame = table(
        plant([("swim", 10), ("turn", 100), ("swim", 10)], movie="fast"),
        plant([("swim", 10), ("turn", 100), ("swim", 10)], movie="slow", track=1),
    )
    report = analyse(frame, fps={"fast": 100.0, "slow": 10.0}, decode="none")

    check("the two complete turn bouts convert to 1 s and 10 s, mean 5.5",
          abs(dwell(report, "turn")["seconds"]["mean"] - 5.5) < TOL
          and dwell(report, "turn")["seconds"]["min"] == 1.0
          and dwell(report, "turn")["seconds"]["max"] == 10.0,
          f"got {dwell(report, 'turn')['seconds']}")
    check("frames stay the same for both, which is what makes seconds necessary",
          abs(dwell(report, "turn")["frames"]["mean"] - 100.0) < TOL)
    check("a single frame rate is not claimed when the movies disagree",
          report["fps"]["uniform"] is None
          and report["fps"]["per_movie"] == {"fast": 100.0, "slow": 10.0})

    partial = analyse(frame, fps={"fast": 100.0}, decode="none")
    check("a movie with no frame rate is named rather than assumed",
          partial["fps"]["missing"] == ["slow"])
    check("bouts with no frame rate are counted and left out of seconds",
          dwell(partial, "turn")["n_without_fps"] == 1
          and abs(dwell(partial, "turn")["seconds"]["mean"] - 1.0) < TOL)
    check("the summary says which movie has no frame rate",
          "no frame rate for slow" in summarise(partial))


# --------------------------------------------------------------------------- #
# what the file has to be
# --------------------------------------------------------------------------- #
def test_a_windows_file_is_refused_with_an_actionable_message() -> None:
    """A window grid has no bouts in it, and the refusal says how to get some."""
    windows = pd.DataFrame({
        "clip_id": ["a", "b"],
        "movie_id": ["movie_a", "movie_a"],
        "track_id": [0, 0],
        "f_start": [0, 50],
        "f_end": [50, 100],
        "skip_reason": ["", ""],
        f"{GROUP}_pred": ["swim", "turn"],
        f"{GROUP}_conf": [0.9, 0.8],
    })
    clips = windows.drop(columns=["skip_reason"])

    def refusal(frame) -> str:
        try:
            analyse(frame, fps=100.0)
        except ValueError as exc:
            return str(exc)
        return ""

    windows_message = refusal(windows)
    clips_message = refusal(clips)

    check("a dense windows file is refused rather than measured",
          windows_message != "")
    check("the refusal says what the rows actually are",
          "dense windows" in windows_message, windows_message)
    check("the refusal names the setting that turns a decode on",
          "predict.decode" in windows_message and
          "min_bout" in windows_message and "viterbi" in windows_message,
          windows_message)
    check("the refusal names the min_bout_frames the decode needs",
          "predict.min_bout_frames" in windows_message, windows_message)
    check("the refusal names the coverage a decode needs",
          "predict.coverage: dense" in windows_message, windows_message)
    check("a clip manifest is refused by the same rule and named as clips",
          clips_message != "" and "clips" in clips_message, clips_message)

    thin = table(plant([("swim", 20), ("turn", 30)])).drop(columns=["n_covering_mean"])
    check("a bout table missing a required column is refused by name",
          "n_covering_mean" in refusal(thin), refusal(thin))


def test_the_report_says_these_are_predictions_and_points_at_the_confusion() -> None:
    """A transition between two classes the model confuses is not evidence."""
    frame = table(plant([
        ("swim", 20), ("turn", 30), ("swim", 40), ("turn", 50),
    ]))
    model = {GROUP: {
        "classes": ["swim", "turn"],
        # 30 of 100 true turns were called swim, so turn -> swim is the least
        # trustworthy row of the matrix.
        "confusion": [[95, 5], [30, 70]],
        "kappa": 0.61,
        "accuracy": 0.83,
        "reliable": True,
    }}
    report = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12,
                     model_report=model, model_report_path="work/suggest_report.json")
    text = summarise(report)

    check("the report says outright that these are decoded predictions",
          any("DECODED PREDICTIONS, not ground truth" in c
              for c in report["caveats"]), f"got {report['caveats']}")
    check("the caveat points at the confusion matrix, not at a docstring",
          any("suggest_report.json" in c for c in report["caveats"]))
    check("the pointer reaches the printed summary",
          "suggest_report.json" in text)
    check("the model's worst confusion is named for the reader",
          report["groups"][GROUP]["model"]["most_confused"]["from"] == "turn"
          and report["groups"][GROUP]["model"]["most_confused"]["to"] == "swim"
          and abs(report["groups"][GROUP]["model"]["most_confused"]["frac"] - 0.3) < TOL)
    check("the printed summary names it too",
          "mistakes turn for swim" in text, text)
    check("the model's agreement score is carried beside it",
          abs(report["groups"][GROUP]["model"]["kappa"] - 0.61) < TOL)

    silent = analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12)
    check("with no report beside the predictions the absence is stated",
          any("there is no suggest_report.json" in c for c in silent["caveats"]),
          f"got {silent['caveats']}")


def test_the_printed_summary_answers_the_four_questions() -> None:
    """What follows what, how long, how much of the record, and on what evidence."""
    frame = table(
        plant([("swim", 20), ("turn", 30), ("swim", 40), ("turn", 50),
               ("swim", 60)], track=0),
        plant([("turn", 25), ("stop", 35), ("turn", 45)], track=1, start=200),
    )
    text = summarise(analyse(frame, fps=100.0, decode="min_bout", min_bout_frames=12))

    check("the printed form asks what follows what",
          "what follows what" in text, text)
    check("a probability is printed beside its count",
          "turn 1.00 (2)" in text, text)
    check("a row denominator is printed",
          "n=2" in text, text)
    check("the printed form asks how long each state lasts",
          "how long each state lasts" in text, text)
    check("the dwell table names the complete, censored and dropped counts",
          "compl" in text and "cens" in text and "drop" in text)
    check("durations are printed in frames and seconds",
          " f  " in text and " s " in text, text)
    check("the evidence behind the counts is printed",
          "pair(s) counted of" in text and "track-final bout(s)" in text, text)
    check("the fitted chain is printed with the unit its lifetimes are in",
          "the Markov chain fitted to those bouts   lifetimes in seconds" in text,
          text)
    check("the printed form asks how much of the record each state holds",
          "how much of the record each state holds" in text, text)
    check("the steady state is printed beside the share measured off the record",
          "steady state" in text and "measured" in text, text)
    check("the rates are printed with their unit",
          "what follows what, and how fast   rates per second" in text, text)
    check("an error bar says what it was resampled over",
          "bootstrap resamples of the 2 track(s)" in text, text)


def test_the_report_is_json_with_no_nan_in_it() -> None:
    """The artefact has to survive a strict reader, so nothing may be NaN."""
    frame = table(
        plant([("swim", 20), ("turn", 30)]),
        # A track with a single bout: no transitions, nothing complete, every
        # statistic undefined.
        plant([("stop", 40)], track=9, start=500),
    )
    report = analyse(frame, fps=None, decode="none")
    text = json.dumps(report, indent=2, allow_nan=False)

    check("the whole report serialises without a NaN",
          "NaN" not in text and "Infinity" not in text)
    check("an undefined statistic is null rather than absent",
          json.loads(text)["groups"][GROUP]["dwell"]["per_class"]["stop"]
          ["frames"]["mean"] is None)
    check("a class nothing followed still appears with a zero denominator",
          report["groups"][GROUP]["transitions"]["row_totals"]["stop"] == 0)
    check("with no frame rate at all the seconds are simply absent",
          report["fps"]["per_movie"] == {}
          and dwell(report, "swim")["seconds"]["n"] == 0)
    check("a summary can still be printed from it",
          "transitions and dwell times" in summarise(report))


# --------------------------------------------------------------------------- #
# against a real project
# --------------------------------------------------------------------------- #
def test_a_project_is_measured_from_its_own_files(root) -> None:
    """``compute`` reads the config, the frame rate and the model report itself."""
    from trackanno.config import load_config
    from trackanno.project import create_project, write_config

    project = root / "measured"
    config_path = create_project(project)
    write_config(config_path, {
        "predict.coverage": "dense",
        "predict.decode": "min_bout",
        "predict.min_bout_frames": 12,
    })
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

    check("the frame rate is read from the movie's own cache.json",
          movie_fps(cfg) == {"movie_a": 50.0}, f"got {movie_fps(cfg)}")

    report = compute(cfg)
    check("the decode settings are read from the config, not guessed",
          report["source"]["decode"] == "min_bout"
          and report["source"]["min_bout_frames"] == 12)
    check("the floor comes through to the dwell section",
          report["groups"][GROUP]["dwell"]["floor_frames"] == 12)
    check("the transitions are the planted ones",
          pairs(report) == {("swimming", "immotile"): 2,
                            ("immotile", "swimming"): 2}, f"got {pairs(report)}")
    check("seconds use the 50 fps the cache.json declares",
          abs(dwell(report, "immotile")["seconds"]["mean"] - 0.8) < TOL,
          f"got {dwell(report, 'immotile')['seconds']['mean']}")
    check("the model report beside the predictions is found and pointed at",
          report["source"]["suggest_report"].endswith("suggest_report.json"))

    path = write_report(cfg, report)
    check("the artefact lands beside the other work-dir files",
          path == work / TRANSITIONS_FILE and path.exists())
    check("what was written reads back as the report that was computed",
          json.loads(path.read_text(encoding="utf-8"))["groups"][GROUP]
          ["transitions"]["counts"] == report["groups"][GROUP]["transitions"]["counts"])
    check("nothing staged is left behind",
          not list(work.glob("*.part")))
