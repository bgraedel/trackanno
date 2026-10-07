"""Smoke tests: the notebook's figures draw from a real transitions report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from trackanno.infer import figures as f
from trackanno.infer.transitions import analyse

PALETTE = {"swim": "#0072B2", "turn": "#E69F00"}
CLASSES = ["swim", "turn"]


def _report() -> dict:
    rows, frame = [], 0
    for i in range(12):
        label, n = ("swim", 30) if i % 2 == 0 else ("turn", 10)
        rows.append({
            "bout_id": f"m:0:{frame}-{frame + n}", "movie_id": "m",
            "track_id": 0, "f_start": frame, "f_end": frame + n,
            "group": "behaviour", "behaviour_pred": label,
            "human_overlap_frac": 0.0, "n_covering_mean": 3.0,
        })
        frame += n
    return analyse(pd.DataFrame(rows), fps=100.0)


REPORT = _report()
G = REPORT["groups"]["behaviour"]
M = G["markov"]


@pytest.mark.parametrize("draw", [
    pytest.param(lambda: f.transition_graph(M, PALETTE, group="g"), id="transition-graph"),
    pytest.param(lambda: f.jump_matrix(G, group="g"), id="jump-matrix"),
    pytest.param(lambda: f.survival(M, PALETTE, group="g"), id="survival"),
    pytest.param(lambda: f.lifetimes_occupancy(M, PALETTE, group="g"), id="lifetimes"),
    pytest.param(lambda: f.usage(pd.Series(["swim"] * 6 + ["turn"] * 6), G, PALETTE,
                                 group="g"), id="usage"),
])
def test_report_figures_draw(draw) -> None:
    assert draw() is not None


BOUTS = pd.DataFrame([
    {"label": "swim", "track_id": 0, "f_start": 0, "f_end": 30, "n_frames": 30},
    {"label": "", "track_id": 0, "f_start": 30, "f_end": 40, "n_frames": 10},
    {"label": "turn", "track_id": 1, "f_start": 0, "f_end": 25, "n_frames": 25},
])


def test_the_ethogram_draws() -> None:
    assert f.ethogram(BOUTS, CLASSES, PALETTE, fps=100.0, movie_id="m") is not None


def test_trajectories_draw() -> None:
    tracks = pd.DataFrame([
        {"frame": t, "track_id": 0, "cx": 5.0 * t, "cy": 1.0 * t}
        for t in range(30)
    ])
    assert f.trajectories(tracks, BOUTS, CLASSES, PALETTE, movie_id="m",
                          min_track_frames=5) is not None


FEATS = pd.DataFrame({
    "_behaviour": ["swim"] * 8 + ["turn"] * 8,
    "net_speed": np.r_[np.full(8, 40.0), np.full(8, 5.0)],
    "wave_freq_hz": np.r_[np.full(8, 25.0), np.full(8, 5.0)],
    "wave_amp_mean": np.ones(16),
})


def test_violins_draw_with_a_missing_column_skipped() -> None:
    fig = f.feature_violins(FEATS, CLASSES, PALETTE,
                            [("net_speed", "net speed"),
                             ("absent_column", "never drawn"),
                             ("wave_freq_hz", "beat frequency")])
    assert fig is not None


def test_the_beat_map_draws_and_refuses_without_waveform_columns() -> None:
    assert f.beat_map(FEATS, CLASSES, PALETTE) is not None
    assert f.beat_map(FEATS[["_behaviour", "net_speed"]], CLASSES, PALETTE) is None


def test_module_plotting_keeps_the_callers_backend() -> None:
    """Drawing must not switch an already-chosen pyplot backend."""
    import matplotlib.pyplot as plt

    before = matplotlib.get_backend()
    try:
        plt.switch_backend("svg")
        f.transition_graph(M, PALETTE, group="g")
        assert matplotlib.get_backend().lower() == "svg", matplotlib.get_backend()
    finally:
        plt.switch_backend(before)


def test_phase_cycles_draws_gated_and_kept_classes() -> None:
    n_phase, K = 12, 4
    mean = np.zeros((n_phase, K, 2))
    mean[:, :, 0] = np.linspace(0, -30, K)
    mean[:, 2:, 1] = 3.0 * np.sin(
        2 * np.pi * np.arange(n_phase) / n_phase)[:, None]
    results = {"swim": (mean, 5, 200, 2), "turn": (None, 1, 0, 0)}
    fig = f.phase_cycles(results, CLASSES, [1, 2, 3], n_phase=n_phase)
    assert fig is not None
    assert len(fig.axes) == 2, "a kept and a gated class share one figure"
