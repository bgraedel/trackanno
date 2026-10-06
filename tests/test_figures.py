"""The downstream notebook's figures, drawn from a real transitions report.

Smoke tests: each figure must come back from the same structures
``transitions.analyse`` emits, not from a hand-shaped dict that drifts from
the real schema.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from harness import check

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


def test_report_figures_draw() -> None:
    check("transition graph", f.transition_graph(M, PALETTE, group="g") is not None)
    check("jump matrix", f.jump_matrix(G, group="g") is not None)
    check("survival", f.survival(M, PALETTE, group="g") is not None)
    check("lifetimes", f.lifetimes_occupancy(M, PALETTE, group="g") is not None)
    labels = pd.Series(["swim"] * 6 + ["turn"] * 6)
    check("usage", f.usage(labels, G, PALETTE, group="g") is not None)


def test_bout_figures_draw() -> None:
    rows = pd.DataFrame([
        {"label": "swim", "track_id": 0, "f_start": 0, "f_end": 30, "n_frames": 30},
        {"label": "", "track_id": 0, "f_start": 30, "f_end": 40, "n_frames": 10},
        {"label": "turn", "track_id": 1, "f_start": 0, "f_end": 25, "n_frames": 25},
    ])
    check("ethogram", f.ethogram(rows, CLASSES, PALETTE, fps=100.0,
                                 movie_id="m") is not None)
    tracks = pd.DataFrame([
        {"frame": t, "track_id": 0, "cx": 5.0 * t, "cy": 1.0 * t}
        for t in range(30)
    ])
    check("trajectories", f.trajectories(tracks, rows, CLASSES, PALETTE,
                                         movie_id="m",
                                         min_track_frames=5) is not None)


def test_feature_figures_draw_and_skip_missing_columns() -> None:
    feats = pd.DataFrame({
        "_behaviour": ["swim"] * 8 + ["turn"] * 8,
        "net_speed": np.r_[np.full(8, 40.0), np.full(8, 5.0)],
        "wave_freq_hz": np.r_[np.full(8, 25.0), np.full(8, 5.0)],
        "wave_amp_mean": np.ones(16),
    })
    fig = f.feature_violins(feats, CLASSES, PALETTE,
                            [("net_speed", "net speed"),
                             ("absent_column", "never drawn"),
                             ("wave_freq_hz", "beat frequency")])
    check("violins draw with a missing column skipped", fig is not None)
    check("beat map", f.beat_map(feats, CLASSES, PALETTE) is not None)
    check("beat map refuses without waveform columns",
          f.beat_map(feats[["_behaviour", "net_speed"]], CLASSES, PALETTE)
          is None)


def test_module_plotting_keeps_the_callers_backend() -> None:
    """The first figlib call in a notebook used to switch the whole session
    to Agg (_pyplot forced it), so every later plt.show() warned
    'FigureCanvasAgg is non-interactive' and nothing displayed. With pyplot
    already imported, drawing must leave the chosen backend alone."""
    import matplotlib
    import matplotlib.pyplot as plt

    before = matplotlib.get_backend()
    try:
        plt.switch_backend("svg")
        f.transition_graph(M, PALETTE, group="g")
        check("the caller's backend survives drawing",
              matplotlib.get_backend().lower() == "svg",
              matplotlib.get_backend())
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
    check("a kept and a gated class share one figure", fig is not None)
    check("both panels exist", len(fig.axes) == 2)
