"""The motif machinery behind the downstream notebook.

The rigid onset alignment, the instantaneous body frame, the fold, the
red-noise oscillation gate, bout-end masking, window labelling, and the kpms
canvas as a smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness import check

from trackanno.infer import motifs as m


def test_align_at_onset_pins_the_onset_pose() -> None:
    """The onset pose lands with its axis on +x and its midpoint at the
    origin; motion after onset is kept rather than normalised away."""
    T, K = 4, 3
    win = np.zeros((T, K, 2))
    for t in range(T):
        win[t, 1] = (0.0, 2.0 * t)          # base, translating up
        win[t, 0] = (0.0, 10.0 + 2.0 * t)   # head 10 px along the heading
        win[t, 2] = (-3.0, 5.0 + 2.0 * t)
    out = m.align_at_onset(win, 0)
    check("head at onset sits on +x", np.allclose(out[0, 0], (5.0, 0.0)),
          str(out[0, 0]))
    check("base at onset mirrors it", np.allclose(out[0, 1], (-5.0, 0.0)))
    check("travel along the heading stays travel",
          np.allclose(out[3, 0], (11.0, 0.0)), str(out[3, 0]))


def test_align_refuses_a_degenerate_axis() -> None:
    win = np.zeros((2, 2, 2))
    win[:, 0] = (1.0, 0.0)
    check("an axis shorter than its own jitter is refused",
          m.align_at_onset(win, 0) is None)


def test_body_frame_removes_locomotion() -> None:
    T, K = 3, 3
    win = np.zeros((T, K, 2))
    for t in range(T):
        win[t, 1] = (7.0 * t, 1.0 * t)          # base travelling
        win[t, 0] = (7.0 * t + 8.0, 1.0 * t)    # head along +x from it
        win[t, 2] = (7.0 * t - 4.0, 1.0 * t + 2.0)
    out = m.body_frame(win)
    check("base is the origin every frame", np.allclose(out[:, 1], 0.0))
    check("head sits on +x at its own distance", np.allclose(out[:, 0], (8.0, 0.0)))
    check("posture survives while travel is gone",
          np.allclose(out[:, 2], (-4.0, 2.0)), str(out[:, 2]))


def test_fold_reflects_only_the_negative_side() -> None:
    inst = np.zeros((2, 5, 2, 2))
    inst[0, :, :, 1] = 3.0
    inst[1, :, :, 1] = -3.0
    folded, note = m.fold_instances(inst, 0)
    check("both sides end positive", (folded[:, :, :, 1] > 0).all())
    check("the split is reported", note == ", folded 1:1", note)
    same, note = m.fold_instances(inst, 0, enabled=False)
    check("disabled leaves everything alone",
          (same == inst).all() and note == "")


def test_beat_gate_rejects_red_noise_and_keeps_a_beat() -> None:
    """Tracking jitter is autocorrelated, so a shuffle null passes it; the
    AR(1) null must not. A real beat must still come back with a phase."""
    rng = np.random.default_rng(0)
    n = 250
    rejected = 0
    for _ in range(10):
        e = rng.normal(0, 1.0, n)
        red = np.empty(n)
        red[0] = e[0]
        for t in range(1, n):
            red[t] = 0.9 * red[t - 1] + e[t]
        red *= 3.0 / red.std()
        if m.beat_phase(red, rng) is None:
            rejected += 1
    check("red noise is rejected", rejected >= 9, f"{rejected}/10")

    t = np.arange(n)
    kept = 0
    for _ in range(10):
        beat = (6 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6))
                + rng.normal(0, 1.5, n))
        if m.beat_phase(beat, rng) is not None:
            kept += 1
    check("a clear beat is kept", kept == 10, f"{kept}/10")

    quiet = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    check("a sub-pixel beat sits below the amplitude floor",
          m.beat_phase(quiet, rng) is None)


def test_the_successor_never_leaks_into_the_motif_tail() -> None:
    """A short bout's window runs past its end; those frames must be NaN, or
    the next behaviour is averaged into this motif."""
    K = 2
    frames = pd.DataFrame([
        {"frame": f, "track_id": 0,
         "kp_x_0": 10.0 + f, "kp_y_0": 0.0, "kp_x_1": 1.0 * f, "kp_y_1": 0.0}
        for f in range(50)
    ])
    rows = pd.DataFrame([{"movie_id": "m", "track_id": 0, "f_start": 0,
                          "n_frames": 5}])
    inst, meta = m.class_instances(
        rows, lambda _id: frames, lambda _id: 10.0, n_keypoints=K,
        window_mode="fixed", pre_s=0.0, post_s=2.0, num_samples=20)
    check("an instance came back", inst is not None and len(inst) == 1)
    check("frames inside the bout are kept", np.isfinite(inst[0, :5]).all())
    check("frames past the bout's end are masked", np.isnan(inst[0, 5:]).all())
    check("the onset sits at the grid's start", meta["onset_frac"] == 0.0)


def test_phase_cycle_draws_the_beat_and_refuses_the_jitter() -> None:
    """The class-level wrap: a beating class yields a cycle, a frozen class
    with red jitter reports nothing oscillating."""
    K = 4
    rng = np.random.default_rng(2)
    bout_rows, frame_rows = [], []
    for cls_i, cls in enumerate(("beat", "frozen")):
        for b in range(4):
            tid = cls_i * 10 + b
            ph = rng.uniform(0, 2 * np.pi)
            e = rng.normal(0, 1.0, 200)
            red = np.empty(200)
            red[0] = e[0]
            for t in range(1, 200):
                red[t] = 0.9 * red[t - 1] + e[t]
            red *= 2.0 / red.std()
            for f in range(200):
                row = {"frame": f, "track_id": tid}
                for k in range(K):
                    lat = 0.0
                    if k >= 2:
                        lat = (5 * np.sin(2 * np.pi * 0.25 * f + ph)
                               if cls == "beat" else red[f])
                    row[f"kp_x_{k}"] = 30.0 - 9.0 * k
                    row[f"kp_y_{k}"] = 100.0 + lat
                frame_rows.append(row)
            bout_rows.append({"movie_id": "m", "track_id": tid, "f_start": 0,
                              "n_frames": 200, "label": cls})
    tracks = pd.DataFrame(frame_rows)
    bouts = pd.DataFrame(bout_rows)

    def cycle(cls):
        return m.phase_cycle(
            bouts[bouts["label"] == cls], lambda _id: tracks,
            n_keypoints=K, ref_node=3, flag_chain=[1, 2, 3])

    mean, n_bouts, _, _ = cycle("beat")
    check("the beating class yields a cycle", mean is not None and n_bouts == 4,
          f"{n_bouts} cleared")
    mean, n_bouts, _, _ = cycle("frozen")
    check("red jitter yields none", mean is None, f"{n_bouts} cleared")


def test_windows_are_labelled_by_the_bout_at_their_centre() -> None:
    bouts = pd.DataFrame([
        {"movie_id": "m", "track_id": 0, "f_start": 0, "f_end": 10, "label": "a"},
        {"movie_id": "m", "track_id": 0, "f_start": 10, "f_end": 20, "label": "b"},
    ])
    feats = pd.DataFrame([
        {"movie_id": "m", "track_id": 0, "f_start": 0, "f_end": 8},
        {"movie_id": "m", "track_id": 0, "f_start": 8, "f_end": 16},
        {"movie_id": "m", "track_id": 0, "f_start": 18, "f_end": 30},
    ])
    out = m.label_windows(feats, bouts)
    check("each covered centre finds its bout",
          out["_behaviour"].tolist() == ["a", "b"],
          str(out["_behaviour"].tolist()))


def test_the_canvas_draws_without_an_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    colors = np.tile([[1.0, 0.5, 0.0, 1.0]], (3, 1))
    Xs = [np.cumsum(np.ones((5, 3, 2)), axis=0), np.zeros((5, 3, 2))]
    fig = m.plot_trajectories(["a", "b"], Xs, colors=colors)
    check("a figure comes back", fig is not None)
    check("its canvas hides the axes", not fig.axes[0].axison)
