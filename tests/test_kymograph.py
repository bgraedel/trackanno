"""Curvature kymographs: the maths, the box-table plumbing and the widget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def close(a: float, b: float, tol: float) -> bool:
    return bool(np.isfinite(a) and abs(a - b) <= tol)


def _wave(n_frames, n_nodes, period, cycles, travelling=True, length=100.0):
    """A flagellum chain (node 0 head, 1.. base to tip) with amplitude a fixed fraction of length."""
    s = np.linspace(0, 1, n_nodes)
    amplitude = 0.10 * length
    kx = np.zeros((n_frames, n_nodes + 1))
    ky = np.zeros((n_frames, n_nodes + 1))
    for t in range(n_frames):
        phase = 2 * np.pi * (t / period)
        kx[t, 1:] = s * length
        along = 2 * np.pi * cycles * s
        ky[t, 1:] = amplitude * (np.sin(along - phase) if travelling
                                 else np.sin(along) * np.cos(phase))
        kx[t, 0], ky[t, 0] = -0.15 * length, 0.0
    return kx, ky


# --------------------------------------------------------------------------- #
# the maths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [20.0, 50.0, 120.0])
def test_a_circle_has_curvature_one_over_r(radius) -> None:
    from trackanno.measure.kymograph import resample_centreline

    theta = np.linspace(0, np.pi / 2, 9)
    _, kappa, length = resample_centreline(
        radius * np.cos(theta), radius * np.sin(theta), 128, smooth_px=0.0)
    assert close(float(np.nanmedian(kappa)), 1.0 / radius, 1e-4), np.nanmedian(kappa)
    assert close(length, radius * np.pi / 2, 1e-3), f"arclength is R*pi/2: {length:.4f}"
    assert float(np.nanstd(kappa)) < 1e-3, "curvature is constant along the arc"


def test_a_straight_chain_has_zero_curvature() -> None:
    from trackanno.measure.kymograph import resample_centreline

    _, kappa, length = resample_centreline(
        np.linspace(0, 100, 8), np.zeros(8), 64, smooth_px=0.0)
    assert float(np.nanmax(np.abs(kappa))) < 1e-9
    assert close(length, 100.0, 1e-6), f"its length is the chord: {length}"


def test_a_sine_crest_has_the_closed_form_curvature() -> None:
    from trackanno.measure.kymograph import resample_centreline

    amplitude, wavelength = 8.0, 60.0
    x = np.linspace(0, wavelength, 25)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    points, kappa, _ = resample_centreline(x, y, 400, smooth_px=0.0)
    crest = int(np.argmax(points[:, 1]))
    expected = -amplitude * (2 * np.pi / wavelength) ** 2
    assert close(float(kappa[crest]), expected, abs(expected) * 0.05), (
        f"-A(2pi/lam)^2: {kappa[crest]:.5f} vs {expected:.5f}")


def test_curvature_is_invariant_to_where_the_cell_is() -> None:
    """Curvature is intrinsic, so rotation and translation change nothing."""
    from trackanno.measure.kymograph import resample_centreline

    theta = np.linspace(0, np.pi / 2, 9)
    x, y = 40.0 * np.cos(theta), 40.0 * np.sin(theta)
    _, reference, _ = resample_centreline(x, y, 64, smooth_px=0.0)

    for angle, dx, dy in ((0.7, 300, -120), (2.1, -40, 900), (-1.3, 0, 0)):
        cos, sin = np.cos(angle), np.sin(angle)
        _, moved, _ = resample_centreline(
            x * cos - y * sin + dx, x * sin + y * cos + dy, 64, smooth_px=0.0)
        assert float(np.nanmax(np.abs(moved - reference))) < 1e-9, f"angle {angle:+.1f}"

    _, mirrored, _ = resample_centreline(x, -y, 64, smooth_px=0.0)
    assert float(np.nanmax(np.abs(mirrored + reference))) < 1e-9, (
        "a mirrored chain flips the sign")


def test_the_chain_comes_from_the_skeleton_not_a_guess() -> None:
    from trackanno.measure.kymograph import chain_from_edges

    # Head, Base, Flag1..Flag5, Tip, with the head not wired into the chain.
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert chain_from_edges(8, edges) == [1, 2, 3, 4, 5, 6, 7]
    flipped = [(b, a) for a, b in reversed(edges)]
    assert chain_from_edges(8, flipped) == chain_from_edges(8, edges), (
        "writing the edges backwards changes nothing")
    assert chain_from_edges(8, edges + [(0, 1)]) == [0, 1, 2, 3, 4, 5, 6, 7], (
        "a head attached to the end extends the chain")


def test_an_unusable_skeleton_gives_no_chain() -> None:
    from trackanno.measure.kymograph import chain_from_edges

    assert chain_from_edges(8, []) == [], "no edges means no derived chain"
    assert chain_from_edges(8, [(1, 2), (2, 3)]) == [], "a too-short chain is refused"
    assert chain_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 99), (2, 2)]) == [0, 1, 2, 3], (
        "out-of-range and self edges are ignored")


def test_the_chain_order_sets_the_sign() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    kx, ky = _wave(20, 9, period=12.0, cycles=1.5)
    forward = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                  order=list(range(1, 10)))
    reverse = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                  order=list(range(9, 0, -1)))
    assert float(np.nanmax(np.abs(forward.kappa + reverse.kappa[:, ::-1]))) < 1e-6, (
        "reversing the chain negates the curvature")
    assert np.allclose(forward.length_px, reverse.length_px, atol=1e-6), (
        "reversing leaves the length alone")


def test_an_explicit_order_overrides_the_fallback() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    kx, ky = _wave(20, 9, period=12.0, cycles=1.5)
    subset = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                 order=[1, 3, 5, 7, 9])
    assert subset.usable and subset.n_nodes == 5, "a subset of the chain is usable"
    assert curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0).n_nodes == 9, (
        "no order falls back to node 1 onwards")


def _bent(offset, n_frames=60):
    """A beater whose wave rides on a static bend of the given sign."""
    from trackanno.measure.kymograph import curvature_kymograph

    s = np.linspace(0, 1, 11)
    kx = np.zeros((n_frames, 12))
    ky = np.zeros((n_frames, 12))
    for t in range(n_frames):
        phase = 2 * np.pi * t / 14.0
        kx[t, 1:] = s * 100.0
        ky[t, 1:] = 8.0 * np.sin(2 * np.pi * 1.5 * s - phase) + offset * s * s * 60.0
        kx[t, 0], ky[t, 0] = -15.0, 0.0
    return curvature_kymograph(kx, ky, n_samples=64, smooth_px=0.0)


def test_a_bent_cell_and_its_mirror_are_brought_into_agreement() -> None:
    from trackanno.measure.kymograph import canonical_sign

    right, left = _bent(+1.0), _bent(-1.0)
    va, vb = canonical_sign(right.kappa), canonical_sign(left.kappa)
    assert va.confident and vb.confident, f"{va.reason} / {vb.reason}"
    assert va.sign == -vb.sign, "a mirrored cell gets the opposite sign"

    canon_a, _ = right.with_sign("canonical")
    canon_b, _ = left.with_sign("canonical")
    assert float(np.nanmean(canon_a.kappa)) * float(np.nanmean(canon_b.kappa)) > 0

    # One factor for the whole clip; per frame would destroy the travelling wave.
    flipped, verdict = left.with_sign("canonical")
    if verdict.flipped:
        ratio = flipped.kappa / np.where(left.kappa == 0, np.nan, left.kappa)
        assert float(np.nanstd(ratio)) < 1e-6, "the flip is a single global factor"

    raw, v = right.with_sign("raw")
    assert raw is right and v.confident, "raw mode changes nothing"


def test_a_symmetric_beater_is_declined_not_guessed() -> None:
    from trackanno.measure.kymograph import canonical_sign

    symmetric = _bent(0.0)
    verdict = canonical_sign(symmetric.kappa)
    assert not verdict.confident, f"ratio {verdict.ratio:.3f}: {verdict.reason}"
    assert "noise" in verdict.reason or "disagree" in verdict.reason, verdict.reason
    unchanged, _ = symmetric.with_sign("canonical")
    assert np.array_equal(np.nan_to_num(unchanged.kappa), np.nan_to_num(symmetric.kappa)), (
        "the data is left exactly as measured")


def _forward_fraction(kymo) -> float:
    """Share of 2D spectral power in the forward diagonal quadrant."""
    field = kymo.kappa - np.nanmean(kymo.kappa)
    field = (field * np.hanning(field.shape[0])[:, None]
             * np.hanning(field.shape[1])[None, :])
    power = np.abs(np.fft.fft2(field)) ** 2
    n_t, n_s = power.shape
    forward = power[1:n_t // 2, 1:n_s // 2].sum()
    backward = power[1:n_t // 2, n_s // 2 + 1:].sum()
    return float(forward / (forward + backward))


def test_a_travelling_wave_reads_as_travelling() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    def kymo(period, travelling):
        return curvature_kymograph(*_wave(60, 9, period=period, cycles=2, travelling=travelling),
                                   n_samples=64, smooth_px=0.0)

    travelling, backwards, standing = kymo(15.0, True), kymo(-15.0, True), kymo(15.0, False)
    assert travelling.usable and standing.usable

    one_way = _forward_fraction(travelling)
    other_way = _forward_fraction(backwards)
    assert one_way < 0.2 or one_way > 0.8, f"power goes one direction: {one_way:.3f}"
    assert abs(other_way - (1.0 - one_way)) < 0.1, "reversing it reverses the direction"
    assert abs(_forward_fraction(standing) - 0.5) < 0.05, (
        "a standing wave splits evenly between the two directions")


def test_the_beat_period_comes_off_the_time_axis() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    travelling = curvature_kymograph(*_wave(60, 9, period=15.0, cycles=2),
                                     n_samples=64, smooth_px=0.0)
    field = travelling.kappa - travelling.kappa.mean(axis=0)
    spectrum = (np.abs(np.fft.rfft(field * np.hanning(len(field))[:, None],
                                   axis=0)) ** 2).sum(axis=1)
    peak = int(np.argmax(spectrum[1:])) + 1
    frequency = peak / travelling.n_frames
    assert close(1.0 / frequency, 15.0, 1.5), f"{1.0 / frequency:.2f} frames"


def test_untrackable_frames_stay_empty() -> None:
    """A gap is NaN, never zero, which would draw a straight flagellum."""
    from trackanno.measure.kymograph import curvature_kymograph

    kx, ky = _wave(40, 9, period=12.0, cycles=1.5)
    kx[10:15] = np.nan
    ky[10:15] = np.nan
    kymo = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0)
    assert np.isnan(kymo.kappa[10:15]).all()
    assert np.isnan(kymo.length_px[10:15]).all()
    assert np.isfinite(kymo.kappa[9]).all() and np.isfinite(kymo.kappa[15]).all(), (
        "the frames around the gap are unaffected")
    assert kymo.usable, "the clip is still usable overall"


def test_a_degenerate_chain_is_refused() -> None:
    from trackanno.measure.kymograph import resample_centreline

    points, kappa, length = resample_centreline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 16)
    assert points is None and kappa is None and np.isnan(length), "too few points"
    points, _, _ = resample_centreline([5.0] * 6, [5.0] * 6, 16)
    assert points is None, "a chain collapsed onto one point"


def test_an_untracked_clip_is_unusable_but_shaped() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    blank = curvature_kymograph(np.full((5, 9), np.nan), np.full((5, 9), np.nan))
    assert not blank.usable
    assert blank.kappa.shape == (5, 128)


def test_kappa_l_is_scale_free() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    kymo = curvature_kymograph(*_wave(50, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0)
    big = curvature_kymograph(*_wave(50, 9, 12.0, 1.5, length=200.0),
                              n_samples=64, smooth_px=0.0)
    assert 0 < big.scale() < kymo.scale(), "a bigger cell has smaller curvature"
    a = float(np.nanpercentile(np.abs(kymo.normalised()), 98))
    b = float(np.nanpercentile(np.abs(big.normalised()), 98))
    assert abs(a - b) / max(a, b) < 0.02, f"{a:.3f} vs {b:.3f}"


def test_one_wild_frame_does_not_set_the_colour_limit() -> None:
    from trackanno.measure.kymograph import curvature_kymograph

    plain = curvature_kymograph(*_wave(50, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0).scale()
    spiked = curvature_kymograph(*_wave(50, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0)
    spiked.kappa[7, 3] = 1e4
    assert spiked.scale() < plain * 1.5, f"{spiked.scale():.4f} vs {plain:.4f}"


# --------------------------------------------------------------------------- #
# the box table the GUI reads
# --------------------------------------------------------------------------- #
N_KPTS, N_FRAMES = 9, 30


@pytest.fixture(scope="module")
def box_rows() -> pd.DataFrame:
    """Target track 7 with frames 12-15 missing and keypoint 5 at low confidence, plus an empty neighbour."""
    kx, ky = _wave(N_FRAMES, N_KPTS - 1, period=10.0, cycles=1.5)
    rows = []
    for t in range(N_FRAMES):
        if 12 <= t < 16:
            continue
        entry = {"row": 0, "t": t, "track_id": 7, "is_target": True, "conf": 0.9}
        for k in range(N_KPTS):
            entry[f"kp_x_{k}"] = kx[t, k]
            entry[f"kp_y_{k}"] = ky[t, k]
            entry[f"kp_c_{k}"] = 0.4 if k == 5 else 0.95
        rows.append(entry)
        neighbour = dict(entry, track_id=8, is_target=False)
        for k in range(N_KPTS):
            neighbour[f"kp_x_{k}"] = np.nan
            neighbour[f"kp_y_{k}"] = np.nan
        rows.append(neighbour)
    return pd.DataFrame(rows)


def test_it_reads_a_clip_box_table(box_rows) -> None:
    from trackanno.measure.kymograph import kymograph_from_boxes

    kymo = kymograph_from_boxes(box_rows, N_KPTS, N_FRAMES, n_samples=48)
    assert kymo.kappa.shape == (N_FRAMES, 48), "one row per clip frame"
    assert kymo.n_nodes == N_KPTS - 1, "the head is dropped"
    assert np.isfinite(kymo.kappa[0]).all(), "only the target track is used"
    assert np.isnan(kymo.kappa[12:16]).all(), "absent frames are placed by frame, not row"
    assert np.isfinite(kymo.kappa[16]).all(), "frames after the gap are filled"


def test_a_confidence_gate_changes_the_fit(box_rows) -> None:
    from trackanno.measure.kymograph import kymograph_from_boxes

    kymo = kymograph_from_boxes(box_rows, N_KPTS, N_FRAMES, n_samples=48)
    dropped = kymograph_from_boxes(box_rows, N_KPTS, N_FRAMES, min_conf=0.5, n_samples=48)
    assert dropped.usable, "a min_conf gate still leaves a fittable chain"
    assert not np.allclose(np.nan_to_num(dropped.kappa), np.nan_to_num(kymo.kappa),
                           atol=1e-9), "the gate really applies"


def test_an_empty_table_or_no_keypoints_is_unusable(box_rows) -> None:
    from trackanno.measure.kymograph import kymograph_from_boxes

    assert not kymograph_from_boxes(pd.DataFrame(), N_KPTS, N_FRAMES).usable
    assert not kymograph_from_boxes(box_rows, 0, N_FRAMES).usable, "a detect project"


# --------------------------------------------------------------------------- #
# the widget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_ramp_diverges_through_a_neutral_midpoint(app, theme) -> None:
    from trackanno.gui.kymograph import diverging_ramp
    from trackanno.gui.theme import apply_theme

    apply_theme(app, theme)
    ramp = diverging_ramp(64, theme)
    assert ramp.shape == (64, 3)
    first, mid, last = ramp[0], ramp[len(ramp) // 2], ramp[-1]
    assert int(first[2]) > int(first[0]) + 40, f"it starts blue: {first}"
    assert int(last[0]) > int(last[2]) + 40, f"it ends red: {last}"
    assert max(mid) - min(mid) < 30, f"the midpoint is neutral: {mid}"
    assert abs(int(mid[0]) - int(mid[2])) < 30, "no hue at zero"
    apply_theme(app, "dark")


def test_the_widget_draws_and_scrubs(app) -> None:
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from trackanno.gui.kymograph import KymographView
    from trackanno.gui.theme import apply_theme, current
    from trackanno.measure.kymograph import curvature_kymograph

    view = KymographView()
    view.resize(600, 220)
    assert view.grab() is not None, "an empty view paints"

    view.set_kymograph(curvature_kymograph(*_wave(60, 9, 12.0, 1.5),
                                           n_samples=64, smooth_px=0.0))
    view.show()
    app.processEvents()
    assert not view.grab().isNull()

    picked: list[int] = []
    view.frame_picked.connect(picked.append)
    plot = view._plot_rect()
    at = QPointF(plot.left() + plot.width() * 0.75, plot.center().y())
    view.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))
    assert len(picked) == 1, picked
    assert 42 <= picked[0] <= 48, f"scrubs to roughly where it was clicked: {picked}"
    assert view.frame == picked[0], "the cursor moved with it"

    before = view._image
    apply_theme(app, "dark" if current() == "light" else "light")
    view.grab()
    assert view._image is not before, "a theme change repaints the field"
    view.close()
    apply_theme(app, "dark")


def _panel(root, pose: bool):
    from fixtures import make_project

    from trackanno.annotate.clips import ClipStore
    from trackanno.annotate.store import AnnotationStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget

    cfg = load_config(make_project(root, pose=pose))
    store = ClipStore(cfg.paths.work_dir)
    return cfg, AnnotationWidget(cfg, store, AnnotationStore(cfg.paths.csv, cfg))


def test_the_panel_shows_and_follows_it_on_a_pose_project(app, root: Path) -> None:
    from fixtures import N_KEYPOINTS

    cfg, panel = _panel(root, pose=True)
    assert cfg.n_keypoints == N_KEYPOINTS
    panel.resize(1000, 700)
    panel.show()
    app.processEvents()

    kymo = panel.kymo_view.kymo
    assert panel.kymo_view.isVisible()
    assert kymo is not None and kymo.usable
    assert kymo.n_frames == panel._clip.shape[0], "one column per frame of the clip"

    # Paused so autoplay cannot move the frame between seek and assert.
    panel.viewer.pause()
    app.processEvents()
    panel.viewer.seek(7)
    app.processEvents()
    assert panel.kymo_view.frame == 7, "the cursor follows the clip"
    panel._scrub_to(3)
    app.processEvents()
    assert panel.viewer._frame == 3, "scrubbing the chart moves the clip"

    panel.next_clip()
    app.processEvents()
    assert panel.kymo_view.kymo is not kymo, "changing clip recomputes it"
    panel.shutdown()
    panel.close()


def test_a_detect_project_hides_it(app, root: Path) -> None:
    _, plain = _panel(root, pose=False)
    plain.show()
    app.processEvents()
    assert not plain.kymo_view.isVisible()
    assert plain.kymo_view.kymo is None, "never computes one"
    plain.shutdown()
    plain.close()


def test_a_dense_centerline_keeps_its_bends() -> None:
    """On 129 close points a per-point tolerance deletes a small bend; a distance cutoff keeps it."""
    from trackanno.measure.kymograph import resample_centreline

    s = np.linspace(0.0, 40.0, 129)                   # 40 px, about 0.3 px apart
    x, y = s, 0.4 * np.sin(2 * np.pi * s / 10.0)      # a 0.4 px bend every 5 px
    truth = 0.4 * (2 * np.pi / 10.0) ** 2
    _, per_point, _ = resample_centreline(x, y, 128, smooth_px=0.6)
    _, by_distance, _ = resample_centreline(x, y, 128, cutoff_px=2.0)
    assert np.nanmax(np.abs(per_point[16:-16])) < 0.2 * truth, "the per-point tolerance flattens it"
    assert np.nanmax(np.abs(by_distance[16:-16])) > 0.9 * truth
