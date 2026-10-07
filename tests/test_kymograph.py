"""Curvature kymographs: the maths, the plumbing and the widget.

The maths is checked against shapes whose curvature is known in closed form,
because "the picture looks like a beat" is not a test. Nothing here needs a GPU,
a movie or a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harness import check


def close(a: float, b: float, tol: float) -> bool:
    return bool(np.isfinite(a) and abs(a - b) <= tol)


# --------------------------------------------------------------------------- #
# the maths
# --------------------------------------------------------------------------- #
def test_curvature_matches_closed_form() -> None:
    """A circle of radius R has curvature 1/R everywhere, a straight line has
    zero, and a sine has -A(2pi/lam)^2 at its crest. If any of these drift the
    kymograph is drawing something that is not curvature."""
    from trackanno.measure.kymograph import resample_centreline

    for radius in (20.0, 50.0, 120.0):
        theta = np.linspace(0, np.pi / 2, 9)
        _, kappa, length = resample_centreline(
            radius * np.cos(theta), radius * np.sin(theta), 128, smooth_px=0.0)
        check(f"circle R={radius:.0f} gives kappa = 1/R",
              close(float(np.nanmedian(kappa)), 1.0 / radius, 1e-4),
              f"{np.nanmedian(kappa):.6f} vs {1/radius:.6f}")
        check(f"...and its arclength is R*pi/2 ({radius:.0f})",
              close(length, radius * np.pi / 2, 1e-3), f"{length:.4f}")
        check(f"...and it is constant along the arc ({radius:.0f})",
              float(np.nanstd(kappa)) < 1e-3, f"{np.nanstd(kappa):.2e}")

    _, kappa, length = resample_centreline(
        np.linspace(0, 100, 8), np.zeros(8), 64, smooth_px=0.0)
    check("a straight chain has zero curvature",
          float(np.nanmax(np.abs(kappa))) < 1e-9, f"{np.nanmax(np.abs(kappa)):.2e}")
    check("...and its length is the chord", close(length, 100.0, 1e-6), f"{length}")

    amplitude, wavelength = 8.0, 60.0
    x = np.linspace(0, wavelength, 25)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    points, kappa, _ = resample_centreline(x, y, 400, smooth_px=0.0)
    crest = int(np.argmax(points[:, 1]))
    expected = -amplitude * (2 * np.pi / wavelength) ** 2
    check("a sine's crest curvature is -A(2pi/lam)^2",
          close(float(kappa[crest]), expected, abs(expected) * 0.05),
          f"{kappa[crest]:.5f} vs {expected:.5f}")


def test_curvature_is_invariant_to_where_the_cell_is() -> None:
    """Curvature is intrinsic, which is the whole reason the kymograph can be
    built in raw image space with no stabilisation, co-swimming frame or body
    axis. If this ever fails, the drawing depends on where the animal drifted
    to and the feature is worthless."""
    from trackanno.measure.kymograph import resample_centreline

    theta = np.linspace(0, np.pi / 2, 9)
    x, y = 40.0 * np.cos(theta), 40.0 * np.sin(theta)
    _, reference, _ = resample_centreline(x, y, 64, smooth_px=0.0)

    for angle, dx, dy in ((0.7, 300, -120), (2.1, -40, 900), (-1.3, 0, 0)):
        cos, sin = np.cos(angle), np.sin(angle)
        _, moved, _ = resample_centreline(
            x * cos - y * sin + dx, x * sin + y * cos + dy, 64, smooth_px=0.0)
        check(f"rotating by {angle:+.1f} rad and shifting changes nothing",
              float(np.nanmax(np.abs(moved - reference))) < 1e-9,
              f"{np.nanmax(np.abs(moved - reference)):.2e}")

    # A mirror flips the sign. Stated as a test because the sign is the whole
    # signal in a diverging plot, so it must be a decision rather than an accident.
    _, mirrored, _ = resample_centreline(x, -y, 64, smooth_px=0.0)
    check("a mirrored chain flips the sign",
          float(np.nanmax(np.abs(mirrored + reference))) < 1e-9)


def _wave(n_frames, n_nodes, period, cycles, travelling=True, length=100.0):
    """A flagellum chain: node 0 is the head, 1.. is the base-to-tip flagellum.

    The amplitude is a fixed FRACTION of the length, so raising ``length``
    produces a genuinely scaled copy of the same shape rather than a stretched
    one. That distinction is the whole point of the kappa*L test below.
    """
    s = np.linspace(0, 1, n_nodes)
    amplitude = 0.10 * length
    kx = np.zeros((n_frames, n_nodes + 1))
    ky = np.zeros((n_frames, n_nodes + 1))
    for t in range(n_frames):
        phase = 2 * np.pi * (t / period)
        kx[t, 1:] = s * length
        # A standing wave has its phase fixed in space and only its amplitude
        # oscillating; a travelling one carries the phase along the arclength.
        along = 2 * np.pi * cycles * s
        ky[t, 1:] = amplitude * (np.sin(along - phase) if travelling
                                 else np.sin(along) * np.cos(phase))
        kx[t, 0], ky[t, 0] = -0.15 * length, 0.0
    return kx, ky


def test_the_chain_comes_from_the_skeleton_not_a_guess() -> None:
    """Which points make up the flagellum, and in which direction, decides the
    sign of every value on the chart. It is read from the skeleton the project
    declares rather than assumed to be 'node 1 onwards', because that assumption
    is only true for one particular keypoint layout."""
    from trackanno.measure.kymograph import chain_from_edges

    # The layout this project actually uses: Head, Base, Flag1..Flag5, Tip, with
    # the head deliberately not wired into the flagellum chain.
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    check("the flagellum chain is recovered in order",
          chain_from_edges(8, edges) == [1, 2, 3, 4, 5, 6, 7],
          repr(chain_from_edges(8, edges)))

    # The same skeleton written back to front must give the same answer, or the
    # sign of the whole plot would depend on how someone typed their config.
    flipped = [(b, a) for a, b in reversed(edges)]
    check("writing the edges backwards changes nothing",
          chain_from_edges(8, flipped) == chain_from_edges(8, edges))

    # A head attached to the base is a side branch, not part of the flagellum.
    check("a branch off the chain does not join it",
          chain_from_edges(8, edges + [(0, 1)]) == [0, 1, 2, 3, 4, 5, 6, 7],
          repr(chain_from_edges(8, edges + [(0, 1)])))

    # A skeleton that says nothing usable hands back to the caller rather than
    # inventing an order.
    check("no edges means no derived chain", chain_from_edges(8, []) == [])
    check("a too-short chain is refused", chain_from_edges(8, [(1, 2), (2, 3)]) == [])
    check("out-of-range and self edges are ignored",
          chain_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 99), (2, 2)]) == [0, 1, 2, 3])


def test_the_chain_order_sets_the_sign() -> None:
    """Walking a bend from the other end negates its curvature. That is real
    geometry, so it has to be pinned by the skeleton rather than left to chance."""
    from trackanno.measure.kymograph import curvature_kymograph

    kx, ky = _wave(20, 9, period=12.0, cycles=1.5)
    forward = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                  order=list(range(1, 10)))
    reverse = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                  order=list(range(9, 0, -1)))
    check("reversing the chain negates the curvature",
          float(np.nanmax(np.abs(forward.kappa + reverse.kappa[:, ::-1]))) < 1e-6,
          f"{np.nanmax(np.abs(forward.kappa + reverse.kappa[:, ::-1])):.2e}")
    check("...and leaves the length alone",
          bool(np.allclose(forward.length_px, reverse.length_px, atol=1e-6)))

    # An explicit order overrides the fallback, and picking a subset works.
    subset = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0,
                                 order=[1, 3, 5, 7, 9])
    check("a subset of the chain is usable", subset.usable and subset.n_nodes == 5)
    check("no order falls back to node 1 onwards",
          curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0).n_nodes == 9)


def test_the_mirror_ambiguity_is_decided_or_declined() -> None:
    """A 2D image cannot tell a cell from its mirror image, so the same beat
    comes out blue in one clip and red in the next. On 20 real clips it was 11
    one way and 9 the other. The sign can only be fixed by an ODD functional of
    curvature - every even one (amplitude, power, wave speed, direction) is
    mirror-invariant - and the physical odd one is the static mode."""
    from trackanno.measure.kymograph import canonical_sign, curvature_kymograph

    # A beater with a real static bend: the wave rides on a constant offset.
    def bent(offset, n_frames=60):
        s = np.linspace(0, 1, 11)
        kx = np.zeros((n_frames, 12))
        ky = np.zeros((n_frames, 12))
        for t in range(n_frames):
            phase = 2 * np.pi * t / 14.0
            kx[t, 1:] = s * 100.0
            ky[t, 1:] = 8.0 * np.sin(2 * np.pi * 1.5 * s - phase) + offset * s * s * 60.0
            kx[t, 0], ky[t, 0] = -15.0, 0.0
        return curvature_kymograph(kx, ky, n_samples=64, smooth_px=0.0)

    right = bent(+1.0)
    left = bent(-1.0)
    va, vb = canonical_sign(right.kappa), canonical_sign(left.kappa)
    check("a clearly bent cell gets a confident verdict", va.confident and vb.confident,
          f"{va.reason} / {vb.reason}")
    check("...and a mirrored one gets the opposite sign", va.sign == -vb.sign,
          f"{va.sign} vs {vb.sign}")

    canon_a, _ = right.with_sign("canonical")
    canon_b, _ = left.with_sign("canonical")
    check("canonicalising brings them into agreement",
          float(np.nanmean(canon_a.kappa)) * float(np.nanmean(canon_b.kappa)) > 0)

    # A symmetric beater has no static mode, so there is genuinely nothing to
    # decide. It must SAY so rather than flip on noise.
    symmetric = bent(0.0)
    verdict = canonical_sign(symmetric.kappa)
    check("a symmetric beater is declined, not guessed", not verdict.confident,
          f"ratio {verdict.ratio:.3f}: {verdict.reason}")
    check("...and it says why", "noise" in verdict.reason or "disagree" in verdict.reason,
          verdict.reason)
    unchanged, _ = symmetric.with_sign("canonical")
    check("...and the data is left exactly as measured",
          bool(np.array_equal(np.nan_to_num(unchanged.kappa),
                              np.nan_to_num(symmetric.kappa))))

    # The flip is one factor for the WHOLE clip. Per frame it would destroy the
    # travelling wave, which is the only thing the plot exists to show.
    flipped, verdict = left.with_sign("canonical")
    if verdict.flipped:
        ratio = flipped.kappa / np.where(left.kappa == 0, np.nan, left.kappa)
        check("the flip is a single global factor",
              float(np.nanstd(ratio)) < 1e-6, f"{np.nanstd(ratio):.2e}")

    raw, v = right.with_sign("raw")
    check("raw mode changes nothing", raw is right and v.confident)


def test_a_travelling_wave_reads_as_travelling() -> None:
    """The point of a kymograph: a wave moving along the flagellum makes
    diagonal stripes, a standing one makes vertical stripes. Measured as the
    arclength shift that best lines up consecutive frames."""
    from trackanno.measure.kymograph import curvature_kymograph

    # Read off the 2D spectrum, which is what "travelling" means: a wave moving
    # one way puts its power in one diagonal quadrant of the (time, arclength)
    # transform, a standing wave splits it evenly between both because it is the
    # sum of two waves travelling in opposite directions.
    #
    # Sliding one frame over the next was tried first and is wrong here: the
    # arclength axis is not periodic, so the correlation wraps the tip round
    # onto the base. Tracking one spatial FFT bin's phase is wrong too - the
    # arclength of a flapping curve changes slightly frame to frame, which
    # wobbles the bin.
    def forward_fraction(kymo) -> float:
        field = kymo.kappa - np.nanmean(kymo.kappa)
        field = (field * np.hanning(field.shape[0])[:, None]
                 * np.hanning(field.shape[1])[None, :])
        power = np.abs(np.fft.fft2(field)) ** 2
        n_t, n_s = power.shape
        forward = power[1:n_t // 2, 1:n_s // 2].sum()
        backward = power[1:n_t // 2, n_s // 2 + 1:].sum()
        return float(forward / (forward + backward))

    travelling = curvature_kymograph(
        *_wave(60, 9, period=15.0, cycles=2, travelling=True),
        n_samples=64, smooth_px=0.0)
    backwards = curvature_kymograph(
        *_wave(60, 9, period=-15.0, cycles=2, travelling=True),
        n_samples=64, smooth_px=0.0)
    standing = curvature_kymograph(
        *_wave(60, 9, period=15.0, cycles=2, travelling=False),
        n_samples=64, smooth_px=0.0)
    check("both produce a full field", travelling.usable and standing.usable)

    one_way = forward_fraction(travelling)
    other_way = forward_fraction(backwards)
    check("a travelling wave puts its power in one direction",
          one_way < 0.2 or one_way > 0.8, f"{one_way:.3f}")
    check("reversing it reverses the direction",
          abs(other_way - (1.0 - one_way)) < 0.1, f"{other_way:.3f} vs {one_way:.3f}")
    check("a standing wave splits evenly between the two",
          abs(forward_fraction(standing) - 0.5) < 0.05,
          f"{forward_fraction(standing):.3f}")

    # The beat period should also be readable straight off the time axis.
    field = travelling.kappa - travelling.kappa.mean(axis=0)
    spectrum = (np.abs(np.fft.rfft(field * np.hanning(len(field))[:, None],
                                   axis=0)) ** 2).sum(axis=1)
    peak = int(np.argmax(spectrum[1:])) + 1
    frequency = peak / travelling.n_frames
    check("and the beat period comes back out of the time axis",
          close(1.0 / frequency, 15.0, 1.5), f"{1.0 / frequency:.2f} frames")


def test_untrackable_frames_stay_empty() -> None:
    """A gap must read as a gap. Filling it with zero would draw a straight
    flagellum where there was no measurement, which is the one lie a curvature
    plot must not tell."""
    from trackanno.measure.kymograph import MIN_NODES, curvature_kymograph, resample_centreline

    kx, ky = _wave(40, 9, period=12.0, cycles=1.5)
    kx[10:15] = np.nan
    ky[10:15] = np.nan
    kymo = curvature_kymograph(kx, ky, n_samples=32, smooth_px=0.0)
    check("lost frames are NaN, not zero",
          bool(np.isnan(kymo.kappa[10:15]).all()))
    check("...and their length is NaN too",
          bool(np.isnan(kymo.length_px[10:15]).all()))
    check("the frames around them are unaffected",
          bool(np.isfinite(kymo.kappa[9]).all() and np.isfinite(kymo.kappa[15]).all()))
    check("the clip is still usable overall", kymo.usable)

    # Too few points for a cubic, and a chain of duplicated points, both of
    # which a tracker that loses a keypoint really does produce.
    points, kappa, length = resample_centreline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 16)
    check(f"fewer than {MIN_NODES} points is refused rather than guessed",
          points is None and kappa is None and np.isnan(length))
    points, _, _ = resample_centreline([5.0] * 6, [5.0] * 6, 16)
    check("a chain collapsed onto one point is refused", points is None)

    blank = curvature_kymograph(np.full((5, 9), np.nan), np.full((5, 9), np.nan))
    check("an entirely untracked clip reports itself unusable", not blank.usable)
    check("...and still has the right shape", blank.kappa.shape == (5, 128))


def test_scale_and_normalisation() -> None:
    """kappa*L makes two cells at different magnifications comparable, and the
    colour limit must not be handed to a single bad frame."""
    from trackanno.measure.kymograph import curvature_kymograph

    kymo = curvature_kymograph(*_wave(50, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0)
    plain = kymo.scale()
    check("the scale is a positive number", plain > 0)

    # Same shape, twice the size: curvature halves, kappa*L does not move.
    big = curvature_kymograph(*_wave(50, 9, 12.0, 1.5, length=200.0),
                              n_samples=64, smooth_px=0.0)
    check("a bigger cell has smaller curvature", big.scale() < plain)
    a = float(np.nanpercentile(np.abs(kymo.normalised()), 98))
    b = float(np.nanpercentile(np.abs(big.normalised()), 98))
    check("but kappa*L is scale-free", abs(a - b) / max(a, b) < 0.02,
          f"{a:.3f} vs {b:.3f}")

    spiked = curvature_kymograph(*_wave(50, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0)
    spiked.kappa[7, 3] = 1e4
    check("one wild frame does not take over the colour limit",
          spiked.scale() < plain * 1.5, f"{spiked.scale():.4f} vs {plain:.4f}")


# --------------------------------------------------------------------------- #
# the plumbing the GUI actually uses
# --------------------------------------------------------------------------- #
def test_it_reads_a_clip_box_table() -> None:
    """The GUI has keypoints as a box table: one row per (frame, track), the
    target track flagged, and missing frames simply absent."""
    from trackanno.measure.kymograph import kymograph_from_boxes

    n_kpts, n_frames = 9, 30
    kx, ky = _wave(n_frames, n_kpts - 1, period=10.0, cycles=1.5)
    rows = []
    for t in range(n_frames):
        if 12 <= t < 16:                       # frames the tracker missed
            continue
        entry = {"row": 0, "t": t, "track_id": 7, "is_target": True, "conf": 0.9}
        for k in range(n_kpts):
            entry[f"kp_x_{k}"] = kx[t, k]
            entry[f"kp_y_{k}"] = ky[t, k]
            entry[f"kp_c_{k}"] = 0.4 if k == 5 else 0.95
        rows.append(entry)
        # A neighbouring track sits in the same crop with no keypoints at all;
        # taking its rows would blank the flagellum on every frame.
        neighbour = dict(entry, track_id=8, is_target=False)
        for k in range(n_kpts):
            neighbour[f"kp_x_{k}"] = np.nan
            neighbour[f"kp_y_{k}"] = np.nan
        rows.append(neighbour)

    kymo = kymograph_from_boxes(pd.DataFrame(rows), n_kpts, n_frames, n_samples=48)
    check("the grid is one row per clip frame", kymo.kappa.shape == (n_frames, 48))
    check("the head is dropped, leaving the flagellum", kymo.n_nodes == n_kpts - 1)
    check("only the target track is used", bool(np.isfinite(kymo.kappa[0]).all()))
    check("absent frames land in the right place, by frame not by row order",
          bool(np.isnan(kymo.kappa[12:16]).all()))
    check("...and the frames after the gap are still filled",
          bool(np.isfinite(kymo.kappa[16]).all()))

    # A low-confidence keypoint is worse than a missing one: it drags the fit.
    dropped = kymograph_from_boxes(pd.DataFrame(rows), n_kpts, n_frames,
                                   min_conf=0.5, n_samples=48)
    check("a min_conf gate still leaves a fittable chain", dropped.usable)
    check("...and it changes the answer, so the gate really applies",
          not np.allclose(np.nan_to_num(dropped.kappa),
                          np.nan_to_num(kymo.kappa), atol=1e-9))

    empty = kymograph_from_boxes(pd.DataFrame(), n_kpts, n_frames)
    check("an empty table is handled", not empty.usable)
    check("a detect project asks for nothing",
          not kymograph_from_boxes(pd.DataFrame(rows), 0, n_frames).usable)


# --------------------------------------------------------------------------- #
# the widget
# --------------------------------------------------------------------------- #
def test_the_widget_draws_and_scrubs(app) -> None:
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from trackanno.gui.kymograph import KymographView, diverging_ramp
    from trackanno.gui.theme import apply_theme
    from trackanno.measure.kymograph import curvature_kymograph

    for theme in ("dark", "light"):
        apply_theme(app, theme)
        ramp = diverging_ramp(64, theme)
        check(f"the {theme} ramp runs pole to pole", ramp.shape == (64, 3))
        first, mid, last = ramp[0], ramp[len(ramp) // 2], ramp[-1]
        check(f"...{theme}: it starts blue", int(first[2]) > int(first[0]) + 40,
              str(first))
        check(f"...{theme}: it ends red", int(last[0]) > int(last[2]) + 40, str(last))
        check(f"...{theme}: the midpoint is neutral",
              max(mid) - min(mid) < 30, str(mid))
        # A rainbow or a hued midpoint is the classic diverging-scale mistake.
        check(f"...{theme}: no hue at zero", int(abs(int(mid[0]) - int(mid[2]))) < 30)

    view = KymographView()
    view.resize(600, 220)
    check("an empty view paints without a kymograph", view.grab() is not None)

    kymo = curvature_kymograph(*_wave(60, 9, 12.0, 1.5), n_samples=64, smooth_px=0.0)
    view.set_kymograph(kymo)
    view.show()
    app.processEvents()
    check("it paints a real field", not view.grab().isNull())

    picked: list[int] = []
    view.frame_picked.connect(picked.append)
    plot = view._plot_rect()
    at = QPointF(plot.left() + plot.width() * 0.75, plot.center().y())
    view.mousePressEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))
    check("clicking the chart scrubs to that frame", len(picked) == 1, repr(picked))
    check("...to roughly where it was clicked", 42 <= picked[0] <= 48, repr(picked))
    check("...and the cursor moved with it", view.frame == picked[0])

    # A theme change must rebuild the image, not keep the old palette.
    from trackanno.gui.theme import current

    before = view._image
    apply_theme(app, "dark" if current() == "light" else "light")
    view.grab()
    check("a theme change repaints the field", view._image is not before)
    view.close()


def test_the_panel_shows_it_only_for_pose(app, root: Path) -> None:
    """End to end: a real pose project, a real clip cache, the real widget."""
    from fixtures import N_KEYPOINTS, make_project

    from trackanno.annotate.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.annotate.store import AnnotationStore

    n_kpts = N_KEYPOINTS

    # The shared fixture, not a config written here: it is built through the real
    # create_project/write_config, so it is block-style YAML that write_config
    # can edit. A hand-written flow-style config parses fine and then cannot be
    # saved, which is a property of the test and not of anything a user has.
    def build(_unused, pose: bool):
        cfg = load_config(make_project(root, pose=pose))
        return cfg, ClipStore(cfg.paths.work_dir)

    cfg, store = build(root / "pose", pose=True)
    check("the pose project knows its keypoint count", cfg.n_keypoints == n_kpts)
    panel = AnnotationWidget(cfg, store, AnnotationStore(cfg.paths.csv, cfg))
    panel.resize(1000, 700)
    panel.show()
    app.processEvents()

    check("the kymograph is shown on a pose project", panel.kymo_view.isVisible())
    check("...and it has data", panel.kymo_view.kymo is not None
          and panel.kymo_view.kymo.usable)
    check("...one column per frame of the clip",
          panel.kymo_view.kymo.n_frames == panel._clip.shape[0],
          f"{panel.kymo_view.kymo.n_frames} vs {panel._clip.shape[0]}")

    # Playing the clip has to move the cursor, and scrubbing has to move the clip.
    # Paused first: autoplay is on by default, so a running timer would advance
    # the frame between the seek and the assertion and the test would blame the
    # cursor for it.
    panel.viewer.pause()
    app.processEvents()
    panel.viewer.seek(7)
    app.processEvents()
    check("the cursor follows the clip", panel.kymo_view.frame == 7,
          str(panel.kymo_view.frame))
    panel._scrub_to(3)
    app.processEvents()
    check("scrubbing the chart moves the clip", panel.viewer._frame == 3,
          str(panel.viewer._frame))

    first = panel.kymo_view.kymo
    panel.next_clip()
    app.processEvents()
    check("changing clip recomputes it", panel.kymo_view.kymo is not first)
    panel.shutdown()
    panel.close()

    # The same project as a detector: no keypoints, so nothing to draw.
    cfg2, store2 = build(root / "detect", pose=False)
    plain = AnnotationWidget(cfg2, store2, AnnotationStore(cfg2.paths.csv, cfg2))
    plain.show()
    app.processEvents()
    check("a detect project hides it entirely", not plain.kymo_view.isVisible())
    check("...and never computes one", plain.kymo_view.kymo is None)
    plain.shutdown()
    plain.close()
