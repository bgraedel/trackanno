"""Display contrast of the annotation window: the lookup table, auto limits and the bar."""

from __future__ import annotations

import numpy as np

from trackanno.gui.contrast import counts_of, limits_of, lut


def test_the_limits_map_to_black_and_white() -> None:
    table = lut(100, 200)
    assert table[[50, 100, 200, 250]].tolist() == [0, 0, 255, 255]
    assert abs(int(table[150]) - 128) <= 1
    assert int(lut(100, 200, gamma=0.5)[150]) > int(table[150]) + 40


def test_the_full_range_is_the_identity() -> None:
    assert np.array_equal(lut(0, 255), np.arange(256))


def test_auto_limits_leave_out_the_padding() -> None:
    rng = np.random.default_rng(0)
    clip = rng.integers(100, 201, size=(4, 32, 32)).astype(np.uint8)
    clip[:, :, :12] = 0
    lo, hi = limits_of(counts_of(clip))
    assert 100 <= lo <= 102 and 198 <= hi <= 200, (lo, hi)


def test_dragging_fixes_the_limits_for_the_next_clip(app) -> None:
    from trackanno.gui.contrast import ContrastBar

    bar = ContrastBar()
    dim = np.full((3, 16, 16), 100, np.uint8)
    dim[:, :8] = 120
    bar.set_clip(dim)
    assert bar.mode == "clip" and (bar.lo, bar.hi) == (100, 120)

    bar.histogram.moved.emit(105.4, 114.6)
    assert bar.mode == "fixed" and (bar.lo, bar.hi) == (105, 115)
    bar.set_clip(dim * 2)
    assert (bar.lo, bar.hi) == (105, 115), "fixed limits stay for the next clip"
    assert bar.mapper()(dim[0], 0).min() == 0

    bar.gamma_slider.setValue(50)
    bar.reset()
    assert bar.mode == "clip" and bar.gamma == 1.0 and (bar.lo, bar.hi) == (200, 240)


def test_auto_frame_stretches_each_frame(app) -> None:
    from trackanno.gui.contrast import ContrastBar

    ramp = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    clip = np.stack([10 + 40 * ramp, 100 + 150 * ramp]).astype(np.uint8)
    bar = ContrastBar(mode="frame")
    bar.set_clip(clip)
    show = bar.mapper()
    for t in range(2):
        out = show(clip[t], t)
        assert out.min() <= 2 and out.max() >= 253, (t, out.min(), out.max())
    bar.set_frame(1)
    assert bar.lo >= 100, "the window shows the frame on screen"


def test_the_readout_is_in_raw_intensity(app) -> None:
    from trackanno.gui.contrast import ContrastBar

    clip = np.zeros((2, 16, 16), np.uint8)
    clip[:, :, :8], clip[:, :, 8:] = 51, 102
    bar = ContrastBar()
    bar.set_clip(clip, {"lo": 1000.0, "hi": 3550.0})
    assert bar.readout.text() == "1510 - 2020", bar.readout.text()


def test_a_window_dragged_to_the_top_stays_on_the_axis(app) -> None:
    from trackanno.gui.contrast import ContrastBar

    bar = ContrastBar()
    bar.set_clip(np.arange(256, dtype=np.uint8).reshape(1, 16, 16))
    bar.histogram.moved.emit(254.8, 255.0)
    assert (bar.lo, bar.hi) == (254, 255), (bar.lo, bar.hi)
