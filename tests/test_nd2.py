"""ND2 axis arithmetic against a mock ND2File whose pixels encode their own plane."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
from trackanno.config import load_config
from trackanno.video.movies import ND2Movie, _nd2_fps

HEIGHT, WIDTH = 8, 6


class FakeND2File:
    """Mimics the bits of nd2.ND2File that ND2Movie touches."""

    def __init__(self, sizes: dict[str, int], period_ms: float | None = 10.0):
        self.sizes = sizes
        self.dtype = np.dtype(np.uint16)
        self.reads: list[int] = []
        n_chan = sizes.get("C", 1)
        self.metadata = types.SimpleNamespace(
            channels=[
                types.SimpleNamespace(channel=types.SimpleNamespace(name=f"CH{i}"))
                for i in range(n_chan)
            ]
        )
        self.experiment = (
            [types.SimpleNamespace(parameters=types.SimpleNamespace(periodMs=period_ms))]
            if period_ms else []
        )

    def read_frame(self, seq_index: int) -> np.ndarray:
        self.reads.append(int(seq_index))
        n_chan = self.sizes.get("C", 1)
        # Pixel value encodes (sequence index, channel).
        frame = np.empty((n_chan, HEIGHT, WIDTH), dtype=np.uint16)
        for c in range(n_chan):
            frame[c] = seq_index * 100 + c
        return frame if n_chan > 1 else frame[0]

    def voxel_size(self):
        return types.SimpleNamespace(x=0.325, y=0.325, z=1.0)

    def events(self):
        return []

    def close(self):
        pass


def make_cfg(tmp: Path, **nd2_options):
    options = {"channel": 0, "z": 0, "positions": "all", "fps": "auto"}
    options.update(nd2_options)
    (tmp / "config.yaml").write_text(
        "paths: {video_dir: ./v, model: ./m.pt, work_dir: ./w, csv: ./a.csv}\n"
        "detection: {task: detect}\n"
        f"nd2: {{channel: {options['channel']!r}, z: {options['z']!r}, "
        f"positions: {options['positions']!r}, fps: {options['fps']!r}}}\n"
        "categories:\n"
        "  - name: behaviour\n    type: single\n    required: true\n"
        "    options:\n      - {name: a, key: '1', color: '#4C9F70'}\n",
        encoding="utf-8",
    )
    return load_config(tmp / "config.yaml")


SIZES = {"P": 3, "T": 10, "C": 2, "Z": 4, "Y": HEIGHT, "X": WIDTH}


@pytest.fixture
def stack(root):
    cfg = make_cfg(root, channel=1, z=2)
    fake = FakeND2File(SIZES)
    return ND2Movie(Path("stack.nd2"), cfg, position=1, _file=fake), fake


def test_geometry_and_axis_discovery(stack):
    movie, _ = stack
    assert movie.id == "stack__P01", "the id encodes the position"
    assert movie.n_frames == 10
    assert (movie.height, movie.width) == (HEIGHT, WIDTH)
    assert movie.channel == 1
    assert abs(movie.pixel_size_um - 0.325) < 1e-9
    assert abs(movie.fps - 100.0) < 1e-9, "fps from the acquisition loop"
    assert movie._seq_axes == ["P", "T", "Z"], "sequence axes exclude C/Y/X"


def test_a_frame_reads_the_right_plane_and_channel(stack):
    movie, fake = stack
    frame = movie.frame(5)
    expected_seq = 1 * 40 + 5 * 4 + 2       # P=1, T=5, Z=2 over shape (3, 10, 4)
    assert fake.reads[-1] == expected_seq
    assert frame.shape == (HEIGHT, WIDTH)
    assert int(frame[0, 0]) == expected_seq * 100 + 1, "the requested channel"


def test_each_position_is_its_own_movie(root):
    cfg = make_cfg(root, channel=1, z=2)
    movie = ND2Movie(Path("stack.nd2"), cfg, position=1, _file=FakeND2File(SIZES))
    other = ND2Movie(Path("stack.nd2"), cfg, position=2, _file=FakeND2File(SIZES))
    other.frame(5)
    assert other._f.reads[-1] == 2 * 40 + 5 * 4 + 2, "a different plane"
    assert other.id != movie.id


def test_z_projection_reads_every_plane(root):
    cfg_max = make_cfg(root, channel=0, z="max")
    proj_file = FakeND2File(SIZES)
    proj = ND2Movie(Path("stack.nd2"), cfg_max, position=0, _file=proj_file)
    out = proj.frame(3)
    assert proj_file.reads[-4:] == [3 * 4 + z for z in range(4)]
    assert int(out[0, 0]) == (3 * 4 + 3) * 100, "the max over z"


def test_a_channel_can_be_named_rather_than_numbered(root):
    cfg_named = make_cfg(root, channel="CH1", z=0)
    named = ND2Movie(Path("stack.nd2"), cfg_named, position=0, _file=FakeND2File(SIZES))
    assert named.channel == 1


def test_a_file_with_no_p_or_c_axis(root):
    flat = {"T": 7, "Y": HEIGHT, "X": WIDTH}
    simple = ND2Movie(Path("plain.nd2"), make_cfg(root, z=0), _file=FakeND2File(flat))
    assert simple.id == "plain", "no P axis gives the plain stem"
    assert simple.frame(4).shape == (HEIGHT, WIDTH)
    assert simple._f.reads[-1] == 4


@pytest.mark.parametrize("kwargs,position", [
    ({}, 9), ({"channel": 5}, 0), ({"z": 9}, 0), ({"channel": "nope"}, 0),
], ids=["position", "channel", "z", "channel-name"])
def test_an_axis_index_that_does_not_exist_is_refused(root, kwargs, position):
    """An out-of-range index raises rather than reading plane 0."""
    with pytest.raises(ValueError):
        ND2Movie(Path("stack.nd2"), make_cfg(root, **kwargs), position=position,
                 _file=FakeND2File(SIZES))


def test_no_timing_metadata_means_no_fps():
    assert _nd2_fps(FakeND2File(SIZES, period_ms=None)) is None
