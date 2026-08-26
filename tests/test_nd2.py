"""ND2 reader checks against a mock ND2File.

There are no .nd2 files in this dataset, and the risky part of the ND2 backend
is arithmetic, not I/O: `nd2.read_frame()` takes a FLAT index over the sequence
axes (everything except C/Y/X, in the file's own axis order), so a wrong ravel
silently returns the wrong position, timepoint or z-plane. The mock below
returns a frame that encodes its own sequence index and channel, which turns
that mapping into something assertable.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
from harness import check
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
        # Pixel value encodes (sequence index, channel) so callers can assert
        # exactly which plane they were handed.
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


def test_geometry_and_axis_discovery(root):
    cfg = make_cfg(root, channel=1, z=2)
    fake = FakeND2File(SIZES)
    movie = ND2Movie(Path("stack.nd2"), cfg, position=1, _file=fake)

    check("id encodes the position", movie.id == "stack__P01", movie.id)
    check("time axis found", movie.n_frames == 10)
    check("frame size found", (movie.height, movie.width) == (HEIGHT, WIDTH))
    check("channel selected", movie.channel == 1)
    check("pixel size read", abs(movie.pixel_size_um - 0.325) < 1e-9)
    check("fps read from the acquisition loop", abs(movie.fps - 100.0) < 1e-9)
    check("sequence axes exclude C/Y/X", movie._seq_axes == ["P", "T", "Z"], str(movie._seq_axes))

    # P=1, T=5, Z=2 over shape (3, 10, 4) -> 1*40 + 5*4 + 2
    frame = movie.frame(5)
    expected_seq = 1 * 40 + 5 * 4 + 2
    check("read_frame got the right sequence index",
          fake.reads[-1] == expected_seq, f"{fake.reads[-1]} != {expected_seq}")
    check("frame shape is 2-D", frame.shape == (HEIGHT, WIDTH), str(frame.shape))
    check("the requested channel was returned",
          int(frame[0, 0]) == expected_seq * 100 + 1, str(int(frame[0, 0])))


def test_each_position_is_its_own_movie(root):
    cfg = make_cfg(root, channel=1, z=2)
    movie = ND2Movie(Path("stack.nd2"), cfg, position=1, _file=FakeND2File(SIZES))
    other = ND2Movie(Path("stack.nd2"), cfg, position=2, _file=FakeND2File(SIZES))
    other.frame(5)
    check("a different position reads a different plane",
          other._f.reads[-1] == 2 * 40 + 5 * 4 + 2)
    check("positions get distinct ids", other.id != movie.id)


def test_z_projection_reads_every_plane(root):
    cfg_max = make_cfg(root, channel=0, z="max")
    proj_file = FakeND2File(SIZES)
    proj = ND2Movie(Path("stack.nd2"), cfg_max, position=0, _file=proj_file)
    out = proj.frame(3)
    check("z projection reads every plane",
          proj_file.reads[-4:] == [3 * 4 + z for z in range(4)], str(proj_file.reads[-4:]))
    check("z projection takes the max",
          int(out[0, 0]) == (3 * 4 + 3) * 100, str(int(out[0, 0])))


def test_a_channel_can_be_named_rather_than_numbered(root):
    cfg_named = make_cfg(root, channel="CH1", z=0)
    named = ND2Movie(Path("stack.nd2"), cfg_named, position=0, _file=FakeND2File(SIZES))
    check("channel selected by name", named.channel == 1)


def test_a_file_with_no_p_or_c_axis(root):
    flat = {"T": 7, "Y": HEIGHT, "X": WIDTH}
    simple = ND2Movie(Path("plain.nd2"), make_cfg(root, z=0), _file=FakeND2File(flat))
    check("no P axis -> plain stem as id", simple.id == "plain", simple.id)
    check("no C axis -> 2-D frames", simple.frame(4).shape == (HEIGHT, WIDTH))
    check("no C axis -> right sequence index", simple._f.reads[-1] == 4)


def test_an_axis_index_that_does_not_exist_is_refused(root):
    """An out-of-range index must raise, not silently read plane 0."""
    for name, kwargs, position in [
        ("out-of-range position", {}, 9),
        ("out-of-range channel", {"channel": 5}, 0),
        ("out-of-range z", {"z": 9}, 0),
        ("unknown channel name", {"channel": "nope"}, 0),
    ]:
        try:
            ND2Movie(Path("stack.nd2"), make_cfg(root, **kwargs), position=position,
                     _file=FakeND2File(SIZES))
            check(f"rejects {name}", False, "no error raised")
        except ValueError:
            check(f"rejects {name}", True)


def test_no_timing_metadata_means_no_fps():
    check("no timing metadata -> no fps", _nd2_fps(FakeND2File(SIZES, period_ms=None)) is None)
