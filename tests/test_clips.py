"""The clip cache on disk: what ClipStore refuses to load and what a failed rebuild keeps."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from trackanno.annotate.clips import ClipStore

N, FRAMES, SIZE = 3, 4, 8


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A consistent hand-made cache: 3 clips of 4 frames at 8x8."""
    work = tmp_path / "work"
    movie = work / "movie_a"
    movie.mkdir(parents=True)
    _write_header(movie, N)
    (movie / "clips.dat").write_bytes(b"\0" * (N * FRAMES * SIZE * SIZE))
    pd.DataFrame({
        "row": range(N), "clip_id": [f"movie_a__t{i}" for i in range(N)],
        "movie_id": ["movie_a"] * N, "track_id": range(N),
        "f_start": [0] * N, "f_end": [FRAMES] * N,
        "crop_x0": [0] * N, "crop_y0": [0] * N, "crop_w": [SIZE] * N,
        "crop_h": [SIZE] * N, "scale": [1.0] * N, "n_present": [FRAMES] * N,
        "mean_conf": [0.9] * N,
    }).to_parquet(movie / "clips.parquet", index=False)
    return work


def _write_header(movie: Path, n_clips: int) -> None:
    header = {"n_clips": n_clips, "frames": FRAMES, "size": SIZE, "dtype": "uint8",
              "movie": {"height": 64, "width": 64, "fps": 100.0}}
    (movie / "cache.json").write_text(json.dumps(header), encoding="utf-8")


def _load(work: Path) -> tuple[int, list[str]]:
    """Clips loaded and warnings raised; an all-skipped store counts as 0."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            loaded = len(ClipStore(work))
        except FileNotFoundError:
            loaded = 0
    return loaded, [str(m.message) for m in caught]


def test_a_consistent_cache_loads(work: Path) -> None:
    assert len(ClipStore(work)) == N


def test_a_cache_that_outgrew_its_manifest_is_refused(work: Path) -> None:
    """An interrupted rebuild: cache.json and clips.dat say 5 clips, the manifest 3."""
    movie = work / "movie_a"
    _write_header(movie, 5)
    (movie / "clips.dat").write_bytes(b"\0" * (5 * FRAMES * SIZE * SIZE))
    loaded, said = _load(work)
    assert loaded == 0, f"loaded {loaded} clips from an inconsistent cache"
    assert any("movie_a" in m and "cache.json" in m for m in said), said


def test_a_truncated_cache_is_refused(work: Path) -> None:
    (work / "movie_a" / "clips.dat").write_bytes(b"\0" * 16)
    loaded, _ = _load(work)
    assert loaded == 0


def test_a_closed_store_refuses_to_remap(work: Path) -> None:
    store = ClipStore(work)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.clip(0)


def test_a_cancelled_rebuild_leaves_the_old_cache_intact(tmp_path: Path) -> None:
    """A rebuild that dies partway leaves every previous file byte-identical."""
    from fixtures import PIPELINE_CONFIG as CONFIG_TEXT, SyntheticMovie

    from trackanno.annotate.clips import (
        build_box_table, choose_crop_size, finalise_geometry, plan_clips,
        specs_to_frame, write_clip_cache,
    )
    from trackanno.config import load_config
    from trackanno.video.movies import Normalizer

    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    movie = SyntheticMovie()
    tracks = movie.tracks()

    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    out_dir = cfg.movie_dir(movie.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_box_table(specs, origins, tracks, cfg, 0).to_parquet(
        out_dir / "clip_boxes.parquet", index=False)
    write_clip_cache(movie, specs, origins, size, out_dir, Normalizer(0.0, 255.0), cfg)
    specs_to_frame(specs).to_parquet(out_dir / "clips.parquet", index=False)

    good = {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}
    before = len(ClipStore(cfg.paths.work_dir))
    assert before == len(specs), "the first build is readable"

    class Boom(Exception):
        pass

    def explode(iterable):
        for i, item in enumerate(iterable):
            if i > 3:
                raise Boom("network share went away")
            yield item

    origins2 = finalise_geometry(plan_clips(tracks, movie.id, movie.n_frames, cfg)[:2],
                                 size, cfg.clips.frames)
    with pytest.raises(Boom):
        write_clip_cache(movie, specs[:2], origins2, size, out_dir,
                         Normalizer(0.0, 255.0), cfg, progress=explode)

    after = {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}
    assert not [n for n in after if n.endswith(".part")], f"staged files left: {list(after)}"
    assert after == good, str(set(after) ^ set(good)) or "contents differ"
    assert len(ClipStore(cfg.paths.work_dir)) == before, "the old cache still loads"


def test_a_16_bit_movie_is_cut_under_one_mapping(tmp_path: Path) -> None:
    import cv2
    import numpy as np
    from fixtures import PIPELINE_CONFIG as CONFIG_TEXT, SyntheticMovie

    from trackanno.annotate.clips import (
        _crop, cache_normalizer, choose_crop_size, finalise_geometry, plan_clips,
        write_clip_cache,
    )
    from trackanno.config import load_config
    from trackanno.video.movies import Normalizer

    (tmp_path / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    cfg = load_config(tmp_path / "config.yaml")
    movie = SyntheticMovie()
    drift = np.arange(movie.n_frames, dtype=np.uint16)[:, None, None] * 8
    movie._data = movie._data.astype(np.uint16) * 16 + 100 + drift
    movie.dtype = np.dtype(np.uint16)

    specs = plan_clips(movie.tracks(), movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    out_dir = cfg.movie_dir(movie.id)
    write_clip_cache(movie, specs, origins, size, out_dir, cache_normalizer(movie), cfg)

    header = json.loads((out_dir / "cache.json").read_text(encoding="utf-8"))
    mapping = Normalizer(header["intensity"]["lo"], header["intensity"]["hi"])
    cache = np.memmap(out_dir / "clips.dat", dtype=np.uint8, mode="r",
                      shape=(len(specs), cfg.clips.frames, size, size))
    s = specs[0]
    for t in (0, s.f_end - s.f_start - 1):
        x0, y0 = origins[0, t]
        crop = _crop(mapping.to_u8(movie.frame(s.f_start + t)), int(x0), int(y0),
                     s.crop_w, s.crop_h, movie.height, movie.width)
        if crop.shape != (size, size):
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        assert np.array_equal(cache[0, t], crop), f"frame {t} is not under the movie's mapping"
    del cache


def test_detection_records_the_cache_range_and_the_cut_reuses_it(tmp_path: Path) -> None:
    import numpy as np
    from fixtures import SyntheticMovie

    from trackanno.annotate.clips import cache_normalizer
    from trackanno.video.movies import Normalizer

    movie = SyntheticMovie()
    movie._data = movie._data.astype(np.uint16) * 16 + 100
    movie.dtype = np.dtype(np.uint16)
    measured = Normalizer.from_movie(movie)
    measured.save(tmp_path / "norm.json")
    loaded = Normalizer.load(tmp_path / "norm.json")
    lo, hi = loaded.cache_range
    assert loaded.cache_range == measured.cache_range
    assert movie._data.min() <= lo < hi <= movie._data.max(), (lo, hi)

    def no_reads(index):
        raise AssertionError("the cut read a frame to find the range")

    movie.frame = no_reads
    mapping = cache_normalizer(movie, loaded)
    assert (float(mapping.lo), float(mapping.hi)) == (lo, hi)
