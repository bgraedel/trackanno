"""A complete, self-contained project for the GUI tests to run against.

Several tests used to load the repository's own ``config.yaml`` and clip cache,
guarded by ``try: ... except: print("skipped")``. That is a bad trade twice
over: on a machine without that data every one of them silently passed without
testing anything, and on a machine with it the suite depended on a network share
staying mounted - so unplugging the network turned a green suite red for reasons
that had nothing to do with the code.

Everything here is generated: a synthetic movie, its tracks, a real clip cache
written by the real ``write_clip_cache``, and a real config. It costs a second
or so to build and is cached per pose flag for the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["make_project", "N_KEYPOINTS"]

N_KEYPOINTS = 8            # Head, Base, Flag1..Flag5, Tip

_CACHE: dict = {}


def _node_names() -> list:
    return (["Head", "Base"]
            + [f"Flag{i}" for i in range(1, N_KEYPOINTS - 2)] + ["Tip"])


def _settings(pose: bool) -> dict:
    """Whatever has to differ from a freshly created project.

    Deliberately applied with the real ``write_config`` to a real
    ``create_project`` template rather than by writing a config by hand: a
    hand-written one in YAML flow style parses fine but cannot be edited by the
    line-oriented writer, so tests that saved settings blew up on the fixture
    and not on anything a user would ever have.
    """
    return {
        "tiff.fps": 100,
        "normalize.mode": "fixed",
        "tracking.min_track_length": 50,
        "detection.task": "pose" if pose else "detect",
        "clips.frames": 50,
        "clips.stride": 50,
        "clips.window_mode": "all",
        "clips.follow": "fixed",
        "clips.crop_from": "union",
        "clips.pad": 1.6,
        "clips.crop_size": 96,
        "clips.max_size": 256,
        "clips.min_clip_frames": 40,
        "ui.annotator": "tester",
    }


def _add_keypoints(tracks):
    """A beating flagellum hung off every detection, in movie pixels."""
    s = np.linspace(0.0, 1.0, N_KEYPOINTS - 1)
    centre_x = (tracks["x1"] + tracks["x2"]) / 2.0
    centre_y = (tracks["y1"] + tracks["y2"]) / 2.0
    phase = 2 * np.pi * tracks["frame"].to_numpy() / 11.0
    tracks["kp_x_0"] = centre_x - 6.0                    # Head
    tracks["kp_y_0"] = centre_y
    tracks["kp_c_0"] = 0.9
    for k in range(N_KEYPOINTS - 1):                     # Base..Tip
        tracks[f"kp_x_{k + 1}"] = centre_x + s[k] * 22.0
        tracks[f"kp_y_{k + 1}"] = centre_y + 5.0 * np.sin(
            2 * np.pi * 1.4 * s[k] - phase)
        tracks[f"kp_c_{k + 1}"] = 0.9
    return tracks


def make_project(root: Path, pose: bool = False) -> Path:
    """Build a project with a real clip cache under ``root``; return its config.

    Cached across calls, so a suite that wants the same flavour twice pays for
    it once. The returned path is stable for the life of the process.
    """
    key = bool(pose)
    cached = _CACHE.get(key)
    if cached is not None and cached.exists():
        return cached

    from test_pipeline import SyntheticMovie

    from trackanno.annotate.clips import (
        build_box_table, choose_crop_size, finalise_geometry, plan_clips,
        specs_to_frame, write_clip_cache,
    )
    from trackanno.config import load_config
    from trackanno.video.movies import Normalizer
    from trackanno.project import create_project, write_config

    project = Path(root) / f"fixture{'_pose' if pose else ''}"
    config_path = create_project(project)
    names = _node_names()
    write_config(
        config_path, _settings(pose),
        keypoints={"nodes": names,
                   "edges": [[names[i], names[i + 1]] for i in range(1, len(names) - 1)]}
        if pose else None,
    )
    cfg = load_config(config_path)

    movie = SyntheticMovie()
    tracks = movie.tracks()
    if pose:
        tracks = _add_keypoints(tracks)

    specs = plan_clips(tracks, movie.id, movie.n_frames, cfg)
    size = choose_crop_size(specs, cfg)
    origins = finalise_geometry(specs, size, cfg.clips.frames)
    out = cfg.movie_dir(movie.id)
    out.mkdir(parents=True, exist_ok=True)
    build_box_table(
        specs, origins, tracks, cfg, N_KEYPOINTS if pose else 0
    ).to_parquet(out / "clip_boxes.parquet", index=False)
    write_clip_cache(movie, specs, origins, size, out, Normalizer(0.0, 255.0), cfg)

    # The same three columns prepare_movie adds to the manifest. Leaving them
    # out made the fixture subtly unlike a real cache, and the annotation store
    # writes them into every CSV row as the reassociation key back to the movie.
    manifest = specs_to_frame(specs)
    manifest["source_file"] = str(movie.path)
    manifest["series"] = movie.series
    manifest["channel"] = movie.channel
    manifest.to_parquet(out / "clips.parquet", index=False)

    # prepare_movie writes this beside the clip cache and several things read it
    # instead of the manifest - the dense prediction grid covers every track in
    # it, not just the ones that became clips. A fixture without it looks like a
    # project prepared by a version that did not keep the tracks.
    tracks.to_parquet(out / "tracks.parquet", index=False)

    # prepare_done.json is what project_status counts, so a fixture without one
    # looks like an unprepared project to the workflow window.
    import json

    (out / "prepare_done.json").write_text(
        json.dumps({"n_clips": len(specs), "movie_id": movie.id}), encoding="utf-8")

    _CACHE[key] = config_path
    return config_path
