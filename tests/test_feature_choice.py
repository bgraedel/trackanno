"""Choosing the model's input columns by name, and refusing the choices that lie.

``suggest.feature_groups`` names the instrument a column came from.
``suggest.feature_columns`` is the finer cut: the columns themselves.
It NARROWS what the groups selected rather than replacing them, so the claims
here are mostly about the ways the two keys can contradict each other,
and about that contradiction being refused where it can still be fixed rather
than resolved silently in favour of whichever cut ran last.

The load-bearing one is the last test: what the fit actually saw has to be what
the payload says it saw, since every later step - prediction and the stale-model refusal - reads the column list back from the file and
never from the config.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from harness import check

GROUP = "behaviour"


def _raises(fn, needle: str) -> str:
    """Run ``fn`` and return the message, insisting it names ``needle``."""
    try:
        fn()
    except Exception as exc:                              # noqa: BLE001
        message = str(exc)
        check(f"the refusal names {needle!r}", needle in message, message)
        return message
    raise AssertionError(f"expected a refusal naming {needle!r}, got none")


def _project(root: Path, name: str, **settings) -> Path:
    """A fresh project with `settings` written into its config."""
    from trackanno.project import create_project, write_config

    path = create_project(Path(root) / name)
    if settings:
        write_config(path, settings)
    return path


def _hand_edit(path: Path, **settings) -> Path:
    """Write settings the way a text editor would: no loader in the way.

    `write_config` parses the candidate before it replaces anything, which is
    what lets the GUI autosave, so a refusal reached through it is the same
    refusal one step earlier. These tests want the loader's own answer on a file
    that already exists in that state, which is what hand-editing config.yaml
    produces.
    """
    from trackanno.project import write_config

    return write_config(path, settings, validate=False)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def test_a_column_list_is_checked_against_the_vocabulary(root: Path) -> None:
    """A typo must not quietly change the feature space the model is fitted in.

    Silently dropping a name nobody recognises would fit on a different set of
    columns from the one asked for and report success, and the difference would
    only ever show as a score that moved for no reason. So the names are checked
    against every column a built table can hold, and a near miss is offered.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config

    path = _project(root, "vocabulary")

    _hand_edit(path, **{"suggest.feature_columns": ["speed_men", "area_mean"]})
    message = _raises(lambda: load_config(path), "speed_men")
    check("...and offers the column that was meant", "speed_mean" in message, message)
    check("...and the same refusal reaches an autosave, before anything is written",
          "speed_men" in _raises(
              lambda: write_config(path, {"suggest.feature_columns": ["speed_men"]}),
              "speed_men"))

    _hand_edit(path, **{"suggest.feature_columns": ["speed_mean", "speed_mean"]})
    _raises(lambda: load_config(path), "more than once")

    _hand_edit(path, **{"suggest.feature_columns": []})
    message = _raises(lambda: load_config(path), "empty list")
    check("...and says how 'no cut at all' is spelled instead",
          "null" in message, message)

    # Embedding columns are matched by shape; their width follows the PCA. An
    # old cnn list is read as the detector embedding 'det'.
    write_config(path, {"suggest.feature_groups": ["kinematic", "cnn"],
                        "suggest.feature_columns": ["speed_mean", "cnn0_mean",
                                                    "emb_det_31_delta"]})
    with pytest.warns(UserWarning, match="emb_det"):
        cfg = load_config(path)
    check("an embedding column is a column, and cnn names map onto it",
          cfg.suggest.feature_columns
          == ["speed_mean", "emb_det_0_mean", "emb_det_31_delta"],
          str(cfg.suggest.feature_columns))

    write_config(path, {"suggest.feature_columns": None,
                        "suggest.feature_groups": ["kinematic"]})
    check("null is how the cut is turned off",
          load_config(path).suggest.feature_columns is None)


def test_naming_a_column_from_a_group_that_is_off_is_refused(root: Path) -> None:
    """The two keys would otherwise disagree about what the model may see.

    Resolving it either way is worse than refusing: honouring the name turns a
    family back on that the Configure page shows as unticked, and honouring the
    group drops a column somebody typed out. Both leave the screen saying
    something the fit does not do.
    """
    from trackanno.config import load_config
    from trackanno.project import write_config

    path = _project(root, "contradiction")
    _hand_edit(path, **{
        "suggest.feature_groups": ["kinematic"],
        "suggest.feature_columns": ["speed_mean", "wave_freq_hz"],
    })
    message = _raises(lambda: load_config(path), "wave_freq_hz")
    check("...and names the group it belongs to", "waveform" in message, message)
    check("...and names both keys",
          "feature_groups" in message and "feature_columns" in message, message)
    check("...and offers both fixes",
          "add the group" in message.lower() and "drop those" in message.lower(),
          message)

    # The same list under the group that owns it loads.
    write_config(path, {"suggest.feature_groups": ["kinematic", "waveform"]})
    cfg = load_config(path)
    check("with the group ticked the same list is fine",
          cfg.suggest.feature_columns == ["speed_mean", "wave_freq_hz"],
          str(cfg.suggest.feature_columns))


def test_a_fresh_project_does_not_cut_by_column(root: Path) -> None:
    """The default has to be the behaviour every project on disk was fitted under.

    A default that named columns would pin new projects to today's schema, and a
    column added in a later FEATURE_VERSION would then never reach a model that
    nobody had edited.
    """
    from trackanno.config import load_config
    from trackanno.project import create_project

    cfg = load_config(create_project(Path(root) / "fresh"))
    check("no column cut in a project a new user makes",
          cfg.suggest.feature_columns is None, repr(cfg.suggest.feature_columns))
    check("...and no embedding or context group is on",
          cfg.suggest.feature_groups == ["kinematic", "pose", "waveform"],
          str(cfg.suggest.feature_groups))

    raw = yaml.safe_load(Path(create_project(Path(root) / "fresh2")).read_text(
        encoding="utf-8"))
    check("...and the shipped file mentions the key, so it can be found",
          "feature_columns" in (raw.get("suggest") or {}),
          str(sorted((raw.get("suggest") or {}))))


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def _labelled_project(root: Path, name: str, **settings):
    """A copy of the shared fixture, labelled, ready to train."""
    from fixtures import make_project
    from trackanno.annotate.store import AnnotationStore
    from trackanno.config import load_config
    from trackanno.project import write_config

    scratch = Path(tempfile.mkdtemp(dir=root, prefix=f"{name}-"))
    fixture = make_project(Path(tempfile.mkdtemp(dir=root, prefix=f"{name}-fix-")))
    project = scratch / "project"
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yaml"
    shutil.copy(fixture, config_path)
    write_config(config_path, {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
        **settings,
    })
    cfg = load_config(config_path)
    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)

    manifests = []
    for movie_dir in sorted(p for p in load_config(fixture).paths.work_dir.iterdir()
                            if p.is_dir()):
        if not (movie_dir / "clips.parquet").exists():
            continue
        (work / movie_dir.name).mkdir(parents=True, exist_ok=True)
        for fname in ("clips.parquet", "tracks.parquet", "cache.json"):
            if (movie_dir / fname).exists():
                shutil.copy(movie_dir / fname, work / movie_dir.name / fname)
        manifests.append(pd.read_parquet(movie_dir / "clips.parquet"))
    manifest = pd.concat(manifests, ignore_index=True)

    store = AnnotationStore(cfg.paths.csv, cfg)
    for i, row in enumerate(manifest.to_dict("records")):
        store.record(row, {GROUP: ("swimming", "immotile")[i % 2]})
    return config_path, cfg


def test_the_named_columns_are_the_ones_the_payload_records(root: Path) -> None:
    """What was fitted and what the file says was fitted have to be one list.

    Every later step reads the columns back from the payload and never from the
    config: prediction builds its matrix by those names, the multi-scale check
    judges them, and the stale-model refusal compares them against the table. A
    cut applied after the payload was written, or only inside the fit, would
    leave all three describing a model that does not exist.
    """
    import joblib

    from trackanno.config import load_config
    from trackanno.learn.suggest import MODEL_FILE, Suggester, train
    from trackanno.project import write_config

    config_path, cfg = _labelled_project(root, "named")
    train(cfg, rebuild_features=True, verbose=False)
    wide = list(joblib.load(cfg.paths.work_dir / MODEL_FILE)["columns"])
    check("the unnarrowed fit uses many columns", len(wide) > 4, str(len(wide)))

    picked = ["speed_mean", "area_mean", "straightness", "immotile_frac"]
    write_config(config_path, {
        "suggest.feature_groups": ["kinematic"],
        "suggest.feature_columns": picked,
    })
    train(load_config(config_path), rebuild_features=False, verbose=False)
    narrow = list(joblib.load(cfg.paths.work_dir / MODEL_FILE)["columns"])

    check("the payload holds exactly what was named",
          narrow == [c for c in wide if c in set(picked)], str(narrow))
    check("...which is fewer than the groups alone would have given",
          len(narrow) < len(wide), f"{len(narrow)} vs {len(wide)}")

    suggester = Suggester(load_config(config_path))
    check("and the model loads against those columns",
          list(suggester.columns) == narrow, str(suggester.columns))


# --------------------------------------------------------------------------- #
# where the fit runs
# --------------------------------------------------------------------------- #
def test_the_fitting_device_is_the_cpu_until_it_is_asked_to_be_something_else(
    root: Path,
) -> None:
    """CatBoost's GPU uses 128 feature borders against the CPU's 254.

    The two therefore fit different models on the same data, so the device is a
    modelling setting rather than a speed one and the default cannot move. The
    refusals cover the two ways it can be set and do nothing: a backend with no
    GPU implementation, and a machine with no GPU.
    """
    from trackanno.config import load_config
    from trackanno.learn.suggest import gpu_devices, resolved_device
    from trackanno.project import create_project, write_config

    cfg = load_config(create_project(Path(root) / "device"))
    check("a fresh project fits on the CPU", cfg.suggest.device == "cpu",
          cfg.suggest.device)
    check("...and resolves to it", resolved_device(cfg) == "cpu")

    path = _project(root, "device-set")
    _hand_edit(path, **{"suggest.device": "cuda"})
    message = _raises(lambda: load_config(path), "cuda")
    check("...offering the three it takes",
          all(v in message for v in ("cpu", "gpu", "auto")), message)

    _hand_edit(path, **{"suggest.device": "gpu", "suggest.model": "hist"})
    message = _raises(lambda: load_config(path), "hist")
    check("...and names both keys",
          "suggest.device" in message and "suggest.model" in message, message)

    # 'auto' answers with what is here; 'gpu' answers only where there is one.
    write_config(path, {"suggest.device": "auto", "suggest.model": "catboost"})
    cfg = load_config(path)
    expected = "gpu" if gpu_devices() else "cpu"
    check("auto resolves to whatever this machine has",
          resolved_device(cfg) == expected, resolved_device(cfg))

    write_config(path, {"suggest.device": "gpu"})
    cfg = load_config(path)
    if gpu_devices():
        check("gpu resolves to the gpu here", resolved_device(cfg) == "gpu")
    else:
        _raises(lambda: resolved_device(cfg), "no GPU")
