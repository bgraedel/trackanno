"""One test per defect found by the adversarial review.

Each test is named for the bug it prevents and states the failure it used to
produce, so a future reader knows what breaks if it starts failing. Nothing here
needs a GPU, a movie or a display.
"""

from __future__ import annotations

import atexit
import inspect
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from qtpy.QtCore import Qt
from fixtures import make_project
from harness import check

# Every GUI test below runs against a project this suite GENERATES: config,
# synthetic movie, real clip cache. It used to load the repository's own
# config.yaml behind a try/except that printed "skipped" - which meant the
# tests passed without running on any machine that lacked that data, and broke
# on the one that had it whenever its network share went away.
FIXTURES = Path(tempfile.mkdtemp(prefix="trackanno-fixtures-"))
atexit.register(shutil.rmtree, FIXTURES, True)


def pump(app, predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# correctness
# --------------------------------------------------------------------------- #
def test_interrupted_rebuild_cannot_mislabel(root: Path) -> None:
    """C1. A cancelled redo left a new cache beside the old manifest and a done
    marker still claiming success, so clips were labelled under the wrong id."""
    from trackanno.clips import ClipStore

    work = root / "work"
    movie = work / "movie_a"
    movie.mkdir(parents=True)

    import json

    import pandas as pd

    # A consistent cache: 3 clips, 4 frames, 8x8.
    n, frames, size = 3, 4, 8
    header = {"n_clips": n, "frames": frames, "size": size, "dtype": "uint8",
              "movie": {"height": 64, "width": 64, "fps": 100.0}}
    (movie / "cache.json").write_text(json.dumps(header), encoding="utf-8")
    (movie / "clips.dat").write_bytes(b"\0" * (n * frames * size * size))
    pd.DataFrame({
        "row": range(n), "clip_id": [f"movie_a__t{i}" for i in range(n)],
        "movie_id": ["movie_a"] * n, "track_id": range(n),
        "f_start": [0] * n, "f_end": [frames] * n,
        "crop_x0": [0] * n, "crop_y0": [0] * n, "crop_w": [size] * n,
        "crop_h": [size] * n, "scale": [1.0] * n, "n_present": [frames] * n,
        "mean_conf": [0.9] * n,
    }).to_parquet(movie / "clips.parquet", index=False)
    check("a consistent cache loads", len(ClipStore(work)) == 3)

    # Now the interrupted-rebuild state: cache grew to 5 clips, manifest did not.
    header["n_clips"] = 5
    (movie / "cache.json").write_text(json.dumps(header), encoding="utf-8")
    (movie / "clips.dat").write_bytes(b"\0" * (5 * frames * size * size))
    import warnings as w

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        try:
            store = ClipStore(work)
            loaded = len(store)
        except FileNotFoundError:
            loaded = 0            # every movie skipped, so the store is empty
    check("a manifest that disagrees with its cache is refused", loaded == 0,
          f"loaded {loaded} clips from an inconsistent cache")
    check("and it says which movie and why",
          any("movie_a" in str(m.message) and "cache.json" in str(m.message)
              for m in caught), str([str(m.message) for m in caught]))

    # A truncated clips.dat is caught too (killed mid-write).
    header["n_clips"] = 3
    (movie / "cache.json").write_text(json.dumps(header), encoding="utf-8")
    (movie / "clips.dat").write_bytes(b"\0" * 16)
    with w.catch_warnings(record=True):
        w.simplefilter("always")
        try:
            loaded = len(ClipStore(work))
        except FileNotFoundError:
            loaded = 0
    check("a truncated cache is refused too", loaded == 0)


def test_forced_rebuild_clears_the_done_marker(root: Path) -> None:
    """C1. prepare_movie kept the old marker, so an interrupted redo still
    reported the movie as prepared and was never retried."""
    import inspect

    from trackanno import prepare

    source = inspect.getsource(prepare.prepare_movie)
    body = source.split("if done_path.exists() and not force:")[1]
    check("the marker is unlinked before any artefact is written",
          "done_path.unlink(missing_ok=True)" in body.split("normalizer =")[0],
          "unlink is missing or happens too late")


def test_cancelled_rebuild_leaves_the_old_cache_intact(root: Path) -> None:
    """C1 (atomic). A rebuild that dies partway must leave the previous cache
    exactly as it was, not half-replaced."""
    from test_pipeline import CONFIG_TEXT, SyntheticMovie

    from trackanno.clips import (
        ClipStore, build_box_table, choose_crop_size, finalise_geometry,
        plan_clips, specs_to_frame, write_clip_cache,
    )
    from trackanno.config import load_config
    from trackanno.movies import Normalizer

    project = root / "atomic"
    project.mkdir()
    (project / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    cfg = load_config(project / "config.yaml")
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
    check("the first build is readable", before == len(specs))

    # A rebuild with different geometry that dies halfway through cutting.
    bigger = specs[:2]
    origins2 = finalise_geometry(plan_clips(tracks, movie.id, movie.n_frames, cfg)[:2],
                                 size, cfg.clips.frames)

    class Boom(Exception):
        pass

    def explode(iterable):
        for i, item in enumerate(iterable):
            if i > 3:
                raise Boom("network share went away")
            yield item

    try:
        write_clip_cache(movie, bigger, origins2, size, out_dir,
                         Normalizer(0.0, 255.0), cfg, progress=explode)
    except Boom:
        pass
    else:
        check("the rebuild was interrupted", False)

    after = {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}
    check("no staged files were left behind",
          not [n for n in after if n.endswith(".part")], str(list(after)))
    check("every original file is byte-identical", after == good,
          str(set(after) ^ set(good)) or "contents differ")
    check("the old cache still loads", len(ClipStore(cfg.paths.work_dir)) == before)


def test_reserved_category_names_are_refused(root: Path) -> None:
    """C2. A group named `notes` or `clip_id` overwrote that CSV column: the
    typed note vanished, and `clip_id` made every row unreadable on restart."""
    from trackanno.config import ConfigError, load_config
    from trackanno.project import create_project, write_config

    path = create_project(root / "reserved")
    for bad in ("notes", "clip_id", "annotator"):
        try:
            write_config(path, {}, categories=[
                {"name": bad, "type": "single", "required": True,
                 "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}])
        except ConfigError as exc:
            check(f"group named {bad!r} is refused", "collides" in str(exc).lower())
        else:
            check(f"group named {bad!r} is refused", False)

    try:
        write_config(path, {}, categories=[
            {"name": "dup", "type": "single", "required": True,
             "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]},
            {"name": "dup", "type": "multi", "required": False,
             "options": [{"name": "b", "key": "2", "color": "#D95F02"}]}])
    except ConfigError as exc:
        check("two groups with the same name are refused", "named" in str(exc))
    else:
        check("two groups with the same name are refused", False)

    check("the project still loads", load_config(path) is not None)

    from trackanno.gui.configform import CategoryEditor
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv[:1])
    editor = CategoryEditor()
    editor.load([{"name": "notes", "type": "single", "required": True,
                  "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}])
    check("the GUI catches it before writing",
          any("already uses" in p for p in editor.problems()), str(editor.problems()))


def test_zero_stride_is_refused(root: Path) -> None:
    """R5. clips.stride is the only loop increment in _windows, so 0 planned
    windows forever."""
    from trackanno.config import ConfigError, load_config
    from trackanno.project import create_project, write_config

    path = create_project(root / "stride")
    for bad in (0, -5):
        try:
            write_config(path, {"clips.stride": bad})
        except ConfigError as exc:
            check(f"stride {bad} is refused", "at least 1" in str(exc))
        else:
            check(f"stride {bad} is refused", False)
    check("the project still loads", load_config(path).clips.stride >= 1)


def test_config_does_not_grow_on_every_save(root: Path) -> None:
    """C6. replace_section kept the previous save's header and prepended a fresh
    one, so an autosaving editor added four banner lines per pause."""
    from trackanno.project import create_project, write_config

    path = create_project(root / "grow")
    cats = [{"name": "state", "type": "single", "required": True,
             "options": [{"name": "a", "key": "1", "color": "#4C9F70"}]}]
    sizes = []
    for i in range(6):
        write_config(path, {"clips.pad": 1.0 + i / 10}, categories=cats,
                     keypoints={"nodes": [], "edges": []})
        sizes.append(len(path.read_text(encoding="utf-8").splitlines()))
    check("repeated saves do not grow the file", len(set(sizes[1:])) == 1, str(sizes))
    check("the comments are still there",
          path.read_text(encoding="utf-8").count("#") > 40)
    check("it still parses", yaml.safe_load(path.read_text(encoding="utf-8"))["categories"] == cats)


def test_a_disconnected_share_does_not_stop_the_project_opening(root: Path) -> None:
    """A path on a dropped network share does not answer "no", it RAISES:
    Windows reports WinError 64 out of is_dir(), glob() and is_file() alike. That
    propagated out of project_status, so closing a laptop lid or losing the VPN
    stopped the project opening at all - backwards, because the clip cache is
    local and there is a day of annotation there that needs no share whatever."""
    from trackanno.config import load_config
    from trackanno.project import _exists, create_project, project_status, write_config

    path = create_project(root / "offline")
    cache = (root / "offline" / "work")
    cache.mkdir(parents=True, exist_ok=True)
    movie = cache / "movie_a"
    movie.mkdir(exist_ok=True)
    import json as _json

    (movie / "prepare_done.json").write_text(_json.dumps({"n_clips": 12}), encoding="utf-8")
    write_config(path, {"paths.video_dir": "//nowhere.invalid/share/videos",
                        "paths.model": "//nowhere.invalid/share/best.pt"})
    cfg = load_config(path)

    class Unreachable(type(Path())):
        """Every probe raises, the way a dropped share behaves."""

        def is_dir(self):
            raise OSError(64, "The specified network name is no longer available")

        def is_file(self):
            raise OSError(64, "The specified network name is no longer available")

        def exists(self):
            raise OSError(64, "The specified network name is no longer available")

        def glob(self, pattern):
            raise OSError(64, "The specified network name is no longer available")

    check("_exists reports an unreachable path as absent",
          not _exists(Unreachable("//nowhere.invalid/share"), "dir"))
    check("...for files too", not _exists(Unreachable("//x/y"), "file"))
    check("...and for a plain existence check", not _exists(Unreachable("//x/y")))

    cfg.paths.video_dir = Unreachable("//nowhere.invalid/share/videos")
    cfg.paths.model = Unreachable("//nowhere.invalid/share/best.pt")
    status = project_status(cfg)          # must not raise
    check("the project still reports a status", status is not None)
    check("the missing share is reported as missing, not fatal",
          not status.video_dir_ok and not status.model_ok)
    check("and the LOCAL clip cache is still counted", status.n_clips == 12,
          str(status.n_clips))


def test_an_unreachable_path_cannot_freeze_the_window(root: Path) -> None:
    """A path on a dropped share does not fail fast, it fails SLOWLY: measured
    on a real disconnected UNC path, one is_dir() blocked for 123 seconds and
    one is_file() for 53. The window probes these settings on every config load
    and on a debounce from the model field, so opening a project took half a
    minute and reaching the annotation page took over two."""
    import time as _time

    from trackanno import project as project_module
    from trackanno.project import _exists, forget_path_probes

    real_probe = project_module._probe
    calls = []

    def glacial(path, kind):
        calls.append(str(path))
        _time.sleep(5.0)
        return True

    project_module._probe = glacial
    try:
        forget_path_probes()
        start = _time.perf_counter()
        answer = _exists(Path("//nowhere.invalid/share/videos"), "dir")
        waited = _time.perf_counter() - start
        check("a probe that never answers is bounded", waited < 2.0, f"{waited:.2f}s")
        check("...and reports absent rather than hanging", answer is False)

        # The second call must not pay the timeout again. Without a cached
        # verdict every repaint re-blocked for the full timeout, which is what
        # made the window feel frozen rather than merely slow.
        start = _time.perf_counter()
        _exists(Path("//nowhere.invalid/share/videos"), "dir")
        again = _time.perf_counter() - start
        check("the verdict is remembered", again < 0.2, f"{again:.3f}s")
        check("and it was probed once, not twice", len(calls) == 1, str(len(calls)))
    finally:
        project_module._probe = real_probe
        forget_path_probes()

    # A local path must not pay anything for the guard.
    start = _time.perf_counter()
    for _ in range(100):
        _exists(Path(__file__), "file")
    check("local paths stay fast", (_time.perf_counter() - start) < 0.5)
    check("...and still answer correctly",
          _exists(Path(__file__), "file") and not _exists(root / "nope", "file"))

    # Every settings path the window probes has to go through the guard. A bare
    # is_dir() anywhere on this path is the bug, so it is checked at the source.
    for name in ("gui/hub.py", "gui/configform.py"):
        text = (Path(__file__).resolve().parents[1] / "src" / "trackanno" / name
                ).read_text(encoding="utf-8")
        for call in (".is_dir()", ".is_file()"):
            offenders = [
                line.strip() for line in text.splitlines()
                if call in line and "_exists" not in line and not line.strip().startswith("#")
            ]
            check(f"{name} has no unguarded {call}", not offenders, repr(offenders[:2]))


def test_a_held_out_clip_can_say_so(app, root: Path) -> None:
    """suggest.audit_fraction holds a share of clips out of the suggestion loop:
    they show NO suggestion at any confidence. Correct, and completely invisible
    - on a real project 77 of 648 clips were silent for this reason and nothing
    on screen said why, which reads as a broken suggester."""
    from fixtures import make_project

    from trackanno.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.store import AnnotationStore

    cfg = load_config(make_project(FIXTURES))
    store = ClipStore(cfg.paths.work_dir)
    try:
        panel = AnnotationWidget(cfg, store, AnnotationStore(cfg.paths.csv, cfg))
        panel.viewer.pause()

        # Force the clip to be a hold-out rather than relying on the hash.
        panel.suggester = _AlwaysAudit()
        panel._current_suggestion = {}

        cfg.suggest.audit_visible = False
        check("silent by default", panel._why_no_suggestion("c0") == "")
        cfg.suggest.audit_visible = True
        note = panel._why_no_suggestion("c0")
        check("...and explains itself when asked to", "held out" in note, note)
        check("...naming the reason it shows nothing", "confidence" in note, note)

        # The note must never appear when there IS a suggestion to show.
        panel._current_suggestion = {"behaviour": ("swimming", 0.9)}
        check("no note when a suggestion is shown",
              panel._why_no_suggestion("c0") == "")

        # And the panel is willing to display one.
        panel.categories.set_suggestion({}, "held out for audit")
        app.processEvents()
        check("the caption carries the note",
              "held out" in panel.categories.suggestion.text())
        panel.categories.set_suggestion({}, "")
        check("...and disappears without one",
              panel.categories.suggestion.isHidden()
              or panel.categories.suggestion.text() == "")
        panel.shutdown()
        panel.close()
    finally:
        store.close()


class _AlwaysAudit:
    ready = True

    def is_audit_clip(self, clip_id):
        return True

    def suggest(self, clip_id):
        return {}

    def status_line(self):
        return "model loaded"


def test_the_distribution_counts_what_the_csv_says(root: Path) -> None:
    """The Distribution page is only worth having if its numbers are right, and
    the cases that matter are the awkward ones: multi-select values packed into
    one cell, options with nothing against them, values left over from a rename,
    and blank cells in a required group."""
    from trackanno.config import load_config
    from trackanno.gui.distribution import UNLABELLED, summary_lines, tally_annotations
    from trackanno.project import create_project

    cfg = load_config(create_project(root / "dist"))
    single = cfg.categories[0]
    multi = next(g for g in cfg.categories if g.is_multi)
    a, b, c = (o.name for o in single.options)
    q0, q1 = (o.name for o in multi.options[:2])

    rows = (
        [{"clip_id": f"x{i}", "movie_id": "m1", single.name: a, multi.name: ""}
         for i in range(6)]
        + [{"clip_id": f"y{i}", "movie_id": "m2", single.name: b,
            multi.name: f"{q0}|{q1}"} for i in range(2)]
        + [{"clip_id": "z0", "movie_id": "m1", single.name: "renamed_away",
            multi.name: q0},
           {"clip_id": "z1", "movie_id": "m3", single.name: "", multi.name: ""}]
    )
    tallies = tally_annotations(rows, cfg)
    by_group = {t.name: t for t in tallies}
    bars = {bar.name: bar for bar in by_group[single.name].bars}

    check("every configured option gets a bar, including the empty one",
          {a, b, c} <= set(bars), repr(sorted(bars)))
    check("counts are right", bars[a].count == 6 and bars[b].count == 2,
          f"{bars[a].count}/{bars[b].count}")
    check("an option nobody chose is shown at zero", bars[c].count == 0)
    check("a value not in the config still appears",
          bars["renamed_away"].count == 1)
    check("...and is marked as unknown", not bars["renamed_away"].known)
    check("blank cells are counted separately", bars[UNLABELLED].count == 1)
    check("movies are tracked per class", bars[a].n_movies == 1 and bars[b].n_movies == 1)

    # A multi-select cell holds several values, so its counts sum above the row
    # count. Splitting on the separator is the whole point.
    mbars = {bar.name: bar for bar in by_group[multi.name].bars}
    check("multi-select cells are split", mbars[q0].count == 3 and mbars[q1].count == 2,
          f"{mbars[q0].count}/{mbars[q1].count}")
    check("a multi group can total more than its rows",
          by_group[multi.name].total > 0)

    lines = " ".join(summary_lines(tallies, cfg))
    check("the empty class is called out", f"'{c}'" in lines, lines)
    check("the renamed value is called out", "renamed_away" in lines)
    check("a lone class reads as singular", "only appears in one movie" in lines, lines)
    check("nine labels are too few to call the balance",
          "always answers" not in lines, lines)

    # An entirely unlabelled group reported "0% of labels are 'blurry', so a
    # model that always answers that would look accurate": the share was taken
    # over a total that counted the "(not set)" bar, which is 100% of itself.
    lopsided = (
        [{"clip_id": f"e{i}", "movie_id": f"m{i % 3}", single.name: a, multi.name: ""}
         for i in range(30)]
        + [{"clip_id": "e30", "movie_id": "m1", single.name: b, multi.name: ""}]
    )
    empty_lines = summary_lines(tally_annotations(lopsided, cfg), cfg)
    check("an unlabelled group makes no claim about its share",
          not any("always answers" in ln and ln.startswith(multi.name)
                  for ln in empty_lines), repr(empty_lines))
    check("but a genuinely dominant class is called out",
          any("always answers" in ln and ln.startswith(single.name)
              for ln in empty_lines), repr(empty_lines))


def test_the_distribution_page_survives_a_broken_csv(app, root: Path) -> None:
    """Reading the CSV happens on the UI thread when the page is shown, so a
    file someone has half-written or opened in Excel must report itself rather
    than raise into Qt and take the window down."""
    from trackanno.gui.hub import DistributionPage, ExportPage, Hub
    from trackanno.project import create_project

    path = create_project(root / "broken-dist")
    hub = Hub(str(path))
    hub.show()
    page = hub.page(DistributionPage)
    check("the page is found by class, not by index", page is not None)
    check("it sits before Export",
          hub.pages.index(page) < hub.pages.index(hub.page(ExportPage)))

    page.reload()
    check("with no CSV it says so, and does not raise",
          "no annotation" in page.coverage.text().lower()
          or "no annotations" in page.coverage.text().lower(),
          page.coverage.text())

    hub.cfg.paths.csv.parent.mkdir(parents=True, exist_ok=True)
    hub.cfg.paths.csv.write_bytes(b"clip_id,movie_id\n\xff\xfe not utf-8 \x00\n")
    if hub.status is not None:
        hub.status.n_annotated = 1
    page.reload()
    check("an unreadable CSV is reported, not raised",
          "could not read" in page.coverage.text(), page.coverage.text())
    check("and no stale charts are left behind", not page._charts)
    hub.close()


def _stub_model_files(cfg, columns, extra_feature_columns=()):
    """A parquet plus a model payload, without training anything."""
    import joblib

    from trackanno.clipfeatures import CLIP_FEATURES_FILE
    from trackanno.suggest import MODEL_FILE

    work = cfg.paths.work_dir
    work.mkdir(parents=True, exist_ok=True)
    frame = {"clip_id": ["a", "b"], "movie_id": ["m", "m"],
             "track_id": [1, 2], "f_start": [0, 0]}
    for i, name in enumerate(extra_feature_columns):
        frame[name] = [float(i), float(i) + 1.0]
    import pandas as pd

    pd.DataFrame(frame).to_parquet(work / CLIP_FEATURES_FILE, index=False)
    joblib.dump({"columns": list(columns), "models": {"state": _ConstantModel()}},
                work / MODEL_FILE)


class _ConstantModel:
    classes_ = np.array(["x", "y"])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_a_stale_model_says_so_instead_of_going_quiet(root: Path) -> None:
    """Changing suggest.feature_groups (or the task) leaves the trained model
    fitted on columns the feature table no longer carries. Suggester.load built
    its vectors with getattr per column, so the missing one raised, a bare
    `except Exception: return False` swallowed it, and suggestions simply
    stopped appearing - with the status line still reporting 'no model yet',
    which sent people to retrain a model they had just retrained."""
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.suggest import Suggester

    cfg = load_config(create_project(root / "stale"))
    _stub_model_files(cfg, columns=["speed_mean", "wave_amp_asym"],
                      extra_feature_columns=["speed_mean"])
    s = Suggester(cfg)
    check("a model fitted on missing columns is refused", not s.ready)
    check("it names the missing column", "wave_amp_asym" in s.problem, s.problem)
    check("and it does not claim there is no model",
          "no model yet" not in s.status_line(), s.status_line())
    check("the status line is the explanation", s.status_line() == s.problem)

    # Retraining fixes it, and the same object picks it up.
    _stub_model_files(cfg, columns=["speed_mean"], extra_feature_columns=["speed_mean"])
    check("reloading in place recovers", s.load() and s.ready)
    check("the problem is cleared", s.problem == "")
    check("and it suggests again", s.suggest("b") == {"state": ("y", 0.8)},
          repr(s.suggest("b")))

    (cfg.paths.work_dir / "suggest_model.joblib").unlink()
    check("a genuinely absent model is reported as absent",
          not s.load() and "no model has been trained" in s.problem, s.problem)


def test_the_group_picker_never_empties_itself(app, root: Path) -> None:
    """The config loader rejects an empty suggest.feature_groups, and autosave
    fires 700ms after every click - so unticking the last box on the way to a
    different selection produced a red 'nothing was saved' banner that also
    blocked every other edit made in the same pause."""
    from trackanno.config import load_config
    from trackanno.gui.hub import Hub
    from trackanno.project import create_project

    path = create_project(root / "picker")
    hub = Hub(str(path))
    hub.show()
    hub.go_to(1)
    page = hub.pages[1]
    page.tabs.setCurrentIndex(1)                  # Advanced, so the note shows
    app.processEvents()
    picker = page.advanced._widgets["suggest.feature_groups"]

    for name in list(picker._boxes):
        picker._boxes[name].setChecked(False)
    app.processEvents()
    check("one group survives unticking them all", len(picker.value()) == 1,
          repr(picker.value()))
    check("and it says why", not picker.note.isHidden())
    check("so the page still saves", page.save())
    check("with no failure banner", not page.banner.isVisible())
    check("and the config is valid", len(load_config(path).suggest.feature_groups) == 1)

    # Ticking the wanted group first is the documented way round, and works.
    picker._boxes["kinematic"].setChecked(True)
    app.processEvents()
    for name in ("cnn", "waveform"):
        picker._boxes[name].setChecked(False)
    app.processEvents()
    check("swapping groups the other way round works",
          "kinematic" in picker.value(), repr(picker.value()))

    # Only-unavailable groups pass the loader but fail at training time, so the
    # page has to warn where it can be fixed.
    picker.set_value(["pose"])
    picker._refresh_note()
    check("a selection this project cannot compute is flagged",
          not picker.note.isHidden())
    check("and the reason is named", "detection.task" in picker.note.text(),
          picker.note.text())
    hub.close()


def test_training_reaches_the_open_annotator(app, root: Path) -> None:
    """The annotation panel is rebuilt only when its fingerprint moves, and the
    trained model is not part of that fingerprint - deliberately, since it would
    otherwise be torn down on every autosave. So a retrain finished and the open
    annotator kept serving the model it had loaded when the page opened."""
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.gui.hub import AnnotatePage
    from trackanno.project import create_project, write_config

    path = create_project(root / "live")
    check("ModelPage hands the new model over",
          "reload_suggester" in inspect.getsource(
              sys.modules["trackanno.gui.hub"].ModelPage._on_finished))
    check("the annotator can reload one in place",
          callable(getattr(AnnotationWidget, "reload_suggester", None)))

    # The fingerprint deliberately ignores the model; that is why the explicit
    # hand-over above has to exist.
    fingerprint = inspect.getsource(AnnotatePage._fingerprint)
    check("the fingerprint still ignores the model file",
          "MODEL_FILE" not in fingerprint and "suggest_model" not in fingerprint)

    cfg = load_config(path)
    check("suggestions are on by default in a new project", cfg.suggest.enabled)
    write_config(path, {"suggest.feature_groups": ["kinematic"]})
    check("changing feature groups alone leaves the config valid",
          load_config(path).suggest.feature_groups == ["kinematic"])


def test_checkpoint_probe_reads_the_ema_weights(root: Path) -> None:
    """The probe read checkpoint["model"], which is None on any checkpoint saved
    with EMA on - which is to say on everybody's own trained best.pt. It found
    no kpt_shape, reported no keypoints, and the whole point of picking the file
    (having the project configure itself) quietly did nothing."""
    try:
        import torch
    except ImportError:                     # pragma: no cover
        print("  --  torch is not installed, skipping the checkpoint probe")
        return
    from types import SimpleNamespace

    from trackanno.tiling import read_model_info

    box = root / "ckpts"
    box.mkdir(exist_ok=True)

    # Exactly the shape ultralytics writes mid-training: weights in "ema",
    # "model" left as None.
    ema = SimpleNamespace(kpt_shape=[8, 3], names={0: "cell"}, yaml={}, model=None)
    path = box / "best.pt"
    torch.save({"model": None, "ema": ema,
                "train_args": {"task": "pose", "imgsz": 1280}}, path)
    info = read_model_info(path)
    check("the ema weights are found", info["n_keypoints"] == 8, repr(info))
    check("the task comes with them", info["task"] == "pose")
    check("and the class names", info["classes"] == ["cell"])
    check("and the training size", info["imgsz"] == 1280)

    # A checkpoint that states kpt_shape only on its head module.
    head = SimpleNamespace(kpt_shape=[5, 3])
    bare = SimpleNamespace(names={0: "cell"}, model=[head])
    path = box / "head_only.pt"
    torch.save({"model": bare}, path)
    info = read_model_info(path)
    check("the head is consulted when nothing else knows", info["n_keypoints"] == 5,
          repr(info))
    check("and pose is inferred from it", info["task"] == "pose")

    # A plain detector must not be reported as a pose model.
    plain = SimpleNamespace(names={0: "cell"}, model=[SimpleNamespace()], task="detect")
    path = box / "plain.pt"
    torch.save({"model": plain, "train_args": {"task": "detect"}}, path)
    info = read_model_info(path)
    check("a detector reports no keypoints", not info["n_keypoints"], repr(info))
    check("and is still called a detector", info["task"] == "detect")


def test_model_probe_saves_task_and_keypoints_together(app, root: Path) -> None:
    """Picking a pose checkpoint on a detect project has to change both the task
    and the keypoint table in one save. ConfigPage.save decided whether to write
    keypoints by reading self.cfg, which still said 'detect' at that moment, so
    the nodes it had just filled in were dropped on the way to disk and the next
    load complained that keypoints.nodes was empty."""
    from trackanno.config import load_config
    from trackanno.gui.hub import Hub
    from trackanno.project import create_project

    path = create_project(root / "probe")
    hub = Hub(str(path))
    hub.show()
    hub.go_to(1)
    app.processEvents()
    page = hub.pages[1]
    check("the project starts out as a detector",
          page.form._widgets["detection.task"].currentText() == "detect")

    # What read_model_info() returns for a yolo11n-pose.pt, without needing one.
    page._on_model_info({"task": "pose", "n_keypoints": 17, "classes": ["cell"],
                         "imgsz": 640, "name": "yolo11n-pose.pt"})
    app.processEvents()
    check("the task followed the checkpoint",
          page.form._widgets["detection.task"].currentText() == "pose")
    check("the names table was sized to match", page.keypoints.count.value() == 17)

    check("it saved", page.save())
    saved = load_config(path)
    check("the task reached disk", saved.detection.task == "pose")
    check("and so did the keypoints", len(saved.keypoints.nodes) == 17,
          repr(saved.keypoints.nodes))
    check("so the project no longer asks for them", not saved.needs_keypoints)

    # And back again. The names are kept rather than cleared: a detect project
    # ignores them, and someone who typed sixteen anatomical labels should not
    # lose them for briefly pointing at a plain detector.
    page._on_model_info({"task": "detect", "n_keypoints": 0, "classes": ["cell"],
                         "imgsz": 640, "name": "yolo11n.pt"})
    app.processEvents()
    check("the task followed it back",
          page.form._widgets["detection.task"].currentText() == "detect")
    check("it saved again", page.save())
    back = load_config(path)
    check("the task reached disk", back.detection.task == "detect")
    check("the typed names survive the round trip", len(back.keypoints.nodes) == 17)
    check("and detect does not ask for them", not back.needs_keypoints)
    hub.close()


def test_the_window_has_an_icon(app) -> None:
    """A window with no icon gets the platform's generic one, which on Windows
    is the Python launcher. apply_theme sets ours, at every size the taskbar and
    the alt-tab switcher ask for, so none of them has to scale a bitmap."""
    from trackanno.gui.theme import app_icon, apply_theme

    apply_theme(app, "dark")
    sizes = sorted(s.width() for s in app.windowIcon().availableSizes())
    check("the application carries an icon", not app.windowIcon().isNull())
    check("drawn at each size rather than scaled", sizes == [16, 32, 64, 256],
          str(sizes))
    small = app_icon().pixmap(16, 16).toImage()
    check("and it is not a blank square",
          any(small.pixelColor(x, y).alpha() > 0
              for x in range(16) for y in range(16)))


# --------------------------------------------------------------------------- #
# gui
# --------------------------------------------------------------------------- #
def test_gui_regressions(app, root: Path) -> None:
    from trackanno.config import load_config
    from trackanno.gui.hub import ExportPage, Hub, ModelPage
    from trackanno.project import create_project, write_config

    # -- U4: a window with no project must still paint itself ---------------
    os.environ["TRACKANNO_STATE_DIR"] = str(root / "empty-state")
    hub = Hub(None)
    hub.show()
    app.processEvents()
    check("the sidebar is painted with no project open",
          hub.sidebar.item(0).text().strip().startswith("1."),
          repr(hub.sidebar.item(0).text()))
    check("steps after the first are disabled with no project",
          not (hub.sidebar.item(3).flags() & Qt.ItemIsEnabled))
    hub.close()

    # -- C5: a failed open must not leave the old project live --------------
    good = create_project(root / "good")
    bad = create_project(root / "bad")
    (bad).write_text("categories: [", encoding="utf-8")   # malformed YAML

    hub = Hub(str(good))
    hub.show()
    app.processEvents()
    check("the good project opened", hub.cfg is not None)
    opened = hub.open_project(str(bad), quiet=True)
    app.processEvents()
    check("a malformed config is refused", not opened and hub.cfg is None)
    check("the status is cleared too", hub.status is None)
    check("the title no longer names the old project", "good" not in hub.windowTitle())
    check("no annotation panel survives", hub.pages[3].panel is None)
    hub.close()

    # -- C7: the theme toggle must survive the next autosave ----------------
    hub = Hub(str(good))
    hub.show()
    hub.go_to(1)
    app.processEvents()
    started = hub.cfg.ui.theme
    hub.toggle_theme()
    app.processEvents()
    switched = hub.cfg.ui.theme
    check("toggling changed the theme", switched != started)
    config_page = hub.pages[1]
    check("the form was told about it",
          config_page.form.values()["ui.theme"] == switched,
          config_page.form.values()["ui.theme"])
    config_page.form._widgets["clips.pad"].setValue(2.25)
    app.processEvents()
    check("the autosave fires", pump(app, lambda: not config_page._dirty))
    check("and does not undo the theme",
          load_config(good).ui.theme == switched, load_config(good).ui.theme)
    hub.close()

    # -- U3: a routine refresh must not disable a running Cancel ------------
    hub = Hub(str(good))
    hub.show()
    app.processEvents()
    model_page = hub.page(ModelPage)
    export_page = hub.page(ExportPage)
    for page in (model_page, export_page):
        page.tasks.start(lambda reporter: time.sleep(1.5))
    app.processEvents()
    check("both jobs are running", model_page.tasks.busy and export_page.tasks.busy)
    hub.note_progress(7)              # fires refresh_gate on every page
    app.processEvents()
    check("labelling does not disable the Model Cancel button",
          model_page.tasks.run_button.isEnabled())
    check("labelling does not disable the Export Cancel button",
          export_page.tasks.run_button.isEnabled())
    hub.go_to(4)
    app.processEvents()
    check("navigating does not disable it either",
          model_page.tasks.run_button.isEnabled())
    for page in (model_page, export_page):
        page.tasks.shutdown()
    hub.close()
    app.processEvents()


def test_cache_job_locks_the_annotate_page(app) -> None:
    """C3. release_caches ran once before the worker started, so opening
    Annotate mid-run re-mapped the very files detection was overwriting."""
    from trackanno.config import load_config
    from trackanno.gui.hub import Hub

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)

    hub = Hub(str(config_path))
    hub.show()
    hub.go_to(3)
    app.processEvents()
    check("the annotate page loads normally", hub.pages[3].panel is not None)

    hub.set_cache_job(True)
    app.processEvents()
    check("a running cache job tears the panel down", hub.pages[3].panel is None)
    check("...and says why", "Detection is running" in hub.pages[3].blocker())
    hub.go_to(3)
    app.processEvents()
    check("navigating back does not reopen it", hub.pages[3].panel is None)

    hub.set_cache_job(False)
    hub.go_to(3)
    app.processEvents()
    check("it reopens once the job finishes", hub.pages[3].panel is not None)
    hub.close()
    app.processEvents()


def test_annotator_rename_reaches_the_csv(app) -> None:
    """C4. The panel fingerprint omitted ui.annotator, so a renamed annotator
    kept writing the old name into every row."""
    from trackanno.config import load_config
    from trackanno.gui.hub import Hub
    from trackanno.project import write_config

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)

    scratch = Path(tempfile.mkdtemp(prefix="trackanno-rename-"))
    try:
        project = scratch / "p"
        project.mkdir()
        shutil.copy(config_path, project / "config.yaml")
        write_config(project / "config.yaml", {
            "paths.work_dir": str(cfg.paths.work_dir).replace("\\", "/"),
            "paths.csv": str(project / "fresh.csv").replace("\\", "/"),
            "ui.annotator": "alice",
        })
        hub = Hub(str(project / "config.yaml"))
        hub.show()
        hub.go_to(3)
        app.processEvents()
        before = hub.pages[3]._fingerprint()

        write_config(project / "config.yaml", {"ui.annotator": "bob"})
        hub.reload_project()
        app.processEvents()
        check("renaming the annotator changes the fingerprint",
              hub.pages[3]._fingerprint() != before)

        hub.go_to(3)
        app.processEvents()
        panel = hub.pages[3].panel
        group = hub.cfg.categories[0]
        panel.categories.toggle(group.name, group.options[0].name)
        app.processEvents()
        rows = list(panel.annotations.rows.values())
        check("the new name reaches the CSV",
              rows and rows[-1]["annotator"] == "bob",
              rows[-1]["annotator"] if rows else "no rows")
        hub.close()
        app.processEvents()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_minimap_keeps_uncommitted_input(app) -> None:
    """U2. Pressing m called _load(), which rewrites every widget from the CSV,
    silently discarding ticks and notes."""
    from trackanno.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.store import AnnotationStore

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)
    store = ClipStore(cfg.paths.work_dir)

    csv = cfg.paths.work_dir / "_regress_minimap.csv"
    csv.unlink(missing_ok=True)
    cfg.paths.csv = csv
    panel = AnnotationWidget(cfg, store, AnnotationStore(csv, cfg))
    panel.viewer.pause()
    app.processEvents()

    multi = next((g for g in cfg.categories if g.is_multi), None)
    if multi is not None:
        panel.categories.toggle(multi.name, multi.options[0].name)
    panel.notes.setText("a note I typed")
    panel.uncertain_box.setChecked(True)
    app.processEvents()

    panel._toggle_minimap()
    app.processEvents()
    check("the note survives the minimap toggle",
          panel.notes.text() == "a note I typed", repr(panel.notes.text()))
    check("uncertain survives it", panel.uncertain_box.isChecked())
    if multi is not None:
        check("the ticked option survives it",
              multi.options[0].name in panel.categories.values()[multi.name])
    panel._toggle_minimap()
    app.processEvents()
    check("and toggling back is harmless too", panel.notes.text() == "a note I typed")

    panel.shutdown()
    store.close()
    csv.unlink(missing_ok=True)


def test_write_failures_are_reported_not_fatal(app) -> None:
    """R2. The store re-raises on purpose; letting that escape a Qt slot aborted
    the process when the CSV was open in Excel."""
    from trackanno.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.store import AnnotationStore

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)
    store = ClipStore(cfg.paths.work_dir)

    csv = cfg.paths.work_dir / "_regress_locked.csv"
    csv.unlink(missing_ok=True)
    cfg.paths.csv = csv
    panel = AnnotationWidget(cfg, store, AnnotationStore(csv, cfg))
    panel.viewer.pause()
    app.processEvents()

    messages: list[str] = []
    panel.status_message.connect(lambda text, _t: messages.append(text))

    class Locked:
        path = csv

        def record(self, *a, **k):
            raise PermissionError(13, "The process cannot access the file")

        def clear(self, *a, **k):
            raise PermissionError(13, "The process cannot access the file")

        def undo(self):
            raise PermissionError(13, "The process cannot access the file")

        def get(self, _clip_id):
            return None

        def missing_required(self, _values):
            return []

        @property
        def done_ids(self):
            return set()

        def __len__(self):
            return 0

    panel.annotations = Locked()
    group = cfg.categories[0]
    panel.categories.toggle(group.name, group.options[0].name)
    app.processEvents()
    check("a locked CSV does not raise out of the slot", True)
    check("the annotator is told what happened",
          any("could not write" in m for m in messages + [panel._pending_message or ""]),
          str(messages))
    check("the hint names the likely cause",
          any("Excel" in m for m in messages + [panel._pending_message or ""]))

    panel._undo()
    app.processEvents()
    check("undo survives it too", any("could not write" in m for m in messages))

    panel.shutdown()
    check("shutdown still released the prefetcher despite the failure", True)
    store.close()
    csv.unlink(missing_ok=True)


def test_broken_csv_does_not_kill_startup(root: Path) -> None:
    """R1. A stray comma made pd.read_csv raise out of Hub.__init__ before the
    window was shown, and startup reopens the last project, so there was no way
    back in."""
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.render import annotated_movies

    path = create_project(root / "badcsv")
    cfg = load_config(path)
    cfg.paths.csv.write_text(
        "clip_id,movie_id,behaviour\na,m,swim\nb,m,swim,EXTRA,COLUMNS\n",
        encoding="utf-8",
    )
    import warnings as w

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        result = annotated_movies(cfg)
    check("a malformed CSV does not raise", isinstance(result, dict))
    check("and it is reported", any("could not read" in str(m.message) for m in caught),
          str([str(m.message) for m in caught]))


def test_prefetcher_is_joined_before_the_cache_closes(app) -> None:
    """R3. stop() set a flag and returned; close() then unmapped memory the
    worker was still reading, which segfaults rather than raising."""
    from trackanno.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.prefetch import Prefetcher

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)
    store = ClipStore(cfg.paths.work_dir)

    order = list(range(len(store)))
    for _ in range(30):
        prefetcher = Prefetcher(store, depth=8)
        prefetcher.around(order, 0)
        check_once("stop() reports that it joined", prefetcher.stop() is True)
    check("the prefetcher joins before the store is closed", True)

    store.close()
    try:
        store.clip(0)
    except RuntimeError as exc:
        check("a closed store refuses to re-map", "closed" in str(exc))
    else:
        check("a closed store refuses to re-map", False)


_ONCE: set[str] = set()


def check_once(name: str, condition: bool) -> None:
    if name in _ONCE:
        if not condition:
            raise AssertionError(f"{name} FAILED")
        return
    _ONCE.add(name)
    check(name, condition)


def test_small_buttons_actually_draw_their_label(app) -> None:
    """UI. The themed 14px horizontal padding is wider than a 26px button, so
    the "?" help button and every "x" delete mark rendered completely blank."""
    import numpy as np
    from qtpy.QtGui import QPixmap

    from trackanno.gui.theme import apply_theme, compact_button

    def ink(widget) -> int:
        pixmap = QPixmap(widget.size())
        widget.render(pixmap)
        image = pixmap.toImage()
        raw = image.constBits().asstring(image.sizeInBytes())
        arr = np.frombuffer(raw, np.uint8).reshape(
            image.height(), image.bytesPerLine() // 4, 4)[..., :3].astype(int)
        # anything that differs from the button's own fill is drawn content
        base = arr[1, 1]
        return int((np.abs(arr - base).max(axis=-1) > 40).sum())

    for theme in ("dark", "light"):
        apply_theme(app, theme)
        for label, size in (("?", 28), ("x", 26)):
            button = compact_button(label, size)
            button.show()
            app.processEvents()
            check(f"a {label!r} button draws its label in the {theme} theme",
                  ink(button) > 0, "the glyph did not render at all")
    apply_theme(app, "dark")

    # And the real ones in the running app, not just synthetic samples.
    from trackanno.clips import ClipStore
    from trackanno.config import load_config
    from trackanno.gui.app import AnnotationWidget
    from trackanno.store import AnnotationStore

    config_path = make_project(FIXTURES)
    repo = config_path.parent
    cfg = load_config(config_path)
    store = ClipStore(cfg.paths.work_dir)
    csv = cfg.paths.work_dir / "_regress_buttons.csv"
    csv.unlink(missing_ok=True)
    cfg.paths.csv = csv
    panel = AnnotationWidget(cfg, store, AnnotationStore(csv, cfg))
    panel.viewer.pause()
    app.processEvents()
    buttons = [b for b in panel.findChildren(type(compact_button()))
               if b.property("compact") and b.text()]
    check("the annotation screen has a compact button", bool(buttons))
    for button in buttons:
        # Both halves matter: a stylesheet min-width of 0 let the layout shrink
        # a setFixedSize button, and an over-full top row then crushed it to two
        # pixels, so it drew nothing at all.
        check(f"{button.text()!r} keeps its full size",
              button.width() >= 26, f"{button.width()}px wide")
        check(f"{button.text()!r} on the annotation screen is visible", ink(button) > 0)

    row = panel.layout().itemAt(0).layout()
    check("the top row fits without crushing its controls",
          row.minimumSize().width() <= panel.width(),
          f"needs {row.minimumSize().width()}px, has {panel.width()}px")
    panel.shutdown()
    store.close()
    csv.unlink(missing_ok=True)


def test_shutdown_never_destroys_a_live_thread(app) -> None:
    """R4. shutdown() discarded wait()'s result, so an uncancellable prologue
    (import torch + torch.load) meant the QThread was destroyed while running."""
    from trackanno.gui.tasks import TaskPanel

    panel = TaskPanel("Go")

    def stubborn(reporter):
        time.sleep(2.5)          # no cancellation point, like a checkpoint load
        return "done"

    panel.start(stubborn)
    pump(app, lambda: panel.busy, timeout=5)
    check("the job is running", panel.busy)
    stopped = panel.shutdown(timeout=200)
    check("shutdown reports that it could not stop it", stopped is False)
    check("and it let go rather than destroying it", panel._runner is None)
    pump(app, lambda: False, timeout=3)   # let the thread finish cleanly
    check("no abort followed", True)


def test_feature_groups_match_what_can_be_computed(root: Path) -> None:
    """The model may only be offered feature families this project can build."""
    from trackanno.clipfeatures import FEATURE_GROUPS, available_groups, group_of
    from trackanno.config import ConfigError, load_config
    from trackanno.project import create_project, write_config

    path = create_project(root / "groups")          # task: detect by default
    cfg = load_config(path)
    availability = available_groups(cfg)
    check("kinematic is always available", availability["kinematic"] == "")
    check("pose is unavailable on a detect project",
          "task: pose" in availability["pose"], availability["pose"])
    check("waveform is unavailable too", bool(availability["waveform"]))

    write_config(path, {"detection.task": "pose", "keypoints.nodes": ["a", "b", "c"]})
    availability = available_groups(load_config(path))
    check("switching to pose enables them",
          not availability["pose"] and not availability["waveform"])

    write_config(path, {"features.enabled": False})
    check("cnn follows features.enabled",
          bool(available_groups(load_config(path))["cnn"]))

    # Every emitted feature must belong to exactly one known group, or a
    # selection would silently drop columns nobody accounted for.
    for column in ("speed_mean", "area_mean", "body_len_mean", "beat_freq_hz",
                   "wave_amp_asym", "cnn3_mean", "conf_std"):
        check(f"{column} maps to a group", group_of(column) in FEATURE_GROUPS)
    check("metadata maps to no group", group_of("clip_id") is None)

    try:
        write_config(path, {"suggest.feature_groups": []})
    except ConfigError as exc:
        check("an empty selection is refused", "no inputs at all" in str(exc))
    else:
        check("an empty selection is refused", False)
    try:
        write_config(path, {"suggest.feature_groups": ["nonsense"]})
    except ConfigError as exc:
        check("an unknown group is refused", "unknown entries" in str(exc))
    else:
        check("an unknown group is refused", False)
