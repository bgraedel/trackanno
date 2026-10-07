"""Re-linking labels to their clips after re-detection renumbered the tracks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation


# --------------------------------------------------------------------------- #
# synthetic frames
# --------------------------------------------------------------------------- #
def clip_id(movie: str, track: int, f_start: int) -> str:
    """The format ``clips.plan_clips`` writes."""
    return f"{movie}__t{track:04d}__f{f_start:06d}"


def manifest_row(movie: str, track: int, f_start: int, x0: int, y0: int,
                 side: int = 64, frames: int = 50) -> dict:
    return {
        "row": 0,
        "clip_id": clip_id(movie, track, f_start),
        "movie_id": movie,
        "track_id": track,
        "f_start": f_start,
        "f_end": f_start + frames,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_w": side,
        "crop_h": side,
        "scale": 1.0,
    }


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    return frame


def label_row(entry: dict, behaviour: str = "swimming", **human) -> dict:
    """A CSV row with the store's identity columns for ``entry``, plus human fields."""
    row = {
        "clip_id": entry["clip_id"],
        "movie_id": entry["movie_id"],
        "source_file": f"/movies/{entry['movie_id']}.tif",
        "series": 0,
        "channel": 0,
        "track_id": entry["track_id"],
        "f_start": entry["f_start"],
        "f_end": entry["f_end"],
        "crop_x0": entry["crop_x0"],
        "crop_y0": entry["crop_y0"],
        "crop_w": entry["crop_w"],
        "crop_h": entry["crop_h"],
        "scale": entry["scale"],
        "behaviour": behaviour,
        "uncertain": "0",
        "notes": "",
        "annotator": "tester",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "dwell_s": "1.50",
        "suggested": "",
        "suggest_conf": "",
        "suggest_shown": "0",
    }
    row.update(human)
    return row


def moved(entry: dict, track: int, dx: int = 0, dy: int = 0, **changes) -> dict:
    """The same animal after a re-prepare: new track id, crop nudged by a pixel."""
    out = dict(entry)
    out["track_id"] = track
    out["clip_id"] = clip_id(entry["movie_id"], track, entry["f_start"])
    out["crop_x0"] = entry["crop_x0"] + dx
    out["crop_y0"] = entry["crop_y0"] + dy
    out.update(changes)
    return out


def pairs(plan) -> list[tuple[str, str]]:
    return [(m.old_clip_id, m.new_clip_id) for m in plan.matched]


# --------------------------------------------------------------------------- #
def test_valid_ids_are_left_completely_alone() -> None:
    clips = [manifest_row("MOVIE_A", 3, 0, 100, 100),
             manifest_row("MOVIE_A", 4, 0, 300, 300),
             manifest_row("MOVIE_B", 3, 50, 10, 10)]
    manifest = make_manifest(clips)
    labels = pd.DataFrame([label_row(c) for c in clips])

    plan = plan_reassociation(labels, manifest)
    assert plan.already_valid == 3, plan.already_valid
    assert (not plan.matched and not plan.unmatched
            and not plan.ambiguous and not plan.conflicted), "nothing is proposed"
    assert not plan.has_repairs

    out = apply_reassociation(labels, plan)
    assert out.equals(labels), "applying an empty plan changes no value"
    assert out is not labels


def test_a_retracked_movie_is_relinked() -> None:
    """Same window, same animal, new track id, crop a pixel or two off: matched by centre."""
    old = [manifest_row("MOVIE_A", 2, 0, 100, 100),
           manifest_row("MOVIE_A", 5, 0, 400, 220),
           manifest_row("MOVIE_A", 9, 50, 80, 640)]
    new = [moved(old[0], 217, dx=1), moved(old[1], 231, dy=-2),
           moved(old[2], 244, dx=2, dy=1)]
    manifest = make_manifest(new)
    labels = pd.DataFrame([
        label_row(old[0], "immotile"),
        label_row(old[1], "swimming"),
        label_row(old[2], "unclear"),
    ])

    plan = plan_reassociation(labels, manifest)
    assert plan.already_valid == 0
    assert len(plan.matched) == 3, f"{len(plan.matched)} matched, {len(plan.unmatched)} unmatched"
    assert [m.new_clip_id for m in plan.matched] == sorted(c["clip_id"] for c in new)
    assert all(0.0 <= m.distance_px <= 3.0 for m in plan.matched), (
        [m.distance_px for m in plan.matched])

    out = apply_reassociation(labels, plan)
    assert bool(out["clip_id"].isin(set(manifest["clip_id"])).all())
    by_id = manifest.set_index("clip_id")
    for _, row in out.iterrows():
        want = by_id.loc[row["clip_id"]]
        for column in ("track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h", "scale"):
            assert row[column] == want[column], (
                f"{column} follows the new clip ({row['clip_id'][-12:]})")
    assert list(out["f_start"]) == list(labels["f_start"]), "the match key is untouched"


def test_the_humans_columns_survive_untouched() -> None:
    old = manifest_row("MOVIE_A", 7, 0, 250, 250)
    manifest = make_manifest([moved(old, 301, dx=1)])
    labels = pd.DataFrame([label_row(
        old, "unclear",
        uncertain="1", notes="out of focus, maybe two cells, comma included",
        annotator="benny", timestamp_utc="2025-11-04T09:12:33+00:00",
        dwell_s="12.75", suggested="swimming", suggest_conf="0.812",
        suggest_shown="1",
    )])

    out = apply_reassociation(labels, plan_reassociation(labels, manifest))
    for column in ("behaviour", "uncertain", "notes", "annotator", "timestamp_utc",
                   "dwell_s", "suggested", "suggest_conf", "suggest_shown",
                   "source_file", "series", "channel"):
        assert out[column].iat[0] == labels[column].iat[0], column
    assert out["clip_id"].iat[0] != labels["clip_id"].iat[0]
    assert list(out.columns) == list(labels.columns)


def test_two_orphans_on_one_clip_are_both_refused() -> None:
    """A merged or split track: two old crops either side of one new one."""
    survivor = manifest_row("MOVIE_A", 88, 0, 500, 500)
    manifest = make_manifest([manifest_row("MOVIE_A", 12, 0, 100, 100), survivor])
    a = manifest_row("MOVIE_A", 3, 0, 496, 500)
    b = manifest_row("MOVIE_A", 4, 0, 503, 500)
    labels = pd.DataFrame([label_row(a, "swimming"), label_row(b, "immotile")])

    plan = plan_reassociation(labels, manifest)
    assert plan.matched == [], plan.matched
    assert plan.ambiguous == sorted([a["clip_id"], b["clip_id"]]), plan.ambiguous
    assert plan.n_orphaned == 2
    assert apply_reassociation(labels, plan).equals(labels)


TIE_LABELS = pd.DataFrame([label_row(manifest_row("MOVIE_A", 3, 0, 100, 100))])


@pytest.mark.parametrize("left_x, resolved", [
    (90, False),     # an exact tie
    (91, False),     # one pixel is within the crop's own drift
    (96, True),      # a lead the crop could not have drifted
])
def test_one_orphan_between_two_candidates(left_x, resolved) -> None:
    plan = plan_reassociation(TIE_LABELS, make_manifest([
        manifest_row("MOVIE_A", 40, 0, left_x, 100),
        manifest_row("MOVIE_A", 41, 0, 110, 100)]))
    if resolved:
        assert len(plan.matched) == 1 and not plan.ambiguous, plan
        assert plan.matched[0].new_clip_id == clip_id("MOVIE_A", 40, 0), "the nearer clip wins"
    else:
        assert plan.matched == [] and len(plan.ambiguous) == 1, plan


LABELLED = manifest_row("MOVIE_A", 3, 0, 105, 100)
SAME_ANIMAL = manifest_row("MOVIE_A", 71, 0, 107, 100)    # 2 px of drift
NEIGHBOUR = manifest_row("MOVIE_A", 72, 0, 104, 100)      # a different animal, 1 px off


def test_a_lead_shorter_than_the_crops_own_drift_decides_nothing() -> None:
    labels = pd.DataFrame([label_row(LABELLED, "immotile")])
    plan = plan_reassociation(labels, make_manifest([SAME_ANIMAL, NEIGHBOUR]))
    assert plan.matched == [], [(m.new_clip_id, m.distance_px) for m in plan.matched]
    assert plan.ambiguous == [LABELLED["clip_id"]], plan.ambiguous
    assert plan.unmatched == []


def test_a_runner_up_beyond_the_margin_does_not_block_the_match() -> None:
    labels = pd.DataFrame([label_row(LABELLED, "immotile")])
    plan = plan_reassociation(
        labels, make_manifest([SAME_ANIMAL, manifest_row("MOVIE_A", 72, 0, 110, 100)]))
    assert len(plan.matched) == 1 and not plan.ambiguous, plan
    assert plan.matched[0].new_clip_id == SAME_ANIMAL["clip_id"]


def test_a_lone_candidate_inside_the_radius_matches() -> None:
    """The margin compares rivals; it is not a second distance threshold."""
    labels = pd.DataFrame([label_row(LABELLED, "immotile")])
    plan = plan_reassociation(labels, make_manifest([SAME_ANIMAL]))
    assert len(plan.matched) == 1 and not plan.ambiguous, plan
    assert plan.matched[0].distance_px == 2.0, plan.matched[0].distance_px


def test_a_zero_margin_refuses_only_exact_ties() -> None:
    labels = pd.DataFrame([label_row(LABELLED, "immotile")])
    plan = plan_reassociation(labels, make_manifest([SAME_ANIMAL, NEIGHBOUR]),
                              min_margin_px=0.0)
    assert len(plan.matched) == 1 and not plan.ambiguous, plan
    assert plan.matched[0].new_clip_id == NEIGHBOUR["clip_id"], "nearest wins by its pixel"


def test_a_clip_that_is_already_labelled_is_not_stolen() -> None:
    """An orphan whose clip has a newer label is a conflict, not a match or a fallback."""
    old = manifest_row("MOVIE_A", 9, 0, 200, 200)
    current = moved(old, 214, dx=1)
    other = manifest_row("MOVIE_A", 260, 0, 900, 900)
    manifest = make_manifest([current, other])
    labels = pd.DataFrame([
        label_row(old, "immotile"),                       # first session, orphaned
        label_row(current, "swimming"),                   # second session, valid
    ])

    plan = plan_reassociation(labels, manifest)
    assert plan.already_valid == 1
    assert plan.matched == [], "it does not fall through to the runner-up either"
    assert plan.conflicted == [old["clip_id"]], plan.conflicted
    assert plan.unmatched == []
    out = apply_reassociation(labels, plan)
    assert not out["clip_id"].duplicated().any()
    assert out.equals(labels)


@pytest.mark.parametrize("dx, dy, matched", [
    (0, 0, True), (5, 0, True), (12, 0, True), (13, 0, False), (400, 0, False),
    (9, 9, False),       # 12.7 px diagonally: the threshold is a radius, not a box
])
def test_the_distance_threshold_is_enforced(dx, dy, matched) -> None:
    old = manifest_row("MOVIE_A", 3, 0, 100, 100)
    plan = plan_reassociation(pd.DataFrame([label_row(old)]),
                              make_manifest([moved(old, 200, dx=dx, dy=dy)]))
    assert bool(plan.matched) == matched, plan


def test_a_wider_threshold_is_honoured() -> None:
    old = manifest_row("MOVIE_A", 3, 0, 100, 100)
    plan = plan_reassociation(pd.DataFrame([label_row(old)]),
                              make_manifest([moved(old, 200, dx=9, dy=9)]),
                              max_centre_px=20.0)
    assert len(plan.matched) == 1


def test_unmatched_labels_are_reported_not_dropped() -> None:
    """A gone movie, a gone window and a far crop are all unmatched and kept."""
    gone_movie = manifest_row("MOVIE_GONE", 3, 0, 100, 100)
    gone_window = manifest_row("MOVIE_A", 3, 500, 100, 100)
    far = manifest_row("MOVIE_A", 3, 0, 100, 100)
    keeper = manifest_row("MOVIE_A", 8, 50, 60, 60)

    manifest = make_manifest([manifest_row("MOVIE_A", 210, 0, 900, 900), keeper])
    labels = pd.DataFrame([
        label_row(gone_movie, "unclear"),
        label_row(gone_window, "immotile"),
        label_row(far, "swimming"),
        label_row(keeper, "swimming"),
    ])

    plan = plan_reassociation(labels, manifest)
    assert plan.already_valid == 1
    assert plan.unmatched == sorted([gone_movie["clip_id"], gone_window["clip_id"],
                                     far["clip_id"]]), plan.unmatched
    assert plan.n_orphaned + plan.already_valid == len(labels)
    out = apply_reassociation(labels, plan)
    assert len(out) == len(labels) and out.equals(labels)


@pytest.fixture
def two_moved():
    old = [manifest_row("MOVIE_A", 3, 0, 100, 100),
           manifest_row("MOVIE_A", 4, 0, 400, 400)]
    manifest = make_manifest([moved(old[0], 210, dx=1), moved(old[1], 211, dy=1)])
    return pd.DataFrame([label_row(c) for c in old]), manifest


def test_plan_and_apply_do_not_mutate_their_input(two_moved) -> None:
    labels, manifest = two_moved
    before = labels.copy(deep=True)

    plan = plan_reassociation(labels, manifest)
    assert labels.equals(before), "planning leaves the frame alone"
    out = apply_reassociation(labels, plan)
    assert labels.equals(before), "applying leaves the frame alone"
    assert not out.equals(before), "the output did change"
    assert out is not labels


def test_planning_is_repeatable_and_order_independent(two_moved) -> None:
    labels, manifest = two_moved
    plan = plan_reassociation(labels, manifest)
    assert pairs(plan_reassociation(labels, manifest)) == pairs(plan)
    reversed_plan = plan_reassociation(labels.iloc[::-1].reset_index(drop=True),
                                       manifest.iloc[::-1].reset_index(drop=True))
    assert pairs(reversed_plan) == pairs(plan)


def test_it_refuses_frames_it_cannot_match_on() -> None:
    manifest = make_manifest([manifest_row("MOVIE_A", 3, 0, 100, 100)])
    thin = pd.DataFrame([{"clip_id": "x", "behaviour": "swimming"}])
    with pytest.raises(KeyError, match="crop_x0"):
        plan_reassociation(thin, manifest)


def test_an_empty_csv_plans_nothing() -> None:
    manifest = make_manifest([manifest_row("MOVIE_A", 3, 0, 100, 100)])
    plan = plan_reassociation(pd.DataFrame(columns=list(manifest.columns)), manifest)
    assert plan.n_orphaned == 0 and plan.already_valid == 0


def test_it_survives_a_csv_read_as_text() -> None:
    """The GUI reads the CSV with ``dtype=str``, so coordinates arrive as strings."""
    old = manifest_row("MOVIE_A", 3, 0, 100, 100)
    manifest = make_manifest([moved(old, 217, dx=1)])
    labels = pd.DataFrame([label_row(old)]).astype(str)

    plan = plan_reassociation(labels, manifest)
    assert len(plan.matched) == 1, plan
    out = apply_reassociation(labels, plan)
    assert out["clip_id"].iat[0] == clip_id("MOVIE_A", 217, 0)
    assert int(out["track_id"].iat[0]) == 217, out["track_id"].iat[0]
    assert out["dwell_s"].iat[0] == "1.50", "the human's text is untouched"


# --------------------------------------------------------------------------- #
# writing it back
# --------------------------------------------------------------------------- #
def test_the_rewrite_is_backed_up_and_atomic(root: Path) -> None:
    from trackanno.annotate.reassociate import backup_labels, write_labels

    old = [manifest_row("MOVIE_A", 3, 0, 100, 100),
           manifest_row("MOVIE_A", 4, 0, 400, 400)]
    manifest = make_manifest([moved(old[0], 210, dx=1), moved(old[1], 211)])
    labels = pd.DataFrame([label_row(old[0], "immotile"),
                           label_row(old[1], "unclear", notes="a, b, c")])

    csv_path = root / "annotations.csv"
    write_labels(csv_path, labels)
    original = csv_path.read_bytes()

    first = backup_labels(csv_path)
    assert first.name == "annotations.csv.bak-1", first.name
    assert first.read_bytes() == original

    plan = plan_reassociation(pd.read_csv(csv_path, dtype=str, keep_default_na=False),
                              manifest)
    assert len(plan.matched) == 2, plan
    repaired = apply_reassociation(
        pd.read_csv(csv_path, dtype=str, keep_default_na=False), plan)
    write_labels(csv_path, repaired)

    assert not list(root.glob("*.csv.tmp")), list(root.glob("*"))
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert len(back) == 2
    assert set(back["clip_id"]) == set(manifest["clip_id"]), list(back["clip_id"])
    assert back["notes"].iat[1] == "a, b, c", back["notes"].iat[1]
    assert list(back.columns) == list(labels.columns)

    second = backup_labels(csv_path)
    assert second.name == "annotations.csv.bak-2", second.name
    assert first.read_bytes() == original, "the first backup is still the pre-repair file"

    # The store appends CRLF rows, so the rewrite must use CRLF too.
    assert csv_path.read_bytes().count(b"\r\n") == 3


def test_the_command_line_relinks_and_keeps_a_backup(root: Path, capsys) -> None:
    """``trackanno repair-links`` plans, reports, honours --dry-run and keeps a backup."""
    from fixtures import make_project

    from trackanno.annotate.clips import ClipStore
    from trackanno.annotate.reassociate import main, write_labels
    from trackanno.config import load_config

    config_path = make_project(root)
    cfg = load_config(config_path)
    store = ClipStore(cfg.paths.work_dir)
    try:
        current = store.manifest.head(2).to_dict("records")
    finally:
        store.close()
    orphans = [moved(entry, int(entry["track_id"]) + 900) for entry in current]
    csv_path = root / "repair-cli.csv"
    write_labels(csv_path, pd.DataFrame([label_row(o) for o in orphans]))
    original = csv_path.read_bytes()

    code = main(["--config", str(config_path), "--csv", str(csv_path), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "2 can be re-linked" in out, out
    assert csv_path.read_bytes() == original, "a dry run writes nothing"
    assert not list(root.glob("repair-cli.csv.bak-*")), "not even a backup"

    code = main(["--config", str(config_path), "--csv", str(csv_path)])
    out = capsys.readouterr().out
    assert code == 0
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert list(back["clip_id"]) == [str(e["clip_id"]) for e in current], list(back["clip_id"])
    backup = root / "repair-cli.csv.bak-1"
    assert backup.exists() and backup.read_bytes() == original
    assert backup.name in out, out


def test_the_trainer_says_how_many_it_ignored(root: Path) -> None:
    """``train_group`` reports labels whose clip_id has no feature row, per class."""
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.learn.suggest import train_group

    features = pd.DataFrame({
        "clip_id": [clip_id("MOVIE_A", 200 + i, 0) for i in range(6)],
        "movie_id": ["MOVIE_A"] * 6,
        "track_id": list(range(200, 206)),
        "f_start": [0] * 6,
        "speed_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    labels = pd.DataFrame({
        "clip_id": list(features["clip_id"]) + [clip_id("MOVIE_A", 9, 0),
                                                clip_id("MOVIE_A", 10, 0),
                                                clip_id("MOVIE_B", 11, 0)],
        "behaviour": ["swimming", "swimming", "immotile", "immotile", "swimming",
                      "immotile", "unclear", "unclear", "swimming"],
    })

    cfg = load_config(create_project(root / "trainer"))
    cfg.suggest.min_per_class = 2
    # One movie, so it stops before fitting: this is about the accounting.
    cfg.suggest.audit_fraction = 0.0
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    assert report.n_orphaned == 3, report.n_orphaned
    assert report.orphaned_by_class == {"swimming": 1, "unclear": 2}, report.orphaned_by_class
    assert report.n_labels == 6, report.n_labels

    payload = json.dumps(asdict(report))
    assert '"n_orphaned": 3' in payload
    assert '"orphaned_by_class"' in payload
