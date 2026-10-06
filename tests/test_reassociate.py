"""Putting labels back onto their clips after the cache was rebuilt.

``clip_id`` embeds the track id, so re-running detection renumbers the tracks
and orphans every label written before it. ``suggest.train`` inner-joins on
``clip_id`` and drops those rows without a word: on one real project that was
145 of 468 labels and 30 of the 39 ``unclear`` ones, which read as a rare-class
modelling problem rather than as data loss.

The stakes decide the tests. A label re-linked onto the WRONG animal is worse
than one left orphaned - the orphan is merely ignored, while the mis-link is
trained on and never questioned again - so most of what follows checks that the
matcher refuses rather than that it matches. Everything here is synthetic: no
project data, no config, no display.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from harness import check


# --------------------------------------------------------------------------- #
# synthetic frames
# --------------------------------------------------------------------------- #
def clip_id(movie: str, track: int, f_start: int) -> str:
    """Exactly the format clips.plan_clips writes."""
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
    """A CSV row for the clip described by a manifest-shaped dict.

    Carries the same identity columns ``store.IDENTITY_COLUMNS`` writes, plus
    the human's own fields - which are the whole point of the exercise and must
    come through a repair untouched.
    """
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


# --------------------------------------------------------------------------- #
def test_valid_ids_are_left_completely_alone() -> None:
    """The common case is that nothing is wrong. A repair that rewrites rows it
    did not need to touch is a repair nobody can safely run twice."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    clips = [manifest_row("MOVIE_A", 3, 0, 100, 100),
             manifest_row("MOVIE_A", 4, 0, 300, 300),
             manifest_row("MOVIE_B", 3, 50, 10, 10)]
    manifest = make_manifest(clips)
    labels = pd.DataFrame([label_row(c) for c in clips])

    plan = plan_reassociation(labels, manifest)
    check("every valid label counts as already valid", plan.already_valid == 3,
          str(plan.already_valid))
    check("...and nothing is proposed", not plan.matched and not plan.unmatched
          and not plan.ambiguous and not plan.conflicted)
    check("...so there is nothing to repair", not plan.has_repairs)

    out = apply_reassociation(labels, plan)
    check("applying an empty plan changes no value", out.equals(labels))
    check("...and still hands back a separate frame", out is not labels)


def test_a_retracked_movie_is_relinked() -> None:
    """The bug itself: same movie, same frame window, same animal, new track id.
    The crop centre drifts by a pixel or two because the tracker's boxes moved,
    which is precisely why the match is nearest-centre and not equality."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    old = [manifest_row("MOVIE_A", 2, 0, 100, 100),
           manifest_row("MOVIE_A", 5, 0, 400, 220),
           manifest_row("MOVIE_A", 9, 50, 80, 640)]
    # Re-prepared: every track renumbered, every crop a pixel or two off.
    new = [moved(old[0], 217, dx=1), moved(old[1], 231, dy=-2),
           moved(old[2], 244, dx=2, dy=1)]
    manifest = make_manifest(new)
    labels = pd.DataFrame([
        label_row(old[0], "immotile"),
        label_row(old[1], "swimming"),
        label_row(old[2], "unclear"),
    ])

    plan = plan_reassociation(labels, manifest)
    check("no label is still valid", plan.already_valid == 0)
    check("all three are re-linked", len(plan.matched) == 3,
          f"{len(plan.matched)} matched, {len(plan.unmatched)} unmatched")
    check("...onto the right clips",
          [m.new_clip_id for m in plan.matched] == sorted(c["clip_id"] for c in new))
    check("...and the plan says how far each moved",
          all(0.0 <= m.distance_px <= 3.0 for m in plan.matched),
          str([m.distance_px for m in plan.matched]))

    out = apply_reassociation(labels, plan)
    check("the CSV now points at clips that exist",
          bool(out["clip_id"].isin(set(manifest["clip_id"])).all()))
    # The geometry has to move with the id, or the row describes a rectangle the
    # new cache never cut and anything mapping it back onto the movie is wrong.
    by_id = manifest.set_index("clip_id")
    for _, row in out.iterrows():
        want = by_id.loc[row["clip_id"]]
        for column in ("track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h", "scale"):
            check(f"{column} follows the new clip ({row['clip_id'][-12:]})",
                  row[column] == want[column], f"{row[column]} vs {want[column]}")
    check("the frame window is untouched, it is the match key",
          list(out["f_start"]) == list(labels["f_start"]))


def test_the_humans_columns_survive_untouched() -> None:
    """The reason the labels are worth rescuing at all. Everything the annotator
    typed or decided is copied through byte for byte; only the machine-generated
    identity moves."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

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
        check(f"{column} is copied through exactly",
              out[column].iat[0] == labels[column].iat[0],
              f"{out[column].iat[0]!r} vs {labels[column].iat[0]!r}")
    check("only the clip_id changed", out["clip_id"].iat[0] != labels["clip_id"].iat[0])
    check("the column order is preserved for the rewrite",
          list(out.columns) == list(labels.columns))


def test_two_orphans_on_one_clip_are_both_refused() -> None:
    """Whichever one wins, the other is silently attached to an animal it never
    described. There is nothing in the CSV that says which is which, so the only
    honest answer is to refuse both and say so."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    survivor = manifest_row("MOVIE_A", 88, 0, 500, 500)
    manifest = make_manifest([manifest_row("MOVIE_A", 12, 0, 100, 100), survivor])
    # Two old clips whose crops sat either side of the surviving one, both well
    # inside the threshold. This is a merged track, or a split one.
    a = manifest_row("MOVIE_A", 3, 0, 496, 500)
    b = manifest_row("MOVIE_A", 4, 0, 503, 500)
    labels = pd.DataFrame([label_row(a, "swimming"), label_row(b, "immotile")])

    plan = plan_reassociation(labels, manifest)
    check("neither is matched", plan.matched == [], str(plan.matched))
    check("both are reported as ambiguous",
          plan.ambiguous == sorted([a["clip_id"], b["clip_id"]]), str(plan.ambiguous))
    check("and they are counted as orphans", plan.n_orphaned == 2)
    out = apply_reassociation(labels, plan)
    check("the CSV is left exactly as it was", out.equals(labels))

    # The same refusal from the other direction: one orphan, two candidates the
    # very same distance away.
    tie_manifest = make_manifest([manifest_row("MOVIE_A", 40, 0, 90, 100),
                                  manifest_row("MOVIE_A", 41, 0, 110, 100)])
    tie_labels = pd.DataFrame([label_row(manifest_row("MOVIE_A", 3, 0, 100, 100))])
    tie = plan_reassociation(tie_labels, tie_manifest)
    check("two equally close candidates are refused too",
          tie.matched == [] and len(tie.ambiguous) == 1, str(tie))

    # Nudge one candidate a single pixel closer and it is still refused: a pixel
    # is what the crop moves by anyway, so that is not what broke the tie.
    nudged = plan_reassociation(
        tie_labels,
        make_manifest([manifest_row("MOVIE_A", 40, 0, 91, 100),
                       manifest_row("MOVIE_A", 41, 0, 110, 100)]),
    )
    check("one pixel of difference does not resolve it",
          nudged.matched == [] and len(nudged.ambiguous) == 1, str(nudged))

    # A lead wider than that movement does resolve it. Stated so that "ambiguous"
    # is known to mean too close to call, not merely crowded.
    broken = plan_reassociation(
        tie_labels,
        make_manifest([manifest_row("MOVIE_A", 40, 0, 96, 100),
                       manifest_row("MOVIE_A", 41, 0, 110, 100)]),
    )
    check("a lead the crop could not have drifted resolves it",
          len(broken.matched) == 1 and not broken.ambiguous,
          str(broken.matched))
    check("...in favour of the nearer clip",
          broken.matched[0].new_clip_id == clip_id("MOVIE_A", 40, 0))


def test_a_lead_shorter_than_the_crops_own_drift_decides_nothing() -> None:
    """The premise the whole module rests on is that a crop moves a pixel or two
    between runs. A winner picked by one pixel is therefore picked by that drift,
    and with a second animal nearby the drift hands the label to the wrong one:
    here the labelled animal's own crop moved 2 px while a neighbour sits 1 px
    from where it used to be, and nearest-wins would re-link onto the neighbour,
    confidently and for good."""
    from trackanno.annotate.reassociate import plan_reassociation

    labelled = manifest_row("MOVIE_A", 3, 0, 105, 100)
    same_animal = manifest_row("MOVIE_A", 71, 0, 107, 100)   # 2 px of drift
    neighbour = manifest_row("MOVIE_A", 72, 0, 104, 100)     # a different animal
    labels = pd.DataFrame([label_row(labelled, "immotile")])

    plan = plan_reassociation(labels, make_manifest([same_animal, neighbour]))
    check("the closer clip does not win by a pixel", plan.matched == [],
          str([(m.new_clip_id, m.distance_px) for m in plan.matched]))
    check("...the orphan is reported as ambiguous",
          plan.ambiguous == [labelled["clip_id"]], str(plan.ambiguous))
    check("...and is not quietly filed as unmatched instead", plan.unmatched == [])

    # 2 px and 5 px from the old centre: a 3 px lead, which the crop cannot have
    # drifted, so the nearest is a real answer and the runner-up does not block it.
    separated = plan_reassociation(
        labels,
        make_manifest([same_animal, manifest_row("MOVIE_A", 72, 0, 110, 100)]),
    )
    check("a runner-up beyond the margin does not block the match",
          len(separated.matched) == 1 and not separated.ambiguous, str(separated))
    check("...and the nearest clip is the one taken",
          separated.matched[0].new_clip_id == clip_id("MOVIE_A", 71, 0))

    # The margin compares rivals; it is not a second threshold on the distance.
    alone = plan_reassociation(labels, make_manifest([same_animal]))
    check("a lone candidate inside the radius still matches",
          len(alone.matched) == 1 and not alone.ambiguous, str(alone))
    check("...at the distance the plan reports", alone.matched[0].distance_px == 2.0,
          str(alone.matched[0].distance_px))

    # Turned off, only a bit-exact tie refuses - the caller gets nearest-wins back
    # rather than an argument the code cannot decide by itself.
    off = plan_reassociation(labels, make_manifest([same_animal, neighbour]),
                             min_margin_px=0.0)
    check("min_margin_px=0 falls back to refusing exact ties only",
          len(off.matched) == 1 and not off.ambiguous, str(off))
    check("...and then the nearest clip wins by its pixel",
          off.matched[0].new_clip_id == clip_id("MOVIE_A", 72, 0))


def test_a_clip_that_is_already_labelled_is_not_stolen() -> None:
    """Re-prepare mid-project, carry on annotating, and some of the orphans now
    describe clips that have a NEWER label. Re-linking would either duplicate a
    clip_id or overwrite the more recent judgement with the older one. This is
    not hypothetical: it accounted for 59 of the 145 orphans on the project that
    prompted the whole exercise."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    old = manifest_row("MOVIE_A", 9, 0, 200, 200)
    current = moved(old, 214, dx=1)
    other = manifest_row("MOVIE_A", 260, 0, 900, 900)
    manifest = make_manifest([current, other])
    labels = pd.DataFrame([
        label_row(old, "immotile"),                       # first session, orphaned
        label_row(current, "swimming"),                   # second session, valid
    ])

    plan = plan_reassociation(labels, manifest)
    check("the newer label is recognised as valid", plan.already_valid == 1)
    check("the orphan is not matched onto it", plan.matched == [])
    check("it is reported as a conflict", plan.conflicted == [old["clip_id"]],
          str(plan.conflicted))
    check("it is NOT silently called unmatched", plan.unmatched == [])
    out = apply_reassociation(labels, plan)
    check("no duplicate clip_id is created", not out["clip_id"].duplicated().any())
    check("the newer label is untouched", out.equals(labels))

    # And the orphan does not fall through onto the next-nearest clip either -
    # that clip is a different animal 900 px away, and a fallback would be the
    # worst possible outcome: a confident, wrong, permanent re-link.
    check("it does not fall through to the runner-up",
          other["clip_id"] not in [m.new_clip_id for m in plan.matched])


def test_the_distance_threshold_is_enforced() -> None:
    """A crop centre moves a pixel or two between runs. It does not move across
    the frame, so anything far away is a different animal - and the threshold is
    the only thing standing between a repair and a fabricated label."""
    from trackanno.annotate.reassociate import plan_reassociation

    old = manifest_row("MOVIE_A", 3, 0, 100, 100)
    for offset, expected in ((0, True), (5, True), (12, True), (13, False), (400, False)):
        manifest = make_manifest([moved(old, 200, dx=offset)])
        plan = plan_reassociation(pd.DataFrame([label_row(old)]), manifest)
        check(f"{offset} px away is {'matched' if expected else 'refused'}",
              bool(plan.matched) == expected, str(plan))

    # Diagonal, so the threshold is a radius rather than a bounding box.
    manifest = make_manifest([moved(old, 200, dx=9, dy=9)])   # 12.7 px
    plan = plan_reassociation(pd.DataFrame([label_row(old)]), manifest)
    check("the threshold is a radius, not a box", not plan.matched, str(plan.matched))

    # The caller can widen it, and doing so must not change anything else.
    plan = plan_reassociation(pd.DataFrame([label_row(old)]), manifest, max_centre_px=20.0)
    check("a wider threshold is honoured", len(plan.matched) == 1)


def test_unmatched_labels_are_reported_not_dropped() -> None:
    """A clip that genuinely no longer exists - the track fell below the length
    cut, the window moved, the movie was removed. The label cannot be repaired,
    and the one thing that must not happen is for it to disappear quietly a
    second time."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

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
    check("the still-valid label is counted", plan.already_valid == 1)
    check("all three orphans are reported",
          plan.unmatched == sorted([gone_movie["clip_id"], gone_window["clip_id"],
                                    far["clip_id"]]),
          str(plan.unmatched))
    check("the totals add up to every orphan",
          plan.n_orphaned + plan.already_valid == len(labels))
    out = apply_reassociation(labels, plan)
    check("an unrepairable label is still in the CSV afterwards",
          len(out) == len(labels) and out.equals(labels))

    # A movie present in the manifest but with no window at that frame is the
    # same answer as a movie that is not there at all: no candidate.
    check("a missing window is unmatched, not matched to a neighbouring window",
          gone_window["clip_id"] in plan.unmatched)


def test_apply_does_not_mutate_its_input() -> None:
    """The caller still has to back the CSV up and write it. If the plan were
    applied in place there would be nothing left to compare against, and a write
    that failed half way would leave the frame in memory already rewritten."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    old = [manifest_row("MOVIE_A", 3, 0, 100, 100),
           manifest_row("MOVIE_A", 4, 0, 400, 400)]
    manifest = make_manifest([moved(old[0], 210, dx=1), moved(old[1], 211, dy=1)])
    labels = pd.DataFrame([label_row(c) for c in old])
    before = labels.copy(deep=True)

    plan = plan_reassociation(labels, manifest)
    out = apply_reassociation(labels, plan)
    check("the input frame is byte for byte what it was", labels.equals(before))
    check("...and the output really did change", not out.equals(before))
    check("...and they are not the same object", out is not labels)

    # Planning must not touch it either, and must be repeatable: the dialog the
    # user confirms has to describe the plan that is then applied.
    again = plan_reassociation(labels, manifest)
    check("planning twice gives the same answer",
          [(m.old_clip_id, m.new_clip_id) for m in again.matched]
          == [(m.old_clip_id, m.new_clip_id) for m in plan.matched])
    check("planning leaves the frame alone", labels.equals(before))

    # Row order must not decide the outcome either.
    shuffled = labels.iloc[::-1].reset_index(drop=True)
    reversed_plan = plan_reassociation(shuffled, manifest.iloc[::-1].reset_index(drop=True))
    check("row order does not change the answer",
          [(m.old_clip_id, m.new_clip_id) for m in reversed_plan.matched]
          == [(m.old_clip_id, m.new_clip_id) for m in plan.matched])


def test_it_refuses_frames_it_cannot_match_on() -> None:
    """A CSV without the identity columns cannot be repaired by any means. Say
    so, rather than matching on whatever happens to be present."""
    from trackanno.annotate.reassociate import plan_reassociation

    manifest = make_manifest([manifest_row("MOVIE_A", 3, 0, 100, 100)])
    thin = pd.DataFrame([{"clip_id": "x", "behaviour": "swimming"}])
    try:
        plan_reassociation(thin, manifest)
    except KeyError as exc:
        check("a CSV without crop geometry is refused with a reason",
              "crop_x0" in str(exc), str(exc))
    else:
        check("a CSV without crop geometry is refused", False)

    # An empty annotation table is a legitimate state, not an error.
    empty = pd.DataFrame(columns=list(manifest.columns))
    plan = plan_reassociation(empty, manifest)
    check("an empty CSV plans nothing", plan.n_orphaned == 0 and plan.already_valid == 0)


def test_it_survives_a_csv_read_as_text() -> None:
    """The GUI reads the CSV with ``dtype=str`` so that untouched values round
    trip byte for byte into the rewritten file. Every coordinate therefore
    arrives as a string, and the matcher has to cope."""
    from trackanno.annotate.reassociate import apply_reassociation, plan_reassociation

    old = manifest_row("MOVIE_A", 3, 0, 100, 100)
    manifest = make_manifest([moved(old, 217, dx=1)])
    labels = pd.DataFrame([label_row(old)]).astype(str)

    plan = plan_reassociation(labels, manifest)
    check("a text CSV still matches", len(plan.matched) == 1, str(plan))
    out = apply_reassociation(labels, plan)
    check("...and the repaired id is right",
          out["clip_id"].iat[0] == clip_id("MOVIE_A", 217, 0))
    check("...and the geometry came across", int(out["track_id"].iat[0]) == 217,
          repr(out["track_id"].iat[0]))
    check("...without disturbing the human's text",
          out["dwell_s"].iat[0] == "1.50", repr(out["dwell_s"].iat[0]))


# --------------------------------------------------------------------------- #
# writing it back
# --------------------------------------------------------------------------- #
def test_the_rewrite_is_backed_up_and_atomic(root: Path) -> None:
    """This overwrites the user's primary data - hours of judgement with no
    other copy - so the old file has to survive under a name of its own, and the
    new one has to appear whole or not at all."""
    from trackanno.annotate.reassociate import (
        apply_reassociation, backup_labels, plan_reassociation, write_labels,
    )

    old = [manifest_row("MOVIE_A", 3, 0, 100, 100),
           manifest_row("MOVIE_A", 4, 0, 400, 400)]
    manifest = make_manifest([moved(old[0], 210, dx=1), moved(old[1], 211)])
    labels = pd.DataFrame([label_row(old[0], "immotile"),
                           label_row(old[1], "unclear", notes="a, b, c")])

    csv_path = root / "annotations.csv"
    write_labels(csv_path, labels)
    original = csv_path.read_bytes()

    first = backup_labels(csv_path)
    check("the backup is numbered from the CSV's own name",
          first.name == "annotations.csv.bak-1", first.name)
    check("...and is an exact copy", first.read_bytes() == original)

    plan = plan_reassociation(pd.read_csv(csv_path, dtype=str, keep_default_na=False),
                              manifest)
    check("the round trip through the file matched both", len(plan.matched) == 2,
          str(plan))
    repaired = apply_reassociation(
        pd.read_csv(csv_path, dtype=str, keep_default_na=False), plan)
    write_labels(csv_path, repaired)

    check("no temporary file is left behind",
          not list(root.glob("*.csv.tmp")), str(list(root.glob("*"))))
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    check("the rewritten file reads back", len(back) == 2)
    check("...pointing at the new clips",
          set(back["clip_id"]) == set(manifest["clip_id"]), str(list(back["clip_id"])))
    check("...with the note containing commas intact",
          back["notes"].iat[1] == "a, b, c", repr(back["notes"].iat[1]))
    check("...and the same columns in the same order",
          list(back.columns) == list(labels.columns))

    # A second repair must not land on top of the first backup.
    second = backup_labels(csv_path)
    check("a second backup takes the next number",
          second.name == "annotations.csv.bak-2", second.name)
    check("...and the first one is still the pre-repair file",
          first.read_bytes() == original)

    # The store appends with the csv module, which writes CRLF. A repaired file
    # with bare newlines would gain mixed line endings on the next annotation.
    check("line endings match what the annotation store appends",
          csv_path.read_bytes().count(b"\r\n") == 3,
          str(csv_path.read_bytes().count(b"\r\n")))


def test_the_command_line_relinks_and_keeps_a_backup(root: Path, capsys) -> None:
    """``trackanno repair-links`` is the only way in, so it has to plan, report,
    honour --dry-run and leave the old file beside the new one."""
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
    # The same animals under track ids the cache no longer has.
    orphans = [moved(entry, int(entry["track_id"]) + 900) for entry in current]
    csv_path = root / "repair-cli.csv"
    write_labels(csv_path, pd.DataFrame([label_row(o) for o in orphans]))
    original = csv_path.read_bytes()

    code = main(["--config", str(config_path), "--csv", str(csv_path), "--dry-run"])
    out = capsys.readouterr().out
    check("a dry run succeeds", code == 0, str(code))
    check("...says what it would re-link", "2 can be re-linked" in out, out)
    check("...and writes nothing", csv_path.read_bytes() == original)
    check("...not even a backup", not list(root.glob("repair-cli.csv.bak-*")))

    code = main(["--config", str(config_path), "--csv", str(csv_path)])
    out = capsys.readouterr().out
    check("the real run succeeds", code == 0, str(code))
    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    check("the labels point at the current clips",
          list(back["clip_id"]) == [str(e["clip_id"]) for e in current],
          str(list(back["clip_id"])))
    backup = root / "repair-cli.csv.bak-1"
    check("the previous file is kept as a numbered .bak",
          backup.exists() and backup.read_bytes() == original)
    check("...and the output says where", backup.name in out, out)


def test_the_trainer_says_how_many_it_ignored(root: Path) -> None:
    """The whole point of the exercise. The inner join in ``train_group`` used
    to drop orphaned labels in silence; the count and its per-class breakdown
    now ride in the report, because a loss concentrated in one class is what
    actually destroys that class."""
    from trackanno.learn.suggest import GroupReport

    fields = GroupReport.__dataclass_fields__
    check("the report carries an orphan count", "n_orphaned" in fields)
    check("...and a per-class breakdown", "orphaned_by_class" in fields)

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

    # A default project config, generated rather than read from the repository:
    # a suite that loads the developer's own config.yaml tests their data.
    from trackanno.config import load_config
    from trackanno.project import create_project
    from trackanno.learn.suggest import train_group

    cfg = load_config(create_project(root / "trainer"))
    cfg.suggest.min_per_class = 2
    # One movie, so it bails before fitting anything: this test is about the
    # accounting, and a real CatBoost fit here would only add seconds and
    # dependencies to a check that has nothing to do with the model.
    cfg.suggest.audit_fraction = 0.0
    _, report = train_group(features, labels, "behaviour", cfg,
                            columns=["speed_mean"], verbose=False)
    check("it counts the labels that never reached the fit", report.n_orphaned == 3,
          str(report.n_orphaned))
    check("...and names the classes they came from",
          report.orphaned_by_class == {"swimming": 1, "unclear": 2},
          str(report.orphaned_by_class))
    check("...leaving the labels that did join intact", report.n_labels == 6,
          str(report.n_labels))

    # asdict() of this goes straight into suggest_report.json, so anything that
    # is not a plain int or str would break the write with a TypeError.
    import json
    from dataclasses import asdict

    payload = json.dumps(asdict(report))
    check("the new fields survive the JSON report", '"n_orphaned": 3' in payload)
    check("...including the breakdown", '"orphaned_by_class"' in payload)
