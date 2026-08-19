"""Put labels back onto their clips after the clip cache was rebuilt.

``clip_id`` embeds the track id (``MOVIE__t0018__f000000``), so re-running
detection renumbers the tracks and every existing ``clip_id`` stops resolving
while the pixels, frame range and crop rectangle are unchanged. Nothing
announces it: ``suggest.train`` inner-joins the CSV to the feature table on
``clip_id``, so orphaned rows vanish from the fit - on one real project 145 of
468 labels, and 30 of the 39 ``unclear`` ones.

A label is therefore re-linked by geometry, from the movie, absolute frame range
and crop rectangle every CSV row also carries in original pixels
(``store.IDENTITY_COLUMNS``) and which identify the animal independently of any
track numbering. Movie and f_start must match EXACTLY - windows are aligned to
the movie's frame grid, not to each track's start, so a near-match would
silently pair adjacent windows - then the nearest crop centre within
``max_centre_px``, the crop moving a pixel or two between runs but never across
the frame.

Refuse rather than guess: a mis-link is trained on and never questioned again,
while an orphan is merely ignored. No Qt and no randomness; the only I/O is at
the bottom, where a repaired frame is backed up and swapped into place.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "Match", "Reassociation", "plan_reassociation", "apply_reassociation",
    "backup_labels", "write_labels",
    "GEOMETRY_COLUMNS", "REQUIRED_COLUMNS", "DEFAULT_MAX_CENTRE_PX",
]

DEFAULT_MAX_CENTRE_PX = 12.0

# What the match itself is made of. Both frames must carry all of it.
REQUIRED_COLUMNS = ["clip_id", "movie_id", "f_start", "crop_x0", "crop_y0", "crop_w", "crop_h"]

# What a re-linked row inherits from the manifest. Rewriting clip_id alone would
# leave the row describing the OLD crop, a rectangle the new cache never cut, so
# every later consumer of those columns would read geometry that is gone.
GEOMETRY_COLUMNS = ["track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h", "scale"]

_INTEGER_GEOMETRY = {"track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h"}

# Distances come from integer pixel coordinates, so genuinely equidistant
# candidates come out bit-identical; this only absorbs the last ulp of the hypot.
_TIE_TOLERANCE = 1e-9


@dataclass
class Match:
    """One orphaned label and the clip it belongs to."""

    old_clip_id: str
    new_clip_id: str
    distance_px: float
    # The manifest's geometry for ``new_clip_id``, so applying a plan needs
    # nothing but the plan. Plain scalars, so it survives pickling to a worker.
    new_geometry: dict = field(default_factory=dict)


@dataclass
class Reassociation:
    """What can be repaired, what cannot, and why not."""

    matched: list[Match]
    unmatched: list[str]        # no candidate at all, or none close enough
    already_valid: int          # label rows whose clip_id is still in the manifest
    ambiguous: list[str]        # two candidates equally close, or two orphans one clip
    # Orphans whose clip is already labelled - what happens when someone
    # re-prepares mid-project and carries on annotating. Re-linking anyway would
    # duplicate a clip_id or overwrite the newer judgement with the older one.
    conflicted: list[str] = field(default_factory=list)

    @property
    def n_orphaned(self) -> int:
        return (len(self.matched) + len(self.unmatched)
                + len(self.ambiguous) + len(self.conflicted))

    @property
    def has_repairs(self) -> bool:
        return bool(self.matched)


def _require(df: pd.DataFrame, what: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"the {what} is missing the column(s) {missing}, which are what a "
            "label is matched on. Nothing can be re-linked without them."
        )


def _centres(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Crop centres in ORIGINAL movie pixels, the one frame both sides share."""
    x0 = pd.to_numeric(df["crop_x0"], errors="coerce").to_numpy(dtype=float)
    y0 = pd.to_numeric(df["crop_y0"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["crop_w"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["crop_h"], errors="coerce").to_numpy(dtype=float)
    return x0 + w / 2.0, y0 + h / 2.0


def _geometry_for(manifest_df: pd.DataFrame, row: int) -> dict:
    out: dict = {}
    for column in GEOMETRY_COLUMNS:
        if column not in manifest_df.columns:
            continue
        value = manifest_df[column].iloc[row]
        if pd.isna(value):
            continue
        out[column] = int(value) if column in _INTEGER_GEOMETRY else float(value)
    return out


def plan_reassociation(
    labels_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    max_centre_px: float = DEFAULT_MAX_CENTRE_PX,
) -> Reassociation:
    """Work out which orphaned labels belong to which clips. Changes nothing.

    Deterministic: the answer depends on the contents of the two frames, never
    on their row order, so the plan shown in a dialog is the plan applied.
    """
    _require(labels_df, "annotation table")
    _require(manifest_df, "clip manifest")

    manifest_ids = manifest_df["clip_id"].astype(str).to_numpy()
    known = set(manifest_ids.tolist())
    label_ids = labels_df["clip_id"].astype(str)
    valid = label_ids.isin(known)
    already_valid = int(valid.sum())

    # A clip that still carries a label of its own is spoken for; excluding it
    # at match time instead would drop the orphan onto the SECOND nearest clip,
    # which is a different animal.
    claimed = set(label_ids[valid].tolist())

    unmatched: list[str] = []
    ambiguous: list[str] = []
    conflicted: list[str] = []
    if already_valid == len(labels_df):
        return Reassociation([], [], already_valid, [], [])

    # (movie_id, f_start) -> manifest row numbers, both parts of the key
    # normalised first: the manifest stores f_start as int32, while the CSV can
    # arrive as int64 or, when read as text so untouched values round-trip byte
    # for byte, as the string "0", which no int key ever meets.
    movie = manifest_df["movie_id"].astype(str).to_numpy()
    starts = pd.to_numeric(manifest_df["f_start"], errors="coerce").to_numpy(dtype=float)
    man_cx, man_cy = _centres(manifest_df)
    buckets: dict[tuple[str, int], list[int]] = {}
    for row in range(len(manifest_df)):
        if not np.isfinite(starts[row]):
            continue
        buckets.setdefault((movie[row], int(starts[row])), []).append(row)

    orphan_rows = np.flatnonzero(~valid.to_numpy())
    orphan_movie = labels_df["movie_id"].astype(str).to_numpy()
    orphan_start = pd.to_numeric(labels_df["f_start"], errors="coerce").to_numpy(dtype=float)
    lab_cx, lab_cy = _centres(labels_df)

    # old clip_id -> (manifest row, distance). Keyed by id, so a CSV holding the
    # same orphan twice repairs both rows rather than looking like a contest.
    tentative: dict[str, tuple[int, float]] = {}
    seen: set[str] = set()
    for row in orphan_rows:
        old = label_ids.iat[int(row)]
        if old in seen:
            continue
        seen.add(old)
        if not np.isfinite(orphan_start[row]) or not np.isfinite(lab_cx[row]) \
                or not np.isfinite(lab_cy[row]):
            unmatched.append(old)
            continue
        candidates = buckets.get((orphan_movie[row], int(orphan_start[row])))
        if not candidates:
            unmatched.append(old)
            continue
        index = np.asarray(candidates, dtype=int)
        distance = np.hypot(man_cx[index] - lab_cx[row], man_cy[index] - lab_cy[row])
        best = float(np.nanmin(distance))
        if not np.isfinite(best) or best > max_centre_px:
            unmatched.append(old)
            continue
        tied = index[distance <= best + _TIE_TOLERANCE]
        if len(tied) > 1:
            # Two clips equidistant from the old crop centre: one is the animal
            # that was labelled, and nothing in the CSV says which.
            ambiguous.append(old)
            continue
        tentative[old] = (int(tied[0]), best)

    # Two orphans landing on one clip means at least one of them is wrong, and
    # the arbitration would be nothing but row order. Refuse both.
    contested = {
        manifest_ids[row] for row, count
        in Counter(row for row, _ in tentative.values()).items() if count > 1
    }

    matched: list[Match] = []
    for old, (row, distance) in tentative.items():
        new = str(manifest_ids[row])
        if new in contested:
            ambiguous.append(old)
        elif new in claimed:
            conflicted.append(old)
        else:
            matched.append(Match(old, new, round(float(distance), 3),
                                 _geometry_for(manifest_df, row)))

    matched.sort(key=lambda m: m.old_clip_id)
    return Reassociation(
        matched=matched,
        unmatched=sorted(unmatched),
        already_valid=already_valid,
        ambiguous=sorted(ambiguous),
        conflicted=sorted(conflicted),
    )


def apply_reassociation(labels_df: pd.DataFrame, plan: Reassociation) -> pd.DataFrame:
    """A NEW frame with the matched rows re-pointed at their clips.

    The input is never touched; the caller still has to back the CSV up and
    write it. Only ``clip_id`` and the geometry move - the category values,
    notes, uncertain flag, annotator, timestamps and suggestion provenance are
    the human's record of what they saw and are copied through exactly.
    """
    out = labels_df.copy(deep=True)
    if not plan.matched:
        return out

    by_old = {m.old_clip_id: m for m in plan.matched}
    ids = out["clip_id"].astype(str)
    rows = np.flatnonzero(ids.isin(by_old).to_numpy())
    if rows.size == 0:
        return out

    replacements: dict[str, list] = {"clip_id": [by_old[ids.iat[int(r)]].new_clip_id
                                                 for r in rows]}
    for column in GEOMETRY_COLUMNS:
        if column not in out.columns:
            continue
        values = [by_old[ids.iat[int(r)]].new_geometry.get(column) for r in rows]
        if any(v is None for v in values):
            # A manifest without this column: leave the old value rather than
            # writing a hole into a row that is otherwise consistent.
            continue
        replacements[column] = values

    for column, values in replacements.items():
        # Cast to the column's own dtype first: the manifest hands back Python
        # ints and floats, and a CSV read with ``dtype=str`` (as the GUI does, so
        # untouched values round-trip) holds a string column that refuses them.
        replacement = pd.Series(values)
        try:
            replacement = replacement.astype(out[column].dtype)
        except (TypeError, ValueError):
            pass
        # By POSITION, not by index label: a frame whose index repeats a label
        # would take the new clip_id on every row sharing it.
        out.iloc[rows, out.columns.get_loc(column)] = replacement.to_numpy()
    return out


# --------------------------------------------------------------------------- #
# writing it back
# --------------------------------------------------------------------------- #
def backup_labels(path: Path) -> Path:
    """Copy the annotation CSV aside, to a name no earlier backup can own.

    This is the user's primary data; a repair that got the matching wrong must
    be undoable with a file rename. Numbered rather than timestamped so repeated
    repairs stack up in order and none can land on top of another.
    """
    path = Path(path)
    n = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak-{n}")
        if not candidate.exists():
            shutil.copyfile(path, candidate)
            return candidate
        n += 1


def write_labels(path: Path, frame: pd.DataFrame) -> None:
    """Replace the CSV in a single rename, the way ``AnnotationStore`` does.

    Writing in place would truncate the annotation log if anything failed part
    way, and there is no second copy. CRLF and no index, matching the ``csv``
    module output the store appends with, or a repaired file gains mixed line
    endings the first time a label is appended.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            frame.to_csv(fh, index=False, lineterminator="\r\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
