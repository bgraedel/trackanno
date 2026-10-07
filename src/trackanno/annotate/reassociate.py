"""``trackanno repair-links``: re-link labels to clips after re-detection renumbered the tracks.

A label matches on exact movie and f_start, then the nearest crop centre; ties are refused.
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
    "backup_labels", "write_labels", "main",
    "GEOMETRY_COLUMNS", "REQUIRED_COLUMNS", "DEFAULT_MAX_CENTRE_PX",
    "DEFAULT_MIN_MARGIN_PX",
]

DEFAULT_MAX_CENTRE_PX = 12.0

# The nearest clip must beat the next by this much; crops jitter a pixel or two.
DEFAULT_MIN_MARGIN_PX = 3.0

# Columns the match uses; both tables need them.
REQUIRED_COLUMNS = ["clip_id", "movie_id", "f_start", "crop_x0", "crop_y0", "crop_w", "crop_h"]

# Columns a re-linked row takes from the manifest along with its clip_id.
GEOMETRY_COLUMNS = ["track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h", "scale"]

_INTEGER_GEOMETRY = {"track_id", "f_end", "crop_x0", "crop_y0", "crop_w", "crop_h"}

# Minimum margin, absorbing float error between equidistant candidates.
_TIE_TOLERANCE = 1e-9


@dataclass
class Match:
    """One orphaned label and the clip it belongs to."""

    old_clip_id: str
    new_clip_id: str
    distance_px: float
    # Manifest geometry for ``new_clip_id``, as plain scalars.
    new_geometry: dict = field(default_factory=dict)


@dataclass
class Reassociation:
    """What can be repaired, what cannot, and why not."""

    matched: list[Match]
    unmatched: list[str]        # no candidate at all, or none close enough
    already_valid: int          # label rows whose clip_id is still in the manifest
    ambiguous: list[str]        # two candidates too alike to separate, or two orphans one clip
    # Orphans whose clip is already labelled.
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
        raise KeyError(f"the {what} is missing the column(s) {missing} needed to match labels")


def _centres(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Crop centres in original movie pixels."""
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
    min_margin_px: float = DEFAULT_MIN_MARGIN_PX,
) -> Reassociation:
    """Plan which orphaned labels belong to which clips, independent of row order; changes nothing.

    A candidate must lead its nearest rival by ``min_margin_px``, else the orphan is ambiguous."""
    _require(labels_df, "annotation table")
    _require(manifest_df, "clip manifest")
    margin = max(float(min_margin_px), _TIE_TOLERANCE)

    manifest_ids = manifest_df["clip_id"].astype(str).to_numpy()
    known = set(manifest_ids.tolist())
    label_ids = labels_df["clip_id"].astype(str)
    valid = label_ids.isin(known)
    already_valid = int(valid.sum())

    # Labelled clips stay candidates, so an orphan never falls to the second nearest.
    claimed = set(label_ids[valid].tolist())

    unmatched: list[str] = []
    ambiguous: list[str] = []
    conflicted: list[str] = []
    if already_valid == len(labels_df):
        return Reassociation([], [], already_valid, [], [])

    # (movie_id, f_start) -> manifest rows; f_start normalised (the CSV may hold text).
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

    # old clip_id -> (manifest row, distance); duplicate rows share one entry.
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
        # Rivals within the margin make it ambiguous; NaN distances never contend.
        contenders = index[distance < best + margin]
        if len(contenders) > 1:
            ambiguous.append(old)
            continue
        tentative[old] = (int(contenders[0]), best)

    # Two orphans on one clip: refuse both.
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
    """A copy of ``labels_df`` with matched rows' ``clip_id`` and geometry updated."""
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
            # The manifest lacks this column: keep the old value.
            continue
        replacements[column] = values

    for column, values in replacements.items():
        # Cast to the column's dtype (the CSV may be read as text).
        replacement = pd.Series(values)
        try:
            replacement = replacement.astype(out[column].dtype)
        except (TypeError, ValueError):
            pass
        # By position, since the index may repeat labels.
        out.iloc[rows, out.columns.get_loc(column)] = replacement.to_numpy()
    return out


# --------------------------------------------------------------------------- #
# writing it back
# --------------------------------------------------------------------------- #
def backup_labels(path: Path) -> Path:
    """Copy the annotation CSV to the next free ``.bak-N`` name."""
    path = Path(path)
    n = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak-{n}")
        if not candidate.exists():
            shutil.copyfile(path, candidate)
            return candidate
        n += 1


def write_labels(path: Path, frame: pd.DataFrame) -> None:
    """Replace the CSV atomically, with CRLF line endings like ``AnnotationStore``."""
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


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def _describe(plan: Reassociation) -> list[str]:
    lines = [
        f"{plan.already_valid} label(s) already point at clips that exist.",
        f"{plan.n_orphaned} point at clips that no longer exist:",
        f"  {len(plan.matched)} can be re-linked by movie, frame range and crop position",
    ]
    if plan.conflicted:
        lines.append(f"  {len(plan.conflicted)} match a clip already labelled since (left alone)")
    if plan.ambiguous:
        lines.append(f"  {len(plan.ambiguous)} are ambiguous, two clips fit equally well (left alone)")
    if plan.unmatched:
        lines.append(f"  {len(plan.unmatched)} have no candidate at all (left alone)")
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from ..config import ConfigError, load_config
    from .clips import ClipStore
    from .store import CsvLockedError, csv_lock

    parser = argparse.ArgumentParser(
        prog="trackanno repair-links",
        description="Re-link labels whose clip no longer exists onto the current clips.",
    )
    parser.add_argument("--config", default="config.yaml",
                        help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--csv", default=None, help="override paths.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    csv_path = Path(args.csv).resolve() if args.csv else Path(cfg.paths.csv)
    if not csv_path.exists():
        print(f"error: no annotation file at {csv_path}", file=sys.stderr)
        return 2
    try:
        store = ClipStore(cfg.paths.work_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        manifest = store.manifest.copy()
    finally:
        store.close()

    # Held from read to write so an open GUI cannot append in between.
    try:
        with csv_lock(csv_path):
            # dtype=str with NA disabled: untouched values must round-trip exactly.
            labels = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            plan = plan_reassociation(labels, manifest)
            if not plan.n_orphaned:
                print(f"All {plan.already_valid} label(s) already point at clips that exist.")
                return 0
            print("\n".join(_describe(plan)))
            if not plan.matched or args.dry_run:
                return 0
            backup = backup_labels(csv_path)
            write_labels(csv_path, apply_reassociation(labels, plan))
    except CsvLockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{len(plan.matched)} label(s) re-linked; backup at {backup}. Retrain to use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
