"""Margin selection wired into a project: config, candidate pool and batch, with a stub model."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackanno.config import ConfigError, Suggest, load_config
from trackanno.project import create_project, write_config

MOVIES = ("movieA", "movieB")
TRACKS = 5
WINDOWS = 20
FRAMES = 50                      # clips.frames in the default config
GROUP = "behaviour"              # the default project's first single-choice group
CLASSES = ("immotile", "swimming")
MARGIN = {"suggest.select_rule": "margin"}


# --------------------------------------------------------------------------- #
# a project, a manifest, and a model that is not a fit
# --------------------------------------------------------------------------- #
class StubModel:
    """Posterior ``[u, 1 - u]`` read off the first feature column, so the ranking is known."""

    classes_ = np.array(CLASSES)

    def predict_proba(self, X):
        u = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([u, 1.0 - u])


def hand_margin(u: float) -> float:
    """``1 - (p1 - p2)`` for two classes, written out from the definition."""
    return 1.0 - abs(2.0 * float(np.float32(u)) - 1.0)


def build_manifest() -> pd.DataFrame:
    """Two movies whose track ids and frames both start at 0, as in a real project."""
    rows = []
    for movie in MOVIES:
        for track in range(TRACKS):
            for window in range(WINDOWS):
                start = window * FRAMES
                rows.append({
                    "clip_id": f"{movie}__t{track:04d}__f{start:06d}",
                    "movie_id": movie,
                    "track_id": track,
                    "f_start": start,
                    "f_end": start + FRAMES,
                })
    return pd.DataFrame(rows)


def _project(root: Path, name: str, **settings):
    """A loadable project with nothing in it but a config."""
    project = Path(root) / name
    config_path = create_project(project)
    updates = {
        "paths.work_dir": str(project / "work").replace("\\", "/"),
        "paths.csv": str(project / "annotations.csv").replace("\\", "/"),
    }
    updates.update(settings)
    write_config(config_path, updates)
    cfg = load_config(config_path)
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _install_model(cfg, manifest, certainty, trained_window: int | None = FRAMES,
                   missing: set | None = None):
    """Write the feature table and model ``Suggester.load`` needs; ``missing`` clips get no row."""
    import joblib

    from trackanno.measure.clipfeatures import CLIP_FEATURES_FILE, FEATURE_VERSION
    from trackanno.learn.suggest import MODEL_FILE

    work = cfg.paths.work_dir
    ids = [c for c in manifest["clip_id"].astype(str) if c not in (missing or set())]
    pd.DataFrame({
        "clip_id": ids,
        "certainty": [float(certainty[c]) for c in ids],
        "filler": np.zeros(len(ids)),
        "_feature_version": FEATURE_VERSION,
    }).to_parquet(work / CLIP_FEATURES_FILE, index=False)

    payload = {
        "columns": ["certainty", "filler"],
        "models": {GROUP: StubModel()},
        "feature_version": FEATURE_VERSION,
    }
    if trained_window is not None:
        payload["trained_window"] = int(trained_window)
    joblib.dump(payload, work / MODEL_FILE)
    return work


def spread_certainty(manifest, contested: str | None = None) -> dict:
    """A distinct posterior per clip in (0.05, 0.95); ``contested`` alone sits at 0.5."""
    ids = manifest["clip_id"].astype(str).tolist()
    values = dict(zip(ids, np.linspace(0.05, 0.95, len(ids))))
    if contested is not None:
        values[contested] = 0.5
    return values


def _suggester(cfg):
    from trackanno.learn.suggest import Suggester

    suggester = Suggester(cfg)
    assert suggester.ready, suggester.problem
    return suggester


def _is_audit(clip_id: str, audit_seed: int = 12345, fraction: float = 0.10) -> bool:
    """``AuditHoldout`` for the default seeds, without a Config."""
    digest = hashlib.blake2b(f"{audit_seed}:{clip_id}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % 10_000) < fraction * 10_000


# --------------------------------------------------------------------------- #
# the default: nothing happens
# --------------------------------------------------------------------------- #
def test_the_default_config_selects_nothing(root: Path) -> None:
    """Only the rule turns selection on; a ready model and any batch size do not."""
    manifest = build_manifest()
    cfg = _project(root, "default")
    _install_model(cfg, manifest, spread_certainty(manifest))
    suggester = _suggester(cfg)

    assert cfg.suggest.select_rule == "off", cfg.suggest.select_rule
    assert cfg.suggest.selection_on is False
    assert all(suggester.selection_batch(manifest, set(), n) == []
               for n in (1, 2, 5, 20, 100)), "no batch size produces a batch"
    assert cfg.ui.shuffle is True, "the queue order stays the shuffle"

    write_config(cfg.path, MARGIN)
    turned_on = _suggester(load_config(cfg.path))
    assert len(turned_on.selection_batch(manifest, set(), 20)) == 20, (
        "setting the rule alone turns selection on")


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def margin_batch(root: Path):
    """A batch of 20 with 40 clips done and one uniquely contested clip."""
    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    done = set(ids[:40])
    contested = next(c for c in ids if c not in done and not _is_audit(c))
    certainty = spread_certainty(manifest, contested=contested)
    cfg = _project(root, "batch", **MARGIN)
    _install_model(cfg, manifest, certainty)
    batch = _suggester(cfg).selection_batch(manifest, done, 20)
    return dict(cfg=cfg, ids=ids, done=done, contested=contested,
                certainty=certainty, batch=batch)


def test_a_rule_returns_a_full_ordered_batch_with_a_reason_per_clip(margin_batch) -> None:
    batch, done = margin_batch["batch"], margin_batch["done"]
    assert len(batch) == 20, len(batch)
    assert len({r["clip_id"] for r in batch}) == 20, "no clip appears twice"
    assert not ({r["clip_id"] for r in batch} & done), "nothing annotated is handed back"
    assert [r["rank"] for r in batch] == list(range(20))
    assert all(r["rule"] == "margin" for r in batch), {r["rule"] for r in batch}
    assert all(r["group"] == GROUP for r in batch)
    assert all(r["predicted_class"] in CLASSES and 0.0 <= r["predicted_proba"] <= 1.0
               for r in batch)


def test_every_batch_record_can_be_written_into_the_csv(margin_batch) -> None:
    from trackanno.annotate.store import decode_provenance, encode_provenance

    batch = margin_batch["batch"]
    encoded = [encode_provenance(r) for r in batch]
    assert all(encoded)
    assert [decode_provenance(cell)["rule"] for cell in encoded] == [r["rule"] for r in batch]


def test_the_batch_is_the_most_contested_clips_in_the_pool(margin_batch) -> None:
    from trackanno.learn.suggest import AuditHoldout

    batch, certainty = margin_batch["batch"], margin_batch["certainty"]
    assert batch[0]["clip_id"] == margin_batch["contested"], batch[0]["clip_id"]
    assert all(a["score"] >= b["score"] for a, b in zip(batch, batch[1:])), (
        [r["score"] for r in batch])
    assert all(abs(r["score"] - hand_margin(certainty[r["clip_id"]])) < 1e-9 for r in batch), (
        "every score is the margin of the clip's own posterior")

    chosen = {r["clip_id"] for r in batch}
    holdout = AuditHoldout.by_hash(margin_batch["cfg"])
    rest = [hand_margin(certainty[c]) for c in margin_batch["ids"]
            if c not in chosen and c not in margin_batch["done"] and c not in holdout]
    assert min(r["score"] for r in batch) >= max(rest), (
        "no unselected candidate is more contested than a selected one")


@pytest.fixture(scope="module")
def margin_project(root: Path):
    """A margin project over the full manifest; read-only for the tests that share it."""
    manifest = build_manifest()
    cfg = _project(root, "shared", **MARGIN)
    _install_model(cfg, manifest, spread_certainty(manifest))
    return cfg, manifest, _suggester(cfg)


def test_the_batch_does_not_depend_on_the_order_the_manifest_arrived_in(
    margin_project
) -> None:
    _, manifest, suggester = margin_project
    forward = suggester.selection_batch(manifest, set(), 15)
    shuffled = manifest.sample(frac=1.0, random_state=7).reset_index(drop=True)
    backwards = suggester.selection_batch(shuffled, set(), 15)

    assert [r["clip_id"] for r in forward] == [r["clip_id"] for r in backwards]
    assert ([(r["rule"], r["score"]) for r in forward]
            == [(r["rule"], r["score"]) for r in backwards])


def test_audit_clips_are_never_ranked(margin_project) -> None:
    from trackanno.learn.suggest import AuditHoldout

    cfg, manifest, suggester = margin_project
    ids = manifest["clip_id"].astype(str).tolist()
    holdout = AuditHoldout.by_hash(cfg)
    assert any(c in holdout for c in ids), "the project has audit clips"
    batch = suggester.selection_batch(manifest, set(), len(ids))
    assert not any(r["clip_id"] in holdout for r in batch)
    assert len(batch) == sum(c not in holdout for c in ids), "every other clip is ranked"


def test_a_batch_near_the_end_of_a_project_shrinks_instead_of_refusing(
    margin_project
) -> None:
    _, manifest, suggester = margin_project
    ids = manifest["clip_id"].astype(str).tolist()

    tail = suggester.selection_batch(manifest, set(ids[:-3]), 20)
    assert 0 < len(tail) <= 3, len(tail)
    assert {r["clip_id"] for r in tail} <= set(ids[-3:])
    assert suggester.selection_batch(manifest, set(ids), 20) == []
    assert suggester.selection_batch(manifest.iloc[:0], set(), 20) == []


def test_a_clip_with_no_feature_row_is_excluded_rather_than_scored(root: Path) -> None:
    """After re-detection a clip may lack a feature row; it is left out of the pool."""
    manifest = build_manifest()
    ids = manifest["clip_id"].astype(str).tolist()
    absent = set(ids[::3])
    cfg = _project(root, "orphans", **MARGIN)
    _install_model(cfg, manifest, spread_certainty(manifest), missing=absent)
    suggester = _suggester(cfg)

    assert not any(suggester._by_clip.get(c) is not None for c in absent)

    batch = suggester.selection_batch(manifest, set(), len(ids))
    chosen = {r["clip_id"] for r in batch}
    assert not (chosen & absent), sorted(chosen & absent)[:3]
    assert batch and chosen <= (set(ids) - absent)
    assert len(batch) <= len(ids) - len(absent), "an oversized request comes back short"


# --------------------------------------------------------------------------- #
# the config refuses what it cannot honour
# --------------------------------------------------------------------------- #
def test_a_rule_without_an_audit_holdout_is_refused_at_load(root: Path) -> None:
    config_path = create_project(Path(root) / "no-audit")
    write_config(config_path, {"suggest.audit_fraction": 0.0})
    assert load_config(config_path).suggest.audit_fraction == 0.0, (
        "an audit fraction of 0 alone is legitimate")

    with pytest.raises(ConfigError) as caught:
        write_config(config_path, MARGIN)
    message = str(caught.value)
    assert "select_rule" in message, message
    assert "audit_fraction" in message, message
    assert "unbiased" in message.lower(), message
    assert "0.10" in message and "'off'" in message, message


def test_a_rule_with_the_suggester_disabled_is_refused(root: Path) -> None:
    config_path = create_project(Path(root) / "disabled")
    write_config(config_path, {"suggest.enabled": False})
    with pytest.raises(ValueError) as caught:
        write_config(config_path, MARGIN)
    assert "select_rule" in str(caught.value) and "enabled" in str(caught.value)


def test_an_unknown_rule_is_refused_naming_the_known_ones(root: Path) -> None:
    config_path = create_project(Path(root) / "choices")
    with pytest.raises(ValueError) as caught:
        write_config(config_path, {"suggest.select_rule": "uncertainty"})
    assert "margin" in str(caught.value) and "off" in str(caught.value)


def test_a_removed_rule_loads_as_margin_with_a_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = Suggest.parse({"select_rule": "entropy"})
    assert parsed.select_rule == "margin"
    assert any("entropy" in str(w.message) for w in caught)


def test_an_unquoted_off_reads_as_off(root: Path) -> None:
    """YAML parses a bare ``off`` as a boolean."""
    config_path = create_project(Path(root) / "bare-off")
    text = config_path.read_text(encoding="utf-8").replace(
        'select_rule: "off"', "select_rule: off")
    config_path.write_text(text, encoding="utf-8")
    assert load_config(config_path).suggest.select_rule == "off"


SUGGEST_KEYS = {"select_rule": "margin", "retrain_every": 12}


def test_the_selection_keys_parse_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = Suggest.parse(dict(SUGGEST_KEYS))
    assert parsed.select_rule == "margin" and parsed.retrain_every == 12


def test_a_removed_selection_key_loads_with_a_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Suggest.parse({"select_rule": "margin", "select_frame_gap": 60,
                       "select_target_quota": 8})
    assert any("select_frame_gap" in str(w.message) for w in caught)


@pytest.mark.parametrize("key", sorted(SUGGEST_KEYS))
def test_a_misspelt_selection_key_is_reported(key) -> None:
    typo = dict(SUGGEST_KEYS)
    typo[key + "s"] = typo.pop(key)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=key + "s"):
            Suggest.parse(typo)


def test_a_refit_after_no_labels_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Suggest.parse({"retrain_every": 0})


# --------------------------------------------------------------------------- #
# the window length
# --------------------------------------------------------------------------- #
def test_a_window_length_mismatch_stops_the_batch(root: Path) -> None:
    manifest = build_manifest()
    certainty = spread_certainty(manifest)

    cfg = _project(root, "skew", **MARGIN)
    _install_model(cfg, manifest, certainty, trained_window=87)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        _suggester(cfg).selection_batch(manifest, set(), 20)
    message = str(caught.value)
    assert "87" in message, message
    assert "50" in message, message
    assert "clips.frames" in message, message

    matched = _project(root, "matched", **MARGIN)
    _install_model(matched, manifest, certainty, trained_window=FRAMES)
    assert len(_suggester(matched).selection_batch(manifest, set(), 20)) == 20


def test_a_model_that_never_recorded_its_window_is_warned_about_not_refused(
    root: Path
) -> None:
    manifest = build_manifest()
    cfg = _project(root, "legacy", **MARGIN)
    _install_model(cfg, manifest, spread_certainty(manifest), trained_window=None)
    suggester = _suggester(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = suggester.selection_batch(manifest, set(), 20)
    assert len(batch) == 20
    assert any("does not record" in str(w.message) for w in caught), (
        [str(w.message)[:60] for w in caught])
