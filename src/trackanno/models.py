"""Detection weights fetched from a release, cached on disk, and self-describing.

A model is named in ``config.yaml`` (``model.source``) rather than being a file
somebody had to copy onto this machine. The first run downloads it into a
content-addressed cache under the state directory; every run after that is a
cache hit and never touches the network.

Hosting is GitHub Releases: the index lives at a release asset, so
``https://github.com/<owner>/<repo>/releases/latest/download/index.json`` always
names the current set, while each entry's ``url`` points at one release's asset
and is therefore immutable.

Three rules, all about not running the wrong code:

* **The registry URL is a machine setting, never a project one.** It lives in the
  state directory beside the recent-projects list. Loading weights runs the
  pickle inside them, so a ``config.yaml`` somebody sends you may name a model
  id but may not decide which server it comes from.
* **https only, and the digest is checked.** A download whose sha256 does not
  match its manifest is deleted rather than cached, and ``model.sha256`` is
  written back into the project on the first success, so a tag that later moves
  to different weights is a loud mismatch rather than a silent substitution.
* **A cache hit never reaches the network.** Offline is the normal case once a
  model has been used, and a miss with no connection fails naming the id, the
  digest and the directory it looked in.

The manifest carries what a checkpoint cannot. ``read_model_info`` already
recovers the task, the keypoint count and the class names; it cannot recover the
keypoint NAMES, the skeleton ``edges``, the ``imgsz`` the model was validated at
or a sensible tile size, and those are most of what a person otherwise types in
by hand. :func:`apply_manifest` writes them into the project and returns what it
changed, rather than editing silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ModelEntry", "registry_url", "set_registry_url", "list_models", "find_model",
    "cache_dir", "cached_path", "fetch", "resolve_model", "apply_manifest",
    "prune_cache", "RegistryError", "is_trusted_url", "local_model_path",
]

REGISTRY_FILE = "registry.json"
CACHE_DIRNAME = "models"
INDEX_NAME = "index.json"
ENV_REGISTRY = "TRACKANNO_MODEL_REGISTRY"
CHUNK = 1 << 20
TIMEOUT = 30.0


class RegistryError(RuntimeError):
    """Anything that stops a model being resolved, phrased for the person."""


# Loopback is the exception to the https rule: a connection that never leaves
# the machine cannot be tampered with in transit, and it is what makes this
# testable without a certificate.
_LOOPBACK = ("http://127.0.0.1", "http://localhost", "http://[::1]")


def is_trusted_url(url: str) -> bool:
    """https anywhere, or plain http to this machine. Nothing else."""
    low = str(url or "").strip().lower()
    return low.startswith("https://") or low.startswith(_LOOPBACK)


@dataclass(frozen=True)
class ModelEntry:
    """One model in the registry, and everything known about it."""

    id: str
    url: str
    version: str = ""
    task: str = ""
    sha256: str = ""
    size: int = 0
    filename: str = ""
    notes: str = ""
    config: dict = field(default_factory=dict)
    assumes: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        bits = [self.id]
        if self.version:
            bits.append(self.version)
        if self.task:
            bits.append(f"({self.task})")
        return " ".join(bits)

    @property
    def name(self) -> str:
        return self.filename or Path(self.url.split("?")[0]).name or f"{self.id}.pt"

    @classmethod
    def parse(cls, d: dict) -> "ModelEntry":
        if not isinstance(d, dict):
            raise RegistryError("every entry in the registry index must be a mapping")
        for required in ("id", "url"):
            if not str(d.get(required, "")).strip():
                raise RegistryError(f"a registry entry is missing {required!r}")
        url = str(d["url"]).strip()
        if not is_trusted_url(url):
            raise RegistryError(
                f"{d['id']}: model urls must be https, got {url!r}. Weights are "
                "executed when they are loaded, so they are not fetched over a "
                "channel that can be rewritten in transit."
            )
        return cls(
            id=str(d["id"]).strip(),
            url=url,
            version=str(d.get("version", "") or "").strip(),
            task=str(d.get("task", "") or "").strip(),
            sha256=str(d.get("sha256", "") or "").strip().lower(),
            size=int(d.get("size") or 0),
            filename=str(d.get("filename", "") or "").strip(),
            notes=str(d.get("notes", "") or "").strip(),
            config=dict(d.get("config") or {}),
            assumes=dict(d.get("assumes") or {}),
        )


# --------------------------------------------------------------------------- #
# where models come from
# --------------------------------------------------------------------------- #
def _state_dir() -> Path:
    from .project import _state_dir as state

    return state()


def registry_url() -> str:
    """The index URL, from the environment or the state directory. "" if unset."""
    from_env = os.environ.get(ENV_REGISTRY, "").strip()
    if from_env:
        return from_env
    path = _state_dir() / REGISTRY_FILE
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("url", "")).strip()
    except (OSError, ValueError, AttributeError):
        return ""


def set_registry_url(url: str) -> None:
    """Point this machine at a registry. Empty clears it."""
    url = str(url or "").strip()
    if url and not is_trusted_url(url):
        raise RegistryError("the registry url must be https")
    directory = _state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REGISTRY_FILE).write_text(json.dumps({"url": url}, indent=2), encoding="utf-8")


def cache_dir() -> Path:
    return _state_dir() / CACHE_DIRNAME


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: float = TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": "trackanno"})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - https enforced above


def list_models(refresh: bool = False) -> list[ModelEntry]:
    """Every model the registry offers, newest index first.

    The index is cached, so the picker still fills in offline. ``refresh`` goes
    to the network and falls back to the cached copy if that fails, since a
    registry being unreachable should not empty a list that was fine yesterday.
    """
    url = registry_url()
    if not url:
        return []
    cached = cache_dir() / INDEX_NAME
    text = ""
    if refresh or not cached.exists():
        try:
            with _get(url) as response:
                text = response.read().decode("utf-8")
            cache_dir().mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
        except Exception:
            text = ""
    if not text:
        try:
            text = cached.read_text(encoding="utf-8")
        except OSError:
            return []
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RegistryError(f"the registry index at {url} is not valid JSON: {exc}") from exc
    entries = data.get("models") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise RegistryError(
            f"the registry index at {url} should be a list of models, or an object "
            "with a 'models' list"
        )
    return [ModelEntry.parse(e) for e in entries]


def find_model(model_id: str, version: str = "") -> ModelEntry | None:
    """Match on id, and on version when one is asked for. None if absent."""
    wanted = str(model_id or "").strip().lower()
    if not wanted:
        return None
    matches = [m for m in list_models() if m.id.lower() == wanted]
    if version:
        exact = [m for m in matches if m.version.lower() == str(version).strip().lower()]
        return exact[0] if exact else None
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def cached_path(sha256: str, name: str = "") -> Path | None:
    """The cached file for a digest, if it is there. Content-addressed, so two
    ids naming the same weights share one copy."""
    if not sha256:
        return None
    directory = cache_dir() / sha256[:16]
    if not directory.is_dir():
        return None
    if name:
        candidate = directory / name
        if candidate.exists():
            return candidate
    files = sorted(p for p in directory.glob("*.pt") if p.is_file())
    return files[0] if files else None


def fetch(entry: ModelEntry, reporter=None, expect_sha256: str = "") -> Path:
    """Return a local path to ``entry``'s weights, downloading them if needed.

    The digest decides the cache location, so a re-download of the same weights
    is a no-op and a changed file cannot overwrite a good one. ``expect_sha256``
    is the project's pin: when it disagrees with what arrives, nothing is cached
    and the mismatch is reported.
    """
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    wanted = (expect_sha256 or entry.sha256 or "").lower()

    hit = cached_path(wanted, entry.name) if wanted else None
    if hit is not None:
        reporter.log(f"{entry.id}: cached at {hit}")
        return hit

    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / f".{entry.id}-{os.getpid()}.part"
    sha = hashlib.sha256()
    total = entry.size or 0
    seen = 0
    try:
        reporter.stage(f"downloading {entry.name}", max(1, total))
        with _get(entry.url, timeout=TIMEOUT) as response, staged.open("wb") as out:
            declared = response.headers.get("Content-Length")
            if not total and declared and declared.isdigit():
                total = int(declared)
                reporter.stage(f"downloading {entry.name}", total)
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                sha.update(block)
                seen += len(block)
                reporter.advance(len(block))
                reporter.raise_if_cancelled()
    except Exception as exc:
        staged.unlink(missing_ok=True)
        raise RegistryError(
            f"could not download {entry.id} from {entry.url}: {exc}\n"
            f"The cache is {directory}; a model already there is used without "
            "any network access."
        ) from exc

    got = sha.hexdigest()
    if wanted and got != wanted:
        staged.unlink(missing_ok=True)
        raise RegistryError(
            f"{entry.id} downloaded from {entry.url} has sha256 {got}, but "
            f"{wanted} was expected. Nothing was cached. Either the release was "
            "changed after this project pinned it, or the download is not what "
            "it claims to be; weights are executed when loaded, so this is "
            "refused rather than warned about."
        )

    target_dir = directory / got[:16]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / entry.name
    staged.replace(target)
    (target_dir / "manifest.json").write_text(
        json.dumps({"id": entry.id, "version": entry.version, "url": entry.url,
                    "sha256": got, "size": seen}, indent=2),
        encoding="utf-8",
    )
    reporter.log(f"{entry.id}: {seen / 1e6:.1f} MB cached at {target}")
    return target


def prune_cache(keep: set[str] | None = None) -> int:
    """Delete cached weights whose digest is not in ``keep``. Returns the count."""
    keep = {d[:16] for d in (keep or set())}
    removed = 0
    directory = cache_dir()
    if not directory.is_dir():
        return 0
    for child in directory.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def local_model_path(cfg) -> Path:
    """The weights already on this disk, without downloading anything.

    For the status readout and the checkpoint probe, which run on every project
    open and must never start a 200 MB transfer. Falls back to ``paths.model``,
    so a project that has not fetched its model yet reads as "not there" rather
    than as an error.
    """
    try:
        return resolve_model(cfg, allow_download=False)
    except Exception:
        return cfg.paths.model


def resolve_model(cfg, reporter=None, allow_download: bool = True) -> Path:
    """The weights this config means, as a local file.

    ``model.source`` empty gives ``paths.model`` unchanged, which is what a
    project built around a local checkpoint uses. Otherwise the id or URL is
    resolved against the cache, downloading once. Everything downstream is
    handed a path and never learns that a network was involved.
    """
    spec = getattr(cfg, "model", None)
    if spec is None or not spec.is_remote:
        return cfg.paths.model

    if spec.is_url:
        entry = ModelEntry(id=Path(spec.source.split("?")[0]).stem or "model",
                           url=spec.source, sha256=spec.sha256)
    else:
        entry = find_model(spec.source, spec.version)
        if entry is None:
            known = ", ".join(m.id for m in list_models()) or "none"
            where = registry_url() or "no registry is configured on this machine"
            raise RegistryError(
                f"model.source is {spec.source!r}"
                + (f" version {spec.version!r}" if spec.version else "")
                + f", which the registry does not offer.\nRegistry: {where}\n"
                f"Available: {known}"
            )

    pinned = spec.sha256 or entry.sha256
    hit = cached_path(pinned, entry.name) if pinned else None
    if hit is not None:
        return hit
    if not allow_download:
        raise RegistryError(
            f"{entry.id} is not in the cache ({cache_dir()}) and downloading is "
            "switched off for this call"
        )
    return fetch(entry, reporter, expect_sha256=spec.sha256)


# --------------------------------------------------------------------------- #
# what the manifest configures
# --------------------------------------------------------------------------- #
# Only these sections may be set from a manifest. A model may say how it should
# be run; it may not point the project at different movies, a different work
# directory or a different annotation log.
_ALLOWED_SECTIONS = ("detection", "tiling", "tracking", "clips", "spectral")


def apply_manifest(config_path, entry: ModelEntry, digest: str = "") -> list[str]:
    """Write the manifest's settings into the project. Returns what changed.

    Returns rather than logs, because the Configure page reports what a model
    decided in the same breath as applying it, and a setting that silently
    changed how clips were cut is the one nobody notices.
    """
    from .config import load_config
    from .project import write_config

    before = load_config(config_path)
    updates: dict[str, object] = {}
    told: list[str] = []

    for section, values in (entry.config or {}).items():
        if section == "keypoints":
            continue
        if section not in _ALLOWED_SECTIONS:
            continue
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            current = getattr(getattr(before, section, None), key, None)
            if isinstance(current, Path):
                continue
            if current != value:
                updates[f"{section}.{key}"] = value
                told.append(f"{section}.{key} {value}")

    if entry.task and before.detection.task != entry.task:
        updates["detection.task"] = entry.task
        told.append(f"task {entry.task}")

    keypoints = None
    kp = (entry.config or {}).get("keypoints") or {}
    nodes = [str(n) for n in (kp.get("nodes") or [])]
    edges = [[str(a), str(b)] for a, b in (kp.get("edges") or [])]
    if nodes and (nodes != list(before.keypoints.nodes)
                  or edges != [list(e) for e in before.keypoints.edges]):
        keypoints = {"nodes": nodes, "edges": edges}
        told.append(f"{len(nodes)} keypoint names")
        if edges:
            told.append(f"{len(edges)} skeleton links")

    updates["model.source"] = entry.id
    updates["model.version"] = entry.version
    if digest or entry.sha256:
        updates["model.sha256"] = (digest or entry.sha256).lower()

    write_config(config_path, updates, keypoints=keypoints)
    return told


def mismatch_warnings(cfg, entry: ModelEntry) -> list[str]:
    """Where the project disagrees with what the model was trained on.

    Reported, never corrected: a model trained at 100 fps still runs on a 30 fps
    movie, and only the person knows whether that is the experiment or a mistake.
    """
    out: list[str] = []
    assumes = entry.assumes or {}
    fps = assumes.get("fps")
    if fps and getattr(cfg.tiff, "fps", None) and abs(float(cfg.tiff.fps) - float(fps)) > 1e-6:
        out.append(
            f"{entry.id} was trained at {fps} fps; tiff.fps is {cfg.tiff.fps}. "
            "Every rate this reports is per second, so the features stay "
            "comparable, but the beat may fall outside what the clip can resolve."
        )
    pixel = assumes.get("pixel_size_um")
    if pixel:
        out.append(
            f"{entry.id} was trained on {pixel} um pixels. A different "
            "magnification changes the apparent size of everything it detects."
        )
    return out
