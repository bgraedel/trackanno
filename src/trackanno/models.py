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
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ModelEntry", "registry_url", "set_registry_url", "list_models", "find_model",
    "cache_dir", "cached_path", "fetch", "resolve_model", "apply_manifest",
    "prune_cache", "RegistryError", "is_trusted_url", "local_model_path",
    "token", "set_token", "RUNNABLE_TASKS", "RUNNABLE_FRAMEWORKS", "span", "describe_span",
    "mismatch_warnings",
]

REGISTRY_FILE = "registry.json"
CACHE_DIRNAME = "models"
INDEX_NAME = "index.json"
ENV_REGISTRY = "TRACKANNO_MODEL_REGISTRY"
CHUNK = 1 << 20
TIMEOUT = 30.0

# What trackanno can drive as a detector. A registry may list anything, and
# often will: segmentation and fluorescence models are useful to publish beside
# these even while this tool cannot run them. They are shown and explained
# rather than hidden, since a registry that silently omits half its contents is
# worse than one that says why.
# What this tool can load and drive. An entry naming anything else is listed
# and downloadable, and something else runs it.
RUNNABLE_FRAMEWORKS = ("ultralytics",)
RUNNABLE_TASKS = ("detect", "pose")


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
    asset: str = ""
    modality: str = ""
    framework: str = ""
    notes: str = ""
    config: dict = field(default_factory=dict)
    preprocess: dict = field(default_factory=dict)
    assumes: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        bits = [self.id]
        if self.version:
            bits.append(self.version)
        kind = " ".join(b for b in (self.framework, self.modality, self.task) if b)
        if kind:
            bits.append(f"({kind})")
        return " ".join(bits)

    @property
    def runnable(self) -> bool:
        """Whether trackanno can use this as its detector."""
        return (self.framework in RUNNABLE_FRAMEWORKS
                and (not self.task or self.task in RUNNABLE_TASKS))

    @property
    def trained_at(self) -> str:
        """What acquisition this model saw, for choosing between two of them."""
        from_ = self.assumes or {}
        bits = [describe_span(from_.get("fps"), "fps"),
                describe_span(from_.get("pixel_size_um"), "um/px")]
        return ", ".join(b for b in bits if b)

    @property
    def unsupported_reason(self) -> str:
        """Why this cannot be used here, or "" when it can."""
        if self.runnable:
            return ""
        runs = f"{'/'.join(RUNNABLE_FRAMEWORKS)} {' and '.join(RUNNABLE_TASKS)} models"
        is_a = " ".join(b for b in (self.framework or "an unstated framework",
                                    self.task) if b)
        return (f"trackanno runs {runs}; {self.id} is {is_a}. It can be found and "
                "downloaded here, and something else has to run it.")

    @property
    def name(self) -> str:
        from_url = Path(self.url.split("?")[0]).name if self.url else ""
        return self.filename or self.asset or from_url or f"{self.id}.pt"

    @classmethod
    def parse(cls, d: dict) -> "ModelEntry":
        if not isinstance(d, dict):
            raise RegistryError("every entry in the registry index must be a mapping")
        if not str(d.get("id", "")).strip():
            raise RegistryError("a registry entry is missing 'id'")
        if not (str(d.get("url", "")).strip() or str(d.get("asset", "")).strip()):
            raise RegistryError(
                f"{d['id']}: an entry needs either a 'url' or an 'asset' naming a "
                "file in the same release"
            )
        url = str(d.get("url", "") or "").strip()
        if url and not is_trusted_url(url):
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
            asset=str(d.get("asset", "") or "").strip(),
            modality=str(d.get("modality", "") or "").strip(),
            framework=str(d.get("framework", "") or "").strip().lower(),
            notes=str(d.get("notes", "") or "").strip(),
            config=dict(d.get("config") or {}),
            preprocess=dict(d.get("preprocess") or {}),
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
    """Point this machine at a registry. Empty clears it.

    Takes ``github://owner/repo[@tag]``, which goes through the API and so works
    on a private repository, or a plain https URL to an index.json.
    """
    url = str(url or "").strip()
    if url and _github_source(url) is None and not is_trusted_url(url):
        raise RegistryError(
            "the registry must be github://owner/repo or an https url to an index.json"
        )
    settings = _settings()
    settings["url"] = url
    directory = _state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REGISTRY_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def cache_dir() -> Path:
    return _state_dir() / CACHE_DIRNAME


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #
def token() -> str:
    """The GitHub token, from the environment or the state file. "" if none.

    Kept beside the registry URL and never in ``config.yaml``, for the same
    reason: a project someone sends you may name a model, and must not be able
    to make your credentials fetch from somewhere else.
    """
    for name in ("TRACKANNO_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return str(_settings().get("token", "") or "").strip()


def _settings() -> dict:
    try:
        data = json.loads((_state_dir() / REGISTRY_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _get(url: str, timeout: float = TIMEOUT, accept: str = ""):
    """One GET, with the token attached only when talking to GitHub.

    Scoped deliberately: a token sent to whatever host an index happens to name
    would leak it to that host. Anything outside api.github.com is fetched
    anonymously, which is all a public release needs.
    """
    headers = {"User-Agent": "trackanno"}
    if accept:
        headers["Accept"] = accept
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("api.github.com", "uploads.github.com"):
        auth = token()
        if auth:
            headers["Authorization"] = f"Bearer {auth}"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - scheme checked above


# --------------------------------------------------------------------------- #
# GitHub releases
# --------------------------------------------------------------------------- #
def _github_source(url: str) -> tuple[str, str, str] | None:
    """``github://owner/repo[@tag]`` -> (owner, repo, tag). None for anything else.

    A private repository has no anonymous download URL, so its assets have to be
    fetched through the API. Naming the repository rather than an asset URL also
    survives re-uploading a release, since assets are resolved by name each time.
    """
    text = str(url or "").strip()
    if not text.lower().startswith("github://"):
        return None
    rest, _, tag = text[len("github://"):].partition("@")
    parts = [p for p in rest.split("/") if p]
    if len(parts) != 2:
        raise RegistryError(
            f"{url!r} should look like github://owner/repo or github://owner/repo@tag"
        )
    return parts[0], parts[1], tag


def _release(owner: str, repo: str, tag: str = "") -> dict:
    where = f"tags/{tag}" if tag else "latest"
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/{where}"
    try:
        with _get(url, accept="application/vnd.github+json") as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        hint = "" if token() else (
            "\nNo GitHub token is set on this machine, which a private "
            "repository needs. Set one with `trackanno.models.set_token(...)` or "
            "the GH_TOKEN environment variable."
        )
        raise RegistryError(f"could not read the release at {url}: {exc}{hint}") from exc


def _asset_url(release: dict, name: str) -> str:
    """The API URL of a named asset, which is what works on a private repo."""
    for asset in release.get("assets") or []:
        if str(asset.get("name", "")) == name:
            return str(asset.get("url", ""))
    offered = ", ".join(str(a.get("name")) for a in release.get("assets") or []) or "none"
    raise RegistryError(
        f"the release {release.get('tag_name', '?')!r} has no asset named {name!r}. "
        f"It offers: {offered}"
    )


def set_token(value: str) -> None:
    """Store a GitHub token for private registries. Empty clears it."""
    settings = _settings()
    settings["token"] = str(value or "").strip()
    directory = _state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REGISTRY_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")


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
            text = _read_index(url)
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


def _read_index(url: str) -> str:
    """The index text, from a release asset or a plain URL."""
    source = _github_source(url)
    if source is None:
        with _get(url) as response:
            return response.read().decode("utf-8")
    owner, repo, tag = source
    release = _release(owner, repo, tag)
    with _get(_asset_url(release, INDEX_NAME), accept="application/octet-stream") as response:
        return response.read().decode("utf-8")


def download_url(entry: "ModelEntry") -> str:
    """Where ``entry``'s weights actually come from.

    An absolute ``url`` is used as given. An ``asset`` name is looked up in the
    registry's own release, which is what lets a PRIVATE repository work: its
    assets have no anonymous download URL and must go through the API.
    """
    if entry.url:
        return entry.url
    source = _github_source(registry_url())
    if source is None:
        raise RegistryError(
            f"{entry.id} names the asset {entry.asset!r}, but the registry is not a "
            "github://owner/repo source, so there is no release to look it up in"
        )
    owner, repo, tag = source
    return _asset_url(_release(owner, repo, tag), entry.asset)


def find_model(model_id: str, version: str = "") -> ModelEntry | None:
    """Match on id, and on version when one is asked for. None if absent."""
    wanted = str(model_id or "").strip().lower()
    if not wanted:
        return None
    matches = [m for m in list_models() if m.id.lower() == wanted]
    if version:
        exact = [m for m in matches if m.version.lower() == str(version).strip().lower()]
        return exact[0] if exact else None
    # No version asked for: the newest. An index keeps one entry per (id,
    # version) in that order, so an id can offer several and a project that
    # pinned an older one keeps getting it.
    return matches[-1] if matches else None


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
    # Any file: weights may be .pt, .onnx, extensionless or a zipped directory.
    files = sorted(f for f in directory.iterdir()
                   if f.is_file() and f.name != "manifest.json")
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
        source_url = download_url(entry)
        with _get(source_url, timeout=TIMEOUT,
                  accept="application/octet-stream") as response, staged.open("wb") as out:
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
            f"could not download {entry.id} ({entry.url or entry.asset}): {exc}\n"
            f"The cache is {directory}; a model already there is used without "
            "any network access."
        ) from exc

    got = sha.hexdigest()
    if wanted and got != wanted:
        staged.unlink(missing_ok=True)
        raise RegistryError(
            f"{entry.id} downloaded from {entry.url or entry.asset} has sha256 "
            f"{got}, but "
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
        json.dumps({"id": entry.id, "version": entry.version,
                    "source": entry.url or entry.asset, "sha256": got, "size": seen},
                   indent=2),
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
# `config` is keyed by consumer: trackanno reads the sections below and ignores
# the rest, so `config.cellpose` reaches a cellpose runner untouched.
# `preprocess` says how pixels must reach the model. Neither is interpreted here
# beyond the sections listed.
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

    # A task this tool cannot drive is reported rather than written: setting
    # detection.task to something the loader rejects would fail the whole apply
    # with a config error, which says nothing about what actually went wrong.
    if entry.task and not entry.runnable:
        told.append(f"NOT applied: {entry.unsupported_reason}")
    elif entry.task and before.detection.task != entry.task:
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

    # A value the loader rejects must cost that value, not the whole apply. A
    # manifest is written elsewhere, so one bad number should not stop a model
    # configuring the rest of the project.
    try:
        write_config(config_path, updates, keypoints=keypoints)
    except Exception:
        from .config import ConfigError

        applied: dict[str, object] = {}
        refused: list[str] = []
        for key, value in updates.items():
            try:
                write_config(config_path, {**applied, key: value}, keypoints=keypoints)
                applied[key] = value
            except (ConfigError, ValueError) as exc:
                refused.append(f"NOT applied: {key} {value!r} ({exc})")
        told = [t for t in told if not any(t.startswith(k) for k in
                                           (r.split()[2] for r in refused))]
        told.extend(refused)
    return told


def span(value) -> tuple[float, float] | None:
    """What a range in ``assumes`` covers, as ``(low, high)``. None if absent.

    Accepts a scalar for a model trained at one scale, and a list for one
    trained across several: ``[0.2, 0.65]`` reads as a range and
    ``[0.325, 0.65, 1.3]`` as discrete values. Both answer the only question
    asked of them, which is whether a given acquisition falls inside what the
    model saw, so the two spellings need not be told apart.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    try:
        numbers = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return (min(numbers), max(numbers)) if numbers else None


def describe_span(value, unit: str) -> str:
    """``0.325 um`` or ``0.2 to 0.65 um``, for a person choosing a model."""
    bounds = span(value)
    if bounds is None:
        return ""
    low, high = bounds
    if low == high:
        return f"{low:g} {unit}"
    return f"{low:g} to {high:g} {unit}"


def _outside(measured, value) -> bool:
    """Whether a measured acquisition falls outside what the model was trained on.

    A tolerance of 1% absorbs the difference between a nominal magnification and
    what the metadata actually reports, which is not a mismatch worth a warning.
    """
    bounds = span(value)
    if bounds is None or measured is None:
        return False
    low, high = bounds
    tol = max(1e-9, 0.01 * max(abs(low), abs(high)))
    return not (low - tol <= float(measured) <= high + tol)


def mismatch_warnings(cfg, entry: ModelEntry, pixel_size_um=None) -> list[str]:
    """Where a project disagrees with what the model was trained on.

    Reported, never corrected: a model trained at 100 fps still runs on a 30 fps
    movie, and only the person knows whether that is the experiment or a
    mistake.

    Silent when they agree, and silent when nothing can be compared. A project
    carries no pixel size of its own, since it is a property of each movie
    (ND2 records it, TIFF does not), so pass ``pixel_size_um`` when one has been
    read. Without it, a model trained at a SINGLE scale still gets a line,
    because that is a real constraint on what it can be pointed at; a
    multi-scale model gets nothing, having no constraint to state.
    """
    out: list[str] = []
    assumes = entry.assumes or {}

    fps = assumes.get("fps")
    project_fps = getattr(getattr(cfg, "tiff", None), "fps", None)
    if _outside(project_fps, fps):
        out.append(
            f"{entry.id} was trained at {describe_span(fps, 'fps')}; this project "
            f"reads TIFFs at {project_fps} fps. Every rate reported is per second, "
            "so the features stay comparable, but the beat may fall outside what a "
            "clip of this length can resolve."
        )

    pixels = assumes.get("pixel_size_um")
    bounds = span(pixels)
    if bounds is not None:
        if _outside(pixel_size_um, pixels):
            out.append(
                f"{entry.id} was trained on {describe_span(pixels, 'um')} pixels; "
                f"these movies are {float(pixel_size_um):g} um. Magnification "
                "changes the apparent size of everything it detects, so check the "
                "first few detections before running the whole batch."
            )
        elif pixel_size_um is None and bounds[0] == bounds[1]:
            out.append(
                f"{entry.id} was trained on {describe_span(pixels, 'um')} pixels "
                "only. At a different magnification, scale detection.imgsz and "
                "tiling.tile with it, or retrain."
            )
    return out
