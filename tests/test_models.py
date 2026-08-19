"""Fetching detection weights from a registry, and what they configure.

Everything here runs against a throwaway HTTP server on the loopback interface,
so no test reaches the internet and none needs a certificate. Loopback is the one
carve-out from the https rule, for exactly that reason.

The stakes are unusual for this project: loading a checkpoint executes the pickle
inside it, so most of what follows checks that the wrong bytes are REFUSED rather
than that the right ones arrive.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading

import pytest

from harness import check
from trackanno import models as M
from trackanno.config import ConfigError, load_config
from trackanno.project import create_project, write_config

WEIGHTS = b"pretend weights" * 5000
DIGEST = hashlib.sha256(WEIGHTS).hexdigest()


class _Server:
    """A directory served over loopback, with a switch to take it offline."""

    def __init__(self, directory):
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(directory)
        )
        handler.log_message = lambda *a, **k: None
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)


def _index(base, digest=DIGEST, model_id="trypanosome-pose"):
    return {"models": [{
        "id": model_id, "version": "v2", "task": "pose",
        "url": f"{base}/{model_id}.pt", "sha256": digest, "size": len(WEIGHTS),
        "notes": "8-point flagellum, brightfield 20x",
        "config": {
            "detection": {"imgsz": 1280, "conf": 0.3},
            "tiling": {"tile": 512},
            "paths": {"video_dir": "/somewhere/else"},
            "keypoints": {
                "nodes": ["Head", "Base", "Flag1", "Flag2", "Flag3", "Flag4", "Flag5", "Tip"],
                "edges": [["Head", "Base"], ["Base", "Flag1"], ["Flag1", "Flag2"]],
            },
        },
        "assumes": {"fps": 100, "pixel_size_um": 0.325},
    }]}


@pytest.fixture(scope="module")
def registry(root):
    """A served release directory, with the machine pointed at its index."""
    served = root / "release"
    served.mkdir()
    (served / "trypanosome-pose.pt").write_bytes(WEIGHTS)
    (served / "other-pose.pt").write_bytes(WEIGHTS)          # same bytes, other id
    server = _Server(served)
    (served / "index.json").write_text(json.dumps(_index(server.base)), encoding="utf-8")
    M.set_registry_url(f"{server.base}/index.json")
    M.list_models(refresh=True)
    yield server
    server.stop()


def test_the_index_is_read_and_cached(registry, root) -> None:
    models = M.list_models(refresh=True)
    check("the registry lists its models", [m.id for m in models] == ["trypanosome-pose"])
    entry = M.find_model("trypanosome-pose")
    check("an entry carries its version and task",
          (entry.version, entry.task) == ("v2", "pose"))
    check("and the notes a person needs to choose", "brightfield" in entry.notes)
    check("a version that does not exist matches nothing",
          M.find_model("trypanosome-pose", "v9") is None)
    check("nor does an unknown id", M.find_model("no-such-model") is None)
    check("the index is cached on disk for the next open",
          (M.cache_dir() / M.INDEX_NAME).exists())


def test_only_a_channel_that_cannot_be_rewritten_is_accepted() -> None:
    """Weights run when they load, so the transport is a security boundary."""
    check("https is fine", M.is_trusted_url("https://example.org/a.pt"))
    check("loopback http is fine, being untamperable in transit",
          M.is_trusted_url("http://127.0.0.1:8000/a.pt")
          and M.is_trusted_url("http://localhost:8000/a.pt"))
    check("plain http to anywhere else is not",
          not M.is_trusted_url("http://example.org/a.pt"))
    check("nor is a file or ftp url",
          not M.is_trusted_url("file:///etc/passwd") and not M.is_trusted_url("ftp://x/a.pt"))

    try:
        M.ModelEntry.parse({"id": "x", "url": "http://example.org/a.pt"})
        refused = False
    except M.RegistryError:
        refused = True
    check("an index offering a plain-http model is refused", refused)


def test_a_config_may_name_a_model_but_not_a_server(root) -> None:
    """The registry URL is a machine setting, so a shared config cannot redirect it."""
    path = create_project(root / "cfg-only")
    write_config(path, {"model.source": "trypanosome-pose"})
    check("a project may name an id", load_config(path).model.source == "trypanosome-pose")
    try:
        write_config(path, {"model.source": "http://evil.example.org/a.pt"})
        refused = False
    except ConfigError:
        refused = True
    check("but may not point the fetch at plain http", refused)
    check("and the section has no key for redirecting the fetch at all",
          set(load_config(path).model.__dict__) == {"source", "version", "sha256"},
          str(sorted(load_config(path).model.__dict__)))


def test_it_downloads_verifies_and_caches(registry, root) -> None:
    path = create_project(root / "fetch")
    write_config(path, {"model.source": "trypanosome-pose", "model.version": "v2"})

    local = M.resolve_model(load_config(path))
    check("the weights arrive", local.exists() and local.read_bytes() == WEIGHTS)
    check("in a directory named for their digest", local.parent.name == DIGEST[:16])
    check("with a manifest beside them", (local.parent / "manifest.json").exists())

    # Content addressing: a second id serving identical bytes shares the copy.
    same = M.ModelEntry(id="other-pose", url=f"{registry.base}/other-pose.pt", sha256=DIGEST)
    check("identical weights under another id reuse the one cached copy",
          M.fetch(same) == local)


def test_a_cache_hit_never_reaches_the_network(registry, root) -> None:
    """Offline is the normal case once a model has been used once."""
    path = create_project(root / "offline")
    write_config(path, {"model.source": "trypanosome-pose"})
    first = M.resolve_model(load_config(path))

    saved = M._get
    M._get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("went to the network"))
    try:
        again = M.resolve_model(load_config(path))
    finally:
        M._get = saved
    check("the second resolve is served from the cache", again == first)


def test_weights_that_are_not_what_they_claim_are_refused(registry, root) -> None:
    """The digest is the whole defence: a mismatch must not reach the cache."""
    served = root / "release"
    swapped = b"different bytes entirely"
    (served / "swapped.pt").write_bytes(swapped)
    # A digest nothing has cached, so the fetch really runs and really fails.
    claimed = "cd" * 32
    entry = M.ModelEntry(id="swapped", url=f"{registry.base}/swapped.pt", sha256=claimed)

    try:
        M.fetch(entry)
        refused = False
        message = ""
    except M.RegistryError as exc:
        refused, message = True, str(exc)
    check("a file whose sha256 disagrees with the manifest is refused", refused)
    check("and the message says what arrived and what was expected",
          hashlib.sha256(swapped).hexdigest()[:16] in message and claimed[:16] in message,
          message[:120])
    check("nothing partial was left in the cache",
          not list(M.cache_dir().glob("*.part")))
    check("and it was not cached under its own digest either",
          M.cached_path(hashlib.sha256(swapped).hexdigest()) is None)


def test_a_pin_survives_the_release_moving(registry, root) -> None:
    """A tag that later points at other weights is a mismatch, not a substitution."""
    path = create_project(root / "pinned")
    write_config(path, {"model.source": "trypanosome-pose", "model.sha256": "ab" * 32})
    try:
        M.resolve_model(load_config(path))
        refused = False
    except M.RegistryError:
        refused = True
    check("weights that do not match the project's pin are refused", refused)


def test_an_unknown_model_says_what_is_on_offer(registry, root) -> None:
    path = create_project(root / "unknown")
    write_config(path, {"model.source": "not-a-model"})
    try:
        M.resolve_model(load_config(path))
        message = ""
    except M.RegistryError as exc:
        message = str(exc)
    check("the failure names the id that was asked for", "not-a-model" in message)
    check("the registry it looked in", registry.base in message)
    check("and what that registry does offer", "trypanosome-pose" in message)


def test_an_empty_source_means_the_local_file(root) -> None:
    """Projects built around a checkpoint on this disk keep working untouched."""
    weights = root / "local.pt"
    weights.write_bytes(b"")
    path = create_project(root / "local", model=weights)
    cfg = load_config(path)
    check("resolve returns paths.model unchanged", M.resolve_model(cfg) == weights.resolve())
    check("and the status probe does not download anything",
          M.local_model_path(cfg) == weights.resolve())


def test_the_manifest_configures_the_project(registry, root) -> None:
    """The point of the exercise: the settings a person otherwise types by hand."""
    path = create_project(root / "manifest")
    entry = M.find_model("trypanosome-pose")
    told = M.apply_manifest(path, entry, DIGEST)
    cfg = load_config(path)

    check("the inference size came from the model", cfg.detection.imgsz == 1280)
    check("so did the confidence it was validated at", abs(cfg.detection.conf - 0.3) < 1e-9)
    check("and a tile size that suits it", cfg.tiling.tile == 512)
    check("the task followed the checkpoint", cfg.detection.task == "pose")
    check("the keypoint NAMES came across, which no checkpoint stores",
          cfg.keypoints.nodes[:2] == ["Head", "Base"] and len(cfg.keypoints.nodes) == 8)
    check("so did the skeleton, which cannot be inferred at all",
          len(cfg.keypoints.edges) == 3, str(cfg.keypoints.edges))
    check("the digest was pinned into the project", cfg.model.sha256 == DIGEST)
    check("and it reported every change rather than editing silently",
          any("imgsz" in t for t in told) and any("keypoint" in t for t in told), str(told))

    check("a manifest cannot repoint the project at other movies",
          cfg.paths.video_dir == (root / "manifest" / "videos").resolve(),
          str(cfg.paths.video_dir))
    check("the comments in config.yaml survived the edit",
          path.read_text(encoding="utf-8").count("#") > 40)


def test_it_warns_where_the_project_disagrees_with_the_model(registry, root) -> None:
    """Reported, never corrected: only the person knows if it is the experiment."""
    path = create_project(root / "assumes")
    write_config(path, {"tiff.fps": 30})
    entry = M.find_model("trypanosome-pose")
    warnings = M.mismatch_warnings(load_config(path), entry)
    check("a frame rate the model was not trained at is called out",
          any("100" in w and "30" in w for w in warnings), str(warnings))
    check("as is a different pixel size", any("0.325" in w for w in warnings))


def test_the_cache_can_be_pruned(registry, root) -> None:
    M.fetch(M.find_model("trypanosome-pose"))
    check("the weights are cached", M.cached_path(DIGEST) is not None)
    check("pruning keeps what is asked for", M.prune_cache(keep={DIGEST}) == 0)
    check("and removes what is not", M.prune_cache(keep=set()) >= 1)
    check("after which the cache is empty", M.cached_path(DIGEST) is None)


# --------------------------------------------------------------------------- #
# private repositories, which have no anonymous download url
# --------------------------------------------------------------------------- #
def test_a_private_release_is_fetched_through_the_api(root, monkeypatch) -> None:
    """A private repo's assets are reachable only with a token, through the API.

    ``github://owner/repo`` resolves the release, finds each asset by name and
    downloads it with ``Accept: application/octet-stream``, which is the one
    combination GitHub serves for a private release. Assets are looked up by
    name every time rather than by id, so re-uploading a release does not
    invalidate the index.
    """
    seen: list[tuple[str, dict]] = []
    release = {
        "tag_name": "v2",
        "assets": [
            {"name": M.INDEX_NAME, "url": "https://api.github.com/repos/o/r/releases/assets/1"},
            {"name": "tryp.pt", "url": "https://api.github.com/repos/o/r/releases/assets/2"},
        ],
    }
    index = json.dumps({"models": [{
        "id": "tryp", "version": "v2", "task": "pose", "asset": "tryp.pt",
        "sha256": DIGEST, "size": len(WEIGHTS),
    }]}).encode("utf-8")

    class _Response:
        def __init__(self, body):
            self.body = body
        def read(self, n=-1):
            out, self.body = (self.body, b"") if n == -1 else (self.body[:n], self.body[n:])
            return out
        @property
        def headers(self):
            return {}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_get(url, timeout=M.TIMEOUT, accept=""):
        seen.append((url, {"accept": accept, "token": M.token()}))
        if url.endswith("/releases/latest"):
            return _Response(json.dumps(release).encode("utf-8"))
        if url.endswith("/assets/1"):
            return _Response(index)
        if url.endswith("/assets/2"):
            return _Response(WEIGHTS)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(M, "_get", fake_get)
    monkeypatch.setenv("GH_TOKEN", "ghp_pretend")
    monkeypatch.setenv(M.ENV_REGISTRY, "github://o/r")

    check("a github:// registry is accepted", M._github_source("github://o/r") == ("o", "r", ""))
    check("and a pinned tag comes through",
          M._github_source("github://o/r@v2") == ("o", "r", "v2"))

    models = M.list_models(refresh=True)
    check("the index is read out of the release", [m.id for m in models] == ["tryp"])
    check("an entry may name an asset instead of a url", models[0].asset == "tryp.pt")
    check("and still knows what to call the file", models[0].name == "tryp.pt")

    M.prune_cache(keep=set())
    path = M.fetch(models[0])
    check("the weights arrive through the API", path.read_bytes() == WEIGHTS)
    check("the asset was requested as a byte stream, which is what serves a "
          "private release",
          any(u.endswith("/assets/2") and o["accept"] == "application/octet-stream"
              for u, o in seen), str(seen[-1]))
    check("and the token was available for it", all(o["token"] for _u, o in seen))


def test_a_token_is_only_ever_sent_to_github(root, monkeypatch) -> None:
    """A registry may name any host; the credential must not follow it there."""
    monkeypatch.setenv("GH_TOKEN", "ghp_pretend")
    sent: dict = {}

    class _Boom(Exception):
        pass

    def capture(request, timeout=None):
        sent.update(dict(request.header_items()))
        raise _Boom()

    monkeypatch.setattr(M.urllib.request, "urlopen", capture)
    for url in ("https://elsewhere.example.org/index.json", "http://127.0.0.1:1/x"):
        sent.clear()
        try:
            M._get(url)
        except _Boom:
            pass
        check(f"no credential goes to {url.split('/')[2]}",
              not any(k.lower() == "authorization" for k in sent), str(sent))

    sent.clear()
    try:
        M._get("https://api.github.com/repos/o/r/releases/latest")
    except _Boom:
        pass
    check("but api.github.com does get one",
          any(k.lower() == "authorization" for k in sent), str(sent))
