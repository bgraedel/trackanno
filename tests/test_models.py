"""Fetching detection weights from a registry, and what they configure.

Runs against a loopback HTTP server. Loading a checkpoint executes its pickle, so
most tests check that the wrong bytes are refused.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading

import pytest

from trackanno.video import models as M
from trackanno.config import ConfigError, load_config
from trackanno.project import create_project, write_config

WEIGHTS = b"pretend weights" * 5000
DIGEST = hashlib.sha256(WEIGHTS).hexdigest()


class _Server:
    """A directory served over loopback."""

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
        "id": model_id, "version": "v2", "task": "pose", "framework": "ultralytics",
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


def test_the_index_is_read_and_cached(registry) -> None:
    models = M.list_models(refresh=True)
    assert [m.id for m in models] == ["trypanosome-pose"]
    entry = M.find_model("trypanosome-pose")
    assert (entry.version, entry.task) == ("v2", "pose")
    assert "brightfield" in entry.notes
    assert (M.cache_dir() / M.INDEX_NAME).exists(), "cached on disk for the next open"


def test_an_unknown_id_or_version_matches_nothing(registry) -> None:
    assert M.find_model("trypanosome-pose", "v9") is None
    assert M.find_model("no-such-model") is None


def test_only_a_channel_that_cannot_be_rewritten_is_trusted() -> None:
    assert M.is_trusted_url("https://example.org/a.pt")
    assert M.is_trusted_url("http://127.0.0.1:8000/a.pt"), "loopback cannot be tampered with"
    assert M.is_trusted_url("http://localhost:8000/a.pt")
    assert not M.is_trusted_url("http://example.org/a.pt")
    assert not M.is_trusted_url("file:///etc/passwd")
    assert not M.is_trusted_url("ftp://x/a.pt")


def test_an_index_offering_a_plain_http_model_is_refused() -> None:
    with pytest.raises(M.RegistryError):
        M.ModelEntry.parse({"id": "x", "url": "http://example.org/a.pt"})


def test_a_config_may_name_a_model_but_not_a_server(root) -> None:
    """The registry URL is a machine setting, so a shared config cannot redirect it."""
    path = create_project(root / "cfg-only")
    write_config(path, {"model.source": "trypanosome-pose"})
    assert load_config(path).model.source == "trypanosome-pose"
    with pytest.raises(ConfigError):
        write_config(path, {"model.source": "http://evil.example.org/a.pt"})
    # The section has no key for redirecting the fetch at all.
    assert set(load_config(path).model.__dict__) == {"source", "version", "sha256"}


def test_it_downloads_verifies_and_caches(registry, root) -> None:
    path = create_project(root / "fetch")
    write_config(path, {"model.source": "trypanosome-pose", "model.version": "v2"})

    local = M.resolve_model(load_config(path))
    assert local.exists() and local.read_bytes() == WEIGHTS
    assert local.parent.name == DIGEST[:16], "a directory named for the digest"
    assert (local.parent / "manifest.json").exists()

    # Content addressing: a second id serving identical bytes shares the copy.
    same = M.ModelEntry(id="other-pose", url=f"{registry.base}/other-pose.pt", sha256=DIGEST)
    assert M.fetch(same) == local


def test_a_cache_hit_never_reaches_the_network(registry, root, monkeypatch) -> None:
    path = create_project(root / "offline")
    write_config(path, {"model.source": "trypanosome-pose"})
    first = M.resolve_model(load_config(path))

    def offline(*a, **k):
        raise AssertionError("went to the network")

    monkeypatch.setattr(M, "_get", offline)
    assert M.resolve_model(load_config(path)) == first


def test_weights_that_are_not_what_they_claim_are_refused(registry, root) -> None:
    """A sha256 mismatch is refused and never reaches the cache."""
    swapped = b"different bytes entirely"
    (root / "release" / "swapped.pt").write_bytes(swapped)
    claimed = "cd" * 32                     # nothing cached under it, so the fetch runs
    entry = M.ModelEntry(id="swapped", url=f"{registry.base}/swapped.pt", sha256=claimed)

    with pytest.raises(M.RegistryError) as caught:
        M.fetch(entry)
    message = str(caught.value)
    assert hashlib.sha256(swapped).hexdigest()[:16] in message, "names what arrived"
    assert claimed[:16] in message, "names what was expected"
    assert not list(M.cache_dir().glob("*.part"))
    assert M.cached_path(hashlib.sha256(swapped).hexdigest()) is None


def test_a_pin_survives_the_release_moving(registry, root) -> None:
    """A tag that later points at other weights is a mismatch, not a substitution."""
    path = create_project(root / "pinned")
    write_config(path, {"model.source": "trypanosome-pose", "model.sha256": "ab" * 32})
    with pytest.raises(M.RegistryError):
        M.resolve_model(load_config(path))


def test_an_unknown_model_says_what_is_on_offer(registry, root) -> None:
    path = create_project(root / "unknown")
    write_config(path, {"model.source": "not-a-model"})
    with pytest.raises(M.RegistryError) as caught:
        M.resolve_model(load_config(path))
    message = str(caught.value)
    assert "not-a-model" in message
    assert registry.base in message
    assert "trypanosome-pose" in message, "what the registry does offer"


def test_an_empty_source_means_the_local_file(root) -> None:
    weights = root / "local.pt"
    weights.write_bytes(b"")
    cfg = load_config(create_project(root / "local", model=weights))
    assert M.resolve_model(cfg) == weights.resolve()
    assert M.local_model_path(cfg) == weights.resolve(), "the status probe downloads nothing"


def test_the_manifest_configures_the_project(registry, root) -> None:
    path = create_project(root / "manifest")
    told = M.apply_manifest(path, M.find_model("trypanosome-pose"), DIGEST)
    cfg = load_config(path)

    assert cfg.detection.imgsz == 1280, told
    assert abs(cfg.detection.conf - 0.3) < 1e-9
    assert cfg.tiling.tile == 512
    assert cfg.detection.task == "pose"
    assert cfg.keypoints.nodes[:2] == ["Head", "Base"] and len(cfg.keypoints.nodes) == 8
    assert len(cfg.keypoints.edges) == 3, cfg.keypoints.edges
    assert cfg.model.sha256 == DIGEST, "the digest is pinned into the project"
    assert any("conf" in t for t in told) and any("keypoint" in t for t in told), (
        f"every change is reported: {told}")
    assert cfg.paths.video_dir == (root / "manifest" / "videos").resolve(), (
        "a manifest cannot repoint the project at other movies")
    assert path.read_text(encoding="utf-8").count("#") > 40, "the comments survive"


def test_it_warns_where_the_project_disagrees_with_the_model(registry, root) -> None:
    """Reported, never corrected."""
    path = create_project(root / "assumes")
    write_config(path, {"tiff.fps": 30})
    warnings = M.mismatch_warnings(load_config(path), M.find_model("trypanosome-pose"))
    assert any("100" in w and "30" in w for w in warnings), f"frame rate: {warnings}"
    assert any("0.325" in w for w in warnings), f"pixel size: {warnings}"


def test_the_cache_can_be_pruned(registry) -> None:
    M.fetch(M.find_model("trypanosome-pose"))
    assert M.cached_path(DIGEST) is not None
    assert M.prune_cache(keep={DIGEST}) == 0, "keeps what is asked for"
    assert M.prune_cache(keep=set()) >= 1, "removes what is not"
    assert M.cached_path(DIGEST) is None


# --------------------------------------------------------------------------- #
# private repositories, which have no anonymous download url
# --------------------------------------------------------------------------- #
def test_a_github_registry_url_parses() -> None:
    assert M._github_source("github://o/r") == ("o", "r", "")
    assert M._github_source("github://o/r@v2") == ("o", "r", "v2")


def test_a_private_release_is_fetched_through_the_api(root, monkeypatch) -> None:
    """Assets are found by name and downloaded as octet-stream with the token."""
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
        "framework": "ultralytics", "sha256": DIGEST, "size": len(WEIGHTS),
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

    models = M.list_models(refresh=True)
    assert [m.id for m in models] == ["tryp"], "the index is read out of the release"
    assert models[0].asset == "tryp.pt", "an entry may name an asset instead of a url"
    assert models[0].name == "tryp.pt"

    M.prune_cache(keep=set())
    assert M.fetch(models[0]).read_bytes() == WEIGHTS
    assert any(u.endswith("/assets/2") and o["accept"] == "application/octet-stream"
               for u, o in seen), seen[-1]
    assert all(o["token"] for _u, o in seen)


def test_a_token_is_only_ever_sent_to_github(monkeypatch) -> None:
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
        with pytest.raises(_Boom):
            M._get(url)
        assert not any(k.lower() == "authorization" for k in sent), f"{url}: {sent}"

    sent.clear()
    with pytest.raises(_Boom):
        M._get("https://api.github.com/repos/o/r/releases/latest")
    assert any(k.lower() == "authorization" for k in sent), sent


# --------------------------------------------------------------------------- #
# models this tool cannot run
# --------------------------------------------------------------------------- #
SEG = M.ModelEntry(id="leish-seg", url="https://example.org/seg.pt",
                   task="segment", modality="fluorescence", framework="ultralytics",
                   config={"detection": {"conf": 0.4}})


def test_a_model_this_tool_cannot_run_is_listed_and_explained() -> None:
    assert SEG.task == "segment"
    assert not SEG.runnable
    assert "segment" in SEG.unsupported_reason and "detect" in SEG.unsupported_reason
    assert "fluorescence" in SEG.label and "ultralytics" in SEG.label, SEG.label


def test_runnable_needs_a_supported_task_and_a_framework() -> None:
    for task in ("pose", "detect"):
        assert M.ModelEntry(id=task, url="https://example.org/p.pt", task=task,
                            framework="ultralytics").runnable
    assert M.ModelEntry(id="u", url="https://example.org/u.pt",
                        framework="ultralytics").runnable, "no task stated is fine"
    assert not M.ModelEntry(id="q", url="https://example.org/q.pt").runnable, (
        "no framework stated means nothing knows how to load it")


def test_an_unrunnable_manifest_applies_its_settings_but_not_its_task(root) -> None:
    path = create_project(root / "seg")
    told = M.apply_manifest(path, SEG)
    cfg = load_config(path)
    assert cfg.detection.conf == 0.4
    assert cfg.detection.task in ("detect", "pose"), cfg.detection.task
    assert any(t.startswith("NOT applied") for t in told), told


def test_the_cache_holds_whatever_the_asset_is(registry, root) -> None:
    """An entry is bytes plus a digest, whatever the format."""
    payload = b"not a torch checkpoint at all" * 100
    (root / "release" / "m2f.tar.gz").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    entry = M.ModelEntry(id="leish-m2f", url=f"{registry.base}/m2f.tar.gz", sha256=sha,
                         task="segment", framework="hf-transformers",
                         preprocess={"input_size": [1024, 1024],
                                     "normalize": {"mean": [0.485], "std": [0.229]}})

    path = M.fetch(entry)
    assert path.read_bytes() == payload
    assert path.name == "m2f.tar.gz"
    assert M.cached_path(sha) == path, "found again without knowing the extension"
    assert entry.preprocess["input_size"] == [1024, 1024]
    assert not entry.runnable and "ultralytics" in entry.unsupported_reason


# --------------------------------------------------------------------------- #
# scale ranges and bad manifest values
# --------------------------------------------------------------------------- #
def test_a_scale_range_reads_as_its_span() -> None:
    assert M.span([0.2, 0.65]) == (0.2, 0.65)
    assert M.span([0.325, 0.65, 1.3]) == (0.325, 1.3), "discrete values read as their span"
    assert M.span(0.325) == (0.325, 0.325)
    assert M.describe_span(0.325, "um") == "0.325 um"
    assert M.describe_span([0.2, 0.65], "um") == "0.2 to 0.65 um"


def test_a_scale_range_is_compared_rather_than_announced(root) -> None:
    """A multi-scale model warns only when a measured pixel size falls outside it."""
    cfg = load_config(create_project(root / "scales"))
    multi = M.ModelEntry(id="multi", url="https://x/m.pt", framework="ultralytics",
                         assumes={"pixel_size_um": [0.2, 0.65], "fps": 100})
    single = M.ModelEntry(id="single", url="https://x/s.pt", framework="ultralytics",
                          assumes={"pixel_size_um": 0.325})

    assert not any("pixel" in w for w in M.mismatch_warnings(cfg, multi))
    assert any("0.325" in w for w in M.mismatch_warnings(cfg, single))
    inside = M.mismatch_warnings(cfg, multi, pixel_size_um=0.5)
    outside = M.mismatch_warnings(cfg, multi, pixel_size_um=1.3)
    assert not any("pixel" in w for w in inside), inside
    assert any("1.3" in w and "0.65" in w for w in outside), outside
    assert multi.trained_at == "100 fps, 0.2 to 0.65 um/px"


def test_a_value_the_loader_rejects_costs_only_that_value(root) -> None:
    path = create_project(root / "badvalue")
    entry = M.ModelEntry(id="sloppy", url="https://x/s.pt", framework="ultralytics",
                         task="pose",
                         config={"detection": {"conf": 0.4},
                                 "tiling": {"tile": 1280, "enabled": "sometimes"}})
    told = M.apply_manifest(path, entry)
    cfg = load_config(path)
    assert cfg.detection.conf == 0.4 and cfg.tiling.tile == 1280
    assert cfg.tiling.enabled == "auto", "the bad value is not applied"
    assert any("NOT applied" in t and "enabled" in t for t in told), told


def test_the_picker_offers_only_what_this_tool_can_run(app, monkeypatch) -> None:
    """Unrunnable entries are counted, not listed."""
    from trackanno.gui.hub import ModelRegistryDialog

    offered = [
        M.ModelEntry(id="pose", url="https://x/p.pt", version="v2", task="pose",
                     framework="ultralytics", notes="the one to use"),
        M.ModelEntry(id="seg", url="https://x/s.pt", task="segment",
                     framework="ultralytics"),
        M.ModelEntry(id="m2f", url="https://x/m.zip", task="segment",
                     framework="hf-transformers"),
    ]
    monkeypatch.setattr(M, "list_models", lambda refresh=False: offered)
    monkeypatch.setattr(M, "registry_url", lambda: "https://example.org/index.json")

    dialog = ModelRegistryDialog()
    listed = [dialog.list.item(i).text() for i in range(dialog.list.count())]
    assert len(listed) == 1 and "pose" in listed[0], listed
    assert "2 further" in dialog.detail.text(), dialog.detail.text()

    dialog.list.setCurrentRow(0)
    assert dialog.ok.isEnabled()
    assert "Download" in dialog.ok.text()
    assert "the one to use" in dialog.detail.text()
    dialog.close()
