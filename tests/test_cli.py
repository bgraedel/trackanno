"""The `trackanno` dispatcher: routing, help and the GUI fallback."""

from __future__ import annotations

import pytest

from trackanno import cli


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_each_command_routes_to_its_parser(name, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([name, "--help"])
    assert exc.value.code == 0
    assert f"usage: trackanno {name}" in capsys.readouterr().out


def test_help_lists_every_command(capsys) -> None:
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert f"  {name}" in out


def test_unknown_command_fails(capsys) -> None:
    assert cli.main(["no-such-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_no_arguments_opens_the_gui(monkeypatch) -> None:
    import trackanno.gui.hub as hub

    calls = []
    monkeypatch.setattr(hub, "main", lambda argv=None: calls.append(argv) or 0)
    assert cli.main([]) == 0
    assert calls == [[]]


def test_a_project_path_opens_the_gui(monkeypatch, tmp_path) -> None:
    import trackanno.gui.hub as hub

    calls = []
    monkeypatch.setattr(hub, "main", lambda argv=None: calls.append(argv) or 0)
    assert cli.main([str(tmp_path)]) == 0
    assert calls == [[str(tmp_path)]]


def test_list_features_shows_groups_and_importance(tmp_path, capsys) -> None:
    import json

    from trackanno.config import load_config
    from trackanno.learn.suggest import REPORT_FILE
    from trackanno.project import create_project

    path = create_project(tmp_path / "proj")
    assert cli.main(["suggest", "--config", str(path), "--list-features"]) == 0
    out = capsys.readouterr().out
    assert "kinematic (trajectory and shape): selected" in out
    assert "no trained model" in out
    assert any(line.split()[:2] == ["*", "speed_mean"] for line in out.splitlines())

    work = load_config(path).paths.work_dir
    work.mkdir(parents=True, exist_ok=True)
    (work / REPORT_FILE).write_text(json.dumps({"behaviour": {"top_features": [
        {"feature": "speed_mean", "importance": 42.5}]}}), encoding="utf-8")
    assert cli.main(["suggest", "--config", str(path), "--list-features"]) == 0
    out = capsys.readouterr().out
    assert "behaviour" in out
    row = next(line for line in out.splitlines() if line.split()[1:2] == ["speed_mean"])
    assert row.endswith("42.5000")


def test_list_features_shows_embedding_and_context_groups(tmp_path, capsys) -> None:
    from trackanno.project import create_project, write_config

    path = create_project(tmp_path / "proj")
    write_config(path, {
        "suggest.embeddings": [{"name": "ae", "source": "file",
                                "path": "emb/{movie_id}.parquet"}],
        "suggest.context_frames": 100,
        "suggest.feature_groups": ["kinematic", "context", "emb_ae"],
    })
    assert cli.main(["suggest", "--config", str(path), "--list-features"]) == 0
    out = capsys.readouterr().out
    assert "emb_ae (file embedding): selected" in out
    assert "context (longer context): selected" in out
    lines = [line.split()[:2] for line in out.splitlines()]
    assert ["*", "emb_ae_0_mean"] in lines
    assert ["*", "ctx_speed_mean"] in lines

