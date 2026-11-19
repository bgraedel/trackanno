"""Project templates: shipped presets (overrides on the default) and saved config.yaml copies."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .yamledit import set_values

__all__ = [
    "Template", "Preset", "BUILTIN_NAME", "BUILTIN_SUMMARY", "templates_dir",
    "list_templates", "find_template", "template_text", "save_template",
    "export_template", "local_files", "delete_template", "list_presets",
    "find_preset", "preset_text",
]

BUILTIN_NAME = "default"
BUILTIN_SUMMARY = "The packaged default: boxes, suggester on."
TEMPLATES_DIRNAME = "templates"
PRESETS_DIRNAME = "presets"

# First line of a saved template; keeps the typed name, not just the file slug.
_HEADER = "# trackanno template: "

# Project-specific keys reset on save.
_PROJECT_PATHS = {
    "paths.video_dir": "./videos",
    "paths.work_dir": "./work",
    "paths.csv": "./annotations.csv",
    "ui.annotator": "",
}

# Kept in a template but only valid on this computer; flagged on export.
_LOCAL_FILES = ("paths.model", "centerline.checkpoint", "centerline.hint_model", "centerline.code")


@dataclass(frozen=True)
class Preset:
    """Overrides on the packaged default, shipped in ``presets/``; ``id`` is the file name."""

    id: str
    name: str
    summary: str = ""
    about: str = ""
    order: int = 0
    overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Template:
    """A named set of settings: the built-in default, a ``preset``, or a saved file at ``path``."""

    name: str
    path: Path | None = None
    preset: str = ""
    summary: str = ""
    about: str = ""

    @property
    def builtin(self) -> bool:
        """Shipped with trackanno, so not deletable."""
        return self.path is None

    @property
    def label(self) -> str:
        if self.preset:
            return f"{self.name} (preset)"
        return f"{self.name} (built in)" if self.builtin else self.name


def templates_dir() -> Path:
    from .project import _state_dir

    return _state_dir() / TEMPLATES_DIRNAME


def _slug(name: str) -> str:
    """A filesystem-safe file name for a typed name."""
    out = re.sub(r"[^A-Za-z0-9._ -]+", "-", str(name)).strip(" .-")
    out = re.sub(r"[\s-]+", "-", out)
    return out.lower() or "template"


def _stored_name(path: Path) -> str:
    """The name in the header line, or the file name if there is none."""
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return path.stem
    if first.startswith(_HEADER):
        return first[len(_HEADER):].strip() or path.stem
    return path.stem


def _preset_files() -> list:
    """The shipped preset files."""
    try:
        directory = resources.files(__package__).joinpath(PRESETS_DIRNAME)
        return sorted(
            (f for f in directory.iterdir() if f.name.endswith(".yaml")),
            key=lambda f: f.name,
        )
    except (OSError, FileNotFoundError, ModuleNotFoundError, TypeError):
        return []


def _read_preset(file) -> Preset | None:
    """One preset file, or None if it is unreadable or not a mapping."""
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    ident = file.name[: -len(".yaml")]
    overrides = data.get("set") or {}
    if not isinstance(overrides, dict):
        return None
    return Preset(
        id=ident,
        name=str(data.get("name") or ident).strip(),
        summary=" ".join(str(data.get("summary") or "").split()),
        about=" ".join(str(data.get("about") or "").split()),
        order=int(data.get("order") or 0),
        overrides={str(k): v for k, v in overrides.items()},
    )


def list_presets() -> list[Preset]:
    """The shipped presets in ladder order; unreadable ones are skipped."""
    found = [p for p in (_read_preset(f) for f in _preset_files()) if p is not None]
    return sorted(found, key=lambda p: (p.order, p.name.lower()))


def find_preset(name: str | None) -> Preset | None:
    """Match a preset by its name, its slug or its file name. None if unknown."""
    if name is None:
        return None
    wanted = str(name).strip().lower()
    if not wanted:
        return None
    slugged = _slug(wanted)
    for preset in list_presets():
        candidates = {preset.id.lower(), preset.name.lower(), _slug(preset.name)}
        if wanted in candidates or slugged in candidates:
            return preset
    return None


def preset_text(preset: Preset | str | None) -> str:
    """The default ``config.yaml`` text with the preset's values set in place."""
    from .project import DEFAULT_CONFIG

    if isinstance(preset, str) or preset is None:
        preset = find_preset(preset)
    if preset is None:
        return DEFAULT_CONFIG
    return set_values(DEFAULT_CONFIG, dict(preset.overrides))


def list_templates() -> list[Template]:
    """The built-in default, then the presets in ladder order, then saved ones."""
    out = [Template(BUILTIN_NAME, summary=BUILTIN_SUMMARY)]
    out.extend(
        Template(p.name, preset=p.id, summary=p.summary, about=p.about)
        for p in list_presets()
    )
    directory = templates_dir()
    try:
        files = sorted(directory.glob("*.yaml"), key=lambda p: p.stem.lower())
    except OSError:
        return out
    out.extend(Template(_stored_name(p), p) for p in files)
    return out


def find_template(name: str | None) -> Template | None:
    """Match a typed name, its slug, or the file name. None if unknown."""
    if name is None:
        return None
    wanted = str(name).strip().lower()
    if not wanted:
        return None
    slugged = _slug(wanted)
    for template in list_templates():
        candidates = {template.name.lower()}
        if template.path is not None or template.preset:
            candidates |= {_slug(template.name)}
        if template.path is not None:
            candidates |= {template.path.stem.lower()}
        if template.preset:
            candidates |= {template.preset.lower()}
        if wanted in candidates or slugged in candidates:
            return template
    return None


def _strip_header(text: str) -> str:
    if text.startswith(_HEADER):
        _, _, text = text.partition("\n")
    return text


def template_text(template: Template | str | Path | None) -> str:
    """The ``config.yaml`` text for a template, its name, or a template file.

    None gives the default; an unknown name raises ValueError."""
    from .project import DEFAULT_CONFIG

    if not template:
        return DEFAULT_CONFIG
    if not isinstance(template, Template):
        found = find_template(str(template))
        if found is None:
            path = Path(template)
            if not path.is_file():
                raise ValueError(f"no template named {str(template)!r}")
            text = _strip_header(path.read_text(encoding="utf-8"))
            return set_values(text, dict(_PROJECT_PATHS))
        template = found
    if template.preset:
        return preset_text(template.preset)
    if template.builtin or template.path is None:
        return DEFAULT_CONFIG
    return _strip_header(template.path.read_text(encoding="utf-8"))


def _write(target: Path, text: str) -> None:
    staged = target.with_name(target.name + ".part")
    staged.write_text(text, encoding="utf-8")
    staged.replace(target)


def save_template(name: str, config_path: str | Path, overwrite: bool = False) -> Template:
    """Save ``config_path``'s settings as template ``name``; refuses clashes and bad configs.

    ``config_path`` can be a project's config.yaml or an exported template file."""
    from .config import load_config

    name = str(name).strip()
    if not name:
        raise ValueError("give the template a name")
    if name.lower() == BUILTIN_NAME:
        raise ValueError(f"{BUILTIN_NAME!r} is the built-in template; choose another name")
    clash = find_preset(name)
    if clash is not None:
        raise ValueError(f"{clash.name!r} is a preset that ships with trackanno; choose another name")

    text = _strip_header(Path(config_path).read_text(encoding="utf-8"))
    text = set_values(text, dict(_PROJECT_PATHS))
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "config.yaml"
        probe.write_text(text, encoding="utf-8")
        load_config(probe)

    directory = templates_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_slug(name)}.yaml"
    if target.exists() and not overwrite:
        raise FileExistsError(f"a template named {target.stem!r} already exists")
    _write(target, f"{_HEADER}{name}\n{text}")
    return Template(name, target)


def export_template(name: str, dest: str | Path) -> Path:
    """Write template ``name`` to ``dest`` as a file to share."""
    template = find_template(name)
    if template is None:
        raise ValueError(f"no template named {name!r}")
    dest = Path(dest)
    _write(dest, f"{_HEADER}{template.name}\n{template_text(template)}")
    return dest


def local_files(path: str | Path) -> dict[str, str]:
    """The settings in a template or config file that name files on this computer."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    remote = bool((data.get("model") or {}).get("source"))
    out = {}
    for key in _LOCAL_FILES:
        section, _, leaf = key.partition(".")
        value = (data.get(section) or {}).get(leaf)
        if value and not (key == "paths.model" and remote):
            out[key] = str(value)
    return out


def delete_template(name: str) -> bool:
    """Remove a saved template. False if it was not there, or ships with trackanno."""
    template = find_template(name)
    if template is None or template.builtin or template.path is None:
        return False
    try:
        template.path.unlink()
    except OSError:
        return False
    return True
