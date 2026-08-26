"""Reusable project settings, saved from one project and applied to the next.

A template is a whole ``config.yaml``, comments included, stored outside any
project so it survives the project it came from.

It does NOT carry the three paths that describe one project on one disk:
``video_dir``, ``work_dir`` and ``csv`` are reset to what a new project uses.
``paths.model`` IS kept, and the New Project form overrides it when a model is
chosen. ``ui.annotator`` is cleared: a template is a protocol, not a person.

Saving copies the project's own file rather than re-serialising it, so the
comments come along unchanged, and parses the result with the real loader, so a
template that would not load cannot be saved.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .yamledit import set_values

__all__ = [
    "Template", "BUILTIN_NAME", "templates_dir", "list_templates", "find_template",
    "template_text", "save_template", "delete_template",
]

BUILTIN_NAME = "default"
TEMPLATES_DIRNAME = "templates"

# The typed name lives in the file, not only in the file name, so a template
# keeps the name it was given ("trypanosome 100 fps", not "trypanosome-100-fps")
# and still describes itself if it is copied to somebody else.
_HEADER = "# trackanno template: "

# Reset on save: these describe one project on one disk, not a way of working.
_PROJECT_PATHS = {
    "paths.video_dir": "./videos",
    "paths.work_dir": "./work",
    "paths.csv": "./annotations.csv",
    "ui.annotator": "",
}


@dataclass(frozen=True)
class Template:
    """A named set of settings. ``path`` is None for the built-in default."""

    name: str
    path: Path | None = None

    @property
    def builtin(self) -> bool:
        return self.path is None

    @property
    def label(self) -> str:
        return f"{self.name} (built in)" if self.builtin else self.name


def templates_dir() -> Path:
    from .project import _state_dir

    return _state_dir() / TEMPLATES_DIRNAME


def _slug(name: str) -> str:
    """A file name that survives every filesystem, from a name a person typed."""
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


def list_templates() -> list[Template]:
    """The built-in first, then saved ones alphabetically."""
    out = [Template(BUILTIN_NAME)]
    directory = templates_dir()
    try:
        files = sorted(directory.glob("*.yaml"), key=lambda p: p.stem.lower())
    except OSError:
        return out
    out.extend(Template(_stored_name(p), p) for p in files)
    return out


def find_template(name: str | None) -> Template | None:
    """Match a typed name, its slug, or the file name. None if unknown.

    All three, because the typed name, the name shown in the picker and the name
    on disk can differ by punctuation.
    """
    if name is None:
        return None
    wanted = str(name).strip().lower()
    if not wanted:
        return None
    slugged = _slug(wanted)
    for template in list_templates():
        candidates = {template.name.lower()}
        if template.path is not None:
            candidates |= {template.path.stem.lower(), _slug(template.name)}
        if wanted in candidates or slugged in candidates:
            return template
    return None


def template_text(template: Template | str | None) -> str:
    """The ``config.yaml`` text a template stands for.

    Accepts a name for convenience. An unknown name, or the built-in, gives the
    packaged default, so a project is always creatable even if the state
    directory has been wiped between sessions.
    """
    from .project import DEFAULT_CONFIG

    if isinstance(template, str) or template is None:
        template = find_template(template)
    if template is None or template.builtin or template.path is None:
        return DEFAULT_CONFIG
    text = template.path.read_text(encoding="utf-8")
    # The header names the template, not the project; a config.yaml written from
    # it should read as an ordinary project file.
    if text.startswith(_HEADER):
        _, _, text = text.partition("\n")
    return text


def save_template(name: str, config_path: str | Path, overwrite: bool = False) -> Template:
    """Store the settings in ``config_path`` under ``name``. Returns the Template.

    Raises ``FileExistsError`` unless ``overwrite``, ``ValueError`` on a name
    that collides with the built-in, and whatever the loader raises if the
    result would not load.
    """
    from .config import load_config

    name = str(name).strip()
    if not name:
        raise ValueError("give the template a name")
    if name.lower() == BUILTIN_NAME:
        raise ValueError(f"{BUILTIN_NAME!r} is the built-in template; choose another name")

    source = Path(config_path)
    text = set_values(source.read_text(encoding="utf-8"), dict(_PROJECT_PATHS))

    # Parse it as a project before storing it: the error reads far better here
    # than three steps later, when the project is created.
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "config.yaml"
        probe.write_text(text, encoding="utf-8")
        load_config(probe)

    directory = templates_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_slug(name)}.yaml"
    if target.exists() and not overwrite:
        raise FileExistsError(f"a template named {target.stem!r} already exists")

    # Written beside its real name and moved into place, so an interrupted save
    # cannot leave a half-written template to be picked up later.
    staged = target.with_suffix(".yaml.part")
    staged.write_text(f"{_HEADER}{name}\n{text}", encoding="utf-8")
    staged.replace(target)
    return Template(name, target)


def delete_template(name: str) -> bool:
    """Remove a saved template. False if it was not there, or is the built-in."""
    template = find_template(name)
    if template is None or template.builtin or template.path is None:
        return False
    try:
        template.path.unlink()
    except OSError:
        return False
    return True
