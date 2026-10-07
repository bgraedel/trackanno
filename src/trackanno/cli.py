"""The `trackanno` command: no arguments opens the GUI, otherwise runs a subcommand."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# name -> (module, description); imported lazily so the GUI starts fast.
COMMANDS: dict[str, tuple[str, str]] = {
    "prepare": ("trackanno.detect.prepare", "detect, track and build the clip cache"),
    "suggest": ("trackanno.learn.suggest", "train the annotation-suggestion model"),
    "predict": ("trackanno.infer.predict", "label every frame with the trained model"),
    "render": ("trackanno.view.render", "draw annotated boxes onto the movies"),
    "transitions": ("trackanno.infer.transitions", "transition rates and dwell times"),
    "repair-links": ("trackanno.annotate.reassociate", "re-link orphaned labels to current clips"),
}

GUI_MODULE = "trackanno.gui.hub"


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: trackanno [PROJECT]          open the GUI",
        "       trackanno <command> [args]   run a pipeline step",
        "",
        "commands:",
        *(f"  {name:<{width}}  {about}" for name, (_, about) in COMMANDS.items()),
        "",
        "Run 'trackanno <command> --help' for a command's options.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv and argv[0] in COMMANDS:
        module = importlib.import_module(COMMANDS[argv[0]][0])
        return module.main(argv[1:])
    if argv and not Path(argv[0]).exists():
        print(usage(), file=sys.stderr)
        print(f"\ntrackanno: unknown command or project {argv[0]!r}", file=sys.stderr)
        return 2
    return importlib.import_module(GUI_MODULE).main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
