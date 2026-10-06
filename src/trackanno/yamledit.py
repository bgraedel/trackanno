"""Surgical edits to config.yaml that keep its comments.

``config.yaml`` is two thirds comment, and those comments document what each knob
does; a ``safe_load`` + ``safe_dump`` round trip throws every one of them away
and reorders the file. So the *text* is edited: the schema is two levels deep and
every leaf the GUI touches is a scalar or a short list, which makes
line-oriented replacement reliable. Structural blocks (``categories``) go through
:func:`replace_section`, which does lose the comments inside that block.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["set_values", "replace_section", "format_scalar"]

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?P<rest>.*)$")
# Unquoted, unambiguous scalars can go into the file as they are.
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/ ]*$")


def format_scalar(value: Any) -> str:
    """Render one value the way a human would type it into the file."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        value = str(value).replace("\\", "/")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {format_scalar(v)}" for k, v in value.items()) + "}"
    text = str(value)
    if text == "":
        return '""'
    # Quote anything that would parse as something other than a string, or that
    # carries YAML punctuation. `1280` as a path would silently become an int.
    if _PLAIN.match(text) and text.lower() not in (
        "true", "false", "yes", "no", "on", "off", "null", "none", "~"
    ):
        try:
            float(text)
        except ValueError:
            return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _split_comment(rest: str) -> tuple[str, str]:
    """Separate a value from the trailing comment that documents it."""
    in_quote = None
    for i, ch in enumerate(rest):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in "\"'":
            in_quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1].isspace()):
            # The run of spaces before the '#' belongs to the comment, or value
            # and documentation weld together.
            value = rest[:i].rstrip()
            return value, rest[len(value):]
    return rest.rstrip(), ""


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    """(first line inside, one past last line inside) for a top-level mapping."""
    start = None
    for i, line in enumerate(lines):
        match = _KEY.match(line)
        if match is None or match.group("indent"):
            continue
        if match.group("key") == section:
            start = i + 1
            continue
        if start is not None:
            return start, i
    return (start, len(lines)) if start is not None else None


def _last_content_line(lines: list[str], start: int, end: int) -> int:
    """End of the section ignoring the blank lines and comments that precede
    the next one - those belong to the *next* section, not this one."""
    last = start
    for i in range(start, end):
        if lines[i].strip() and not lines[i].lstrip().startswith("#"):
            last = i + 1
    return last


def set_values(text: str, updates: dict[str, Any]) -> str:
    """Apply ``{"section.key": value}`` (or ``{"key": value}``) to YAML text.

    Existing keys are rewritten in place, keeping indentation and the trailing
    comment; missing keys are appended to their section, missing sections to the
    file."""
    lines = text.splitlines()
    by_section: dict[str, dict[str, Any]] = {}
    for dotted, value in updates.items():
        section, _, key = dotted.rpartition(".")
        by_section.setdefault(section, {})[key] = value

    for section, entries in by_section.items():
        remaining = dict(entries)
        if section:
            bounds = _section_bounds(lines, section)
            if bounds is None:
                block = ["", f"{section}:"] + [
                    f"  {k}: {format_scalar(v)}" for k, v in remaining.items()
                ]
                lines.extend(block)
                continue
            start, end = bounds
        else:
            start, end = 0, len(lines)

        indent = None
        for i in range(start, end):
            match = _KEY.match(lines[i])
            if match is None:
                continue
            key = match.group("key")
            if section and not match.group("indent"):
                continue
            if not section and match.group("indent"):
                continue
            if indent is None:
                indent = match.group("indent")
            if key not in remaining:
                continue
            _, comment = _split_comment(match.group("rest"))
            lines[i] = f"{match.group('indent')}{key}: {format_scalar(remaining.pop(key))}{comment}"

        if remaining:
            indent = indent if indent is not None else ("  " if section else "")
            insert_at = _last_content_line(lines, start, end)
            new = [f"{indent}{k}: {format_scalar(v)}" for k, v in remaining.items()]
            lines[insert_at:insert_at] = new

    return "\n".join(lines) + ("\n" if text.endswith("\n") or not text else "")


def replace_section(text: str, section: str, value: Any, header: str = "") -> str:
    """Replace a whole top-level block with freshly dumped YAML.

    For ``categories`` and ``keypoints``, whose shape the GUI can change.
    Comments *inside* the block do not survive - hence ``header``, and hence not
    the scalar sections."""
    body = yaml.safe_dump(
        {section: value}, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).rstrip("\n")
    block = ([line for line in header.splitlines()] if header else []) + body.splitlines()

    lines = text.splitlines()
    bounds = _section_bounds(lines, section)
    if bounds is None:
        return "\n".join(lines + [""] + block) + "\n"

    start, end = bounds
    # Drop the comment block immediately above `section:` before writing a new
    # one, or every save prepends another copy of the injected header.
    head = _drop_leading_comments(lines[: start - 1])
    # Keep the `section:` line itself; stop before the blanks introducing the
    # next one.
    return "\n".join(
        head + block + lines[_last_content_line(lines, start, end):]
    ) + "\n"


def _drop_leading_comments(head: list[str]) -> list[str]:
    """Strip the comment block directly above a section heading - only a
    contiguous run touching it, so the *previous* section's comments survive."""
    cut = len(head)
    while cut > 0 and head[cut - 1].lstrip().startswith("#"):
        cut -= 1
    return head[:cut]
