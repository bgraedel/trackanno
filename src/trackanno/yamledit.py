"""Line-based edits to config.yaml that keep its comments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["set_values", "replace_section", "format_scalar"]

_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?P<rest>.*)$")
# Unquoted, unambiguous scalars can go into the file as they are.
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/ ]*$")


def _native(value: Any) -> Any:
    """A numpy scalar as the Python value it holds."""
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return value.item()
    return value


def format_scalar(value: Any) -> str:
    """Render one value the way a human would type it into the file."""
    value = _native(value)
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
    # Quote anything that would parse as something other than a string.
    if _PLAIN.match(text) and text.lower() not in (
        "true", "false", "yes", "no", "on", "off", "null", "none", "~"
    ):
        try:
            float(text)
        except ValueError:
            return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _split_comment(rest: str) -> tuple[str, str]:
    """Separate a value from its trailing comment (with the spaces before it)."""
    in_quote = None
    for i, ch in enumerate(rest):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in "\"'":
            in_quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1].isspace()):
            value = rest[:i].rstrip()
            return value, rest[len(value):]
    return rest.rstrip(), ""


def _is_content(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], i: int) -> int:
    """One past the last line of the value of the key on line ``i``."""
    own = _indent(lines[i])
    end = i + 1
    for j in range(i + 1, len(lines)):
        line = lines[j]
        if not _is_content(line):
            continue
        ind = _indent(line)
        if ind > own or (ind == own and line.lstrip().startswith("- ")):
            end = j + 1
        else:
            break
    return end


def _find_key(lines: list[str], start: int, end: int, key: str,
              indent: int | None) -> int | None:
    """Line index of ``key`` among the direct children in ``[start, end)``."""
    child = indent
    for i in range(start, end):
        if not _is_content(lines[i]):
            continue
        match = _KEY.match(lines[i])
        ind = _indent(lines[i])
        if child is None:
            child = ind
        if ind != child or match is None:
            continue
        if match.group("key") == key:
            return i
    return None


def _children(lines: list[str], i: int) -> tuple[int, int, int | None]:
    """(start, end, child indent) of the block under the key on line ``i``."""
    end = _block_end(lines, i)
    for j in range(i + 1, end):
        if _is_content(lines[j]):
            return i + 1, end, _indent(lines[j])
    return i + 1, end, None


def _locate(lines: list[str], path: list[str]) -> int | None:
    """Line index of the key at dotted ``path``, or None."""
    start, end, indent = 0, len(lines), 0
    found = None
    for depth, key in enumerate(path):
        found = _find_key(lines, start, end, key, indent)
        if found is None:
            return None
        if depth < len(path) - 1:
            start, end, indent = _children(lines, found)
            if indent is None:
                return None
    return found


def _write_line(lines: list[str], i: int, value: Any) -> None:
    """Set the key on line ``i`` to ``value``, replacing any block under it."""
    match = _KEY.match(lines[i])
    _, comment = _split_comment(match.group("rest"))
    end = _block_end(lines, i)
    lines[i:end] = [f"{match.group('indent')}{match.group('key')}: "
                    f"{format_scalar(value)}{comment}"]


def _last_content_line(lines: list[str], start: int, end: int) -> int:
    """End of a block ignoring the comments that introduce the next one."""
    last = start
    for i in range(start, end):
        if _is_content(lines[i]):
            last = i + 1
    return last


def _nest(value: Any, keys: list[str]) -> Any:
    for key in reversed(keys):
        value = {key: value}
    return value


def _set_one(lines: list[str], path: list[str], value: Any) -> None:
    at = _locate(lines, path)
    if at is not None:
        _write_line(lines, at, value)
        return
    # Insert under the deepest parent that exists.
    for depth in range(len(path) - 1, 0, -1):
        at = _locate(lines, path[:depth])
        if at is None:
            continue
        start, end, indent = _children(lines, at)
        if indent is None:
            # The parent holds a flow value or nothing: merge into it.
            current = _current(lines, path[:depth])
            merged = dict(current) if isinstance(current, dict) else {}
            target = merged
            for key in path[depth:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = value
            _write_line(lines, at, merged)
            return
        insert_at = _last_content_line(lines, start, end)
        lines[insert_at:insert_at] = [
            f"{' ' * indent}{path[depth]}: {format_scalar(_nest(value, path[depth + 1:]))}"]
        return
    if len(path) == 1:
        lines.append(f"{path[0]}: {format_scalar(value)}")
        return
    lines.extend(["", f"{path[0]}:",
                  f"  {path[1]}: {format_scalar(_nest(value, path[2:]))}"])


def _current(lines: list[str], path: list[str]):
    try:
        data = yaml.safe_load("\n".join(lines)) or {}
    except yaml.YAMLError:
        return None
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def set_values(text: str, updates: dict[str, Any]) -> str:
    """Apply ``{"section.key": value}`` to YAML text in place; missing keys are added."""
    lines = text.splitlines()
    for dotted, value in updates.items():
        _set_one(lines, dotted.split("."), _native(value))
    return "\n".join(lines) + ("\n" if text.endswith("\n") or not text else "")


def replace_section(text: str, section: str, value: Any, header: str = "") -> str:
    """Replace a top-level block with dumped YAML under ``header``; its comments are lost."""
    body = yaml.safe_dump(
        {section: value}, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).rstrip("\n")
    block = (header.splitlines() if header else []) + body.splitlines()

    lines = text.splitlines()
    at = _locate(lines, [section])
    if at is None:
        return "\n".join(lines + [""] + block) + "\n"
    end = _block_end(lines, at)
    head = _drop_leading_comments(lines[:at])
    return "\n".join(head + block + lines[end:]) + "\n"


def _drop_leading_comments(head: list[str]) -> list[str]:
    """Strip the comment run directly above a section heading."""
    cut = len(head)
    while cut > 0 and head[cut - 1].lstrip().startswith("#"):
        cut -= 1
    return head[:cut]
