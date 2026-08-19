"""The named-assertion helper the suites are written in.

A plain ``assert`` reports the expression; these tests want to report the CLAIM,
because most of them are one scenario making a dozen related statements and
"``window.position == 1`` is False" does not say which of them broke. ``check``
carries the sentence, so a failure reads as the rule that stopped holding.

It also counts, which is why a run reports checks rather than only tests: one
end-to-end scenario is a single pytest test and thirty separate claims.
"""

from __future__ import annotations

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}".rstrip())
    PASSED.append(name)
    print(f"  ok  {name}")
