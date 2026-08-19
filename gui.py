#!/usr/bin/env python
"""The whole pipeline in one window.  ->  uv run gui.py --help

Deliberately not named trackanno.py: a script in the repo root shadows the
installed package of the same name for every other entry point.
"""

from trackanno.gui.hub import main

if __name__ == "__main__":
    raise SystemExit(main())
