#!/usr/bin/env python
"""Entry point 1: detection + tracking + clip cache.  ->  uv run prepare.py --help"""

from trackanno.detect.prepare import main

if __name__ == "__main__":
    raise SystemExit(main())
