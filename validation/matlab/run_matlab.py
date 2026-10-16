"""Run the MATLAB pipeline on every input cells.py wrote, in parallel MATLAB processes.

    uv run python validation/matlab/run_matlab.py --cells OUT --pipeline PATH/flagella-analysis-pipeline
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", required=True, help="folder cells.py wrote")
    ap.add_argument("--pipeline", required=True, help="the pipeline's folder (holds MATLAB_scripts)")
    ap.add_argument("--matlab", default="matlab", help="the matlab executable")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    cells = Path(args.cells).resolve()
    meta = pd.read_csv(cells / "cells.csv")
    fps, scale = float(meta.fps.iloc[0]), float(meta.pixel_size_um.iloc[0])
    todo = sorted(str(f.parent) for f in cells.glob("*/*.tif_analysis/rawcoordinates.txt")
                  if not (f.parent / "compared.mat").exists())
    print(f"{len(todo)} folders on {args.workers} MATLAB processes")
    procs = []
    for w in range(args.workers):
        mine = todo[w::args.workers]
        if not mine:
            continue
        queue = cells / f"queue_{w}.txt"
        queue.write_text("\n".join(mine) + "\n", encoding="ascii")
        call = (f"cd('{HERE}'); run_pipeline('{queue}', '{Path(args.pipeline)}', {fps:g}, {scale:g})")
        log = open(cells / f"matlab_{w}.log", "w")
        procs.append((subprocess.Popen([args.matlab, "-batch", call], stdout=log, stderr=log), log))
    for proc, log in procs:
        proc.wait()
        log.close()
    failed = list(cells.glob("*/*.tif_analysis/failed.txt"))
    print(f"done; {len(failed)} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
