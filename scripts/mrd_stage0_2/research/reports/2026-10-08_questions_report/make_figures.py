"""Regenerate every figure of the 2026-10-08 questions report.

Each section has one figure module, figs_<NN>_<topic>.py, whose main() reads stored EXP
artifacts (through report_style.exp_dir()/diag_dir()) and writes figures/<name>.pdf plus a
.png preview. This script imports the modules in section order and calls main().

Usage (from the repo root or this directory, with the lab env's python):
    python make_figures.py            # all sections
    python make_figures.py 03 07      # only the listed sections (section 2b is "02b"; section 5's protein
                                      # explainer figures fig05g-fig05k and fig05m are "05p"; its CD63
                                      # and total-ADT figures fig05n, fig05o, fig05q, fig05r are "05q")

build.sh calls this before compiling report.tex. Nothing here writes to EXP.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# RAPIDS/cupy (pulled in by anndata in figs_03) needs CONDA_PREFIX; default it to this interpreter's env.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

# Section number -> figure module, in report order.
MODULES = {
    "01": "figs_01_cohort",
    "02": "figs_02_shape",
    "02b": "figs_02b_d26",
    "03": "figs_03_aggregate",
    "04": "figs_04_threshold",
    "05": "figs_05_pc8",
    "05p": "figs_05p_protein_explainer",  # section 5's protein explainers (runs the Pavan update's make_figures_v3.py)
    "05q": "figs_05q_cd63_depth",  # section 5's CD63 and total-ADT follow-up (D29): fig05n, fig05o, fig05q, fig05r
    "06": "figs_06_inductive",
    "07": "figs_07_curves",
    "08": "figs_08_dr_vs_direct",
    "09": "figs_09_timing",
}


def main(argv: list[str]) -> int:
    wanted = argv or list(MODULES)
    unknown = [nn for nn in wanted if nn not in MODULES]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}; choose from {', '.join(MODULES)}", file=sys.stderr)
        return 2
    failed = []
    for nn in wanted:
        name = MODULES[nn]
        t0 = time.time()
        print(f"[make_figures] {nn}: {name}.main()", flush=True)
        try:
            importlib.import_module(name).main()
        except Exception as exc:  # keep going so one broken module does not block the rest
            failed.append(nn)
            print(f"[make_figures] {nn} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"[make_figures] {nn}: done in {time.time() - t0:.1f} s", flush=True)
    if failed:
        print(f"[make_figures] failed sections: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
