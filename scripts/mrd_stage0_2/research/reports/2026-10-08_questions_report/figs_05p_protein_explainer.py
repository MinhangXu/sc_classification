"""Section 5 explainer figures for the user's 2026-10-10 protein questions (asks A-20261010-14..24).

One source for the report and the Pavan update: this module runs
research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v3.py, which draws the update's
fig5-fig7 and, into figures/ here, the report versions (no title or footer; the LaTeX caption carries them):

  fig05g_celltype_counts      full report Figure 19 (replaces fig05c in the text); unbuilt core appendix
  fig05h_rho_schematic_ab     full report Figure 20, panels A-B (how one rho is computed)
  fig05i_rho_schematic_cd     full report Figure 20 continued, panels C-D (three levels; what the sign means)
  fig05j_two_checks           full report Figure 22; unbuilt core appendix
  fig05k_rho_schematic_core   unbuilt core appendix: panels A and D of Figure 20, lettered A and B
  fig05m_celltype_core        core report page 7, Figure 6 bottom: Figure 19 compact (one column per type)

The core's protein-page figure (page 8, Figure 7 = fig05l_protein_core) is drawn by figs_05_pc8.py with this
script's loaders; the core's Table 1 is typed from table3_rho_three_levels_v3.csv.

It also rewrites the update's table3_rho_three_levels_v3.{md,csv}, whose numbers section 5's Table 3 copies.
Reads EXP read-only (D27, D18, the stored IFN-alpha PCA/FA K=40 scores and the cohort cache; never the raw
AnnData). About 20 s.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (HERE.parents[1] / "team_updates" / "2026-10-09_protein_association_for_pavan" / "make_figures_v3.py")


def main() -> None:
    spec = importlib.util.spec_from_file_location("make_figures_v3", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
