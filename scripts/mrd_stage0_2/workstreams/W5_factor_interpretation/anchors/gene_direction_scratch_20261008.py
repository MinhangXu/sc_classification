"""Regression anchor, not a pipeline module. Frozen: do not edit, import or extend.

Provenance
- Source: agent scratch script ``factcheck_agent/gene_direction.py`` from the
  2026-10-08 fact-check session (session scratchpad; the original file was not kept).
- Copied here on 2026-10-08 (wave-1 package P4_workstreams_core). The only change is
  path resolution: the hard-coded EXP constant and cohort-cache path were replaced by
  ``experiment_dir()`` and ``cohort_cache()`` from ``sc_classification.utils.paths``
  through the standard workstream import header (``parents[5]`` because this file
  sits one level deeper, in ``anchors/``). The computation is unchanged.
- Context: the fact-check corrections recorded in devlog entry
  2026-10-08_team-report-devlog-pivot (ask A-20261008-06); F9 (devlog entry
  2026-10-02_f9-ifna-pc8-loadings) holds the PC8 loadings themselves.

Purpose
Direction of the IFN-alpha PC8 top-loading genes in malignant vs normal cells: for 13
genes (both PC8 loading poles: IFI27, PSME2, PSMA3, CNP, LAP3, USP18 vs TXNIP, IFITM2,
TENT5A, NCOA7, TRIM5, ELF1, plus BST2), the within-patient malignant-minus-normal mean
of cohort-z-scored log1p expression from the cohort cache, for each supported patient.
Gene level only; it does not use Stage 1 scores or Stage 2 coefficients.

Numbers it reproduced on 2026-10-08 (agent session)
- Negative-loading genes are mostly higher in malignant cells, with patient-specific
  gene contributions: TXNIP P02 1.17, P09 0.75, about 0 in P03/P04/P07; IFITM2 P04
  0.79, P07 0.77; TENT5A P02 0.87, P07 0.86; NCOA7 P09 1.66, P02 0.64, P04 0.55;
  TRIM5 P03 0.57, P04 0.46; ELF1 P03 0.54 but P02 -0.35.
- Positive-loading genes show small, inconsistent shifts: IFI27 from -0.36 (P07) to
  +0.11 (P04); LAP3 from -0.53 (P07) to +0.55 (P02).
- Context, not computed here: Stage 2 weights PC8 negatively in every nonzero fold, so
  the malignant direction is toward the TXNIP/IFITM2/TENT5A/NCOA7 pole.
Caveat: scratch; descriptive only, not causal. Orientation statements must also name the
penalty (AGENTS.md R6).

Consumer
W5.2 (oriented per-patient gene-direction table for PC8 and FA7) and the PC8 card of
W5.1 (D18, slug factor_cards_v1) must reproduce these values before reporting new
numbers.

Inputs (EXP-relative): the cohort cache (study_config cohort_cache_relpath; read in
backed mode, 13 gene columns plus obs patient and CN.label).

Run manually from the repo root with the lab env (CPU, minutes):
    python scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/anchors/gene_direction_scratch_20261008.py

Original scratch docstring (verbatim):
Scratch (not a repo artifact): within-patient malignant-minus-normal mean of z-scored log1p expression for PC8 top genes.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[5]  # workstreams/<W#_slug>/anchors/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache
import numpy as np, pandas as pd, anndata as ad
EXP = experiment_dir()
cache = str(cohort_cache())
genes = ["IFI27","PSME2","PSMA3","CNP","LAP3","USP18","TXNIP","IFITM2","TENT5A","NCOA7","TRIM5","ELF1","BST2"]
a = ad.read_h5ad(cache, backed="r")
obs = a.obs[["patient","CN.label"]].copy()
present = [g for g in genes if g in a.var_names]
X = a[:, present].X
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
a.file.close()
X = np.asarray(X, float); Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1, X.std(0))
pat = obs["patient"].astype(str).to_numpy(); y = (obs["CN.label"].astype(str) == "cancer").to_numpy()
out = pd.DataFrame({p: Z[(pat == p) & y].mean(0) - Z[(pat == p) & ~y].mean(0) for p in ["P01","P02","P03","P04","P06","P07","P09"]}, index=present)
pd.set_option("display.width", 200)
print(out.round(2).to_string())
