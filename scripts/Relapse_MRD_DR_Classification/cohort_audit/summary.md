# Relapse/MRD Cohort Audit

- Input: `/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad`
- Total focus cells (`MRD`/`Relapse`, `cancer`/`normal`, known patient): 113208
- Patients with both coarse `MRD` and `Relapse`: 9
- Patients with all four classes present: 8

## Patients with paired MRD and Relapse

- P08: MRD_cancer=0, MRD_normal=1411, Relapse_cancer=568, Relapse_normal=874, MRD_tech=CITE, Relapse_tech=Multi
- P13: MRD_cancer=4, MRD_normal=5770, Relapse_cancer=3787, Relapse_normal=552, MRD_tech=CITE, Relapse_tech=CITE
- P05: MRD_cancer=8, MRD_normal=4351, Relapse_cancer=4166, Relapse_normal=169, MRD_tech=CITE, Relapse_tech=Multi
- P04: MRD_cancer=102, MRD_normal=1026, Relapse_cancer=7459, Relapse_normal=15, MRD_tech=CITE, Relapse_tech=Multi
- P07: MRD_cancer=22, MRD_normal=2163, Relapse_cancer=4320, Relapse_normal=97, MRD_tech=CITE, Relapse_tech=Multi
- P01: MRD_cancer=65, MRD_normal=18899, Relapse_cancer=1484, Relapse_normal=1255, MRD_tech=CITE,Multi, Relapse_tech=Multi
- P03: MRD_cancer=758, MRD_normal=3280, Relapse_cancer=5586, Relapse_normal=169, MRD_tech=CITE,Multi, Relapse_tech=Multi
- P02: MRD_cancer=230, MRD_normal=2100, Relapse_cancer=6460, Relapse_normal=4791, MRD_tech=CITE,Multi, Relapse_tech=Multi
- P09: MRD_cancer=375, MRD_normal=16751, Relapse_cancer=1045, Relapse_normal=610, MRD_tech=CITE, Relapse_tech=Multi

## Key caveats

- `MRD_cancer` is the limiting class for several patients (`P05`, `P07`, `P13`, and `P08` has zero MRD cancer cells).
- Most paired patients follow the expected tech pattern of MRD=`CITE` and Relapse=`Multi`, but `P13` has both timepoints in `CITE` and `P01`/`P02`/`P03` include mixed-tech MRD cells.
- These outputs should be treated as the source of truth for patient inclusion and CV feasibility thresholds in downstream runners.

