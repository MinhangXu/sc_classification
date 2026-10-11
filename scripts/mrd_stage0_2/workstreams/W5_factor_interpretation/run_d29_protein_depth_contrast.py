#!/usr/bin/env python
"""D29 (W5.3 follow-up): PC8 and FA7 x surface protein, two sensitivity questions D27 left open.

Asks A-20261010-27, -29..-32 (research/asks/2026-10-10_cd63-every-card-isotypes.md):
(1) Total-ADT sensitivity of the two checks. CLR subtracts each cell's mean log1p count over all 170
    antibodies, so a cell's total ADT count enters every CLR value; when a factor also tracks total ADT
    counts, every protein and isotype picks up a correlation with the factor through that one path. For
    the IFN-alpha PC8 and FA7 cards (oriented as in D18/D27), every ADT feature (161 markers + 9
    isotypes) and both labels, this recomputes D27's Spearman rho pooled, inside each supported patient
    and inside each D27 supported patient x predicted cell type group, raw and partial on the cell's
    log total ADT count (rank-based partial correlation: residualize the three ranks' x and y on the
    rank of log total ADT within the stratum, then correlate the residuals).
(2) A direct malignant-vs-healthy contrast, no factor involved: inside each supported patient x
    predicted cell type with at least 10 malignant and 10 healthy cells (Figure 19's 21 comparisons), and
    again inside one sample (372 of P09's 375 malignant cells are in one of its four samples),
    the AUC of each ADT feature's CLR (probability that a malignant cell has more than a healthy cell of
    the same patient and type; 0.5 = no difference), raw and after residualizing the CLR rank on the rank
    of log total ADT within the group (both labels together).
It also records how each isotype's CLR tracks total ADT and cell type, and how PC8 and FA7 track total
ADT and total RNA counts inside each patient.

Reads (EXP-relative, read only): analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/
(d27_ifna_pc8_fa7_adt.csv, d27_card_summary.csv, d27_strata.csv; D27 values are the regression targets);
stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa}/k=40/seed=42/
scores.npy; the cohort cache (obs patient, CN.label, predicted.annotation, nCount_RNA; obsm['ADT'] raw
counts). The raw AnnData is never loaded (AGENTS R8). Nothing is refit; no Stage 2 (c_policy:
not_applicable).

Writes (new slug protein_depth_contrast_v1; with --out-root DIR everything goes under
DIR/scorecards/stage2_diagnostic/protein_depth_contrast_v1/ and EXP is only read):
  d29_rho_depth.csv            factor x label x feature: pooled, median within patient (+ n same sign),
                               median inside patient x type (+ n same sign), raw and partial
  d29_rho_depth_strata.csv     the same per stratum (pooled, patient, patient x type), long
  d29_isotype_reference.csv    factor x label x level: largest isotype |rho|, raw and partial, and which
  d29_depth_axes.csv           factor x label x patient: rho with log total ADT and with log nCount_RNA;
                               isotype x label: rho of CLR with log total ADT, and healthy-cell type eta2
  d29_mal_vs_healthy.csv       level (patient | sample) x feature x patient (x sample) x type: n, median CLR
                               difference, AUC raw and adjusted, and the AUC of log total ADT itself
  d29_mal_vs_healthy_summary.csv  per feature: groups, groups with malignant lower, median AUC (raw, adj.)
  d29_regression_checks.csv, protein_depth_contrast_v1_{findings.md,manifest.json,artifact_ledger.csv}

Usage (lab env; about a minute on CPU, BLAS threads capped at 4):
  python scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py
  ... --resume      # skip if every output exists and reads back; recompute otherwise
  ... --out-root <scratch>/d29_try
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import cohort_cache, experiment_dir, git_state, study_config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SLUG = "protein_depth_contrast_v1"
ANALYSIS_ID = "D29"
D27_SLUG = "factor_modality_assoc_v1"
SCRIPT_REL = "scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
FACTORS = {  # name -> (card id, method, 0-based score column)
    "PC8": (f"{PANEL}__pca__k40__f008", "pca", 7),
    "FA7": (f"{PANEL}__fa__k40__f007", "fa", 6),
}
LABELS = ("normal", "malignant")  # D27 names; "normal" = healthy
CONTRAST_MIN = 10  # malignant-vs-healthy groups: >= 10 cells of each label (Figure 19's 21 comparisons)
TOL = 1e-6  # D27 writes its CSVs at 6 significant digits (factor_modality_assoc.FLOAT_FORMAT)
FLOAT_FORMAT = "%.8g"
OUTPUTS = ("d29_rho_depth.csv", "d29_rho_depth_strata.csv", "d29_isotype_reference.csv", "d29_depth_axes.csv",
           "d29_mal_vs_healthy.csv", "d29_mal_vs_healthy_summary.csv", "d29_regression_checks.csv")


def log(msg: str) -> None:
    print("[{}] {}".format(datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


class Paths:
    def __init__(self, exp: Path, out_root: Optional[Path] = None):
        self.exp = exp
        self.out_root = Path(os.path.abspath(out_root)) if out_root else None
        base = self.out_root if self.out_root else exp / "analysis"
        self.out = base / "scorecards" / "stage2_diagnostic" / SLUG
        if self.out.name != SLUG or base not in self.out.parents:
            raise RuntimeError("refusing to write outside the {} namespace: {}".format(SLUG, self.out))
        self.d27 = exp / "analysis" / "scorecards" / "stage2_diagnostic" / D27_SLUG

    def rel(self, path: Path) -> str:
        path = Path(os.path.abspath(path))
        try:
            return path.relative_to(Path(os.path.abspath(self.exp))).as_posix()
        except ValueError:
            if self.out_root is None:
                raise
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------


def _std_ranks(r: np.ndarray) -> np.ndarray:
    r = r - r.mean(axis=0)
    sd = np.sqrt((r ** 2).mean(axis=0))
    out = r / np.where(sd > 0, sd, 1.0)
    out[:, sd <= 0] = np.nan
    return out


def spearman_raw_partial(f: np.ndarray, Y: np.ndarray, z: np.ndarray) -> tuple:
    """Spearman rho of f with each column of Y, raw and partial on z (all ranks taken within the rows given).

    Partial: residualize rank(f) and rank(Y[:, j]) on [1, rank(z)] by least squares, then Pearson of the
    residuals (the rank-based partial correlation)."""
    rf = rankdata(f)[:, None]
    rY = rankdata(Y, axis=0, method="average")
    rz = rankdata(z)
    raw = (np.nan_to_num(_std_ranks(rf)).T @ np.nan_to_num(_std_ranks(rY)))[0] / len(f)
    Z = np.column_stack([np.ones(len(f)), rz])
    proj = Z @ np.linalg.pinv(Z)
    ef = rf - proj @ rf
    eY = rY - proj @ rY
    part = (np.nan_to_num(_std_ranks(ef)).T @ np.nan_to_num(_std_ranks(eY)))[0] / len(f)
    const = np.ptp(rY, axis=0) == 0
    raw[const] = np.nan
    part[const] = np.nan
    return np.clip(raw, -1, 1), np.clip(part, -1, 1)


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a = b) for independent draws (Mann-Whitney U / (n_a n_b))."""
    r = rankdata(np.concatenate([a, b]))
    return float((r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def residual_ranks(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    ry, rz = rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(len(y)), rz])
    return ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]


def same_sign(values: np.ndarray, med: float) -> int:
    return int(((np.sign(values) == np.sign(med)) & np.isfinite(values) & (np.sign(med) != 0)).sum())


def eta2(y: np.ndarray, groups: np.ndarray) -> float:
    df = pd.DataFrame({"y": y, "g": groups})
    tot = ((df["y"] - df["y"].mean()) ** 2).sum()
    between = df.groupby("g")["y"].apply(lambda s: len(s) * (s.mean() - df["y"].mean()) ** 2).sum()
    return float(between / tot) if tot > 0 else float("nan")


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def load(paths: Paths) -> Dict[str, object]:
    import anndata as ad

    cfg = study_config()
    path = paths.exp / cfg.get("cohort_cache_relpath", "")
    if not path.is_file():
        path = cohort_cache(must_exist=True)
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    if adata.n_obs != int(cfg["cohort"]["n_cells"]):
        raise AssertionError("unexpected cohort size {}".format(adata.n_obs))
    cells = pd.DataFrame({
        "patient": obs["patient"].astype(str).to_numpy(),
        "sample": obs["sample"].astype(str).to_numpy(),
        "label": np.where(obs["CN.label"].astype(str).eq("cancer"), "malignant", "normal"),
        "cell_type": obs["predicted.annotation"].astype(str).to_numpy(),
        "log_rna": np.log1p(pd.to_numeric(obs["nCount_RNA"]).to_numpy(dtype=np.float64)),
    })
    adt = adata.obsm["ADT"]
    names = [str(c) for c in adt.columns]
    counts = np.asarray(adt.to_numpy(), dtype=np.float64)
    adata.file.close()
    logged = np.log1p(counts)
    clr = logged - logged.mean(axis=1, keepdims=True)  # = factor_cards.clr_per_cell (D18, D27)
    cells["log_adt"] = np.log1p(counts.sum(axis=1))
    cs = pd.read_csv(paths.d27 / "d27_card_summary.csv").set_index("card_id")
    for name, (card, method, col) in FACTORS.items():
        sc = paths.exp / "stage1_dr" / f"panel_id={PANEL}" / f"method={method}" / "k=40" / "seed=42" / "scores.npy"
        x = np.asarray(np.load(sc), dtype=np.float64)[:, col]
        cells[name] = int(cs.loc[card, "orientation_sign"]) * (x - x.mean()) / x.std()
    st = pd.read_csv(paths.d27 / "d27_strata.csv")
    pct = st[st["kind"] == "patient_celltype"][["label", "patient", "cell_type", "n_cells"]].reset_index(drop=True)
    return {"cells": cells, "clr": clr, "names": names, "supported": list(cfg["supported_patients"]), "pct": pct,
            "cohort_path": path}


# --------------------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------------------


def rho_depth(data: Dict[str, object]) -> tuple:
    cells, clr, names = data["cells"], data["clr"], data["names"]
    long_rows: List[pd.DataFrame] = []
    for fname in FACTORS:
        for lab in LABELS:
            strata = [("pooled", "all", "", (cells["label"] == lab).to_numpy())]
            strata += [("patient", p, "", ((cells["label"] == lab) & (cells["patient"] == p)).to_numpy())
                       for p in data["supported"]]
            g = data["pct"][data["pct"]["label"] == lab]
            strata += [("patient_celltype", r["patient"], r["cell_type"],
                        ((cells["label"] == lab) & (cells["patient"] == r["patient"])
                         & (cells["cell_type"] == r["cell_type"])).to_numpy()) for _, r in g.iterrows()]
            for kind, pat, ct, m in strata:
                raw, part = spearman_raw_partial(cells.loc[m, fname].to_numpy(), clr[m], cells.loc[m, "log_adt"].to_numpy())
                long_rows.append(pd.DataFrame({"factor": fname, "label": lab, "kind": kind, "patient": pat,
                                               "cell_type": ct, "n_cells": int(m.sum()), "feature": names,
                                               "rho_raw": raw, "rho_partial": part}))
    long = pd.concat(long_rows, ignore_index=True)
    long["is_isotype"] = long["feature"].str.startswith("Isotype")
    long["feature_short"] = long["feature"].str.replace("^ADT-", "", regex=True)
    rows = []
    for (fname, lab, feat), g in long.groupby(["factor", "label", "feature"], sort=False):
        row = {"factor": fname, "label": lab, "feature": feat, "feature_short": g["feature_short"].iloc[0],
               "is_isotype": bool(g["is_isotype"].iloc[0])}
        for col in ("raw", "partial"):
            v = g[f"rho_{col}"]
            row[f"rho_pooled_{col}"] = float(v[g["kind"] == "pooled"].iloc[0])
            pv = v[g["kind"] == "patient"].to_numpy()
            tv = v[g["kind"] == "patient_celltype"].to_numpy()
            row[f"rho_within_median_{col}"] = float(np.nanmedian(pv))
            row[f"n_patients_same_sign_{col}"] = same_sign(pv, np.nanmedian(pv))
            row[f"rho_pct_median_{col}"] = float(np.nanmedian(tv))
            row[f"n_pct_same_sign_{col}"] = same_sign(tv, np.nanmedian(tv))  # vs the in-type median (report Table 3)
            row[f"n_pct_same_sign_as_within_{col}"] = same_sign(tv, np.nanmedian(pv))  # D27's definition
        row["n_pct_strata"] = int((g["kind"] == "patient_celltype").sum())
        rows.append(row)
    return pd.DataFrame(rows), long


def isotype_reference(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    iso = summary[summary["is_isotype"]]
    for (fname, lab), g in iso.groupby(["factor", "label"], sort=False):
        for level, stem in (("pooled", "rho_pooled"), ("within_patient", "rho_within_median"),
                            ("within_type", "rho_pct_median")):
            row = {"factor": fname, "label": lab, "level": level}
            for col in ("raw", "partial"):
                a = g[f"{stem}_{col}"].abs()
                row[f"iso_max_{col}"] = float(a.max())
                row[f"iso_max_{col}_set_by"] = g.loc[a.idxmax(), "feature"].replace("Isotype-", "")
                b = g.loc[g["feature"] != "Isotype-RTK2071", f"{stem}_{col}"].abs()
                row[f"iso_max_{col}_without_RTK2071"] = float(b.max())
            rows.append(row)
    return pd.DataFrame(rows)


def depth_axes(data: Dict[str, object]) -> pd.DataFrame:
    cells, clr, names = data["cells"], data["clr"], data["names"]
    rows = []
    for fname in FACTORS:
        for lab in LABELS:
            for p in ["all"] + data["supported"]:
                m = (cells["label"] == lab).to_numpy()
                if p != "all":
                    m &= (cells["patient"] == p).to_numpy()
                f = cells.loc[m, fname].to_numpy()
                rows.append({"kind": "factor", "name": fname, "label": lab, "patient": p, "n_cells": int(m.sum()),
                             "rho_log_total_adt": float(pd.Series(f).corr(pd.Series(cells.loc[m, "log_adt"].to_numpy()), method="spearman")),
                             "rho_log_ncount_rna": float(pd.Series(f).corr(pd.Series(cells.loc[m, "log_rna"].to_numpy()), method="spearman")),
                             "eta2_celltype": np.nan})
    for j, n in enumerate(names):
        if not n.startswith("Isotype"):
            continue
        for lab in LABELS:
            m = (cells["label"] == lab).to_numpy()
            per = []
            for p in data["supported"]:
                mm = m & (cells["patient"] == p).to_numpy()
                per.append(pd.Series(clr[mm, j]).corr(pd.Series(cells.loc[mm, "log_adt"].to_numpy()), method="spearman"))
            rows.append({"kind": "isotype", "name": n, "label": lab, "patient": "median_within",
                         "n_cells": int(m.sum()), "rho_log_total_adt": float(np.nanmedian(per)),
                         "rho_log_ncount_rna": np.nan,
                         "eta2_celltype": eta2(clr[m, j], cells.loc[m, "cell_type"].to_numpy())})
    return pd.DataFrame(rows)


def mal_vs_healthy(data: Dict[str, object]) -> tuple:
    """Two levels. patient: supported patient x predicted type (Figure 19's 21 comparisons). sample: the same inside
    one sample (P09's malignant cells are almost all in one of its four samples; every other supported patient has
    one sample, so only P09's groups change)."""
    cells, clr, names = data["cells"], data["clr"], data["names"]
    rows = []
    for level in ("patient", "sample"):
        for p in data["supported"]:
            units = [None] if level == "patient" else sorted(cells.loc[cells["patient"] == p, "sample"].unique())
            for unit in units:
                for ct in sorted(cells["cell_type"].unique()):
                    m = ((cells["patient"] == p) & (cells["cell_type"] == ct)).to_numpy()
                    if unit is not None:
                        m &= (cells["sample"] == unit).to_numpy()
                    is_mal = (cells.loc[m, "label"] == "malignant").to_numpy()
                    n_mal, n_h = int(is_mal.sum()), int((~is_mal).sum())
                    if n_mal < CONTRAST_MIN or n_h < CONTRAST_MIN:
                        continue
                    z = cells.loc[m, "log_adt"].to_numpy()
                    auc_depth = auc(z[is_mal], z[~is_mal])
                    X = clr[m]
                    for j, n in enumerate(names):
                        y = X[:, j]
                        res = residual_ranks(y, z)
                        rows.append({"level": level, "feature": n, "is_isotype": n.startswith("Isotype"), "patient": p,
                                     "sample": unit if unit is not None else "all", "cell_type": ct,
                                     "n_malignant": n_mal, "n_healthy": n_h,
                                     "median_clr_diff": float(np.median(y[is_mal]) - np.median(y[~is_mal])),
                                     "auc_raw": auc(y[is_mal], y[~is_mal]), "auc_adjusted": auc(res[is_mal], res[~is_mal]),
                                     "auc_log_total_adt": auc_depth})
    t = pd.DataFrame(rows)
    t["feature_short"] = t["feature"].str.replace("^ADT-", "", regex=True)
    summ = (t.groupby(["level", "feature", "feature_short", "is_isotype"], sort=False)
            .agg(n_groups=("auc_raw", "size"), n_lower_raw=("auc_raw", lambda s: int((s < 0.5).sum())),
                 median_auc_raw=("auc_raw", "median"), n_lower_adjusted=("auc_adjusted", lambda s: int((s < 0.5).sum())),
                 median_auc_adjusted=("auc_adjusted", "median"))
            .reset_index())
    return t, summ


def regression_checks(paths: Paths, summary: pd.DataFrame, data: Dict[str, object], mvh: pd.DataFrame) -> pd.DataFrame:
    d27 = pd.read_csv(paths.d27 / "d27_ifna_pc8_fa7_adt.csv")
    card_to_f = {v[0]: k for k, v in FACTORS.items()}
    d27["factor"] = d27["card_id"].map(card_to_f)
    d27 = d27.rename(columns={c: f"d27_{c}" for c in d27.columns if c not in ("factor", "label", "adt_marker")})
    m = summary.merge(d27, left_on=["factor", "label", "feature"], right_on=["factor", "label", "adt_marker"],
                      how="left", validate="one_to_one")
    rows = []
    for ours, theirs in (("rho_pooled_raw", "rho_pooled"), ("rho_within_median_raw", "rho_within_median"),
                         ("rho_pct_median_raw", "rho_pct_median"), ("n_patients_same_sign_raw", "n_patients_same_sign"),
                         ("n_pct_same_sign_as_within_raw", "n_pct_same_sign"), ("n_pct_strata", "n_pct_strata")):
        diff = float((m[ours] - m[f"d27_{theirs}"]).abs().max())
        rows.append({"check_id": f"d27_{theirs}", "description": f"raw {ours} reproduces D27 {theirs} ({len(m)} rows)",
                     "max_abs_diff": diff, "tolerance": TOL,
                     "passed": bool(diff <= TOL and m[f"d27_{theirs}"].notna().all())})
    n_pct = data["pct"].groupby("label").size().to_dict()
    rows.append({"check_id": "strata_counts", "description": "D27 strata: 71 healthy and 11 malignant patient x type groups",
                 "max_abs_diff": float(abs(n_pct.get("normal", 0) - 71) + abs(n_pct.get("malignant", 0) - 11)),
                 "tolerance": 0.0, "passed": n_pct == {"malignant": 11, "normal": 71}})
    n_groups = int(mvh[mvh["level"] == "patient"].groupby(["patient", "cell_type"]).ngroups)
    rows.append({"check_id": "contrast_groups", "description": "21 supported patient x type groups with >= 10 cells of each label (Figure 19)",
                 "max_abs_diff": float(abs(n_groups - 21)), "tolerance": 0.0, "passed": n_groups == 21})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# findings, manifest, ledger
# --------------------------------------------------------------------------------------

FOCUS = ("CD63", "CD82", "CD99", "CD48", "CD116", "CD33", "CD146", "GPR56", "CD117", "CD71", "CD34", "CD22")


def _f(v, d: int = 3) -> str:
    return "nan" if pd.isna(v) else f"{v:+.{d}f}"


def write_findings(paths: Paths, summary: pd.DataFrame, iso: pd.DataFrame, axes: pd.DataFrame, mvh_s: pd.DataFrame,
                   checks: pd.DataFrame, head: str, runtime: float) -> Path:
    L = ["# D29 total-ADT sensitivity and malignant-vs-healthy contrast (protein_depth_contrast_v1): findings", "",
         f"Generated by `{SCRIPT_REL}` at git {head} in {runtime:.0f} s. EXP read only; nothing refit; c_policy not_applicable.", "",
         "## How to read", "",
         "- rho: Spearman of the oriented factor score (cohort z, D18 sign) with CLR ADT, inside one label. *raw* = D27's value "
         "(reproduced, see checks); *partial* = the same with each cell's log total ADT count held fixed (rank residuals within the stratum).",
         "- within patient: median over the 7 supported patients (n same sign); in type: median over D27's supported patient x "
         "predicted cell type groups with >= 30 cells (71 healthy, 11 malignant).",
         "- AUC (malignant vs healthy): inside one supported patient x predicted type with >= 10 cells of each label; P(malignant "
         "cell has more than a healthy cell); 0.5 = no difference; adjusted = after removing the rank of log total ADT within the group.",
         "- Descriptive only: a screen, not a test; the malignant in-type groups come from P02, P03, P04 and P09.", "",
         "## Isotype reference (largest |rho| over the 9 isotypes)", "",
         "| factor | label | level | raw (set by) | partial (set by) | partial without RTK2071 |", "|---|---|---|---|---|---|"]
    for _, r in iso.iterrows():
        L.append(f"| {r['factor']} | {r['label']} | {r['level']} | {r['iso_max_raw']:.3f} ({r['iso_max_raw_set_by']}) | "
                 f"{r['iso_max_partial']:.3f} ({r['iso_max_partial_set_by']}) | {r['iso_max_partial_without_RTK2071']:.3f} |")
    L += ["", "## Factor vs total counts inside each patient (Spearman)", "",
          "| factor | label | patient | n | log total ADT | log nCount_RNA |", "|---|---|---|---|---|---|"]
    for _, r in axes[axes["kind"] == "factor"].iterrows():
        L.append(f"| {r['name']} | {r['label']} | {r['patient']} | {r['n_cells']} | {_f(r['rho_log_total_adt'])} | {_f(r['rho_log_ncount_rna'])} |")
    L += ["", "## Isotype CLR vs log total ADT (median within patient) and healthy-cell type eta2", "",
          "| isotype | label | rho with log total ADT | cell-type eta2 |", "|---|---|---|---|"]
    for _, r in axes[axes["kind"] == "isotype"].iterrows():
        L.append(f"| {r['name']} | {r['label']} | {_f(r['rho_log_total_adt'])} | {r['eta2_celltype']:.3f} |")
    L += ["", "## Focus proteins: raw vs partial", "",
          "| factor | label | protein | within raw (k/7) | within partial (k/7) | in type raw (k/n) | in type partial (k/n) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in summary[summary["feature_short"].isin(FOCUS)].iterrows():
        L.append(f"| {r['factor']} | {r['label']} | {r['feature_short']} | {_f(r['rho_within_median_raw'])} ({r['n_patients_same_sign_raw']}/7) | "
                 f"{_f(r['rho_within_median_partial'])} ({r['n_patients_same_sign_partial']}/7) | "
                 f"{_f(r['rho_pct_median_raw'])} ({r['n_pct_same_sign_raw']}/{r['n_pct_strata']}) | "
                 f"{_f(r['rho_pct_median_partial'])} ({r['n_pct_same_sign_partial']}/{r['n_pct_strata']}) |")
    L += ["", "## Malignant vs healthy, same predicted type (patient level: Figure 19's 21 groups; sample level: also the same sample)", "",
          "| level | protein | malignant lower (raw) | median AUC raw | malignant lower (adjusted) | median AUC adjusted |",
          "|---|---|---|---|---|---|"]
    for _, r in mvh_s[mvh_s["feature_short"].isin(FOCUS) | mvh_s["is_isotype"]].iterrows():
        L.append(f"| {r['level']} | {r['feature_short']} | {r['n_lower_raw']}/{r['n_groups']} | {r['median_auc_raw']:.3f} | "
                 f"{r['n_lower_adjusted']}/{r['n_groups']} | {r['median_auc_adjusted']:.3f} |")
    L += ["", "## Regression checks", "", "| check | max abs diff | passed |", "|---|---|---|"]
    for _, r in checks.iterrows():
        L.append(f"| {r['check_id']} | {r['max_abs_diff']:.2e} | {bool(r['passed'])} |")
    path = paths.out / f"{SLUG}_findings.md"
    path.write_text("\n".join(L) + "\n")
    return path


def write_ledger(paths: Paths) -> Path:
    kinds = {".csv": "table", ".json": "manifest", ".md": "doc"}
    ledger = paths.out / f"{SLUG}_artifact_ledger.csv"
    rows = [{"path": paths.rel(p), "kind": kinds.get(p.suffix, "other"), "size_bytes": p.stat().st_size}
            for p in sorted(paths.out.glob("*")) if p.is_file() and p != ledger]
    pd.DataFrame(rows).to_csv(ledger, index=False)
    return ledger


def outputs_ok(paths: Paths) -> bool:
    for name in OUTPUTS + (f"{SLUG}_findings.md", f"{SLUG}_manifest.json", f"{SLUG}_artifact_ledger.csv"):
        p = paths.out / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
        if p.suffix == ".csv":
            try:
                if pd.read_csv(p).empty:
                    return False
            except Exception:
                return False
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", default=None)
    ap.add_argument("--out-root", default=None, help="write under DIR/scorecards/stage2_diagnostic/%s/ instead of EXP/analysis" % SLUG)
    ap.add_argument("--resume", action="store_true", help="skip when every output exists and reads back")
    args = ap.parse_args(argv)
    t0 = time.time()
    exp = Path(os.path.abspath(args.experiment_dir)) if args.experiment_dir else experiment_dir()
    paths = Paths(exp, Path(args.out_root) if args.out_root else None)
    if args.resume and outputs_ok(paths):
        log("resume: every output exists and reads back; nothing to do")
        return 0
    if not args.resume and (paths.out / f"{SLUG}_manifest.json").is_file():
        raise SystemExit(f"{paths.rel(paths.out)} already holds a finished run; R3: bump the slug, or pass --resume")
    paths.out.mkdir(parents=True, exist_ok=True)
    data = load(paths)
    log("cohort loaded: {} cells, {} ADT features".format(len(data["cells"]), len(data["names"])))
    summary, long = rho_depth(data)
    log("rho raw/partial: {} rows, {} strata rows".format(len(summary), len(long)))
    iso = isotype_reference(summary)
    axes = depth_axes(data)
    mvh, mvh_s = mal_vs_healthy(data)
    log("malignant vs healthy groups: {}".format({lv: g.groupby(["sample", "patient", "cell_type"]).ngroups for lv, g in mvh.groupby("level")}))
    checks = regression_checks(paths, summary, data, mvh)
    for name, frame in (("d29_rho_depth.csv", summary), ("d29_rho_depth_strata.csv", long),
                        ("d29_isotype_reference.csv", iso), ("d29_depth_axes.csv", axes),
                        ("d29_mal_vs_healthy.csv", mvh), ("d29_mal_vs_healthy_summary.csv", mvh_s),
                        ("d29_regression_checks.csv", checks)):
        tmp = paths.out / (name + ".tmp")
        frame.to_csv(tmp, index=False, float_format=FLOAT_FORMAT)
        tmp.replace(paths.out / name)
    head, dirty = git_state()
    head_s = "{}{}".format(head, "+dirty" if dirty else "")
    runtime = time.time() - t0
    manifest = {
        "slug": SLUG, "analysis_id": ANALYSIS_ID, "created_by": SCRIPT_REL,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_state": {"head": head, "dirty": dirty},
        "inputs": {"d27": paths.rel(paths.d27), "cohort_cache": paths.rel(Path(data["cohort_path"])) if str(data["cohort_path"]).startswith(str(exp)) else "cohort_cache()",
                   "stage1_scores": f"stage1_dr/panel_id={PANEL}/method={{pca,fa}}/k=40/seed=42/scores.npy"},
        "parameters": {"factors": {k: v[0] for k, v in FACTORS.items()}, "orientation": "D18 orientation_sign (PC8 -1, FA7 +1)",
                       "adt_normalization": "CLR per cell over all 170 ADT features; no dsb",
                       "depth_covariate": "log1p(total raw ADT count over the 170 features)",
                       "partial": "rank-based: residualize ranks on [1, rank(log total ADT)] within each stratum",
                       "strata": "D27 (supported patients; patient x predicted type groups >= 30 cells)",
                       "contrast_min_cells_each_label": CONTRAST_MIN, "c_policy": "not_applicable"},
        "regression_checks_passed": bool(checks["passed"].all()), "runtime_seconds": round(runtime, 1),
    }
    write_findings(paths, summary, iso, axes, mvh_s, checks, head_s, runtime)
    (paths.out / f"{SLUG}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_ledger(paths)
    log("regression checks passed: {}".format(manifest["regression_checks_passed"]))
    if not checks["passed"].all():
        log("failed: {}".format(list(checks.loc[~checks["passed"], "check_id"])))
        return 3
    log("done in {:.0f}s -> {}".format(runtime, paths.rel(paths.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
