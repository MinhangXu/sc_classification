"""D18 generalized factor cards (W5.1): pure computation, rendering and bundle schema.

One card describes one factor of one stored Stage 1 representation (panel x method x K,
seed 42, transductive) together with the globally selected Stage 2 LOPO probe fitted on it.
Every card carries the misinterpretation guards the team agreed on:

- Orientation comes from the probe: orientation_sign = sign(beta) when beta is nonzero with
  the same sign in every supported fold where it is active. Positive = toward malignant.
  beta = 0 in every supported fold -> ``unoriented_all_zero``; any sign flip ->
  ``unoriented_sign_flip``. Unoriented cards keep raw loadings and leave
  ``oriented_loading`` empty.
- beta is always shown beside the univariate held-out lift (factor alone, sign fixed on the
  training folds) and the selection frequency, so an L1 zero is never read as "no signal".
- Folds whose whole coefficient vector is zero are labelled (lift = 0 by construction).
- Cross-method matches use score correlation (Hungarian at the same panel x K, best |r|
  across K), many-to-one R2, principal angles and Tucker phi of the loadings. Factor index
  is never compared across representations.
- Degenerate factors (zero score variance or zero loadings) are flagged (degenerate_flag);
  effective_k counts the non-degenerate factors. A blown-up or collapsed score scale is a
  separate caveat (scale_outlier), not degeneracy: Stage 2 standardizes scores, and the
  column scale of oblique (factosig_promax) factors is arbitrary.
- FA / FactoSig loadings are generative (how expected expression moves with the factor),
  not score weights. factosig_promax loadings are a pattern matrix with arbitrary column
  scale, so promax cards also rank genes by structure (corr(gene, score)) and report the
  factor score correlations (Phi).
- Fixed-model ablation dAP is attribution inside the fitted model, not a causal effect.

Pure numpy / pandas / scipy. Nothing here reads experiment paths; the runner is
run_d18_factor_cards.py. Bundle contract: factor_cards_v1, schema_version 1.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

# --------------------------------------------------------------------------------------
# constants and schema
# --------------------------------------------------------------------------------------

SCHEMA_VERSION = 1
SLUG = "factor_cards_v1"
METHODS = ("pca", "fa", "factosig", "factosig_promax")
OBLIQUE_METHODS = ("factosig_promax",)
GENERATIVE_METHODS = ("fa", "factosig", "factosig_promax")

DEGENERATE_SD = 1e-8  # absolute score SD below which a factor is degenerate
SCALE_OUTLIER_RATIO = 10.0  # score SD vs the representation's median SD (either way)
NEAR_DUPLICATE_R = 0.9  # oblique fits: max |score r| with another factor
DENSE_NONZERO_FRACTION = 0.5  # layer4 "sparse" threshold; above it a fit is dense
LOW_DETERMINACY = 0.707  # Grice: two valid score sets can be uncorrelated below this
SMALL_EIGENGAP = 0.10  # relative gap to a neighbour; <= blocks naming the factor alone
MATCH_ABS_R = 0.5  # F9c match threshold
NORMAL_MIN_CELLS = 200  # D14 eta2 stratum parameters (not the support rule)
MALIGNANT_MIN_CELLS = 20
ADT_STRATUM_MIN = {"normal": NORMAL_MIN_CELLS, "malignant": MALIGNANT_MIN_CELLS}
COEF_EPS = 1e-12
FLOAT_FORMAT = "%.10g"  # bundle CSV precision (regression checks run on full-precision tables)

CARD_ID_PATTERN = re.compile(r"^(?P<panel>.+)__(?P<method>pca|fa|factosig|factosig_promax)__k(?P<k>\d+)__f(?P<factor>\d{3})$")

ORIENTATION_STATUSES = ("oriented", "unoriented_all_zero", "unoriented_sign_flip")

CAVEAT_TEXT: Dict[str, str] = {
    "transductive_basis": "Stage 1 was fit on all 13 patients, including each held-out patient; LOPO lift is label transfer on a transductive basis, not inductive deployment.",
    "unoriented_all_zero": "beta is 0 in every supported fold, so the probe gives no orientation; loadings are shown with their arbitrary raw sign.",
    "unoriented_sign_flip": "beta changes sign across supported folds, so the factor has no stable orientation; loadings are shown with their arbitrary raw sign.",
    "has_all_zero_folds": "At least one supported fold fit an all-zero model (no factor selected); its lift is 0 by construction, not evidence of absent signal.",
    "dense_fit": "Dense fit (median fold nonzero fraction above 0.5): per-factor betas are partial effects among many correlated contributions and are not interpretable one factor at a time.",
    "l1_selected_absent": "The factor is never selected by the fitted probe; check its univariate held-out lift before reading the zero as absence of signal (L1 at a fixed C keeps one of several near-equivalent axes).",
    "zero_variance": "Degenerate factor: zero score variance (and loadings), beyond the representation's effective K.",
    "scale_outlier": "Blown-up or collapsed factor scale (score SD more than 10x away from the representation median). Not degeneracy: Stage 2 standardizes scores, and oblique (promax) column scale is arbitrary.",
    "near_duplicate_oblique": "Oblique fit: this factor's scores correlate |r| >= 0.9 with another factor; beta is a partial effect within a near-duplicate pair.",
    "low_determinacy": "Factor-score determinacy below 0.707: equally valid score sets can be uncorrelated, so do not name this factor as a unique program.",
    "small_eigengap": "Relative eigengap to a neighbouring factor is 10% or less; the individual axis is not identifiable (rotation within the near-degenerate block).",
    "generative_loadings": "FA / FactoSig loadings are generative (how expected expression moves with the factor), not the weights that compute the score.",
    "oblique_pattern_loadings": "factosig_promax loadings are a pattern matrix with arbitrary column scale; read them together with the structure correlations and Phi.",
    "ablation_is_attribution": "Ablation dAP and single-factor lift are attributions inside the fitted model, not causal effects.",
    "small_patient_support": "At least one supported patient has fewer than 25 malignant cells; its lift is noisy.",
    "reconstruction_mismatch": "Reconstructed held-out AP differs from the stored atlas AP by more than 1e-4 in at least one supported fold. When the only active coefficients are tiny (|beta| < 1e-4) the stored float32 probabilities tie and shift the stored AP; otherwise the fold coefficients may not match the atlas selection.",
}
ALWAYS_FLAGS = ("transductive_basis", "ablation_is_attribution")

# Bundle files: required columns (A4's factor explorer reads these; extra columns allowed).
BUNDLE_FILES: Dict[str, Tuple[str, ...]] = {
    "factor_cards_index.csv": (
        "card_id",
        "panel_id",
        "method",
        "k",
        "factor",
        "is_selected_driver",
        "n_active_folds",
        "sign_consistency",
        "orientation_sign",
        "eta2_normal",
        "eta2_malignant",
        "degenerate_flag",
        "effective_k",
        "determinacy",
        "matched_cards",
        "caveat_flags",
    ),
    "factor_cards_genes.csv": ("card_id", "gene", "loading", "oriented_loading", "rank", "pole", "structure_corr"),
    "factor_cards_patient_lift.csv": (
        "card_id",
        "patient",
        "univariate_lift",
        "univariate_auroc",
        "n_malignant",
        "prevalence",
    ),
    "factor_cards_fold_coefs.csv": ("card_id", "heldout_patient", "penalty", "C", "coef"),
}
OPTIONAL_FILES: Dict[str, Tuple[str, ...]] = {
    "factor_cards_matches.csv": (
        "card_id",
        "other_representation_id",
        "other_method",
        "other_k",
        "relation",
        "matched_factor",
        "matched_card_id",
        "score_r",
        "r2_many_to_one",
        "angle_deg",
        "tucker_phi_aligned",
    ),
    "factor_cards_subspace.csv": ("representation_id", "other_representation_id", "n_canonical", "cc_min", "cc_median"),
    "factor_cards_score_dist.csv": ("card_id", "patient", "label", "n_cells", "oriented", "mean", "q50"),
    "factor_cards_adt.csv": ("card_id", "label", "adt_marker", "is_isotype", "spearman_pooled", "spearman_patient_stratified"),
    "factor_cards_celltype.csv": ("card_id", "label", "cell_type", "n_cells", "mean_oriented_z"),
    "factor_cards_phi.csv": ("representation_id", "factor_i", "factor_j", "score_r"),
    "factor_cards_shortlist.csv": ("representation_id", "stage0_panel_id", "stage1_method", "k", "shortlist_reasons"),
}
MANIFEST_NAME = "factor_cards_manifest.json"
MANIFEST_KEYS = ("schema_version", "slug", "inputs", "git_state", "created_by", "created_utc", "parameters", "counts")
POLES_ORIENTED = ("malignant", "normal")
POLES_RAW = ("positive", "negative")


def card_id(panel_id: str, method: str, k: int, factor: int) -> str:
    """Return ``<panel_id>__<method>__k<K>__f<NNN>`` (factor is 1-based)."""
    return "{}__{}__k{}__f{:03d}".format(panel_id, method, int(k), int(factor))


def parse_card_id(value: str) -> Dict[str, object]:
    match = CARD_ID_PATTERN.match(str(value))
    if match is None:
        raise ValueError("Not a factor card id: {!r}".format(value))
    return {
        "panel_id": match.group("panel"),
        "method": match.group("method"),
        "k": int(match.group("k")),
        "factor": int(match.group("factor")),
    }


def representation_id(panel_id: str, method: str, k: int, seed: int = 42) -> str:
    return "{}|dr|{}|{}|{}".format(panel_id, method, int(k), int(seed))


def feature_ids(k: int) -> List[str]:
    return ["factor_{:03d}".format(i) for i in range(1, int(k) + 1)]


# --------------------------------------------------------------------------------------
# metrics (exact sklearn equivalents, faster for many columns)
# --------------------------------------------------------------------------------------


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """sklearn.metrics.average_precision_score for a binary label, without validation overhead.

    Thresholds are the distinct score values (ties grouped). A constant score returns the
    prevalence. Returns NaN when there is no positive.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0 or y.size == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    distinct = np.flatnonzero(np.diff(s_sorted))
    thresholds = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y_sorted)[thresholds].astype(np.float64)
    fps = (thresholds + 1).astype(np.float64) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auroc_columns(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ROC AUC per column (Mann-Whitney with average ranks; equals sklearn roc_auc_score)."""
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0, method="average")
    rank_sum = ranks[y].sum(axis=0)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def zscore_columns(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Population z-score per column; degenerate columns (SD <= DEGENERATE_SD) become 0.

    Returns (z, mean, sd, nondegenerate).
    """
    x = np.asarray(matrix, dtype=np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    nondegenerate = sd > DEGENERATE_SD
    z = (x - mean) / np.where(nondegenerate, sd, 1.0)
    z[:, ~nondegenerate] = 0.0
    return z, mean, sd, nondegenerate


# --------------------------------------------------------------------------------------
# cohort container
# --------------------------------------------------------------------------------------


@dataclass
class Cohort:
    """Per-cell arrays in Stage 1 score row order."""

    patients: np.ndarray  # str, (n,)
    labels: np.ndarray  # int 0/1, 1 = malignant, (n,)
    supported: Tuple[str, ...]
    cell_types: Optional[np.ndarray] = None  # str, (n,)
    adt: Optional["AdtRanks"] = None

    def __post_init__(self) -> None:
        self.patients = np.asarray(self.patients).astype(str)
        self.labels = np.asarray(self.labels).astype(np.int8)
        if self.patients.shape != self.labels.shape:
            raise ValueError("patients and labels must have the same length")
        if self.cell_types is not None:
            self.cell_types = np.asarray(self.cell_types).astype(str)
        self.supported = tuple(self.supported)

    @property
    def n_cells(self) -> int:
        return int(self.patients.size)


# --------------------------------------------------------------------------------------
# ADT (surface protein) helpers
# --------------------------------------------------------------------------------------


def clr_per_cell(counts: np.ndarray) -> np.ndarray:
    """Centered log-ratio per cell: log1p(x) minus the cell's mean log1p over all ADT features.

    The input must be RAW counts (obsm['ADT'] in the cohort cache). Isotypes are included in
    the per-cell centering, as in Seurat's CLR over the whole ADT assay.
    """
    logged = np.log1p(np.asarray(counts, dtype=np.float64))
    return logged - logged.mean(axis=1, keepdims=True)


def _standardized_ranks(matrix: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(matrix, dtype=np.float64), axis=0, method="average")
    mean = ranks.mean(axis=0)
    sd = ranks.std(axis=0)
    out = (ranks - mean) / np.where(sd > 0, sd, 1.0)
    out[:, sd <= 0] = np.nan
    return out


@dataclass
class AdtRanks:
    """Precomputed standardized ADT ranks for within-label Spearman correlations."""

    names: List[str]
    is_isotype: np.ndarray
    label_rows: Dict[str, np.ndarray]  # label -> cell indices
    label_ranks: Dict[str, np.ndarray]  # label -> (n_label, m) standardized ranks
    strata_rows: Dict[Tuple[str, str], np.ndarray]  # (label, patient) -> cell indices
    strata_ranks: Dict[Tuple[str, str], np.ndarray]


def build_adt_ranks(
    adt_clr: np.ndarray,
    names: Sequence[str],
    patients: np.ndarray,
    labels: np.ndarray,
    stratum_min: Mapping[str, int] = ADT_STRATUM_MIN,
) -> AdtRanks:
    names = [str(n) for n in names]
    is_isotype = np.array([n.lower().startswith("isotype") for n in names], dtype=bool)
    patients = np.asarray(patients).astype(str)
    labels = np.asarray(labels).astype(int)
    label_rows: Dict[str, np.ndarray] = {}
    label_ranks: Dict[str, np.ndarray] = {}
    strata_rows: Dict[Tuple[str, str], np.ndarray] = {}
    strata_ranks: Dict[Tuple[str, str], np.ndarray] = {}
    for label_name, label_value in (("normal", 0), ("malignant", 1)):
        rows = np.flatnonzero(labels == label_value)
        label_rows[label_name] = rows
        label_ranks[label_name] = _standardized_ranks(adt_clr[rows])
        for patient in sorted(pd.unique(patients[rows])):
            prow = rows[patients[rows] == patient]
            if prow.size >= int(stratum_min.get(label_name, 1)):
                strata_rows[(label_name, patient)] = prow
                strata_ranks[(label_name, patient)] = _standardized_ranks(adt_clr[prow])
    return AdtRanks(names, is_isotype, label_rows, label_ranks, strata_rows, strata_ranks)


def adt_spearman(scores: np.ndarray, adt: AdtRanks) -> Dict[str, Dict[str, np.ndarray]]:
    """Within-label Spearman rho between every factor score and every ADT marker.

    Returns {label: {"pooled": (m, K), "stratified": (m, K), "n_cells": int}}. "pooled" ranks
    all cells of that label together (patients mixed); "stratified" is the cell-weighted mean
    of within-patient rhos over patients with enough cells in that label.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for label_name, rows in adt.label_rows.items():
        s = _standardized_ranks(scores[rows])
        pooled = (np.nan_to_num(adt.label_ranks[label_name]).T @ np.nan_to_num(s)) / max(rows.size, 1)
        pooled[np.isnan(adt.label_ranks[label_name]).all(axis=0), :] = np.nan
        pooled[:, np.isnan(s).all(axis=0)] = np.nan
        acc = np.zeros_like(pooled)
        weight = np.zeros_like(pooled)
        for (lab, _patient), prow in adt.strata_rows.items():
            if lab != label_name:
                continue
            sp = _standardized_ranks(scores[prow])
            ap = adt.strata_ranks[(lab, _patient)]
            rho = (np.nan_to_num(ap).T @ np.nan_to_num(sp)) / prow.size
            valid = (~np.isnan(ap).all(axis=0))[:, None] & (~np.isnan(sp).all(axis=0))[None, :]
            acc += np.where(valid, rho * prow.size, 0.0)
            weight += np.where(valid, prow.size, 0.0)
        stratified = np.where(weight > 0, acc / np.where(weight > 0, weight, 1.0), np.nan)
        out[label_name] = {"pooled": pooled, "stratified": stratified, "n_cells": int(rows.size)}
    return out


# --------------------------------------------------------------------------------------
# Stage 2 fold coefficients: stability and orientation
# --------------------------------------------------------------------------------------


def fold_coefficient_matrix(coefs: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long fold coefficients to (heldout_patient x factor_###) plus per-fold settings.

    ``coefs`` needs heldout_patient, feature_id, coefficient, penalty, C, l1_ratio.
    """
    if coefs is None or coefs.empty:
        empty = pd.DataFrame(columns=feature_ids(k), dtype=float)
        return empty, pd.DataFrame(columns=["penalty", "C", "l1_ratio"])
    frame = coefs.copy()
    frame["coefficient"] = pd.to_numeric(frame["coefficient"], errors="coerce").fillna(0.0)
    matrix = (
        frame.pivot_table(index="heldout_patient", columns="feature_id", values="coefficient", aggfunc="first")
        .reindex(columns=feature_ids(k))
        .fillna(0.0)
        .sort_index()
    )
    settings = frame.groupby("heldout_patient")[["penalty", "C", "l1_ratio"]].first().reindex(matrix.index)
    return matrix, settings


@dataclass
class FoldSummary:
    n_folds: int
    n_active_folds: np.ndarray  # (K,)
    sign_consistency: np.ndarray  # (K,) fraction of active folds with the majority sign; NaN if never active
    orientation_sign: np.ndarray  # (K,) +1 / -1 / 0
    orientation_status: List[str]
    n_active_folds_all: np.ndarray
    n_folds_all: int
    median_abs_coef: np.ndarray
    fold_all_zero: Dict[str, bool]
    fold_n_active: Dict[str, int]
    fold_nonzero_fraction: Dict[str, float]
    median_nonzero_fraction: float
    dense: bool


def summarize_folds(matrix: pd.DataFrame, supported: Sequence[str]) -> FoldSummary:
    k = matrix.shape[1]
    values = matrix.to_numpy(dtype=np.float64) if k else np.zeros((len(matrix), 0))
    patients = [str(p) for p in matrix.index]
    sup_rows = [i for i, p in enumerate(patients) if p in set(supported)]
    sup = values[sup_rows] if sup_rows else np.zeros((0, k))
    active = np.abs(sup) > COEF_EPS
    n_active = active.sum(axis=0).astype(int)
    pos = (sup > COEF_EPS).sum(axis=0)
    neg = (sup < -COEF_EPS).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        consistency = np.where(n_active > 0, np.maximum(pos, neg) / np.maximum(n_active, 1), np.nan)
    orientation = np.zeros(k, dtype=int)
    status: List[str] = []
    for j in range(k):
        if n_active[j] == 0:
            status.append("unoriented_all_zero")
        elif pos[j] > 0 and neg[j] > 0:
            status.append("unoriented_sign_flip")
        else:
            orientation[j] = 1 if pos[j] > 0 else -1
            status.append("oriented")
    all_active = (np.abs(values) > COEF_EPS).sum(axis=0).astype(int)
    median_abs = np.zeros(k)
    for j in range(k):
        if n_active[j] > 0:
            median_abs[j] = float(np.median(np.abs(sup[active[:, j], j])))
    fold_all_zero = {}
    fold_n_active = {}
    fold_frac = {}
    for i, p in enumerate(patients):
        n_act = int((np.abs(values[i]) > COEF_EPS).sum())
        fold_n_active[p] = n_act
        fold_all_zero[p] = n_act == 0
        fold_frac[p] = n_act / k if k else float("nan")
    sup_fracs = [fold_frac[patients[i]] for i in sup_rows]
    median_frac = float(np.median(sup_fracs)) if sup_fracs else float("nan")
    return FoldSummary(
        n_folds=len(sup_rows),
        n_active_folds=n_active,
        sign_consistency=consistency,
        orientation_sign=orientation,
        orientation_status=status,
        n_active_folds_all=all_active,
        n_folds_all=len(patients),
        median_abs_coef=median_abs,
        fold_all_zero=fold_all_zero,
        fold_n_active=fold_n_active,
        fold_nonzero_fraction=fold_frac,
        median_nonzero_fraction=median_frac,
        dense=bool(median_frac > DENSE_NONZERO_FRACTION) if not math.isnan(median_frac) else False,
    )


# --------------------------------------------------------------------------------------
# held-out univariate lift and fitted-model attribution
# --------------------------------------------------------------------------------------


def _train_stats(scores: np.ndarray, train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xt = scores[train]
    mean = xt.mean(axis=0)
    sd = xt.std(axis=0)
    return mean, np.where(sd > 0, sd, 1.0)  # StandardScaler keeps scale 1 for zero variance


def heldout_univariate(scores: np.ndarray, cohort: Cohort) -> pd.DataFrame:
    """Per factor x supported patient: univariate held-out AP / lift / AUROC.

    The sign of each factor is fixed on the training folds (sign of the standardized
    malignant-minus-normal mean gap among training cells), so ``univariate_lift`` uses no
    held-out label. ``univariate_lift_sign_agnostic`` = max(AP(+s), AP(-s)) - prevalence is the
    F6 quantity (optimistic: picks the sign on the held-out patient) kept for regression.
    """
    k = scores.shape[1]
    y_all = cohort.labels.astype(bool)
    rows = []
    for patient in cohort.supported:
        held = cohort.patients == patient
        train = ~held
        y = y_all[held]
        if y.sum() == 0 or y.sum() == y.size:
            continue
        mean, sd = _train_stats(scores, train)
        z_train_mal = (scores[train & y_all].mean(axis=0) - mean) / sd
        z_train_norm = (scores[train & ~y_all].mean(axis=0) - mean) / sd
        train_sign = np.where(z_train_mal - z_train_norm >= 0, 1, -1)
        held_scores = scores[held]
        z_held = (held_scores - mean) / sd
        prevalence = float(y.mean())
        oriented = held_scores * train_sign
        auroc = auroc_columns(y, oriented)
        for j in range(k):
            ap = average_precision(y, oriented[:, j])
            ap_pos = ap if train_sign[j] > 0 else average_precision(y, held_scores[:, j])
            ap_neg = ap if train_sign[j] < 0 else average_precision(y, -held_scores[:, j])
            rows.append(
                {
                    "factor": j + 1,
                    "patient": patient,
                    "univariate_auprc": ap,
                    "univariate_lift": ap - prevalence,
                    "univariate_auroc": float(auroc[j]),
                    "univariate_lift_sign_agnostic": max(ap_pos, ap_neg) - prevalence,
                    "train_orientation_sign": int(train_sign[j]),
                    "heldout_malignant_mean_z": float(z_held[y, j].mean()),
                    "heldout_normal_mean_z": float(z_held[~y, j].mean()),
                    "n_malignant": int(y.sum()),
                    "n_cells": int(y.size),
                    "prevalence": prevalence,
                }
            )
    return pd.DataFrame(rows)


def fitted_attribution(
    scores: np.ndarray, cohort: Cohort, coef_matrix: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """D8-style fixed-model attribution per supported fold.

    The held-out linear score is sum_j beta_j * z_j with z from the training-fold
    StandardScaler. Returns (per factor x patient: coefficient, fitted single-factor lift,
    ablation dAP, full-model AP/lift) and (per patient reconstruction).
    """
    k = scores.shape[1]
    y_all = cohort.labels.astype(bool)
    factor_rows = []
    fold_rows = []
    for patient in cohort.supported:
        if patient not in coef_matrix.index:
            continue
        held = cohort.patients == patient
        y = y_all[held]
        if y.sum() == 0 or y.sum() == y.size:
            continue
        mean, sd = _train_stats(scores, ~held)
        z_held = (scores[held] - mean) / sd
        beta = coef_matrix.loc[patient].to_numpy(dtype=np.float64)
        contrib = z_held * beta
        full = contrib.sum(axis=1)
        prevalence = float(y.mean())
        full_ap = average_precision(y, full)
        fold_rows.append(
            {
                "patient": patient,
                "fold_model_ap": full_ap,
                "fold_model_lift": full_ap - prevalence,
                "fold_n_active": int((np.abs(beta) > COEF_EPS).sum()),
            }
        )
        for j in range(k):
            if abs(beta[j]) > COEF_EPS:
                ablated = average_precision(y, full - contrib[:, j])
                single = average_precision(y, contrib[:, j])
            else:
                ablated = full_ap
                single = prevalence
            factor_rows.append(
                {
                    "factor": j + 1,
                    "patient": patient,
                    "fitted_coef": float(beta[j]),
                    "fitted_full_model_ap": full_ap,
                    "fitted_full_model_lift": full_ap - prevalence,
                    "fitted_ablated_ap": ablated,
                    "fitted_ap_drop": full_ap - ablated,
                    "fitted_single_factor_lift": single - prevalence,
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(fold_rows)


# --------------------------------------------------------------------------------------
# patient effect, distributions, cell type
# --------------------------------------------------------------------------------------


def eta2_by_label(scores: np.ndarray, cohort: Cohort) -> Dict[str, np.ndarray]:
    """eta2 of patient within each label (between-patient SS / total SS about the grand mean).

    Strata follow D14: normal patients with >= 200 normal cells, malignant patients with
    >= 20 malignant cells; computed here from the scores (no dependence on D14/D16 files).
    """
    out: Dict[str, np.ndarray] = {}
    for label_name, label_value, min_cells in (("normal", 0, NORMAL_MIN_CELLS), ("malignant", 1, MALIGNANT_MIN_CELLS)):
        mask = cohort.labels == label_value
        pats = cohort.patients[mask]
        names = [p for p in sorted(pd.unique(pats)) if int((pats == p).sum()) >= min_cells]
        if len(names) < 2:
            out[label_name] = np.full(scores.shape[1], np.nan)
            continue
        keep = np.isin(pats, names)
        x = scores[mask][keep]
        groups = pats[keep]
        grand = x.mean(axis=0)
        ss_total = ((x - grand) ** 2).sum(axis=0)
        ss_between = np.zeros(scores.shape[1])
        for name in names:
            g = x[groups == name]
            ss_between += g.shape[0] * (g.mean(axis=0) - grand) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            out[label_name] = np.where(ss_total > 0, ss_between / ss_total, 0.0)
    return out


def eta2_groups(column_matrix: np.ndarray, groups: np.ndarray, min_cells: int = 1) -> np.ndarray:
    names = [g for g in pd.unique(groups) if int((groups == g).sum()) >= min_cells]
    if len(names) < 2:
        return np.full(column_matrix.shape[1], np.nan)
    keep = np.isin(groups, names)
    x = column_matrix[keep]
    grp = groups[keep]
    grand = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(x.shape[1])
    for name in names:
        g = x[grp == name]
        ss_between += g.shape[0] * (g.mean(axis=0) - grand) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_total > 0, ss_between / ss_total, 0.0)


def score_distribution(z: np.ndarray, cohort: Cohort, orientation: np.ndarray) -> pd.DataFrame:
    """Per factor x patient x label quantiles of the cohort-z score, oriented when possible."""
    signed = z * np.where(orientation == 0, 1, orientation)
    rows = []
    for patient in sorted(pd.unique(cohort.patients)):
        pmask = cohort.patients == patient
        for label_name, label_value in (("normal", 0), ("malignant", 1)):
            mask = pmask & (cohort.labels == label_value)
            n = int(mask.sum())
            if n == 0:
                continue
            x = signed[mask]
            qs = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
            means = x.mean(axis=0)
            sds = x.std(axis=0)
            for j in range(z.shape[1]):
                rows.append(
                    {
                        "factor": j + 1,
                        "patient": patient,
                        "label": label_name,
                        "n_cells": n,
                        "oriented": bool(orientation[j] != 0),
                        "mean": float(means[j]),
                        "sd": float(sds[j]),
                        "q10": float(qs[0, j]),
                        "q25": float(qs[1, j]),
                        "q50": float(qs[2, j]),
                        "q75": float(qs[3, j]),
                        "q90": float(qs[4, j]),
                    }
                )
    return pd.DataFrame(rows)


def celltype_means(z: np.ndarray, cohort: Cohort, orientation: np.ndarray, min_cells: int = 20) -> pd.DataFrame:
    if cohort.cell_types is None:
        return pd.DataFrame()
    signed = z * np.where(orientation == 0, 1, orientation)
    rows = []
    for label_name, label_value in (("normal", 0), ("malignant", 1)):
        lmask = cohort.labels == label_value
        for ctype in sorted(pd.unique(cohort.cell_types[lmask])):
            mask = lmask & (cohort.cell_types == ctype)
            n = int(mask.sum())
            if n < min_cells:
                continue
            means = signed[mask].mean(axis=0)
            for j in range(z.shape[1]):
                rows.append({"factor": j + 1, "label": label_name, "cell_type": ctype, "n_cells": n, "mean_oriented_z": float(means[j])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Stage 1 structure: determinacy, effective K, degeneracy, eigengaps
# --------------------------------------------------------------------------------------


def factor_determinacy(loadings: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """rho_k = sqrt([L^T Sigma^-1 L]_kk) for an orthogonal unit-variance factor model.

    Sigma = L L^T + diag(psi). Uses Woodbury: L^T Sigma^-1 L = M (I + M)^-1, M = L^T Psi^-1 L.
    Equals sqrt(d_k / (1 + d_k)) when M is diagonal (sklearn FA convention). Zero columns -> 0.
    """
    lam = np.asarray(loadings, dtype=np.float64)
    psi = np.maximum(np.asarray(psi, dtype=np.float64), 1e-6)
    m = lam.T @ (lam / psi[:, None])
    k = m.shape[0]
    inner = m @ np.linalg.inv(np.eye(k) + m)
    return np.sqrt(np.clip(np.diag(inner), 0.0, 1.0))


def psi_whitened_d(loadings: np.ndarray, psi: np.ndarray) -> np.ndarray:
    lam = np.asarray(loadings, dtype=np.float64)
    psi = np.maximum(np.asarray(psi, dtype=np.float64), 1e-6)
    return np.sum(lam ** 2 / psi[:, None], axis=0)


def relative_eigengap(values: np.ndarray) -> np.ndarray:
    """min(|v_k - v_k-1|, |v_k - v_k+1|) / v_k in the stored factor order (NaN for v_k <= 0)."""
    v = np.asarray(values, dtype=np.float64)
    gaps = np.full(v.size, np.nan)
    for i in range(v.size):
        if not v[i] > 0:
            continue
        neighbours = [abs(v[i] - v[j]) for j in (i - 1, i + 1) if 0 <= j < v.size and v[j] > 0]
        if neighbours:
            gaps[i] = min(neighbours) / v[i]
    return gaps


@dataclass
class Structure:
    effective_k: int
    nondegenerate: np.ndarray
    score_sd: np.ndarray
    zero_variance: np.ndarray
    scale_outlier: np.ndarray
    determinacy: np.ndarray
    determinacy_method: str
    eigengap: np.ndarray
    eigengap_basis: str
    near_duplicate_r: np.ndarray  # oblique: max |r| with another factor (NaN otherwise)
    near_duplicate_partner: np.ndarray


def stage1_structure(
    method: str,
    scores: np.ndarray,
    loadings: np.ndarray,
    diagnostics: Optional[Mapping] = None,
    score_corr: Optional[np.ndarray] = None,
) -> Structure:
    diagnostics = diagnostics or {}
    extras = (diagnostics.get("extras") or {}).get("uns") or {}
    sd = np.asarray(scores, dtype=np.float64).std(axis=0)
    load_norm = np.linalg.norm(np.asarray(loadings, dtype=np.float64), axis=0)
    zero_var = (sd <= DEGENERATE_SD) | (load_norm <= DEGENERATE_SD)
    nondeg = ~zero_var
    med = float(np.median(sd[nondeg])) if nondeg.any() else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sd / med
    scale_outlier = nondeg & ((ratio > SCALE_OUTLIER_RATIO) | (ratio < 1.0 / SCALE_OUTLIER_RATIO))
    k = scores.shape[1]
    determinacy = np.full(k, np.nan)
    det_method = "not_applicable_pca_exact_scores"
    eigengap = np.full(k, np.nan)
    gap_basis = "none"
    if method == "pca":
        variances = np.asarray(extras.get("explained_variance") or diagnostics.get("score_variance") or sd ** 2, dtype=np.float64)
        eigengap = relative_eigengap(variances)
        gap_basis = "pca_explained_variance"
    elif method in ("fa", "factosig"):
        psi = extras.get("noise_variance_per_feature")
        if psi is not None and len(psi) == loadings.shape[0]:
            psi_arr = np.asarray(psi, dtype=np.float64)
            det_method = "psi_saved"
        else:
            communality = np.sum(np.asarray(loadings, dtype=np.float64) ** 2, axis=1)
            psi_arr = np.maximum(1.0 - communality, 1e-3)
            det_method = "psi_approx_1_minus_communality"
        determinacy = factor_determinacy(loadings, psi_arr)
        determinacy[zero_var] = np.nan
        if method == "fa":
            d = psi_whitened_d(loadings, psi_arr)
            d[zero_var] = 0.0
            eigengap = relative_eigengap(d)
            gap_basis = "psi_whitened_d"
    else:
        det_method = "not_computed_oblique_phi_unsaved"
    near_r = np.full(k, np.nan)
    partner = np.full(k, -1, dtype=int)
    if method in OBLIQUE_METHODS and score_corr is not None:
        c = np.abs(np.asarray(score_corr, dtype=np.float64)).copy()
        np.fill_diagonal(c, -1.0)
        partner = np.argmax(c, axis=1) + 1
        near_r = c[np.arange(k), partner - 1]
    eigengap[zero_var] = np.nan
    return Structure(
        effective_k=int(nondeg.sum()),
        nondegenerate=nondeg,
        score_sd=sd,
        zero_variance=zero_var,
        scale_outlier=scale_outlier,
        determinacy=determinacy,
        determinacy_method=det_method,
        eigengap=eigengap,
        eigengap_basis=gap_basis,
        near_duplicate_r=near_r,
        near_duplicate_partner=partner,
    )


def structure_correlations(gene_z: np.ndarray, scores_z: np.ndarray) -> np.ndarray:
    """corr(gene, score) (p x K) from z-scored genes and z-scored scores (same rows)."""
    g = np.asarray(gene_z, dtype=np.float64)
    gz, _, _, _ = zscore_columns(g)
    return (gz.T @ scores_z) / gz.shape[0]


# --------------------------------------------------------------------------------------
# cross-representation matching
# --------------------------------------------------------------------------------------


@dataclass
class Basis:
    """Cohort z-scores of one representation plus an orthonormal basis of its span."""

    rep_id: str
    panel_id: str
    method: str
    k: int
    z: np.ndarray
    nondegenerate: np.ndarray
    q: np.ndarray
    loadings: np.ndarray


def make_basis(rep_id: str, panel_id: str, method: str, k: int, scores: np.ndarray, loadings: np.ndarray) -> Basis:
    z, _, _, nondeg = zscore_columns(scores)
    sub = z[:, nondeg]
    if sub.shape[1]:
        u, s, _ = np.linalg.svd(sub, full_matrices=False)
        keep = s > (s[0] * 1e-8 if s.size else 0)
        q = u[:, keep]
    else:
        q = np.zeros((z.shape[0], 0))
    return Basis(rep_id, panel_id, method, int(k), z, nondeg, q, np.asarray(loadings, dtype=np.float64))


def canonical_correlations(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return np.zeros(0)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def tucker_phi(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= DEGENERATE_SD or nb <= DEGENERATE_SD:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def match_factors(a: Basis, b: Basis) -> pd.DataFrame:
    """Match every factor of ``a`` against representation ``b`` (same panel, same gene order).

    Columns: factor, best_factor, best_r, hungarian_factor, hungarian_r (same K only),
    r2_many_to_one (R2 of the factor on all of b's scores), angle_deg (to span(b)),
    tucker_phi_aligned (loading congruence with the matched factor, oriented by sign(r)),
    relation, matched_factor, score_r.
    """
    n = a.z.shape[0]
    corr = (a.z.T @ b.z) / n
    corr[~a.nondegenerate, :] = 0.0
    corr[:, ~b.nondegenerate] = 0.0
    abs_corr = np.abs(corr)
    best = np.argmax(abs_corr, axis=1)
    same_k = a.k == b.k
    hung = np.full(a.k, -1, dtype=int)
    if same_k:
        rows, cols = linear_sum_assignment(-abs_corr)
        hung[rows] = cols
    proj = b.q.T @ a.z if b.q.shape[1] else np.zeros((0, a.k))
    norm2 = np.sum(a.z ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(norm2 > 0, np.sum(proj ** 2, axis=0) / norm2, np.nan)
    r2 = np.clip(r2, 0.0, 1.0)
    out = []
    for f in range(a.k):
        if not a.nondegenerate[f]:
            continue
        matched = hung[f] if same_k else best[f]
        r_match = float(corr[f, matched])
        phi = tucker_phi(a.loadings[:, f], b.loadings[:, matched])
        out.append(
            {
                "factor": f + 1,
                "best_factor": int(best[f]) + 1,
                "best_r": float(corr[f, best[f]]),
                "hungarian_factor": int(hung[f]) + 1 if same_k else np.nan,
                "hungarian_r": float(corr[f, hung[f]]) if same_k else np.nan,
                "relation": "hungarian_same_k" if same_k else "best_abs_r",
                "matched_factor": int(matched) + 1,
                "score_r": r_match,
                "abs_r": abs(r_match),
                "r2_many_to_one": float(r2[f]),
                "angle_deg": float(np.degrees(np.arccos(np.sqrt(r2[f])))) if not np.isnan(r2[f]) else np.nan,
                "tucker_phi_aligned": float(np.sign(r_match) * phi) if not np.isnan(phi) else np.nan,
            }
        )
    return pd.DataFrame(out)


def subspace_summary(a: Basis, b: Basis, selected: Optional[Sequence[int]] = None) -> Dict[str, object]:
    cc = canonical_correlations(a.q, b.q)
    row: Dict[str, object] = {
        "representation_id": a.rep_id,
        "other_representation_id": b.rep_id,
        "k": a.k,
        "other_k": b.k,
        "other_method": b.method,
        "rank": int(a.q.shape[1]),
        "other_rank": int(b.q.shape[1]),
        "n_canonical": int(cc.size),
        "cc_min": float(cc.min()) if cc.size else np.nan,
        "cc_median": float(np.median(cc)) if cc.size else np.nan,
        "cc_max": float(cc.max()) if cc.size else np.nan,
        "n_cc_below_0p9": int((cc < 0.9).sum()),
        "selected_rank": 0,
        "selected_cc_min": np.nan,
        "selected_cc_median": np.nan,
    }
    if selected:
        cols = [i - 1 for i in selected if a.nondegenerate[i - 1]]
        if cols:
            u, s, _ = np.linalg.svd(a.z[:, cols], full_matrices=False)
            qs = u[:, s > s[0] * 1e-8]
            scc = canonical_correlations(qs, b.q)
            row.update(
                {
                    "selected_rank": int(qs.shape[1]),
                    "selected_cc_min": float(scc.min()) if scc.size else np.nan,
                    "selected_cc_median": float(np.median(scc)) if scc.size else np.nan,
                }
            )
    return row


# --------------------------------------------------------------------------------------
# representation -> card tables
# --------------------------------------------------------------------------------------


@dataclass
class RepInput:
    panel_id: str
    method: str
    k: int
    scores: np.ndarray
    loadings: np.ndarray
    genes: List[str]
    coefs: pd.DataFrame
    diagnostics: Mapping = field(default_factory=dict)
    expected_ap: Optional[Mapping[str, float]] = None  # stored atlas held-out AP per patient
    gene_z: Optional[np.ndarray] = None  # z-scored panel genes (oblique fits: structure corr)
    shortlist_reasons: str = ""
    panel_label: str = ""
    seed: int = 42

    @property
    def rep_id(self) -> str:
        return representation_id(self.panel_id, self.method, self.k, self.seed)


def loading_semantics(method: str) -> str:
    return {
        "pca": "pca_component_weights",
        "fa": "fa_generative_loadings",
        "factosig": "factosig_varimax_generative_loadings",
        "factosig_promax": "promax_pattern_arbitrary_scale",
    }.get(method, "unknown")


def _top_gene_rows(
    cid: str,
    genes: Sequence[str],
    values: np.ndarray,
    loadings: np.ndarray,
    structure: Optional[np.ndarray],
    orientation: int,
    ranked_by: str,
    top_n: int,
    semantics: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    signed = values * (orientation if orientation != 0 else 1)
    poles = POLES_ORIENTED if orientation != 0 else POLES_RAW
    for pole, direction in ((poles[0], 1.0), (poles[1], -1.0)):
        cand = np.flatnonzero(signed * direction > 0)
        order = cand[np.argsort(-(signed[cand] * direction), kind="mergesort")][:top_n]
        for rank, gi in enumerate(order, start=1):
            rows.append(
                {
                    "card_id": cid,
                    "gene": genes[gi],
                    "loading": float(loadings[gi]),
                    "oriented_loading": float(loadings[gi] * orientation) if orientation != 0 else np.nan,
                    "rank": rank,
                    "pole": pole,
                    "structure_corr": float(structure[gi] * (orientation if orientation != 0 else 1)) if structure is not None else np.nan,
                    "ranked_by": ranked_by,
                    "loading_semantics": semantics,
                }
            )
    return rows


def caveat_flags_for(
    method: str,
    status: str,
    fold: FoldSummary,
    j: int,
    structure: Structure,
    small_patient: bool,
    reconstruction_error: float,
    supported: Sequence[str] = (),
) -> List[str]:
    flags = list(ALWAYS_FLAGS)
    if status != "oriented":
        flags.append(status)
    if any(fold.fold_all_zero.get(p, False) for p in supported):
        flags.append("has_all_zero_folds")
    if fold.dense:
        flags.append("dense_fit")
    if fold.n_active_folds[j] == 0:
        flags.append("l1_selected_absent")
    if structure.zero_variance[j]:
        flags.append("zero_variance")
    if structure.scale_outlier[j]:
        flags.append("scale_outlier")
    if method in OBLIQUE_METHODS:
        flags.append("oblique_pattern_loadings")
        if not np.isnan(structure.near_duplicate_r[j]) and structure.near_duplicate_r[j] >= NEAR_DUPLICATE_R:
            flags.append("near_duplicate_oblique")
    if method in GENERATIVE_METHODS:
        flags.append("generative_loadings")
    if not np.isnan(structure.determinacy[j]) and structure.determinacy[j] < LOW_DETERMINACY:
        flags.append("low_determinacy")
    if not np.isnan(structure.eigengap[j]) and structure.eigengap[j] <= SMALL_EIGENGAP:
        flags.append("small_eigengap")
    if small_patient:
        flags.append("small_patient_support")
    if not np.isnan(reconstruction_error) and reconstruction_error > 1e-4:
        flags.append("reconstruction_mismatch")
    unknown = [f for f in flags if f not in CAVEAT_TEXT]
    if unknown:
        raise KeyError("Unknown caveat flags: {}".format(unknown))
    return flags


def build_representation_tables(
    rep: RepInput,
    cohort: Cohort,
    top_genes: int = 20,
    match_bases: Sequence[Basis] = (),
    adt_top: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Compute every card table for one representation (all K factors)."""
    scores = np.asarray(rep.scores, dtype=np.float64)
    loadings = np.asarray(rep.loadings, dtype=np.float64)
    if loadings.shape[0] != len(rep.genes) and loadings.shape[1] == len(rep.genes):
        loadings = loadings.T
    k = int(rep.k)
    if scores.shape != (cohort.n_cells, k) or loadings.shape != (len(rep.genes), k):
        raise ValueError(
            "{}: scores {} / loadings {} do not match n_cells={} genes={} K={}".format(
                rep.rep_id, scores.shape, loadings.shape, cohort.n_cells, len(rep.genes), k
            )
        )
    cids = [card_id(rep.panel_id, rep.method, k, f) for f in range(1, k + 1)]
    z, _, _, _ = zscore_columns(scores)
    score_corr = (z.T @ z) / z.shape[0] if rep.method in OBLIQUE_METHODS else None
    structure = stage1_structure(rep.method, scores, loadings, rep.diagnostics, score_corr)
    coef_matrix, settings = fold_coefficient_matrix(rep.coefs, k)
    fold = summarize_folds(coef_matrix, cohort.supported)
    uni = heldout_univariate(scores, cohort)
    fitted, fold_fit = fitted_attribution(scores, cohort, coef_matrix)
    eta = eta2_by_label(scores, cohort)
    eta_ct = eta2_groups(scores[cohort.labels == 0], cohort.cell_types[cohort.labels == 0], 20) if cohort.cell_types is not None else np.full(k, np.nan)
    orientation = fold.orientation_sign
    mal_label = cohort.labels == 1
    label_dir = np.sign(z[mal_label].mean(axis=0) - z[~mal_label].mean(axis=0)).astype(int)

    # reconstruction against the stored atlas AP
    recon_err = float("nan")
    if rep.expected_ap and not fold_fit.empty:
        errs = [
            abs(float(rep.expected_ap[p]) - float(ap))
            for p, ap in zip(fold_fit["patient"], fold_fit["fold_model_ap"])
            if p in rep.expected_ap and not pd.isna(rep.expected_ap[p])
        ]
        recon_err = max(errs) if errs else float("nan")
    small_patient = bool((uni.groupby("patient")["n_malignant"].first() < 25).any()) if not uni.empty else False

    # patient lift
    lift = uni.merge(fitted, on=["factor", "patient"], how="left") if not fitted.empty else uni.copy()
    lift.insert(0, "card_id", [cids[f - 1] for f in lift["factor"]])

    # fold coefficients (all folds)
    fold_rows = []
    fold_fit_map = fold_fit.set_index("patient").to_dict("index") if not fold_fit.empty else {}
    for patient in coef_matrix.index:
        beta = coef_matrix.loc[patient].to_numpy(dtype=np.float64)
        st = settings.loc[patient]
        info = fold_fit_map.get(patient, {})
        for j in range(k):
            fold_rows.append(
                {
                    "card_id": cids[j],
                    "heldout_patient": patient,
                    "penalty": st["penalty"],
                    "C": float(st["C"]),
                    "l1_ratio": st["l1_ratio"],
                    "coef": float(beta[j]),
                    "is_nonzero": bool(abs(beta[j]) > COEF_EPS),
                    "supported_fold": patient in cohort.supported,
                    "fold_all_zero": bool(fold.fold_all_zero[patient]),
                    "fold_n_active": int(fold.fold_n_active[patient]),
                    "fold_nonzero_fraction": float(fold.fold_nonzero_fraction[patient]),
                    "fold_model_ap": info.get("fold_model_ap", np.nan),
                    "fold_model_lift": info.get("fold_model_lift", np.nan),
                }
            )
    fold_df = pd.DataFrame(fold_rows)

    # genes
    semantics = loading_semantics(rep.method)
    struct_corr = None
    if rep.method in OBLIQUE_METHODS and rep.gene_z is not None:
        struct_corr = structure_correlations(rep.gene_z, z)
    gene_rows: List[Dict[str, object]] = []
    for j in range(k):
        if structure.zero_variance[j]:
            continue
        sc = struct_corr[:, j] if struct_corr is not None else None
        gene_rows += _top_gene_rows(cids[j], rep.genes, loadings[:, j], loadings[:, j], sc, int(orientation[j]), "loading", top_genes, semantics)
        if sc is not None:
            gene_rows += _top_gene_rows(cids[j], rep.genes, sc, loadings[:, j], sc, int(orientation[j]), "structure", top_genes, semantics)
    genes_df = pd.DataFrame(gene_rows)

    # matches
    match_rows = []
    sub_rows = []
    selected = [j + 1 for j in range(k) if fold.n_active_folds[j] > 0]
    if match_bases:
        a = make_basis(rep.rep_id, rep.panel_id, rep.method, k, scores, loadings)
        for b in match_bases:
            if b.rep_id == rep.rep_id:
                continue
            m = match_factors(a, b)
            if m.empty:
                continue
            m.insert(0, "card_id", [cids[f - 1] for f in m["factor"]])
            m["other_representation_id"] = b.rep_id
            m["other_method"] = b.method
            m["other_k"] = b.k
            m["matched_card_id"] = [card_id(b.panel_id, b.method, b.k, f) for f in m["matched_factor"]]
            match_rows.append(m)
            sub_rows.append(subspace_summary(a, b, selected))
    matches_df = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    subspace_df = pd.DataFrame(sub_rows)

    # distributions, cell type, ADT, phi
    dist = score_distribution(z, cohort, orientation)
    dist.insert(0, "card_id", [cids[f - 1] for f in dist["factor"]])
    ct = celltype_means(z, cohort, orientation)
    if not ct.empty:
        ct.insert(0, "card_id", [cids[f - 1] for f in ct["factor"]])
    adt_rows = []
    iso_max: Dict[str, np.ndarray] = {}
    if cohort.adt is not None:
        rho = adt_spearman(scores, cohort.adt)
        names = cohort.adt.names
        iso = cohort.adt.is_isotype
        for label_name, res in rho.items():
            pooled = res["pooled"] * np.where(orientation == 0, 1, orientation)[None, :]
            strat = res["stratified"] * np.where(orientation == 0, 1, orientation)[None, :]
            iso_abs = np.abs(pooled[iso]) if iso.any() else np.full((1, k), np.nan)
            iso_max[label_name] = np.array([np.nanmax(c) if np.isfinite(c).any() else np.nan for c in iso_abs.T])
            for j in range(k):
                if structure.zero_variance[j]:
                    continue
                col = np.abs(np.where(iso, np.nan, pooled[:, j]))
                if np.all(np.isnan(col)):
                    continue
                order = np.argsort(-np.nan_to_num(col, nan=-1.0), kind="mergesort")[:adt_top]
                for rank, mi in enumerate(order, start=1):
                    adt_rows.append(
                        {
                            "card_id": cids[j],
                            "label": label_name,
                            "rank": rank,
                            "adt_marker": names[mi],
                            "is_isotype": bool(iso[mi]),
                            "spearman_pooled": float(pooled[mi, j]),
                            "spearman_patient_stratified": float(strat[mi, j]),
                            "n_cells": int(res["n_cells"]),
                            "oriented": bool(orientation[j] != 0),
                        }
                    )
    adt_df = pd.DataFrame(adt_rows)
    phi_rows = []
    if score_corr is not None:
        iu = np.triu_indices(k, 1)
        for i, j2 in zip(*iu):
            phi_rows.append({"representation_id": rep.rep_id, "factor_i": int(i) + 1, "factor_j": int(j2) + 1, "score_r": float(score_corr[i, j2])})
    phi_df = pd.DataFrame(phi_rows, columns=list(OPTIONAL_FILES["factor_cards_phi.csv"]))

    # index
    first_setting = settings.iloc[0] if len(settings) else pd.Series({"penalty": np.nan, "C": np.nan, "l1_ratio": np.nan})
    uni_sup = uni.groupby("factor") if not uni.empty else None
    fit_grp = fitted.groupby("factor") if not fitted.empty else None
    supported_zero = [p for p in cohort.supported if fold.fold_all_zero.get(p, False)]
    index_rows = []
    for j in range(k):
        f = j + 1
        status = fold.orientation_status[j]
        flags = caveat_flags_for(rep.method, status, fold, j, structure, small_patient, recon_err, cohort.supported)
        m_ids: List[str] = []
        if not matches_df.empty:
            mm = matches_df.loc[(matches_df["factor"] == f) & (matches_df["abs_r"] >= MATCH_ABS_R)].sort_values("abs_r", ascending=False)
            m_ids = list(mm["matched_card_id"])
        u = uni_sup.get_group(f) if uni_sup is not None and f in uni_sup.groups else pd.DataFrame()
        fg = fit_grp.get_group(f) if fit_grp is not None and f in fit_grp.groups else pd.DataFrame()
        # degenerate = no usable score (zero variance / zero loadings); scale_outlier stays a caveat
        degenerate = bool(structure.zero_variance[j])
        index_rows.append(
            {
                "card_id": cids[j],
                "panel_id": rep.panel_id,
                "method": rep.method,
                "k": k,
                "factor": f,
                "is_selected_driver": bool(fold.n_active_folds[j] > 0 and not fold.dense),
                "n_active_folds": int(fold.n_active_folds[j]),
                "sign_consistency": float(fold.sign_consistency[j]),
                "orientation_sign": int(orientation[j]),
                "eta2_normal": float(eta["normal"][j]),
                "eta2_malignant": float(eta["malignant"][j]),
                "degenerate_flag": degenerate,
                "effective_k": int(structure.effective_k),
                "determinacy": float(structure.determinacy[j]),
                "matched_cards": ";".join(m_ids),
                "caveat_flags": ";".join(flags),
                # extras
                "representation_id": rep.rep_id,
                "panel_label": rep.panel_label,
                "n_genes": len(rep.genes),
                "shortlist_reasons": rep.shortlist_reasons,
                "orientation_status": status,
                "n_folds": int(fold.n_folds),
                "n_active_folds_all": int(fold.n_active_folds_all[j]),
                "n_folds_all": int(fold.n_folds_all),
                "fit_penalty": first_setting.get("penalty"),
                "fit_C": first_setting.get("C"),
                "fit_l1_ratio": first_setting.get("l1_ratio"),
                "fit_dense": bool(fold.dense),
                "median_fold_nonzero_fraction": float(fold.median_nonzero_fraction),
                "n_all_zero_folds": len(supported_zero),
                "all_zero_folds": ";".join(supported_zero),
                "median_abs_coef": float(fold.median_abs_coef[j]) if fold.n_active_folds[j] else 0.0,
                "max_fitted_ap_drop": float(fg["fitted_ap_drop"].max()) if not fg.empty else np.nan,
                "median_univariate_lift": float(u["univariate_lift"].median()) if not u.empty else np.nan,
                "max_univariate_lift": float(u["univariate_lift"].max()) if not u.empty else np.nan,
                "n_patients_univariate_lift_ge_0p05": int((u["univariate_lift"] >= 0.05).sum()) if not u.empty else 0,
                "label_direction_sign": int(label_dir[j]),
                "eta2_celltype_normal": float(eta_ct[j]),
                "determinacy_method": structure.determinacy_method,
                "score_sd": float(structure.score_sd[j]),
                "eigengap_rel": float(structure.eigengap[j]),
                "eigengap_basis": structure.eigengap_basis,
                "near_duplicate_max_abs_r": float(structure.near_duplicate_r[j]),
                "near_duplicate_partner": int(structure.near_duplicate_partner[j]) if structure.near_duplicate_partner[j] > 0 else np.nan,
                "adt_isotype_max_abs_rho_normal": float(iso_max.get("normal", np.full(k, np.nan))[j]),
                "adt_isotype_max_abs_rho_malignant": float(iso_max.get("malignant", np.full(k, np.nan))[j]),
                "reconstruction_max_abs_error": recon_err,
                "loading_semantics": semantics,
                "card_md": "cards/{}.md".format(cids[j]),
            }
        )
    index_df = pd.DataFrame(index_rows)
    return {
        "index": index_df,
        "genes": genes_df,
        "patient_lift": lift,
        "fold_coefs": fold_df,
        "matches": matches_df,
        "subspace": subspace_df,
        "score_dist": dist,
        "celltype": ct,
        "adt": adt_df,
        "phi": phi_df,
    }


TABLE_FILES = {
    "index": "factor_cards_index.csv",
    "genes": "factor_cards_genes.csv",
    "patient_lift": "factor_cards_patient_lift.csv",
    "fold_coefs": "factor_cards_fold_coefs.csv",
    "matches": "factor_cards_matches.csv",
    "subspace": "factor_cards_subspace.csv",
    "score_dist": "factor_cards_score_dist.csv",
    "celltype": "factor_cards_celltype.csv",
    "adt": "factor_cards_adt.csv",
    "phi": "factor_cards_phi.csv",
}


# --------------------------------------------------------------------------------------
# card markdown
# --------------------------------------------------------------------------------------


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "n/a"
    return "{:.{d}f}".format(v, d=digits)


def _pp(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else "{:+.2f}".format(100.0 * v)


def _md_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def render_card_markdown(
    card: Mapping[str, object],
    genes: pd.DataFrame,
    lift: pd.DataFrame,
    folds: pd.DataFrame,
    matches: pd.DataFrame,
    dist: pd.DataFrame,
    adt: pd.DataFrame,
    celltype: pd.DataFrame,
    phi_partners: Optional[pd.DataFrame] = None,
    gene_rows_per_pole: int = 12,
) -> str:
    cid = str(card["card_id"])
    method = str(card["method"])
    k = int(card["k"])
    f = int(card["factor"])
    status = str(card.get("orientation_status", ""))
    sign = int(card["orientation_sign"])
    flags = [x for x in str(card.get("caveat_flags", "")).split(";") if x]
    title = "{} | {} K={} | factor {}".format(card.get("panel_label") or card["panel_id"], method, k, f)
    out: List[str] = ["# " + title, "", "Card id: `{}`".format(cid), ""]
    out.append(
        "Representation `{}` ({} genes). Stage 2 probe: the globally selected LOPO setting, {} C={}{}{}.".format(
            card.get("representation_id"),
            card.get("n_genes"),
            card.get("fit_penalty"),
            card.get("fit_C"),
            "" if pd.isna(card.get("fit_l1_ratio")) else " l1_ratio={}".format(card.get("fit_l1_ratio")),
            " (dense fit)" if bool(card.get("fit_dense", False)) else "",
        )
    )
    if card.get("shortlist_reasons"):
        out.append("Shortlist reason: {}.".format(card.get("shortlist_reasons")))
    out += ["", "## Orientation", ""]
    if status == "oriented":
        out.append(
            "sign(beta) = {:+d} in every supported fold where the factor is active ({} of {} folds). "
            "Oriented score = {:+d} x raw score; a higher oriented score pushes the probe toward malignant. "
            "The `malignant` pole lists genes whose expected expression rises with the oriented factor.".format(
                sign, int(card["n_active_folds"]), int(card["n_folds"]), sign
            )
        )
    elif status == "unoriented_all_zero":
        out.append(
            "Unoriented: beta = 0 in all {} supported folds. Genes are listed by raw loading sign "
            "(`positive` / `negative`), which is arbitrary. Pooled label direction (descriptive, not the "
            "probe): malignant cells sit on the {} side of the raw score.".format(
                int(card["n_folds"]), "positive" if int(card.get("label_direction_sign", 0)) >= 0 else "negative"
            )
        )
    else:
        out.append(
            "Unoriented: beta changes sign across supported folds (sign consistency {}). Genes are listed by "
            "raw loading sign, which is arbitrary.".format(_fmt(card.get("sign_consistency"), 2))
        )
    out += ["", "## Selection, beta and held-out lift (supported LOPO folds)", ""]
    out.append(
        "Univariate lift = held-out AP of this factor alone (sign fixed on the training folds) minus the "
        "patient's malignant prevalence. Fitted single-factor lift and ablation dAP are attributions inside "
        "the fitted probe (not causal). All lifts in percentage points."
    )
    out.append("")
    frows = []
    lift_map = lift.set_index("patient").to_dict("index") if not lift.empty else {}
    sup_folds = folds.loc[folds["supported_fold"].astype(bool)] if not folds.empty else folds
    for _, row in sup_folds.sort_values("heldout_patient").iterrows():
        p = row["heldout_patient"]
        lr = lift_map.get(p, {})
        beta_txt = "0 (all-zero fold)" if bool(row["fold_all_zero"]) else _fmt(row["coef"], 4)
        frows.append(
            [
                p,
                beta_txt,
                "yes" if bool(row["is_nonzero"]) else "no",
                _pp(lr.get("univariate_lift")),
                _fmt(lr.get("univariate_auroc"), 3),
                _pp(lr.get("fitted_single_factor_lift")),
                _pp(lr.get("fitted_ap_drop")),
                _pp(row.get("fold_model_lift")),
                lr.get("n_malignant", ""),
                _fmt(lr.get("prevalence"), 3),
            ]
        )
    out.append(
        _md_table(
            ["held-out", "beta", "active", "univariate lift", "univariate AUROC", "fitted single-factor lift", "ablation dAP", "full-model lift", "n malignant", "prevalence"],
            frows,
        )
    )
    out.append("")
    out.append(
        "Selection frequency {} / {} supported folds; sign consistency {}; all-zero supported folds: {}. "
        "Unsupported-patient folds (training fits only, no held-out lift) with beta != 0: {} of {}.".format(
            int(card["n_active_folds"]),
            int(card["n_folds"]),
            _fmt(card.get("sign_consistency"), 2),
            (card.get("all_zero_folds") if isinstance(card.get("all_zero_folds"), str) and card.get("all_zero_folds") else "none"),
            int(card.get("n_active_folds_all", 0)) - int(card["n_active_folds"]),
            int(card.get("n_folds_all", 0)) - int(card["n_folds"]),
        )
    )
    out += ["", "## Genes from both poles", ""]
    out.append("Loading semantics: `{}`.".format(card.get("loading_semantics")))
    if method in GENERATIVE_METHODS:
        out.append("FA / FactoSig loadings are generative (expected expression vs factor), not score weights.")
    if genes.empty:
        out.append("")
        out.append("No genes listed (degenerate factor).")
    for ranked_by in ("loading", "structure"):
        g = genes.loc[genes["ranked_by"] == ranked_by] if not genes.empty else genes
        if g.empty:
            continue
        poles = sorted(g["pole"].unique(), key=lambda p: (POLES_ORIENTED + POLES_RAW).index(p))
        label = "pattern / loading" if ranked_by == "loading" else "structure, corr(gene, score)"
        out += ["", "Ranked by {}:".format(label), ""]
        cols = []
        for pole in poles:
            sub = g.loc[g["pole"] == pole].sort_values("rank").head(gene_rows_per_pole)
            val_col = "oriented_loading" if (ranked_by == "loading" and sign != 0) else ("structure_corr" if ranked_by == "structure" else "loading")
            cols.append((pole, ["{} ({})".format(r["gene"], _fmt(r[val_col], 3)) for _, r in sub.iterrows()]))
        depth = max((len(c[1]) for c in cols), default=0)
        rows = [[i + 1] + [c[1][i] if i < len(c[1]) else "" for c in cols] for i in range(depth)]
        out.append(_md_table(["rank"] + ["{} pole".format(c[0]) for c in cols], rows))
    if phi_partners is not None and not phi_partners.empty:
        out += ["", "Factor score correlations (Phi, oblique fit), strongest partners:", ""]
        out.append(_md_table(["partner factor", "score r"], [[int(r["partner"]), _fmt(r["score_r"], 3)] for _, r in phi_partners.iterrows()]))
    out += ["", "## Patient effect and score distributions", ""]
    out.append(
        "eta2 of patient within label (D14 strata: >=200 normal / >=20 malignant cells per patient): normal {}, "
        "malignant {}. eta2 of predicted cell type within normal cells: {}.".format(
            _fmt(card.get("eta2_normal")), _fmt(card.get("eta2_malignant")), _fmt(card.get("eta2_celltype_normal"))
        )
    )
    if not dist.empty:
        out.append("")
        out.append("Median {}cohort-z score by patient (normal / malignant):".format("oriented " if sign != 0 else "raw "))
        out.append("")
        piv = dist.pivot_table(index="patient", columns="label", values="q50", aggfunc="first")
        npiv = dist.pivot_table(index="patient", columns="label", values="n_cells", aggfunc="first")
        rows = []
        for p in piv.index:
            rows.append(
                [
                    p,
                    _fmt(piv.loc[p].get("normal"), 2),
                    _fmt(piv.loc[p].get("malignant"), 2) if "malignant" in piv.columns else "n/a",
                    "" if pd.isna(npiv.loc[p].get("malignant", np.nan)) else int(npiv.loc[p].get("malignant")),
                ]
            )
        out.append(_md_table(["patient", "normal q50", "malignant q50", "n malignant"], rows))
    if not celltype.empty:
        top = celltype.loc[celltype["label"] == "normal"].sort_values("mean_oriented_z", ascending=False)
        if not top.empty:
            out.append("")
            out.append(
                "Normal cell types with the highest / lowest mean {}z: {} / {}.".format(
                    "oriented " if sign != 0 else "raw ",
                    ", ".join("{} ({})".format(r["cell_type"], _fmt(r["mean_oriented_z"], 2)) for _, r in top.head(3).iterrows()),
                    ", ".join("{} ({})".format(r["cell_type"], _fmt(r["mean_oriented_z"], 2)) for _, r in top.tail(3).iloc[::-1].iterrows()),
                )
            )
    out += ["", "## Cross-method and cross-K matches (same panel; never by index)", ""]
    if matches.empty:
        out.append("No matches computed.")
    else:
        m = matches.sort_values(["other_k", "other_method"])
        rows = [
            [
                "{} K={}".format(r["other_method"], int(r["other_k"])),
                r["relation"],
                int(r["matched_factor"]),
                _fmt(r["score_r"], 3),
                _fmt(r["r2_many_to_one"], 3),
                _fmt(r["angle_deg"], 1),
                _fmt(r["tucker_phi_aligned"], 3),
            ]
            for _, r in m.iterrows()
        ]
        out.append(_md_table(["other representation", "relation", "matched factor", "score r", "R2 on whole basis", "angle to span (deg)", "Tucker phi (aligned)"], rows))
        out.append("")
        out.append("Tucker phi bands: >= 0.95 equal, 0.85-0.94 fair. R2 is many-to-one: how much of this factor the other basis spans.")
    out += ["", "## Surface protein (ADT, CLR per cell, within-label Spearman)", ""]
    if adt.empty:
        out.append("Not computed.")
    else:
        for label_name in ("normal", "malignant"):
            a = adt.loc[adt["label"] == label_name].sort_values("rank").head(6)
            if a.empty:
                continue
            out.append("{} cells ({} cells; {}rho):".format(label_name.capitalize(), int(a["n_cells"].iloc[0]), "oriented " if sign != 0 else ""))
            out.append("")
            out.append(
                _md_table(
                    ["marker", "rho pooled", "rho patient-stratified"],
                    [[r["adt_marker"], _fmt(r["spearman_pooled"], 3), _fmt(r["spearman_patient_stratified"], 3)] for _, r in a.iterrows()],
                )
            )
            out.append("")
        out.append(
            "Isotype control max |rho|: normal {}, malignant {}. Confounds: every malignant cell is "
            "recipient-origin (normal cells are almost all donor), and patient is nested in batch, so pooled "
            "within-label rho still mixes patients; the patient-stratified column averages within-patient rho.".format(
                _fmt(card.get("adt_isotype_max_abs_rho_normal")), _fmt(card.get("adt_isotype_max_abs_rho_malignant"))
            )
        )
    out += ["", "## Stage 1 structure", ""]
    out.append(
        "effective_k {} of K={}; degenerate {}; score SD {}; determinacy {} ({}); relative eigengap {} ({}).".format(
            int(card["effective_k"]),
            k,
            _fmt(card.get("degenerate_flag")),
            _fmt(card.get("score_sd")),
            _fmt(card.get("determinacy")),
            card.get("determinacy_method"),
            _fmt(card.get("eigengap_rel")),
            card.get("eigengap_basis"),
        )
    )
    out += ["", "## Caveats", ""]
    for flag in flags:
        out.append("- `{}`: {}".format(flag, CAVEAT_TEXT.get(flag, "")))
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# bundle schema validation
# --------------------------------------------------------------------------------------


def validate_bundle(bundle_dir: Path, check_cards: bool = True) -> List[str]:
    """Return a list of schema errors for a factor_cards_v1 bundle (empty = valid)."""
    bundle_dir = Path(bundle_dir)
    errors: List[str] = []
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        errors.append("missing {}".format(MANIFEST_NAME))
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError as exc:
            errors.append("manifest is not JSON: {}".format(exc))
            manifest = {}
        for key in MANIFEST_KEYS:
            if key not in manifest:
                errors.append("manifest missing key {}".format(key))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append("manifest schema_version {} != {}".format(manifest.get("schema_version"), SCHEMA_VERSION))
        for name, info in (manifest.get("inputs") or {}).items():
            path = str(info.get("path", "")) if isinstance(info, dict) else str(info)
            if path.startswith("/") or path.startswith("~"):
                errors.append("manifest input {} is not EXP-relative: {}".format(name, path))
    frames: Dict[str, pd.DataFrame] = {}
    for name, cols in list(BUNDLE_FILES.items()) + list(OPTIONAL_FILES.items()):
        path = bundle_dir / name
        if not path.is_file():
            if name in BUNDLE_FILES:
                errors.append("missing {}".format(name))
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            errors.append("{} is empty (no header)".format(name))
            continue
        frames[name] = frame
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            errors.append("{} missing columns {}".format(name, missing))
    index = frames.get("factor_cards_index.csv")
    if index is not None and "card_id" in index.columns:
        ids = index["card_id"].astype(str)
        if ids.duplicated().any():
            errors.append("duplicate card_id in index")
        bad = [c for c in ids if CARD_ID_PATTERN.match(c) is None]
        if bad:
            errors.append("{} malformed card ids, e.g. {}".format(len(bad), bad[:3]))
        if {"panel_id", "method", "k", "factor"} <= set(index.columns):
            rebuilt = [card_id(p, m, k, f) for p, m, k, f in zip(index["panel_id"], index["method"], index["k"], index["factor"])]
            if list(ids) != rebuilt:
                errors.append("card_id does not match panel_id/method/k/factor")
        if "orientation_sign" in index.columns and not index["orientation_sign"].isin([-1, 0, 1]).all():
            errors.append("orientation_sign outside {-1,0,1}")
        if "caveat_flags" in index.columns:
            unknown = sorted({f for s in index["caveat_flags"].fillna("") for f in str(s).split(";") if f and f not in CAVEAT_TEXT})
            if unknown:
                errors.append("unknown caveat flags {}".format(unknown))
        if {"orientation_sign", "n_active_folds"} <= set(index.columns):
            if ((index["orientation_sign"] != 0) & (index["n_active_folds"] == 0)).any():
                errors.append("oriented card with zero active folds")
        for col in ("eta2_normal", "eta2_malignant"):
            if col in index.columns:
                v = pd.to_numeric(index[col], errors="coerce").dropna()
                if ((v < -1e-9) | (v > 1 + 1e-9)).any():
                    errors.append("{} outside [0,1]".format(col))
        id_set = set(ids)
        for name in list(BUNDLE_FILES) + list(OPTIONAL_FILES):
            frame = frames.get(name)
            if frame is None or "card_id" not in frame.columns or name == "factor_cards_index.csv":
                continue
            orphan = set(frame["card_id"].astype(str)) - id_set
            if orphan:
                errors.append("{} has {} card ids not in the index".format(name, len(orphan)))
        genes = frames.get("factor_cards_genes.csv")
        if genes is not None and "pole" in genes.columns:
            badp = set(genes["pole"].astype(str)) - set(POLES_ORIENTED + POLES_RAW)
            if badp:
                errors.append("unknown poles {}".format(sorted(badp)))
            if {"pole", "oriented_loading"} <= set(genes.columns):
                raw = genes["pole"].isin(POLES_RAW)
                if genes.loc[raw, "oriented_loading"].notna().any():
                    errors.append("unoriented genes carry oriented_loading")
                if genes.loc[~raw, "oriented_loading"].isna().any():
                    errors.append("oriented genes lack oriented_loading")
        if check_cards and "card_md" in index.columns:
            missing_md = [p for p in index["card_md"].astype(str) if not (bundle_dir / p).is_file()]
            if missing_md:
                errors.append("{} card markdown files missing, e.g. {}".format(len(missing_md), missing_md[:2]))
    return errors


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path) -> Dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, frame in tables.items():
        name = TABLE_FILES[key]
        if frame is None or (frame.empty and len(frame.columns) == 0):
            frame = pd.DataFrame(columns=list(BUNDLE_FILES.get(name) or OPTIONAL_FILES.get(name) or ("card_id",)))
        frame.to_csv(out_dir / name, index=False, float_format=FLOAT_FORMAT)
        counts[name] = int(len(frame))
    return counts


def concat_tables(parts: Sequence[Mapping[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for key in TABLE_FILES:
        frames = [p[key] for p in parts if key in p and p[key] is not None and not p[key].empty]
        out[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list((BUNDLE_FILES.get(TABLE_FILES[key]) or OPTIONAL_FILES.get(TABLE_FILES[key]) or ())))
    return out


def render_all_cards(tables: Mapping[str, pd.DataFrame], cards_dir: Path) -> int:
    """Write cards/<card_id>.md for every index row. Returns the number written."""
    cards_dir = Path(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)

    def grouped(key: str) -> Dict[str, pd.DataFrame]:
        frame = tables.get(key)
        if frame is None or frame.empty or "card_id" not in frame.columns:
            return {}
        return {cid: g for cid, g in frame.groupby("card_id", sort=False)}

    by = {key: grouped(key) for key in ("genes", "patient_lift", "fold_coefs", "matches", "score_dist", "adt", "celltype")}
    phi = tables.get("phi")
    phi_by_rep = {rid: g for rid, g in phi.groupby("representation_id")} if phi is not None and not phi.empty else {}
    empty = pd.DataFrame()
    n = 0
    for _, card in tables["index"].iterrows():
        cid = card["card_id"]
        partners = None
        rep_phi = phi_by_rep.get(card["representation_id"])
        if rep_phi is not None:
            f = int(card["factor"])
            sub = rep_phi.loc[(rep_phi["factor_i"] == f) | (rep_phi["factor_j"] == f)].copy()
            sub["partner"] = np.where(sub["factor_i"] == f, sub["factor_j"], sub["factor_i"])
            sub["abs_r"] = sub["score_r"].abs()
            partners = sub.sort_values("abs_r", ascending=False).head(3)
        text = render_card_markdown(
            card,
            by["genes"].get(cid, empty),
            by["patient_lift"].get(cid, empty),
            by["fold_coefs"].get(cid, empty),
            by["matches"].get(cid, empty),
            by["score_dist"].get(cid, empty),
            by["adt"].get(cid, empty),
            by["celltype"].get(cid, empty),
            partners,
        )
        (cards_dir / "{}.md".format(cid)).write_text(text)
        n += 1
    return n


# --------------------------------------------------------------------------------------
# bundle schema document (A4 builds against this text)
# --------------------------------------------------------------------------------------

BUNDLE_SCHEMA_MD = """# factor_cards_v1 bundle schema (schema_version 1)

Producer: `scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py` (D18, W5.1).
Consumer: the W7.2 factor explorer. Location: `analysis/scorecards/stage2_diagnostic/factor_cards_v1/`
(EXP-relative). The app reads only the CSV and JSON files; `cards/*.md` are for people and slides.
Required columns are fixed for schema_version 1; files may carry extra columns, and consumers
must ignore columns they do not know. All paths inside the bundle are relative to the bundle dir
or to the experiment dir, never absolute.

## Identifiers

- `card_id` = `<panel_id>__<method>__k<K>__f<NNN>` (factor 1-based, zero-padded to 3 digits;
  method in pca | fa | factosig | factosig_promax). Regex:
  `^(.+)__(pca|fa|factosig|factosig_promax)__k(\\d+)__f(\\d{3})$`.
- `representation_id` = `<panel_id>|dr|<method>|<K>|42` (the atlas id).
- `matched_cards` and `matched_card_id` may name cards of representations outside the shortlist;
  such ids have no row in the index and must be shown as plain text, not links.

## factor_cards_manifest.json

Keys: `schema_version` (1), `slug`, `inputs` ({name: {path (EXP-relative), size_bytes, mtime_utc}}),
`git_state` ({head, dirty}), `created_by` (repo-relative script path), `created_utc`, `parameters`,
`counts` (rows per file, n_cards, n_representations), `runtime_seconds`. Extra keys in v1:
`analysis_id`, `task`, `compute_seconds`, `merge_only`, `part_worker_seconds_sum`,
`regression_checks_passed`.

## factor_cards_index.csv (one row per card)

Required:

| column | type | meaning |
|---|---|---|
| card_id | str | see above |
| panel_id | str | Stage 0 panel id |
| method | str | Stage 1 method |
| k | int | requested K |
| factor | int | 1-based factor index within this representation only |
| is_selected_driver | bool | beta != 0 in >=1 supported LOPO fold AND the fit is not dense (median supported-fold nonzero fraction <= 0.5) |
| n_active_folds | int | supported folds (of n_folds, normally 7) with beta != 0 |
| sign_consistency | float | share of active supported folds with the majority sign; empty when never active |
| orientation_sign | int | +1 / -1 = sign(beta), stable over active supported folds (positive = toward malignant); 0 = unoriented |
| eta2_normal, eta2_malignant | float | eta2 of patient within label (normal patients >=200 cells, malignant >=20) |
| degenerate_flag | bool | zero score variance / zero loadings (a blown-up or collapsed score scale is the separate caveat `scale_outlier`) |
| effective_k | int | non-degenerate factors in this representation |
| determinacy | float | factor-score determinacy rho (fa: saved psi; factosig: psi approx 1-h2); empty for pca (exact scores) and promax (Phi unsaved) |
| matched_cards | str | ';'-separated card ids with abs(score r) >= 0.5: Hungarian match at the same K for other methods, best-abs(r) match at other K, sorted by abs(r) |
| caveat_flags | str | ';'-separated codes (vocabulary below) |

Extra columns (stable in v1, may be extended): representation_id, panel_label, n_genes,
shortlist_reasons, orientation_status (oriented | unoriented_all_zero | unoriented_sign_flip),
n_folds, n_active_folds_all, n_folds_all, fit_penalty, fit_C, fit_l1_ratio, fit_dense,
median_fold_nonzero_fraction, n_all_zero_folds, all_zero_folds, median_abs_coef,
max_fitted_ap_drop, median_univariate_lift, max_univariate_lift,
n_patients_univariate_lift_ge_0p05, label_direction_sign (pooled malignant-minus-normal sign of
the raw score; descriptive, not the probe), eta2_celltype_normal, determinacy_method, score_sd,
eigengap_rel, eigengap_basis, near_duplicate_max_abs_r, near_duplicate_partner,
adt_isotype_max_abs_rho_normal, adt_isotype_max_abs_rho_malignant,
reconstruction_max_abs_error, loading_semantics, card_md (relative path).

## factor_cards_genes.csv (top genes per pole)

Required: card_id, gene, loading (raw), oriented_loading (orientation_sign x loading; empty when
unoriented), rank (1-based within card x ranked_by x pole), pole (`malignant` | `normal` when
oriented; `positive` | `negative` by raw sign when unoriented), structure_corr (corr(gene, score),
oriented when possible; promax only, else empty).
Extra: ranked_by (`loading` | `structure`; promax cards carry both lists), loading_semantics.
Degenerate factors have no gene rows.

## factor_cards_patient_lift.csv (card x supported patient)

Required: card_id, patient, univariate_lift (held-out AP of the factor alone, sign fixed on the
training folds, minus prevalence), univariate_auroc (same sign), n_malignant, prevalence.
Extra: univariate_auprc, univariate_lift_sign_agnostic (F6 definition: max over both signs on the
held-out patient; optimistic), train_orientation_sign, heldout_malignant_mean_z,
heldout_normal_mean_z, n_cells, fitted_coef, fitted_full_model_ap, fitted_full_model_lift,
fitted_ablated_ap, fitted_ap_drop (D8 ablation dAP), fitted_single_factor_lift (D8).

## factor_cards_fold_coefs.csv (card x LOPO fold, all 13 folds)

Required: card_id, heldout_patient, penalty, C, coef.
Extra: l1_ratio, is_nonzero, supported_fold, fold_all_zero (whole coefficient vector zero),
fold_n_active, fold_nonzero_fraction, fold_model_ap, fold_model_lift (reconstructed; supported folds).

## Optional files (present in v1)

- factor_cards_matches.csv: card_id, other_representation_id, other_method, other_k, relation
  (`hungarian_same_k` | `best_abs_r`), matched_factor, matched_card_id, score_r, abs_r,
  r2_many_to_one, angle_deg, tucker_phi_aligned, best_factor, best_r, hungarian_factor, hungarian_r.
- factor_cards_subspace.csv: representation_id, other_representation_id, k, other_k, other_method,
  rank, other_rank, n_canonical, cc_min, cc_median, cc_max, n_cc_below_0p9, selected_rank,
  selected_cc_min, selected_cc_median (principal angles as canonical correlations).
- factor_cards_score_dist.csv: card_id, factor, patient, label, n_cells, oriented, mean, sd, q10,
  q25, q50, q75, q90 (cohort-z score, multiplied by orientation_sign when oriented).
- factor_cards_adt.csv: card_id, label, rank, adt_marker, is_isotype, spearman_pooled,
  spearman_patient_stratified, n_cells, oriented (top markers by |pooled rho| excluding isotypes;
  ADT = CLR per cell of raw counts).
- factor_cards_celltype.csv: card_id, factor, label, cell_type, n_cells, mean_oriented_z.
- factor_cards_phi.csv: representation_id, factor_i, factor_j, score_r (oblique fits only).
- factor_cards_shortlist.csv: the pre-declared shortlist with reasons.
- factor_cards_regression_checks.csv: check_id, description, source, n_compared, max_abs_diff,
  tolerance, passed.

## Caveat flag vocabulary

{flags}
"""


def bundle_schema_markdown() -> str:
    flags = "\n".join("- `{}`: {}".format(k, v) for k, v in CAVEAT_TEXT.items())
    return BUNDLE_SCHEMA_MD.replace("{flags}", flags)
