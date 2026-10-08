"""Stage 1 refit utilities for W2 basis variants (analysis D17, tasks W2.1 / W2.2).

Fit and project PCA and FA Stage 1 bases on a chosen set of cells, with train-only gene
standardization, so that held-out (or left-out) cells are projected rather than fitted.
Dependencies: numpy, scipy and scikit-learn only. No absolute paths (research_index E07).

Conventions copied from the Stage 1 producer
(scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py) and the D11 refit
template (run_basis_refit_target in
scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py):
- input = covered-gene expression already z-scored on all cells
  (stage1_direct_gene/.../transform=zscore/features.npz), so a refit on a cell subset first
  re-standardizes the genes on the fit cells only (population SD; a zero-variance gene keeps
  scale 1 and becomes 0: "feature_zscore_with_zero_variance_to_zero");
- PCA = sklearn PCA(svd_solver='randomized', random_state=seed);
- FA = sklearn FactorAnalysis(svd_method='randomized', random_state=seed).
An unweighted refit on all cells reproduces the stored transductive FA bases (scores and
loadings) and the stored PCA loadings. It does not reproduce the stored PCA scores: those are
sklearn's randomized-SVD U * S from fit_transform, whereas Stage1Basis.transform returns the
projection X @ V onto the loadings (PC8 correlates 0.991 at K10 to 0.999 at K40 with that
projection; D17 d17_reference_axes.csv).

Fit scopes (label every basis by scope, AGENTS R6):
- transductive: all cells;
- inductive: all cells of the training patients (the held-out patient is projected);
- leave_out: all cells except the listed patients (they are projected);
- normal_only: normal cells only (malignant cells are projected);
- balanced: patient-equal weights. PCA uses the weighted covariance and eigh. FA has no
  sample weights in sklearn, so it is fit either on a patient-stratified equal-n subsample
  (stratified_subsample) or, exactly, on moment-matched pseudo-data whose covariance equals
  the weighted covariance (fa_from_covariance; FA maximum likelihood depends on the data
  only through the covariance; with sklearn's randomized SVD and a rescaled tol this
  reproduces a fit on the original rows up to the sign of each component: sklearn's
  svd_flip sign rule depends on the pseudo-data rows; noise variances match exactly).

Persistence (Stage1Basis.save / load): gene order, gene mean / SD of the fit cells, the
components (K x genes, standardized gene space), the model mean, FA noise variance, PCA
explained variance and a fit_info dict. Scores are recomputed with Stage1Basis.transform,
which applies exactly sklearn's PCA.transform / FactorAnalysis.transform from the stored
parameters (no pickles).

Alignment helpers compare a variant basis with a reference basis: score correlation matrix,
Hungarian matching on |r|, many-to-one R^2 of a reference factor on all variant factors,
principal angles between loading subspaces and the containment of one loading vector in a
subspace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment

__all__ = [
    "MODULE_VERSION",
    "METHODS",
    "Stage1Basis",
    "fit_standardizer",
    "apply_standardizer",
    "patient_balanced_weights",
    "stratified_subsample",
    "random_drop_mask",
    "label_matched_drop_mask",
    "fit_scope_mask",
    "weighted_covariance",
    "pseudo_data_from_covariance",
    "fa_from_covariance",
    "fit_basis",
    "score_correlation",
    "hungarian_match",
    "many_to_one_r2",
    "principal_angles_deg",
    "subspace_containment",
    "top_factor_indices",
    "align_scores",
]

MODULE_VERSION = "1.0.0"
METHODS = ("pca", "fa")
_BASIS_NPZ = "basis.npz"
_BASIS_JSON = "basis.json"

ArrayLike = Union[np.ndarray, Sequence[float]]


# --------------------------------------------------------------------------------------
# standardization, weights and fit scopes
# --------------------------------------------------------------------------------------


def _as_2d(X: Any) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be 2-D (cells x genes)")
    return arr


def _row_index(n: int, rows: Any) -> np.ndarray:
    """Integer row index from None (all), a boolean mask or an integer index."""
    if rows is None:
        return np.arange(n)
    rows = np.asarray(rows)
    if rows.dtype == bool:
        if rows.size != n:
            raise ValueError("boolean row mask has the wrong length")
        return np.flatnonzero(rows)
    idx = rows.astype(int).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError("row index out of range")
    return idx


def _normalise_weights(weights: Any, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError("weights must have one value per fit row")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    return w / w.sum()


def fit_standardizer(X: Any, weights: Optional[ArrayLike] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(mean, scale) per gene on the given rows; population SD, optionally weighted.

    A gene with zero (or numerically zero) variance keeps scale 1, so it standardizes to 0,
    as in sklearn StandardScaler and the Stage 1 producer.
    """
    Xa = _as_2d(X)
    if weights is None:
        mean = Xa.mean(axis=0)
        var = Xa.var(axis=0)
    else:
        w = _normalise_weights(weights, Xa.shape[0])
        mean = w @ Xa
        var = w @ (Xa - mean) ** 2
    scale = np.sqrt(var)
    scale[~np.isfinite(scale) | (scale < 10 * np.finfo(np.float64).eps)] = 1.0
    return mean, scale


def apply_standardizer(X: Any, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (_as_2d(X) - np.asarray(mean, dtype=np.float64)) / np.asarray(scale, dtype=np.float64)


def patient_balanced_weights(patients: Any, rows: Any = None) -> np.ndarray:
    """Per-row weights for the fit rows so every patient contributes the same total weight.

    Returns one weight per selected row: 1 / (n_patients * n_cells_of_that_patient), which sums
    to 1 overall and to 1 / n_patients within each patient.
    """
    pat = np.asarray(patients).astype(str)
    idx = _row_index(pat.size, rows)
    sub = pat[idx]
    names, inverse, counts = np.unique(sub, return_inverse=True, return_counts=True)
    return 1.0 / (names.size * counts[inverse].astype(np.float64))


def stratified_subsample(
    patients: Any,
    *,
    n_per_patient: Optional[int] = None,
    seed: int = 0,
    rows: Any = None,
) -> np.ndarray:
    """Sorted row indices with the same number of cells from every patient (no replacement).

    n_per_patient defaults to the smallest patient's cell count among the selected rows.
    Deterministic for a given seed.
    """
    pat = np.asarray(patients).astype(str)
    idx = _row_index(pat.size, rows)
    names, counts = np.unique(pat[idx], return_counts=True)
    n_take = int(counts.min()) if n_per_patient is None else int(n_per_patient)
    if n_take <= 0 or n_take > int(counts.min()):
        raise ValueError(f"n_per_patient must be in [1, {int(counts.min())}] (smallest patient)")
    rng = np.random.default_rng(int(seed))
    chosen: List[np.ndarray] = []
    for name in names:
        members = idx[pat[idx] == name]
        chosen.append(rng.choice(members, size=n_take, replace=False))
    return np.sort(np.concatenate(chosen))


def random_drop_mask(n: int, n_drop: int, *, seed: int = 0, rows: Any = None) -> np.ndarray:
    """Boolean keep-mask over n cells that drops n_drop cells chosen uniformly from `rows`.

    The size-matched random control for a leave-patients-out refit.
    """
    idx = _row_index(int(n), rows)
    if not 0 <= int(n_drop) < idx.size:
        raise ValueError("n_drop must be smaller than the number of eligible rows")
    rng = np.random.default_rng(int(seed))
    keep = np.ones(int(n), dtype=bool)
    keep[rng.choice(idx, size=int(n_drop), replace=False)] = False
    return keep


def label_matched_drop_mask(labels: Any, n_drop: int, n_drop_positive: int, *, seed: int = 0) -> np.ndarray:
    """Boolean keep-mask that drops n_drop cells, exactly n_drop_positive of them with label 1.

    The composition-matched random control for a leave-patients-out refit: it matches both the
    number of dropped cells and the number of dropped malignant cells, so leverage beyond it is
    not explained by how many malignant cells leave the fit. Positives and negatives are drawn
    uniformly without replacement within their label.
    """
    y = np.asarray(labels).astype(int).ravel()
    n_drop, n_pos = int(n_drop), int(n_drop_positive)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y != 1)
    if not 0 <= n_pos <= n_drop:
        raise ValueError("n_drop_positive must lie in [0, n_drop]")
    if n_pos > pos.size or n_drop - n_pos > neg.size or n_drop >= y.size:
        raise ValueError("cannot drop that many cells of one label")
    rng = np.random.default_rng(int(seed))
    keep = np.ones(y.size, dtype=bool)
    keep[rng.choice(pos, size=n_pos, replace=False)] = False
    keep[rng.choice(neg, size=n_drop - n_pos, replace=False)] = False
    return keep


def fit_scope_mask(
    patients: Any,
    *,
    exclude_patients: Iterable[str] = (),
    labels: Any = None,
    label_value: Optional[int] = None,
) -> np.ndarray:
    """Boolean fit-cell mask: drop the listed patients and, optionally, keep one label only.

    Inductive LOPO for held-out patient h: exclude_patients=[h]. Normal-only:
    labels=y, label_value=0.
    """
    pat = np.asarray(patients).astype(str)
    mask = ~np.isin(pat, [str(p) for p in exclude_patients])
    if label_value is not None:
        if labels is None:
            raise ValueError("label_value needs labels")
        lab = np.asarray(labels).ravel()
        if lab.size != pat.size:
            raise ValueError("labels and patients differ in length")
        mask &= lab == label_value
    return mask


# --------------------------------------------------------------------------------------
# covariance-based fits
# --------------------------------------------------------------------------------------


def weighted_covariance(Xs: Any, weights: Optional[ArrayLike] = None) -> np.ndarray:
    """(Weighted) population covariance of the rows of Xs about their (weighted) mean."""
    Xa = _as_2d(Xs)
    if weights is None:
        w = np.full(Xa.shape[0], 1.0 / Xa.shape[0])
    else:
        w = _normalise_weights(weights, Xa.shape[0])
    centred = Xa - w @ Xa
    cov = (centred * w[:, None]).T @ centred
    return 0.5 * (cov + cov.T)


def _eigh_desc(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _orient_rows(components: np.ndarray) -> np.ndarray:
    """Deterministic sign: the largest-|value| entry of each component row is positive."""
    comp = np.array(components, dtype=np.float64, copy=True)
    for i in range(comp.shape[0]):
        j = int(np.argmax(np.abs(comp[i])))
        if comp[i, j] < 0:
            comp[i] = -comp[i]
    return comp


def pseudo_data_from_covariance(C: Any) -> np.ndarray:
    """2p x p zero-mean matrix whose population covariance equals C (C symmetric PSD).

    Rows are +/- sqrt(p * lambda_j) v_j for the eigenpairs of C; negative eigenvalues from
    round-off are clipped to 0.
    """
    Cm = np.asarray(C, dtype=np.float64)
    if Cm.ndim != 2 or Cm.shape[0] != Cm.shape[1]:
        raise ValueError("C must be square")
    vals, vecs = _eigh_desc(Cm)
    p = Cm.shape[0]
    half = (np.sqrt(np.clip(vals, 0.0, None) * p)[:, None] * vecs.T)  # (p, p): row j = sqrt(p lambda_j) v_j
    return np.vstack([half, -half])


def fa_from_covariance(
    C: Any,
    k: int,
    *,
    seed: int = 42,
    svd_method: str = "randomized",
    n_samples: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-2,
):
    """sklearn FactorAnalysis fitted to a covariance matrix through moment-matched pseudo-data.

    FA maximum likelihood depends on the data only through the sample covariance, and sklearn's
    fit uses only np.var(X) and the SVD of X / sqrt(n * psi); both are functions of the
    pseudo-data covariance, which equals C exactly. sklearn's randomized SVD draws its test
    matrix on the gene side, so its subspaces also depend only on X^T X: with the same seed the
    iterates match a fit on the original rows up to the sign of each component (svd_flip picks
    signs from the row-side singular vectors, which differ for pseudo-data); noise_variance_ and
    the loadings up to column sign match. sklearn stops when the log-likelihood, which is
    scaled by the number of rows, gains less than tol; pass n_samples (the number of cells the
    covariance summarises) to rescale tol so the stopping rule matches a fit on those cells.
    Returns the fitted estimator; its mean_ is 0.
    """
    from sklearn.decomposition import FactorAnalysis

    Y = pseudo_data_from_covariance(C)
    tol_eff = float(tol) if n_samples is None else float(tol) * Y.shape[0] / float(n_samples)
    fa = FactorAnalysis(n_components=int(k), svd_method=svd_method, random_state=int(seed), max_iter=int(max_iter), tol=tol_eff)
    fa.fit(Y)
    return fa


# --------------------------------------------------------------------------------------
# the basis object
# --------------------------------------------------------------------------------------


@dataclass
class Stage1Basis:
    """A fitted Stage 1 basis with everything needed to project new cells.

    components: (K, n_genes) in the standardized gene space of the fit cells.
    gene_mean / gene_scale: standardization of the fit cells (applied before projection).
    model_mean: mean subtracted by the DR model in standardized space (PCA mean_, FA mean_).
    noise_variance: FA uniquenesses (n_genes,), None for PCA.
    explained_variance: PCA eigenvalues (K,), None for FA.
    """

    method: str
    k: int
    gene_names: List[str]
    gene_mean: np.ndarray
    gene_scale: np.ndarray
    components: np.ndarray
    model_mean: np.ndarray
    noise_variance: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None
    fit_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.k = int(self.k)
        self.gene_names = [str(g) for g in self.gene_names]
        self.gene_mean = np.asarray(self.gene_mean, dtype=np.float64).ravel()
        self.gene_scale = np.asarray(self.gene_scale, dtype=np.float64).ravel()
        self.components = np.asarray(self.components, dtype=np.float64)
        self.model_mean = np.asarray(self.model_mean, dtype=np.float64).ravel()
        p = len(self.gene_names)
        if self.components.shape != (self.k, p):
            raise ValueError(f"components shape {self.components.shape} != ({self.k}, {p})")
        for name in ("gene_mean", "gene_scale", "model_mean"):
            if getattr(self, name).shape != (p,):
                raise ValueError(f"{name} must have one value per gene")
        if self.method == "fa":
            if self.noise_variance is None:
                raise ValueError("FA basis needs noise_variance")
            self.noise_variance = np.asarray(self.noise_variance, dtype=np.float64).ravel()
            if self.noise_variance.shape != (p,):
                raise ValueError("noise_variance must have one value per gene")
        if self.explained_variance is not None:
            self.explained_variance = np.asarray(self.explained_variance, dtype=np.float64).ravel()

    # ---- projection ------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def projection_matrix(self) -> np.ndarray:
        """(n_genes, K) matrix P with scores = (standardized X - model_mean) @ P."""
        if self.method == "pca":
            return self.components.T.copy()
        W = self.components
        Wpsi = W / self.noise_variance
        cov_z = np.linalg.inv(np.eye(self.k) + Wpsi @ W.T)
        return Wpsi.T @ cov_z

    def transform(self, X: Any, gene_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Scores (cells x K) for raw input rows; identical to sklearn PCA/FA .transform."""
        if gene_names is not None and [str(g) for g in gene_names] != self.gene_names:
            raise ValueError("gene order differs from the basis gene order")
        Xa = _as_2d(X)
        if Xa.shape[1] != self.n_genes:
            raise ValueError(f"X has {Xa.shape[1]} genes, basis has {self.n_genes}")
        Xs = apply_standardizer(Xa, self.gene_mean, self.gene_scale)
        return (Xs - self.model_mean) @ self.projection_matrix()

    def loadings_common(self) -> np.ndarray:
        """(n_genes, K) loadings on the common (input) gene scale: gene_scale * components^T.

        In the generative reading x = mean + scale * (W^T z + e), so loadings in input units are
        the standardized loadings times the fit cells' gene SD; this is what principal angles
        between bases fitted on different cell sets compare.
        """
        return self.components.T * self.gene_scale[:, None]

    def factor_variance_order(self) -> np.ndarray:
        """Factor order by importance: PCA explained variance; FA sum of squared loadings."""
        if self.method == "pca" and self.explained_variance is not None:
            key = self.explained_variance
        else:
            key = np.sum(self.components ** 2, axis=1)
        return np.argsort(-key, kind="stable")

    # ---- persistence -----------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> Path:
        """Write basis.npz (arrays) and basis.json (gene order, method, K, fit_info)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {
            "gene_mean": self.gene_mean,
            "gene_scale": self.gene_scale,
            "components": self.components,
            "model_mean": self.model_mean,
        }
        if self.noise_variance is not None:
            arrays["noise_variance"] = self.noise_variance
        if self.explained_variance is not None:
            arrays["explained_variance"] = self.explained_variance
        tmp_npz = out / (_BASIS_NPZ + ".tmp.npz")
        np.savez(tmp_npz, **arrays)
        tmp_npz.replace(out / _BASIS_NPZ)
        meta = {
            "module": "stage1_refit",
            "module_version": MODULE_VERSION,
            "method": self.method,
            "k": self.k,
            "n_genes": self.n_genes,
            "gene_names": self.gene_names,
            "arrays": sorted(arrays),
            "fit_info": self.fit_info,
            "status": "done",
        }
        tmp_json = out / (_BASIS_JSON + ".tmp")
        tmp_json.write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
        tmp_json.replace(out / _BASIS_JSON)
        return out

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Stage1Basis":
        src = Path(directory)
        meta = json.loads((src / _BASIS_JSON).read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            raise ValueError(f"{src}: basis.json is not marked done")
        with np.load(src / _BASIS_NPZ) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        return cls(
            method=meta["method"],
            k=int(meta["k"]),
            gene_names=list(meta["gene_names"]),
            gene_mean=arrays["gene_mean"],
            gene_scale=arrays["gene_scale"],
            components=arrays["components"],
            model_mean=arrays["model_mean"],
            noise_variance=arrays.get("noise_variance"),
            explained_variance=arrays.get("explained_variance"),
            fit_info=dict(meta.get("fit_info", {})),
        )

    @staticmethod
    def is_saved(directory: Union[str, Path]) -> bool:
        """True when a complete, loadable basis exists in directory (empty/corrupt -> False)."""
        try:
            Stage1Basis.load(directory)
            return True
        except Exception:
            return False


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------


def fit_basis(
    X: Any,
    method: str,
    k: int,
    *,
    rows: Any = None,
    weights: Optional[ArrayLike] = None,
    seed: int = 42,
    gene_names: Optional[Sequence[str]] = None,
    pca_solver: str = "randomized",
    fa_svd_method: str = "randomized",
    fit_info: Optional[Dict[str, Any]] = None,
) -> Stage1Basis:
    """Fit a PCA or FA basis on the selected rows of X with train-only standardization.

    rows: None (all), boolean mask or integer index of the fit cells.
    weights: optional per-fit-row weights (patient_balanced_weights); then the standardizer is
    weighted, PCA is the eigen-decomposition of the weighted covariance (explained_variance =
    weighted population eigenvalues, not sklearn's n - 1 convention) and FA is fitted to the
    weighted covariance through pseudo-data (fa_from_covariance, same svd_method and seed, tol
    rescaled to the number of fit cells). Unweighted fits use sklearn
    PCA(svd_solver=pca_solver) / FactorAnalysis(svd_method=fa_svd_method) with random_state=seed,
    the stored Stage 1 settings.
    """
    from sklearn.decomposition import PCA, FactorAnalysis

    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    Xa = _as_2d(X)
    idx = _row_index(Xa.shape[0], rows)
    if idx.size <= int(k):
        raise ValueError("need more fit cells than factors")
    names = [str(g) for g in gene_names] if gene_names is not None else [f"gene_{j + 1:04d}" for j in range(Xa.shape[1])]
    if len(names) != Xa.shape[1]:
        raise ValueError("gene_names length does not match X")
    Xfit = Xa[idx]
    w = None if weights is None else _normalise_weights(weights, idx.size)
    mean, scale = fit_standardizer(Xfit, w)
    Xs = apply_standardizer(Xfit, mean, scale)
    info: Dict[str, Any] = {
        "n_fit_cells": int(idx.size),
        "seed": int(seed),
        "weighted": w is not None,
    }
    noise = None
    explained = None
    if method == "pca":
        if w is None:
            model = PCA(n_components=int(k), svd_solver=pca_solver, random_state=int(seed))
            model.fit(Xs)
            components = np.asarray(model.components_, dtype=np.float64)
            model_mean = np.asarray(model.mean_, dtype=np.float64)
            explained = np.asarray(model.explained_variance_, dtype=np.float64)
            info["solver"] = f"sklearn PCA svd_solver={pca_solver}"
            info["explained_variance_ratio"] = np.asarray(model.explained_variance_ratio_).tolist()
        else:
            C = weighted_covariance(Xs, w)
            vals, vecs = _eigh_desc(C)
            components = _orient_rows(vecs[:, : int(k)].T)
            model_mean = np.zeros(Xa.shape[1])
            explained = vals[: int(k)]
            info["solver"] = "weighted covariance + numpy eigh"
            info["explained_variance_ratio"] = (vals[: int(k)] / max(vals.clip(min=0).sum(), 1e-300)).tolist()
    else:
        if w is None:
            model = FactorAnalysis(n_components=int(k), svd_method=fa_svd_method, random_state=int(seed))
            model.fit(Xs)
            info["solver"] = f"sklearn FactorAnalysis svd_method={fa_svd_method}"
        else:
            C = weighted_covariance(Xs, w)
            model = fa_from_covariance(C, int(k), seed=int(seed), svd_method=fa_svd_method, n_samples=int(idx.size))
            info["solver"] = (
                f"sklearn FactorAnalysis on weighted-covariance pseudo-data, svd_method={fa_svd_method}, "
                f"tol rescaled to {int(idx.size)} cells"
            )
        components = np.asarray(model.components_, dtype=np.float64)
        model_mean = np.asarray(model.mean_, dtype=np.float64)
        noise = np.asarray(model.noise_variance_, dtype=np.float64)
        info["n_iter"] = int(getattr(model, "n_iter_", -1))
        info["loglike_last"] = float(model.loglike_[-1]) if getattr(model, "loglike_", None) else None
    if fit_info:
        info.update(fit_info)
    return Stage1Basis(
        method=method,
        k=int(k),
        gene_names=names,
        gene_mean=mean,
        gene_scale=scale,
        components=components,
        model_mean=model_mean,
        noise_variance=noise,
        explained_variance=explained,
        fit_info=info,
    )


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------


def score_correlation(A: Any, B: Any, rows: Any = None) -> np.ndarray:
    """Pearson correlation matrix (A columns x B columns) over the selected rows.

    A zero-variance column (e.g. a degenerate FA factor) gets correlation 0 with everything.
    """
    Aa, Ba = _as_2d(A), _as_2d(B)
    if Aa.shape[0] != Ba.shape[0]:
        raise ValueError("A and B need the same rows")
    idx = _row_index(Aa.shape[0], rows)
    Ac = Aa[idx] - Aa[idx].mean(axis=0)
    Bc = Ba[idx] - Ba[idx].mean(axis=0)
    an = np.linalg.norm(Ac, axis=0)
    bn = np.linalg.norm(Bc, axis=0)
    tiny = 1e-12 * max(1.0, float(np.sqrt(idx.size)))
    an_safe = np.where(an > tiny, an, np.inf)
    bn_safe = np.where(bn > tiny, bn, np.inf)
    return (Ac / an_safe).T @ (Bc / bn_safe)


def hungarian_match(R: Any) -> Dict[int, Tuple[int, float]]:
    """One-to-one matching of reference factors (rows) to variant factors (columns) that
    maximises the total |r|. Returns {ref_index: (variant_index, signed r)}."""
    Rm = np.nan_to_num(np.asarray(R, dtype=np.float64))
    ri, ci = linear_sum_assignment(-np.abs(Rm))
    return {int(r): (int(c), float(Rm[r, c])) for r, c in zip(ri, ci)}


def many_to_one_r2(target: Any, B: Any, rows: Any = None) -> float:
    """R^2 of the OLS regression (with intercept) of one target score on all columns of B."""
    t = np.asarray(target, dtype=np.float64).ravel()
    Ba = _as_2d(B)
    if Ba.shape[0] != t.size:
        raise ValueError("target and B need the same rows")
    idx = _row_index(t.size, rows)
    tt = t[idx] - t[idx].mean()
    Bc = Ba[idx] - Ba[idx].mean(axis=0)
    sst = float(tt @ tt)
    if sst <= 0:
        return float("nan")
    coef, *_ = np.linalg.lstsq(Bc, tt, rcond=None)
    resid = tt - Bc @ coef
    return float(1.0 - (resid @ resid) / sst)


def principal_angles_deg(A: Any, B: Any) -> np.ndarray:
    """Principal angles (degrees, ascending) between the column spaces of A and B (genes x k).

    Zero columns are dropped first (degenerate FA factors).
    """
    Aa, Ba = _as_2d(A), _as_2d(B)
    Aa = Aa[:, np.linalg.norm(Aa, axis=0) > 1e-12]
    Ba = Ba[:, np.linalg.norm(Ba, axis=0) > 1e-12]
    if Aa.shape[1] == 0 or Ba.shape[1] == 0:
        return np.array([90.0])
    return np.sort(np.degrees(subspace_angles(Aa, Ba)))


def subspace_containment(v: Any, B: Any) -> float:
    """||projection of v onto span(B)|| / ||v||: 1 = v lies in the subspace, 0 = orthogonal."""
    vv = np.asarray(v, dtype=np.float64).ravel()
    Ba = _as_2d(B)
    Ba = Ba[:, np.linalg.norm(Ba, axis=0) > 1e-12]
    nv = float(np.linalg.norm(vv))
    if nv == 0 or Ba.shape[1] == 0:
        return float("nan") if nv == 0 else 0.0
    Q, _ = np.linalg.qr(Ba)
    return float(np.linalg.norm(Q.T @ vv) / nv)


def top_factor_indices(order: Sequence[int], m: int) -> np.ndarray:
    return np.asarray(list(order)[: int(m)], dtype=int)


def align_scores(
    ref_scores: Any,
    var_scores: Any,
    ref_factor: int,
    *,
    rows: Any = None,
    ref_loadings: Any = None,
    var_loadings: Any = None,
    ref_order: Optional[Sequence[int]] = None,
    var_order: Optional[Sequence[int]] = None,
    top_m: int = 8,
) -> Dict[str, Any]:
    """Compare a variant basis with the reference basis, focused on one reference factor.

    Scores are compared over `rows` (e.g. the variant's fit cells); loadings (genes x K, common
    gene scale) give principal angles (top_m factors by each basis's own importance order, and
    all K) and the containment of the reference factor's loading in the variant subspaces.
    """
    R = score_correlation(ref_scores, var_scores, rows)
    match = hungarian_match(R)
    j_h, r_h = match[int(ref_factor)]
    row = np.abs(np.nan_to_num(R[int(ref_factor)]))
    j_b = int(np.argmax(row))
    abs_matched = np.asarray([abs(match[i][1]) for i in sorted(match)])
    out: Dict[str, Any] = {
        "ref_factor": int(ref_factor),
        "matched_factor": int(j_h),
        "matched_r": float(r_h),
        "matched_abs_r": float(abs(r_h)),
        "best_factor": j_b,
        "best_abs_r": float(row[j_b]),
        "many_to_one_r2": many_to_one_r2(_as_2d(ref_scores)[:, int(ref_factor)], var_scores, rows),
        "mean_matched_abs_r_all": float(abs_matched.mean()),
        "r_ref_factor_vs_variant": [float(x) for x in np.nan_to_num(R[int(ref_factor)])],
    }
    if ref_order is not None:
        top = top_factor_indices(ref_order, top_m)
        out[f"mean_matched_abs_r_top{top_m}"] = float(np.mean([abs(match[int(i)][1]) for i in top if int(i) in match]))
    if ref_loadings is not None and var_loadings is not None:
        RL, VL = _as_2d(ref_loadings), _as_2d(var_loadings)
        ang_all = principal_angles_deg(RL, VL)
        out["angle_allK_max_deg"] = float(ang_all.max())
        out["angle_allK_mean_cos2"] = float(np.mean(np.cos(np.radians(ang_all)) ** 2))
        if ref_order is not None and var_order is not None:
            rt = top_factor_indices(ref_order, top_m)
            vt = top_factor_indices(var_order, top_m)
            ang_top = principal_angles_deg(RL[:, rt], VL[:, vt])
            out[f"angle_top{top_m}_max_deg"] = float(ang_top.max())
            out[f"angle_top{top_m}_mean_cos2"] = float(np.mean(np.cos(np.radians(ang_top)) ** 2))
            out[f"ref_axis_containment_top{top_m}"] = subspace_containment(RL[:, int(ref_factor)], VL[:, vt])
            rank = [int(i) for i in var_order].index(int(j_h)) + 1
            out["matched_factor_importance_rank"] = rank
        out["ref_axis_containment_allK"] = subspace_containment(RL[:, int(ref_factor)], VL)
        a, b = RL[:, int(ref_factor)], VL[:, int(j_h)]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        out["matched_loading_abs_cos"] = float(abs(a @ b) / denom) if denom > 0 else float("nan")
    return out
