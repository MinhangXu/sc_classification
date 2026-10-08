"""Regularization contract for Stage 2 logistic probes (AGENTS R5; analysis D19, task W3.1).

This module is the single shared implementation of the C policy. W2, W4 and W6 code import
it, so the public API below is fixed; changes go through a devlog entry with type: refactor.
Dependencies: numpy and scikit-learn only.

Conventions copied from scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py
- C grid: 1 / logspace(-4, 4, 17), i.e. C from 1e4 down to 1e-4 in half-decade steps.
- class_weight='balanced'; SimpleImputer(fill_value=0) then StandardScaler fitted on the
  training cells of each fold (population SD; a constant column keeps scale 1, so it is 0).
- Same objective. The stored Set A grids ran on the cuML backend (QN / OWL-QN, with
  l1_strength = l1_ratio / C, l2_strength = (1 - l1_ratio) / C, penalties normalised by the
  sample size). That is the scikit-learn objective, so C means the same thing on CPU and GPU.
  Here L1 / elastic net default to ProxNewtonLogistic ('newton_cd', an exact proximal-Newton
  solver of that objective; saga, the runner's CPU backend, is available but can need
  thousands of epochs on heavy-tailed standardized scores) and L2 to lbfgs.

Objective (scikit-learn parametrisation, rho = l1_ratio, rho = 1 for L1)

    min_w,b   C * sum_i s_i * logloss_i(w, b)  +  rho * ||w||_1  +  (1 - rho) / 2 * ||w||_2^2

L1 entry rule. With balanced weights s_i = n / (2 n_class) the optimal intercept at w = 0
satisfies sigmoid(b) = 1/2, and the loss gradient for a standardized feature j is
(n / 4) * (mean_normal_j - mean_malignant_j). w = 0 is optimal (all-zero probe) iff
C * |grad_j| <= rho for every j, so feature j can enter at w = 0 only if

    |standardized training class-mean gap_j|  >  4 * rho / (C * n_train).

For L1 the threshold is 4 / (C * n_train). For elastic net it is 4 * l1_ratio / (C * n_train):
a smaller l1_ratio makes entry easier, not harder. sklearn's C does not scale with n, so a
fixed C over-penalises small training sets; c_scaled_to_n keeps the threshold fixed.

An all-zero probe scores the constant-score no-skill AP = prevalence, so its lift is exactly 0
by construction. Never read it as "no signal".
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_C_GRID",
    "C_GRID_LOWER_BOUND",
    "C_GRID_UPPER_BOUND",
    "DEFAULT_MIN_MALIGNANT",
    "COEF_NONZERO_TOL",
    "standardize_train",
    "balanced_sample_weights",
    "l1_entry_threshold",
    "l1_entry_margins",
    "l1_entry_c",
    "c_scaled_to_n",
    "c_at_grid_lower_bound",
    "optimal_intercept",
    "kkt_margins",
    "supported_patients",
    "ProxNewtonLogistic",
    "SOLVERS",
    "NEWTON_CD_MAX_FEATURES",
    "fit_probe",
    "probe_coef",
    "probe_scores",
    "heldout_lift",
    "tune_c_inner_lopo",
    "c_policy_record",
]

CONTRACT_VERSION = "1.0.0"


def _runner_c_grid(log10_min: float = -4.0, log10_max: float = 4.0, n: int = 17) -> Tuple[float, ...]:
    """Exactly run_stage2_mrd_multiobjective_scorecard.c_grid_from_args (same float values)."""
    alpha_grid = np.logspace(float(log10_min), float(log10_max), int(n))
    return tuple(float(1.0 / alpha) for alpha in alpha_grid)


#: The runner's C grid, in the runner's order (1e4 first, 1e-4 last).
DEFAULT_C_GRID: Tuple[float, ...] = _runner_c_grid()
C_GRID_LOWER_BOUND: float = min(DEFAULT_C_GRID)
C_GRID_UPPER_BOUND: float = max(DEFAULT_C_GRID)
#: Support rule (research/study_config.json): >= 10 malignant cells and both classes.
DEFAULT_MIN_MALIGNANT = 10
#: The runner's --coef-nonzero-tol.
COEF_NONZERO_TOL = 1e-8
DEFAULT_MAX_ITER = 5000

_PENALTY_ALIASES = {"l1": "l1", "lasso": "l1", "l2": "l2", "ridge": "l2", "elasticnet": "elasticnet", "en": "elasticnet"}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _norm_penalty(penalty: str) -> str:
    key = str(penalty).strip().lower()
    if key not in _PENALTY_ALIASES:
        raise ValueError(f"Unknown penalty {penalty!r}; expected l1, l2 or elasticnet")
    return _PENALTY_ALIASES[key]


def _rho(penalty: str = "l1", l1_ratio: Optional[float] = None) -> float:
    """The L1 share of the penalty: 1 for L1, l1_ratio for elastic net, 0 for L2."""
    pen = _norm_penalty(penalty)
    if pen == "l1":
        return 1.0
    if pen == "l2":
        return 0.0
    if l1_ratio is None or not np.isfinite(float(l1_ratio)):
        raise ValueError("elasticnet needs a finite l1_ratio")
    rho = float(l1_ratio)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("l1_ratio must lie in [0, 1]")
    return rho


def _ratio_from_args(l1_ratio: Optional[float]) -> float:
    """l1_entry_* helpers: None (or NaN) means pure L1."""
    if l1_ratio is None:
        return 1.0
    value = float(l1_ratio)
    if not np.isfinite(value):
        return 1.0
    if not 0.0 < value <= 1.0:
        raise ValueError("l1_ratio must lie in (0, 1] for an entry threshold")
    return value


def _as_2d(Z: Any) -> np.ndarray:
    arr = np.asarray(Z, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Z must be 2-D (cells x features)")
    return arr


def _as_binary(y: Any) -> np.ndarray:
    arr = np.asarray(y).ravel()
    if arr.dtype == bool:
        return arr.astype(int)
    values = set(np.unique(arr).tolist())
    if not values <= {0, 1}:
        raise ValueError("y must be binary 0/1 (1 = malignant)")
    return arr.astype(int)


def standardize_train(Z: Any, Z_eval: Any = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The runner's fit_preprocess_transform: impute NaN -> 0, StandardScaler on training rows.

    Returns (Z_train_std, Z_eval_std or None) in float64. A zero-variance column keeps scale 1
    and becomes all zeros, as in sklearn.
    """
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(_as_2d(Z)))
    evaluated = None
    if Z_eval is not None:
        evaluated = scaler.transform(imputer.transform(_as_2d(Z_eval)))
    return np.asarray(train, dtype=np.float64), None if evaluated is None else np.asarray(evaluated, dtype=np.float64)


def balanced_sample_weights(y: Any) -> np.ndarray:
    """sklearn class_weight='balanced' per sample: n / (n_classes * n_class). Sums to n."""
    y_arr = _as_binary(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    weight = {int(c): y_arr.size / (classes.size * float(k)) for c, k in zip(classes, counts)}
    return np.asarray([weight[int(v)] for v in y_arr], dtype=np.float64)


# --------------------------------------------------------------------------------------
# entry rule
# --------------------------------------------------------------------------------------


def l1_entry_threshold(C: float, n_train: int, l1_ratio: Optional[float] = None) -> float:
    """Standardized class-mean gap a feature must exceed to enter at w = 0.

    L1 (l1_ratio None): 4 / (C * n_train). Elastic net: 4 * l1_ratio / (C * n_train).
    Valid for class_weight='balanced' with a fitted, unpenalised intercept.
    """
    C = float(C)
    n_train = int(n_train)
    if C <= 0 or n_train <= 0:
        raise ValueError("C and n_train must be positive")
    return 4.0 * _ratio_from_args(l1_ratio) / (C * n_train)


def _zero_point_gradient(Zs: np.ndarray, y: np.ndarray, class_weight: Optional[str]) -> np.ndarray:
    """|d loss / d w_j| at w = 0 with the optimal intercept (sum, not mean, over cells)."""
    if class_weight == "balanced":
        s = balanced_sample_weights(y)
    elif class_weight in (None, "none", "None"):
        s = np.ones(y.size, dtype=np.float64)
    else:
        raise ValueError("class_weight must be 'balanced' or None")
    p0 = float(np.sum(s * y) / np.sum(s))  # sigmoid(b*) at w = 0
    return np.abs((s * (p0 - y)) @ Zs)


def l1_entry_margins(
    Z: Any,
    y: Any,
    C: float,
    *,
    l1_ratio: Optional[float] = None,
    standardize: bool = True,
    class_weight: Optional[str] = "balanced",
) -> np.ndarray:
    """Per-feature entry margin at w = 0: |standardized training class-mean gap| / threshold.

    Z holds the TRAINING cells only (raw Stage 1 scores when standardize=True; already
    train-standardized otherwise). A margin above 1 means the feature can leave zero at w = 0;
    if every margin is <= 1 the fitted probe is all-zero (the KKT condition, exact up to
    solver tolerance). With several margins above 1, the largest enters first; the final
    active set also depends on the other features, so a per-feature margin is not a
    prediction of the final support.

    Computed from the loss gradient, so it also covers class_weight=None
    (threshold 1 / (C * n * pi * (1 - pi)) in gap units); for 'balanced' it equals
    |gap_j| / l1_entry_threshold(C, n, l1_ratio).
    """
    Zs = standardize_train(Z)[0] if standardize else _as_2d(Z)
    y_arr = _as_binary(y)
    if Zs.shape[0] != y_arr.size:
        raise ValueError("Z and y have different numbers of rows")
    if y_arr.min() == y_arr.max():
        raise ValueError("y needs both classes")
    rho = _ratio_from_args(l1_ratio)
    return float(C) * _zero_point_gradient(Zs, y_arr, class_weight) / rho


def l1_entry_c(
    Z: Any,
    y: Any,
    *,
    l1_ratio: Optional[float] = None,
    standardize: bool = True,
    class_weight: Optional[str] = "balanced",
) -> np.ndarray:
    """Per-feature critical C above which the feature can enter at w = 0 (inf for a zero gap).

    min(l1_entry_c(...)) is the C at which the L1 / elastic-net path first leaves all-zero.
    """
    unit = l1_entry_margins(Z, y, 1.0, l1_ratio=l1_ratio, standardize=standardize, class_weight=class_weight)
    with np.errstate(divide="ignore"):
        return np.where(unit > 0, 1.0 / unit, np.inf)


def c_scaled_to_n(C_ref: float, n_ref: int, n_train: int) -> float:
    """C that keeps the L1 entry threshold of (C_ref, n_ref) at a training size n_train.

    C_ref * n_ref / n_train. Use it when training n changes (subsets, subsampling, inductive
    folds, learning curves) and a fixed-threshold comparison is wanted; the R5 default is still
    to retune C by inner LOPO (tune_c_inner_lopo).
    """
    if int(n_train) <= 0 or int(n_ref) <= 0:
        raise ValueError("n_ref and n_train must be positive")
    return float(C_ref) * float(n_ref) / float(n_train)


def c_at_grid_lower_bound(C: float, c_grid: Sequence[float] = DEFAULT_C_GRID, rtol: float = 1e-9) -> bool:
    """True when C equals the smallest C of the grid (strongest regularization)."""
    return bool(np.isclose(float(C), float(min(c_grid)), rtol=rtol, atol=0.0))


def optimal_intercept(
    Zs: Any,
    y: Any,
    coef: Any,
    *,
    class_weight: Optional[str] = "balanced",
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Intercept that solves the (unpenalised) intercept score equation for fixed coef.

    sum_i s_i * (sigmoid(b + z_i . w) - y_i) = 0, solved by safeguarded Newton. Use it when a
    stored coefficient vector has no intercept column (coefficient_paths.csv).
    """
    Zs = _as_2d(Zs)
    y_arr = _as_binary(y)
    s = balanced_sample_weights(y_arr) if class_weight == "balanced" else np.ones(y_arr.size)
    eta = Zs @ np.asarray(coef, dtype=np.float64).ravel()
    target = float(np.sum(s * y_arr))
    lo, hi = -50.0 - float(np.max(np.abs(eta))), 50.0 + float(np.max(np.abs(eta)))
    b = 0.0
    for _ in range(int(max_iter)):
        p = 1.0 / (1.0 + np.exp(-(b + eta)))
        f = float(np.sum(s * p)) - target
        if abs(f) <= tol * max(1.0, target):
            break
        if f > 0:
            hi = b
        else:
            lo = b
        d = float(np.sum(s * p * (1.0 - p)))
        step = b - f / d if d > 0 else 0.5 * (lo + hi)
        b = step if lo < step < hi else 0.5 * (lo + hi)
    return float(b)


def kkt_margins(
    Zs: Any,
    y: Any,
    coef: Any,
    C: float,
    *,
    l1_ratio: Optional[float] = None,
    intercept: Optional[float] = None,
    class_weight: Optional[str] = "balanced",
) -> np.ndarray:
    """C * |grad_j| / rho at a given (coef, intercept) on train-standardized features Zs.

    For a converged L1 / elastic-net solution every ZERO feature has margin <= 1; a zero
    feature with margin in (0.9, 1] is one small C increase away from entering, and margin > 1
    means the stored solution is not optimal at this C (solver tolerance). For nonzero
    features the value is 1 (L1) or 1 + (1 - rho) |w_j| / rho (elastic net). At coef = 0 this
    equals l1_entry_margins(Zs, y, C, standardize=False).
    """
    Zs = _as_2d(Zs)
    y_arr = _as_binary(y)
    w = np.asarray(coef, dtype=np.float64).ravel()
    if w.size != Zs.shape[1]:
        raise ValueError("coef length does not match Z columns")
    s = balanced_sample_weights(y_arr) if class_weight == "balanced" else np.ones(y_arr.size)
    b = optimal_intercept(Zs, y_arr, w, class_weight=class_weight) if intercept is None else float(intercept)
    p = 1.0 / (1.0 + np.exp(-(b + Zs @ w)))
    grad = (s * (p - y_arr)) @ Zs
    return float(C) * np.abs(grad) / _ratio_from_args(l1_ratio)


def supported_patients(
    y: Any,
    patients: Any,
    min_malignant: int = DEFAULT_MIN_MALIGNANT,
    require_both_classes: bool = True,
) -> List[str]:
    """Patients with >= min_malignant malignant cells (and >= 1 normal cell), sorted."""
    y_arr = _as_binary(y)
    pat = np.asarray(patients).astype(str)
    out = []
    for patient in sorted(np.unique(pat)):
        mask = pat == patient
        n_mal = int(y_arr[mask].sum())
        n_norm = int(mask.sum() - n_mal)
        if n_mal >= int(min_malignant) and (n_norm > 0 or not require_both_classes):
            out.append(str(patient))
    return out


# --------------------------------------------------------------------------------------
# probe fitting and metrics
# --------------------------------------------------------------------------------------


class ProxNewtonLogistic(ClassifierMixin, BaseEstimator):
    """Binary logistic regression, scikit-learn objective, solved by proximal Newton.

        min_{w,b}  C * sum_i s_i * logloss_i  +  rho * ||w||_1  +  (1 - rho) / 2 * ||w||^2

    rho = 1 (l1), l1_ratio (elasticnet) or 0 (l2); s_i = balanced class weights; the intercept
    is not penalised. This is the objective of sklearn's saga/lbfgs and of cuML's QN with
    penalty normalisation, so the optimum is the same; only the algorithm differs. Each outer
    step solves the penalised quadratic model (exact Hessian of the weighted loss, features plus
    intercept) by cyclic coordinate descent, then backtracks on the true objective (glmnet /
    newGLMNET style). With few features (K <= ~100) it is fast and converges to a tight KKT
    point, unlike saga, whose step size collapses on heavy-tailed standardized scores.

    fit(X, y, coef_init=None, intercept_init=None) accepts a warm start. Converged when the
    C-scaled KKT violation is <= tol, or the step stalls. n_iter_ counts Newton steps.
    """

    def __init__(
        self,
        penalty: str = "l1",
        C: float = 1.0,
        l1_ratio: Optional[float] = None,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 200,
        tol: float = 1e-7,
        cd_max_sweeps: int = 2000,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.cd_max_sweeps = cd_max_sweeps

    # objective pieces -------------------------------------------------------------------
    @staticmethod
    def _loss(eta: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
        return float(np.sum(s * (np.logaddexp(0.0, eta) - y * eta)))

    def _penalty_value(self, w: np.ndarray, lam1: float, lam2: float) -> float:
        return lam1 * float(np.sum(np.abs(w))) + 0.5 * lam2 * float(w @ w)

    @staticmethod
    def _cd_subproblem(H: np.ndarray, g: np.ndarray, beta: np.ndarray, lam1: float, lam2: float,
                       max_sweeps: int, tol: float) -> np.ndarray:
        """argmin_u g.u + u'Hu/2 + lam1*|w+u_w|_1 + lam2/2*|w+u_w|^2 (last coordinate = intercept)."""
        k = beta.size - 1
        u = np.zeros_like(beta)
        Hu = np.zeros_like(beta)
        diag = np.diag(H)
        Hcols = [H[:, j].copy() for j in range(k + 1)]
        for _ in range(int(max_sweeps)):
            max_change = 0.0
            for j in range(k):
                A = diag[j] + lam2
                z0 = beta[j] + u[j]
                if A <= 1e-300:
                    znew = 0.0
                else:
                    c = A * z0 - (g[j] + Hu[j] + lam2 * z0)
                    znew = math.copysign(max(abs(c) - lam1, 0.0), c) / A
                delta = znew - z0
                if delta != 0.0:
                    u[j] += delta
                    Hu += Hcols[j] * delta
                    max_change = max(max_change, abs(delta) * math.sqrt(max(A, 0.0)))
            A = diag[k]
            if A > 1e-300:
                delta = -(g[k] + Hu[k]) / A
                u[k] += delta
                Hu += Hcols[k] * delta
                max_change = max(max_change, abs(delta) * math.sqrt(A))
            if max_change <= tol:
                break
        return u

    def _kkt_violation(self, beta: np.ndarray, g: np.ndarray, rho: float) -> float:
        """C-scaled KKT residual: max over features (subgradient) and the intercept."""
        C = float(self.C)
        w, gw = beta[:-1], C * g[:-1]
        res = np.where(
            w != 0.0,
            np.abs(gw + rho * np.sign(w) + (1.0 - rho) * w),
            np.maximum(np.abs(gw) - rho, 0.0),
        )
        return float(max(np.max(res) if res.size else 0.0, abs(C * g[-1])))

    # sklearn API ------------------------------------------------------------------------
    def fit(self, X: Any, y: Any, coef_init: Any = None, intercept_init: Optional[float] = None) -> "ProxNewtonLogistic":
        X = _as_2d(X)
        y_arr = _as_binary(y)
        if y_arr.min() == y_arr.max():
            raise ValueError("y needs both classes")
        pen = _norm_penalty(self.penalty)
        rho = _rho(pen, self.l1_ratio)
        C = float(self.C)
        if C <= 0:
            raise ValueError("C must be positive")
        lam1, lam2 = rho / C, (1.0 - rho) / C
        s = balanced_sample_weights(y_arr) if self.class_weight == "balanced" else np.ones(y_arr.size)
        n, k = X.shape
        Xa = np.empty((n, k + 1), dtype=np.float64)
        Xa[:, :k] = X
        Xa[:, k] = 1.0
        beta = np.zeros(k + 1)
        if coef_init is not None:
            beta[:k] = np.asarray(coef_init, dtype=np.float64).ravel()
        if intercept_init is not None:
            beta[k] = float(np.asarray(intercept_init).ravel()[0])
        elif coef_init is None:
            pw = float(np.sum(s * y_arr) / np.sum(s))
            beta[k] = math.log(pw / (1.0 - pw))
        eta = Xa @ beta
        F = self._loss(eta, y_arr, s) + self._penalty_value(beta[:k], lam1, lam2)
        n_iter = 0
        for n_iter in range(1, int(self.max_iter) + 1):
            p = 0.5 * (1.0 + np.tanh(0.5 * eta))
            g = Xa.T @ (s * (p - y_arr))
            if self._kkt_violation(beta, g, rho) <= float(self.tol):
                n_iter -= 1
                break
            h = s * p * (1.0 - p)
            H = (Xa * h[:, None]).T @ Xa
            u = self._cd_subproblem(H, g, beta, lam1, lam2, self.cd_max_sweeps, 1e-3 * float(self.tol))
            w_old, w_new = beta[:k], beta[:k] + u[:k]
            D = float(g @ u) + self._penalty_value(w_new, lam1, lam2) - self._penalty_value(w_old, lam1, lam2)
            if not np.isfinite(D) or D >= 0.0:
                break  # no descent direction left: converged to working precision
            t = 1.0
            accepted = False
            while t >= 1e-12:
                cand = beta + t * u
                eta_c = Xa @ cand
                F_c = self._loss(eta_c, y_arr, s) + self._penalty_value(cand[:k], lam1, lam2)
                if F_c <= F + 1e-4 * t * D:
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                break
            step = float(np.max(np.abs(t * u)))
            beta, eta, F = cand, eta_c, F_c
            if step <= 1e-12:
                break
        self.coef_ = beta[:k][None, :].copy()
        self.intercept_ = np.array([beta[k]])
        self.n_iter_ = np.array([n_iter])
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = k
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return _as_2d(X) @ self.coef_.ravel() + float(self.intercept_[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        eta = self.decision_function(X)
        p1 = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)


SOLVERS = ("newton_cd", "saga", "lbfgs", "liblinear")


#: Above this many features the default l1 / elasticnet solver switches from newton_cd (which
#: forms a (K+1) x (K+1) Hessian per step) to saga, e.g. for direct-gene panels.
NEWTON_CD_MAX_FEATURES = 200


def _default_solver(penalty: str, n_features: int = 0) -> str:
    """lbfgs for l2; newton_cd for l1 / elasticnet up to NEWTON_CD_MAX_FEATURES features, else saga."""
    if penalty == "l2":
        return "lbfgs"
    return "newton_cd" if int(n_features) <= NEWTON_CD_MAX_FEATURES else "saga"


def _make_logreg(pen: str, C: float, ratio: Optional[float], solver: str, random_state: int, max_iter: int,
                 tol: Optional[float], class_weight: Optional[str]) -> Any:
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}")
    if solver == "newton_cd":
        return ProxNewtonLogistic(
            penalty=pen, C=float(C), l1_ratio=ratio, class_weight=class_weight,
            max_iter=min(int(max_iter), 200), tol=1e-7 if tol is None else float(tol),
        )
    kwargs: Dict[str, Any] = {
        "penalty": pen, "C": float(C), "solver": solver, "class_weight": class_weight,
        "random_state": int(random_state), "max_iter": int(max_iter), "tol": 1e-4 if tol is None else float(tol),
    }
    if pen == "elasticnet":
        kwargs["l1_ratio"] = ratio
    return LogisticRegression(**kwargs)


def fit_probe(
    Z: Any,
    y: Any,
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float] = None,
    random_state: int = 42,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: Optional[str] = None,
    tol: Optional[float] = None,
    class_weight: Optional[str] = "balanced",
    standardize: bool = True,
) -> Pipeline:
    """Fit one Stage 2 probe on training cells; returns a fitted sklearn Pipeline.

    Steps: 'impute' (fill 0) and 'scale' (StandardScaler) on the training cells when
    standardize=True, then 'logreg' (class_weight='balanced'). predict_proba on raw held-out
    scores therefore applies the training-fold standardization.

    Solver (all minimise the same objective as the runner's sklearn and cuML backends):
    'newton_cd' (default for l1 / elasticnet with <= NEWTON_CD_MAX_FEATURES features;
    ProxNewtonLogistic, tol = C-scaled KKT residual, default 1e-7), 'lbfgs' (default for l2),
    'saga' (the runner's CPU backend and the default for wider l1 / elasticnet inputs such as
    direct genes; tol default 1e-4; slow on heavy-tailed scores) or 'liblinear' (l1 / l2 only;
    penalises the intercept).
    Convergence warnings are silenced; read est.named_steps['logreg'].n_iter_ if needed.
    """
    pen = _norm_penalty(penalty)
    ratio = _rho(pen, l1_ratio) if pen == "elasticnet" else None
    X = _as_2d(Z)
    model = _make_logreg(pen, C, ratio, solver or _default_solver(pen, X.shape[1]), random_state, max_iter, tol, class_weight)
    steps: List[Tuple[str, Any]] = []
    if standardize:
        steps += [("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("scale", StandardScaler())]
    steps.append(("logreg", model))
    pipe = Pipeline(steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        pipe.fit(X, _as_binary(y))
    return pipe


def probe_coef(estimator: Any) -> np.ndarray:
    """1-D coefficient vector (standardized feature units) of a fit_probe result."""
    model = estimator.named_steps["logreg"] if isinstance(estimator, Pipeline) else estimator
    return np.asarray(model.coef_, dtype=np.float64).ravel()


def probe_scores(estimator: Any, Z_eval: Any) -> np.ndarray:
    """Malignant-class probability for held-out cells."""
    return np.asarray(estimator.predict_proba(_as_2d(Z_eval))[:, 1], dtype=np.float64)


def heldout_lift(scores: Any, y: Any) -> Tuple[float, float, float]:
    """(AP, prevalence, lift = AP - prevalence) for one held-out set.

    AP is sklearn average_precision_score. A constant score has AP = prevalence, so lift 0.
    Returns NaN AP / lift when y has no malignant cell (prevalence 0) or no normal cell.
    """
    y_arr = _as_binary(y)
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size != y_arr.size:
        raise ValueError("scores and y differ in length")
    prevalence = float(y_arr.mean()) if y_arr.size else float("nan")
    if y_arr.size == 0 or y_arr.min() == y_arr.max():
        return float("nan"), prevalence, float("nan")
    ap = float(average_precision_score(y_arr, s))
    return ap, prevalence, ap - prevalence


def _fold_score(scores: np.ndarray, y: np.ndarray, scoring: str) -> Tuple[float, float, float]:
    if scoring == "average_precision":
        return heldout_lift(scores, y)
    if scoring == "roc_auc":
        if y.min() == y.max():
            return float("nan"), float(y.mean()), float("nan")
        auc = float(roc_auc_score(y, scores))
        return auc, float(y.mean()), auc - 0.5
    raise ValueError("scoring must be 'average_precision' (lift) or 'roc_auc' (AUROC - 0.5)")


# --------------------------------------------------------------------------------------
# nested C tuning
# --------------------------------------------------------------------------------------


def _inner_splits(
    y: np.ndarray,
    patients: np.ndarray,
    *,
    min_malignant: int,
    random_state: int,
    n_fallback_splits: int,
) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]], bool, str]:
    """[(label, train_idx, eval_idx)], fallback flag, reason."""
    inner = supported_patients(y, patients, min_malignant=min_malignant)
    splits: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for patient in inner:
        test = patients == patient
        train = ~test
        if np.unique(y[train]).size < 2:
            continue
        splits.append((patient, np.flatnonzero(train), np.flatnonzero(test)))
    if len(splits) >= 2:
        return splits, False, ""
    reason = f"{len(splits)} supported training patient(s) (< 2): stratified {n_fallback_splits}-fold CV"
    n_min = int(min(np.bincount(y, minlength=2)))
    n_splits = int(max(2, min(int(n_fallback_splits), n_min)))
    if n_min < 2:
        raise ValueError("fallback CV needs >= 2 cells of each class in the training set")
    strata = np.char.add(patients.astype(str), np.char.add("|", y.astype(str)))
    _, strata_counts = np.unique(strata, return_counts=True)
    if strata_counts.min() < n_splits:
        # patient x label strata too small for this many folds: stratify on label only
        strata = y.astype(str)
        reason += " (label strata)"
    else:
        reason += " (patient x label strata)"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    fallback = []
    for i, (tr, te) in enumerate(skf.split(np.zeros(y.size), strata)):
        fallback.append((f"cv{i + 1}", tr, te))
    return fallback, True, reason


def _evaluate_fold_path(
    Z: np.ndarray,
    y: np.ndarray,
    split: Tuple[str, np.ndarray, np.ndarray],
    c_grid: Sequence[float],
    *,
    penalty: str,
    l1_ratio: Optional[float],
    random_state: int,
    max_iter: int,
    solver: Optional[str],
    scoring: str,
) -> List[Dict[str, Any]]:
    """One inner fold over the whole C grid: standardize once on the inner-training cells,
    then fit C in ascending order (strong to weak regularization). newton_cd fits are
    warm-started from the previous C; other solvers refit from scratch."""
    label, tr, te = split
    Ztr, Zte = standardize_train(Z[tr], Z[te])
    ytr, yte = y[tr], y[te]
    chosen = solver or _default_solver(penalty, Z.shape[1])
    rows: List[Dict[str, Any]] = []
    coef_prev = intercept_prev = None
    for C in sorted(float(c) for c in c_grid):
        model = _make_logreg(penalty, C, l1_ratio, chosen, random_state, max_iter, None, "balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            if isinstance(model, ProxNewtonLogistic):
                model.fit(Ztr, ytr, coef_init=coef_prev, intercept_init=intercept_prev)
                coef_prev, intercept_prev = model.coef_.ravel().copy(), float(model.intercept_[0])
            else:
                model.fit(Ztr, ytr)
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        metric, prevalence, lift = _fold_score(model.predict_proba(Zte)[:, 1], yte, scoring)
        rows.append(
            {
                "C": float(C),
                "inner_fold": label,
                "n_inner_train": int(tr.size),
                "n_inner_eval": int(te.size),
                "n_inner_eval_malignant": int(yte.sum()),
                "metric": metric,
                "prevalence": prevalence,
                "lift": lift,
                "n_nonzero": int(np.sum(np.abs(coef) > COEF_NONZERO_TOL)),
            }
        )
    return rows


def tune_c_inner_lopo(
    Z: Any,
    y: Any,
    patients: Any,
    *,
    penalty: str = "l1",
    l1_ratio: Optional[float] = None,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    min_malignant: int = DEFAULT_MIN_MALIGNANT,
    scoring: str = "average_precision",
    random_state: int = 42,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: Optional[str] = None,
    n_jobs: int = 1,
    n_fallback_splits: int = 5,
    tie_atol: float = 1e-12,
) -> Dict[str, Any]:
    """Choose C by inner leave-one-patient-out over the training patients (nested; R5).

    Z, y, patients are the OUTER training cells only (raw scores; every inner fit standardizes
    on its own inner-training cells). Inner folds: each training patient meeting the support
    rule (>= min_malignant malignant cells and both classes) is held out once; the probe is fit
    on all other training cells, including unsupported patients. Criterion per C: the median
    over inner folds of lift = AP - prevalence (scoring='average_precision'), or AUROC - 0.5
    (scoring='roc_auc'). Ties (|diff| <= tie_atol) go to the sparser probe (lower mean nonzero
    count over inner folds), then the smaller C. With fewer than 2 supported training
    patients, falls back to stratified n_fallback_splits-fold CV on the training cells
    (patient x label strata when large enough, else label), same criterion.

    Returns dict: best_C, best_score, scores_by_C {C: median lift}, nonzero_by_C {C: mean
    nonzero}, per_fold (list of rows), n_inner_folds, inner_folds, fallback, fallback_reason,
    penalty, l1_ratio, criterion, tie_break, c_grid, n_train, n_train_malignant.
    Each inner fold standardizes once and walks the grid in ascending C (newton_cd warm-starts
    along the path). Deterministic: newton_cd and lbfgs are deterministic, saga is seeded by
    random_state; n_jobs (parallel over inner folds) only changes speed.
    """
    Z_arr = _as_2d(Z)
    y_arr = _as_binary(y)
    pat = np.asarray(patients).astype(str)
    if not (Z_arr.shape[0] == y_arr.size == pat.size):
        raise ValueError("Z, y and patients must have the same number of rows")
    pen = _norm_penalty(penalty)
    ratio = _rho(pen, l1_ratio) if pen == "elasticnet" else None
    grid = [float(c) for c in c_grid]
    if not grid:
        raise ValueError("empty c_grid")
    splits, fallback, reason = _inner_splits(
        y_arr, pat, min_malignant=min_malignant, random_state=random_state, n_fallback_splits=n_fallback_splits
    )
    kwargs = dict(penalty=pen, l1_ratio=ratio, random_state=random_state, max_iter=max_iter, solver=solver, scoring=scoring)
    if int(n_jobs) == 1:
        per_split = [_evaluate_fold_path(Z_arr, y_arr, split, grid, **kwargs) for split in splits]
    else:
        from joblib import Parallel, delayed  # scikit-learn dependency

        per_split = Parallel(n_jobs=int(n_jobs))(delayed(_evaluate_fold_path)(Z_arr, y_arr, split, grid, **kwargs) for split in splits)
    rows = [row for split_rows in per_split for row in split_rows]

    scores_by_C: Dict[float, float] = {}
    nonzero_by_C: Dict[float, float] = {}
    for C in grid:
        lifts = np.asarray([r["lift"] for r in rows if r["C"] == C], dtype=np.float64)
        nnz = np.asarray([r["n_nonzero"] for r in rows if r["C"] == C], dtype=np.float64)
        finite = lifts[np.isfinite(lifts)]
        scores_by_C[C] = float(np.median(finite)) if finite.size else float("nan")
        nonzero_by_C[C] = float(nnz.mean()) if nnz.size else float("nan")
    valid = [C for C in grid if np.isfinite(scores_by_C[C])]
    if not valid:
        raise ValueError("no C produced a finite inner score")
    top = max(scores_by_C[C] for C in valid)
    tied = [C for C in valid if abs(scores_by_C[C] - top) <= tie_atol]
    best_C = sorted(tied, key=lambda C: (nonzero_by_C[C], C))[0]
    for r in rows:
        r["penalty"] = pen
        r["l1_ratio"] = ratio
    return {
        "best_C": float(best_C),
        "best_score": float(scores_by_C[best_C]),
        "best_nonzero_mean": float(nonzero_by_C[best_C]),
        "scores_by_C": scores_by_C,
        "nonzero_by_C": nonzero_by_C,
        "per_fold": rows,
        "n_inner_folds": len(splits),
        "inner_folds": [s[0] for s in splits],
        "fallback": bool(fallback),
        "fallback_reason": reason,
        "penalty": pen,
        "l1_ratio": ratio,
        "criterion": "median_inner_lift" if scoring == "average_precision" else "median_inner_auroc_minus_0.5",
        "tie_break": "sparser (mean inner nonzero) then smaller C",
        "c_grid": grid,
        "at_grid_lower_bound": c_at_grid_lower_bound(best_C, grid),
        "n_train": int(y_arr.size),
        "n_train_malignant": int(y_arr.sum()),
    }


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------


def c_policy_record(**kw: Any) -> Dict[str, Any]:
    """Manifest block describing the C policy; keyword arguments override the defaults.

    Typical keys to pass: c_policy (global_selected | per_fold_tuned | nested | fixed_plus_l2 |
    not_applicable), training_n_changed, penalties, l1_ratio, notes.
    """
    record: Dict[str, Any] = {
        "contract": "scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py",
        "contract_version": CONTRACT_VERSION,
        "c_policy": "per_fold_tuned",
        "training_n_changed": False,
        "c_grid": list(DEFAULT_C_GRID),
        "c_grid_definition": "1/logspace(-4, 4, 17) (runner c_grid_from_args)",
        "c_grid_lower_bound": C_GRID_LOWER_BOUND,
        "selection": "inner leave-one-patient-out over supported training patients",
        "criterion": "median inner held-out lift (AP - prevalence)",
        "tie_break": "sparser (mean inner nonzero) then smaller C",
        "support_rule": {"min_malignant_cells": DEFAULT_MIN_MALIGNANT, "require_both_classes": True},
        "fallback": "stratified 5-fold CV on training cells (patient x label strata) when < 2 supported training patients",
        "class_weight": "balanced",
        "standardization": "SimpleImputer(0) + StandardScaler fitted on training cells of each (inner) fold",
        "solver": "newton_cd (proximal Newton) for l1/elasticnet with <= 200 features, saga above; lbfgs for l2 "
        "(all minimise the sklearn objective, which equals the cuML QN C semantics)",
        "max_iter": DEFAULT_MAX_ITER,
        "coef_nonzero_tol": COEF_NONZERO_TOL,
        "l1_entry_rule": "feature enters at w=0 iff |standardized class-mean gap| > 4*l1_ratio/(C*n_train) (l1_ratio=1 for L1)",
    }
    record.update(kw)
    return record
