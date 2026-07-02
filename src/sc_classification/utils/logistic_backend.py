from __future__ import annotations

import importlib.metadata as importlib_metadata
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


LOGGER = logging.getLogger(__name__)


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except Exception:
        return "not-installed"


def _to_numpy(x: Any) -> np.ndarray:
    """Convert numpy/cupy/pandas outputs to a host NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())
    if hasattr(x, "get"):
        # cupy arrays commonly expose .get() for host transfer.
        return np.asarray(x.get())
    return np.asarray(x)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute sklearn-compatible 'balanced' class weights per sample."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    n_samples = float(y_arr.shape[0])
    n_classes = float(classes.shape[0])
    by_class: Dict[Any, float] = {
        cls: n_samples / (n_classes * float(cnt)) for cls, cnt in zip(classes, counts)
    }
    return np.asarray([by_class[v] for v in y_arr], dtype=np.float32)


@dataclass
class LogisticRegressionAdapter:
    """Unified thin wrapper around sklearn/cuml logistic regression."""

    model: Any
    backend_used: str
    supports_predict_proba: bool = True
    use_balanced_sample_weight: bool = False
    constructor_kwargs: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionAdapter":
        fit_sig = inspect.signature(self.model.fit)
        fit_kwargs: Dict[str, Any] = {}
        if self.use_balanced_sample_weight and "sample_weight" in fit_sig.parameters:
            fit_kwargs["sample_weight"] = _balanced_sample_weights(y)
        start = time.perf_counter()
        self.model.fit(X, y, **fit_kwargs)
        LOGGER.debug(
            "LogisticRegression fit completed backend=%s n_samples=%d n_features=%d seconds=%.3f used_sample_weight=%s",
            self.backend_used,
            int(np.asarray(X).shape[0]),
            int(np.asarray(X).shape[1]) if np.asarray(X).ndim > 1 else 1,
            time.perf_counter() - start,
            bool("sample_weight" in fit_kwargs),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _to_numpy(self.model.predict(X)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.supports_predict_proba:
            proba = self.model.predict_proba(X)
            return _to_numpy(proba)
        # cuML may return only decision_function in some versions/settings.
        scores = _to_numpy(self.model.decision_function(X))
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    @property
    def coef_(self) -> np.ndarray:
        return _to_numpy(self.model.coef_)


def _build_sklearn_lr(
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    max_iter: int,
    n_jobs: int,
    class_weight: Optional[str],
) -> LogisticRegressionAdapter:
    model = SklearnLogisticRegression(
        penalty=penalty,
        solver="saga",
        C=float(C),
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        class_weight=class_weight,
        random_state=int(random_state),
        max_iter=int(max_iter),
        n_jobs=int(n_jobs),
    )
    return LogisticRegressionAdapter(
        model=model,
        backend_used="cpu",
        constructor_kwargs={
            "penalty": penalty,
            "C": float(C),
            "l1_ratio": l1_ratio if penalty == "elasticnet" else None,
            "class_weight": class_weight,
            "max_iter": int(max_iter),
            "n_jobs": int(n_jobs),
        },
    )


def _build_cuml_lr(
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    max_iter: int,
    class_weight: Optional[str],
) -> LogisticRegressionAdapter:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression

    # cuML support can vary by RAPIDS version; keep kwargs dynamic/safe.
    kwargs: Dict[str, Any] = {
        "penalty": penalty,
        "C": float(C),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "output_type": "numpy",
    }
    if class_weight is not None:
        kwargs["class_weight"] = class_weight
    if penalty == "elasticnet" and l1_ratio is not None:
        kwargs["l1_ratio"] = float(l1_ratio)

    model_sig = inspect.signature(CuMLLogisticRegression.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in model_sig.parameters}
    dropped_kwargs = sorted(set(kwargs) - set(valid_kwargs))
    if dropped_kwargs:
        LOGGER.warning("cuML LogisticRegression ignored unsupported kwargs: %s", dropped_kwargs)
    LOGGER.debug(
        "Building cuML LogisticRegression sklearn=%s cuml=%s kwargs=%s",
        package_version("scikit-learn"),
        package_version("cuml"),
        valid_kwargs,
    )
    model = CuMLLogisticRegression(**valid_kwargs)
    use_balanced_sample_weight = class_weight == "balanced" and "class_weight" not in valid_kwargs

    return LogisticRegressionAdapter(
        model=model,
        backend_used="gpu",
        use_balanced_sample_weight=use_balanced_sample_weight,
        constructor_kwargs=valid_kwargs,
    )


def make_logistic_regression(
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    max_iter: int = 5000,
    n_jobs: int = -1,
    backend: str = "cpu",
    strict_gpu: bool = False,
    class_weight: Optional[str] = "balanced",
) -> LogisticRegressionAdapter:
    """
    Create a logistic-regression model using sklearn (CPU) or cuML (GPU).

    Parameters
    ----------
    backend
        One of: "cpu", "gpu", "auto".
    strict_gpu
        If True and backend is "gpu"/"auto", do not fall back to CPU.
    """
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "auto"}:
        raise ValueError(f"Unknown backend '{backend}'. Expected one of: cpu, gpu, auto.")
    class_weight_norm = None if class_weight in {None, "", "none", "None"} else str(class_weight)

    if backend_norm == "cpu":
        return _build_sklearn_lr(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=n_jobs,
            class_weight=class_weight_norm,
        )

    try:
        return _build_cuml_lr(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight_norm,
        )
    except Exception as exc:
        if backend_norm == "gpu" or strict_gpu:
            LOGGER.exception(
                "cuML LogisticRegression backend failed with strict GPU requested. sklearn=%s cuml=%s",
                package_version("scikit-learn"),
                package_version("cuml"),
            )
            raise
        LOGGER.warning(
            "cuML LogisticRegression backend failed; falling back to sklearn CPU. sklearn=%s cuml=%s error=%r",
            package_version("scikit-learn"),
            package_version("cuml"),
            exc,
        )
        return _build_sklearn_lr(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=n_jobs,
            class_weight=class_weight_norm,
        )
