# Utils notes

## `logistic_backend.py`

`logistic_backend.py` provides `make_logistic_regression(...)`, a unified factory for CPU/GPU logistic regression used by benchmark and classifier code.

### Key behavior

- Returns a thin adapter with a stable interface:
  - `.fit(X, y)`
  - `.predict(X)`
  - `.predict_proba(X)`
  - `.coef_`
  - `.backend_used` (`"cpu"` or `"gpu"`)
- Applies balanced-class weighting:
  - sklearn path uses `class_weight="balanced"`
  - cuML path uses computed `sample_weight` to match the same behavior
- Supports fallback control:
  - `backend="auto"` tries cuML then falls back to sklearn
  - `strict_gpu=True` disables fallback and raises on GPU path errors
