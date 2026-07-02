# Plan1c K40 Analysis Questions and Hypotheses

This note captures the current pooled-analysis framing before per-patient extension.

## Core priorities

- Primary classification priority: maximize malignant-cell capture (high malignant recall).
- Secondary priorities: keep strong AUC and AP while reducing selected factor count.
- Practical tolerance: some false positives (normal -> cancer) are acceptable, but still monitored.

## Why track both AUC and AP

- AUC evaluates rank separation across all thresholds (global discrimination).
- AP emphasizes positive-class quality (cancer precision-recall behavior).
- For this project, model selection should not rely on only one metric.

## Pooled execution order

1. Build pooled selection table (DR x regularizer x downsampling).
2. Build downsampling effect report (`none` vs `random`) on selected points.
3. Build predictive signatures with `S = L @ w`.
4. Compare selected models via signature similarity and overlap analyses.

## Data-driven questions with biological interpretation

- Sparsity-performance frontier shape:
  - Flat frontier: compact and redundant malignant signal.
  - Sharp drop: distributed signal requiring more latent factors.
- Regularizer behavior:
  - L1 strongest screening baseline.
  - L2 dense baseline for diffuse signal.
  - Elastic net indicates correlated predictive factors when it dominates at matched sparsity.
- Downsampling sensitivity:
  - If selected signatures and metrics are stable between `none` and `random`, signal is robust to donor balancing.
  - If unstable, donor composition may drive model behavior.

## Cross-DR comparison principle

- Do not compare factor IDs directly across DR methods.
- Compare in gene space using predictive signatures `S = L @ w`.
- Optional interpretability layer: factor-level matching can be used later, but model-level signature comparison is primary.

## Planned next step

- Apply the same framework per patient after pooled framework is validated.
