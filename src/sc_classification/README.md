# `sc_classification` source package notes

## GPU logistic regression backend (new)

`sc_classification` now includes a shared logistic regression backend adapter at:

- `sc_classification.utils.logistic_backend`

It supports:

- `backend="cpu"`: sklearn (`solver="saga"`)
- `backend="gpu"`: cuML logistic regression
- `backend="auto"`: try cuML first, then fallback to sklearn

Strict GPU mode is available via `strict_gpu=True` to fail fast instead of falling back.

## Where it is used

- `scripts/dr_feature_screening/plan1c_supervised/run_plan1c_supervised_latent_benchmark.py`
  - New CLI flags:
    - `--ml-backend {cpu,gpu,auto}`
    - `--strict-gpu`
- `sc_classification.classification_methods.lr_lasso.LRLasso`
  - New init args:
    - `ml_backend="cpu"`
    - `strict_gpu=False`

## Environment for RAPIDS + FactoSig

Use:

- `env-rapids-factosig.yml`

This environment is designed to keep RAPIDS/cuML and `factosig` (PyTorch CUDA) in the same conda env and installs both repos in editable mode:

- `-e .` (this repo)
- `-e ../factosig` (sibling repo)
