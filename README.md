# Parameter_prediction

This repository provides a  data-processing and machine-learning pipeline
for predicting cohesive-zone parameters from simulation traces. The code
extracts features from simulation CSVs, pairs them with target values, and
trains/evaluates ML models (SVR, Random Forest, simple ANN) to reproduce the
baseline experiments.

This README explains the project layout, how to reproduce the main steps,
and where outputs are written.

## Project overview

- Extract features (Pmax, delta_star, At, initial slope m) from simulation CSV
  files located in `data/Simulation/` or `Simulation/`.
- Combine extracted features with target variables (stored in
  `data/target_features/increasing_targets.csv`).
- Run hyperparameter grids / CV for SVR and Random Forest, train final models,
  and evaluate with R² and MAPE.

Key scripts

- `predict.py` — feature extraction, builds `data/features_only.csv` and
  `data/combined_dataset.csv`. Also demonstrates minimal end-to-end flow.
- `src/data_loading.py` — (earlier project version) dataset assembly helper.
- `src/model_prediction.py` — model tuning, training and evaluation (SVR and RF
  implementations, saving to `models/`).

## Repository layout

Important files and folders:

- `predict.py` — main feature extraction + combine script (entry point).
- `src/` — supporting modules: `data_loading.py`, `model_prediction.py`.
- `data/` — inputs and outputs
  - `Simulation/` — many `sample_*.csv` files (simulation traces)
  - `target_features/` — `increasing_targets.csv` (targets per sample)
  - `features_only.csv` — extracted features (created by `predict.py`)
  - `combined_dataset.csv` — features + targets (created by `predict.py`)
  - `svr_grid_results.csv`, `rf_trees_cv_results.csv` — tuning outputs
  - `output_results/` — optional directory containing past result CSVs
- `models/` — (created by `src/model_prediction.py`) saved model files

## Quickstart

1) Create and activate a Python environment (example for macOS / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

2) Install minimal dependencies (these match the project's imports):

```bash
pip install pandas numpy scikit-learn joblib openpyxl
```

3) Extract features from simulation CSVs and build the combined dataset:

```bash
# from project root
python predict.py

# This writes `data/features_only.csv` and `data/combined_dataset.csv`
```

4) Run model tuning / training (example using the script in `src`):

```bash
python src/model_prediction.py

# This will produce tuning CSVs in `data/` and save models into `models/`
```

## Inputs and outputs

Inputs
- Simulation traces: `Simulation/sample_*.csv` (each should contain
  displacement & force columns)
- Targets: `data/target_features/increasing_targets.csv`

Outputs
- `data/features_only.csv` — extracted per-sample features
- `data/combined_dataset.csv` — features joined with targets (ready for ML)
- `data/svr_grid_results.csv`, `data/rf_trees_cv_results.csv` — tuning results
- `models/` — saved trained models and scalers (`*.joblib`)

## Notes, assumptions and tips

- The feature extraction in `predict.py` attempts to be robust to slightly
  different CSV column names; it looks for numeric columns and prefers names
  `displacement`/`force` when available.
- If files are malformed or missing required columns the current code raises
  an error; you can catch and continue if you prefer to skip bad files.
- The ML scripts standardize features before training — this is required for
  SVR and recommended for ANN and RF here.
- For research-grade experiments, increase cross-validation folds, use nested
  CV for hyperparameter selection, and log random seeds and package versions
  for reproducibility.

## How I validated this README

- I inspected `predict.py` and `src/model_prediction.py` to list the expected
  inputs/outputs and commands.

## Next steps (suggested)

- Add a `requirements.txt` (pip freeze) or `pyproject.toml` for reproducible
  installs.
- Add a small driver script or `Makefile` to automate the sequence
  (extract -> combine -> tune -> train -> evaluate).
- Add unit tests for feature extraction edge cases (missing columns, short
  traces) under a `tests/` folder.

## Contact / author

Describe author or contact info here (optional).

---

If you'd like, I can also:

- create a `requirements.txt` from the environment
- add a small `Makefile` or runner script to reproduce the full pipeline
- add a short usage example that runs end-to-end on one sample

Tell me which of those you'd prefer and I'll add it.
