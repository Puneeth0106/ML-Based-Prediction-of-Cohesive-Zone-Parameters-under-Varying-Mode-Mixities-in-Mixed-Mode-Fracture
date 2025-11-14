# Parameter_prediction

This repository provides a  data-processing and machine-learning pipeline
for predicting cohesive-zone parameters from simulation traces. The code
extracts features from simulation CSVs, pairs them with target values, and
trains/evaluates ML models (SVR, Random Forest, simple ANN) to reproduce the
baseline experiments.


## Project overview

- Extract features (Pmax, delta_star, At, initial slope m) from simulation CSV
  files located in `data/Simulation/` 
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
  - `output_results/` — directory containing output result CSVs
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

# Parameter_prediction

This repository contains a reproducible data-processing and machine-learning
pipeline for predicting fracture-mechanics outputs (for example:
`sigmaI`, `GI`, `sigmaII`, `GII`) from simulation-derived input features
(`delta_star`, `Pmax`, `At`, `m`). It includes feature extraction, model
training and hyperparameter tuning (SVR, Random Forest, XGBoost), plotting of
predicted-vs-actual, and a small inference helper to load saved models and
produce predictions on new samples.

**Visual Workflow**

![Workflow diagram](images/workflow.png)

- **Data -> Features -> Train/Tune -> Save -> Predict & Plot**
- Example result plots are saved under `figures/` (e.g. `figures/svr_test.png`).

**Quick Start**

- **Create and activate a venv (macOS / zsh)**:

```bash
python -m venv dvenv
source dvenv/bin/activate
python -m pip install -U pip
```

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Extract features** (build a combined dataset):

```bash
python predict.py
# produces `data/features_only.csv` and `data/combined_dataset.csv`
```

- **Train models and run tuning**:

```bash
python src/model_prediction.py
# produces tuning CSVs in `data/output_results/`, saved models in `models/`, and figures in `figures/`
```

- **Run inference** against saved models:

```bash
python src/predict_from_models.py --sample-file data/Simulation/sample_1.csv --models-dir models --out data/output_results/predictions_from_models.csv
```

**Project layout (high level)**

- **`predict.py`**: Feature extraction from `data/Simulation/*.csv` and assembly of `combined_dataset.csv`.
- **`src/data_loading.py`**: Helpers to load and validate datasets.
- **`src/model_prediction.py`**: Training entrypoint — tuning functions for SVR, RF and XGB, evaluation and plotting helpers.
- **`src/predict_from_models.py`**: Loads joblib artifacts from `models/` and runs predictions on sample CSVs or rows.
- **`data/output_results/`**: Tuning CSVs and saved summaries (e.g., `svr_grid_results.csv`, `rf_trees_cv_results.csv`, `xgb_grid_results.csv`).
- **`models/`**: Saved models as joblib dictionaries containing `{'model','scaler','features','targets'}`.
- **`figures/`**: Saved PNGs of predicted vs actual comparisons.

**Detailed workflow**

1. Data preparation
   - Raw simulation traces: `data/Simulation/sample_*.csv`.
   - Targets: `data/target_features/increasing_targets.csv`.
   - Run `python predict.py` to extract features and create `data/combined_dataset.csv`.

2. Training & tuning (`src/model_prediction.py`)
   - Loads `data/combined_dataset.csv` and splits into train/test.
   - Standardizes inputs with `StandardScaler`.
   - SVR tuning: can be run against an explicit list of `(epsilon, C)` pairs or a default grid. Results written to `data/output_results/svr_grid_results.csv`.
   - RF tuning: evaluates `n_estimators` values and writes `data/output_results/rf_trees_cv_results.csv`.
   - XGBoost tuning: grid over `(n_estimators, max_depth, learning_rate)` and writes `data/output_results/xgb_grid_results.csv`.
   - Best models are fitted and saved to `models/`.

3. Evaluation & visualization
   - `evaluate_model(...)` prints per-target Train/Test R² and Test MAPE.
   - `plot_pred_vs_actual(...)` saves 2×2 predicted-vs-actual figures in `figures/`.

4. Inference
   - `src/predict_from_models.py` loads joblib model files and applies scaling + prediction to new samples; output CSV is written to `data/output_results/`.

**File examples & important outputs**

- `data/output_results/combined_dataset.csv`: features + target table used for training.
- `data/output_results/svr_grid_results.csv`: per-(epsilon,C) tuning rows.
- `data/output_results/svr_grid_combined.csv`: aggregated (epsilon,C) summary (created by analysis helper).
- `models/svr_model.joblib`: saved SVR model with scaler and metadata.
- `figures/svr_test.png`: example predicted vs actual for SVR on test data.

## Results (example run)

The repository includes example outputs from a recent training run under `figures/` and `data/output_results/`.

### Key numeric summaries

SVR aggregated (per (epsilon, C)) — `data/output_results/svr_grid_combined.csv`:

| epsilon | C   | mean_R2  | mean_MAPE_percent |
|--------:|:----|---------:|------------------:|
| 0.01    | 0.1 | -15.3216 | 11.9350           |
| 0.02    | 0.7 |  -9.6127 |  7.4810           |
| 0.03    | 1.0 |  -8.6550 |  6.8161           |
| 0.04    | 2.0 |  -5.1397 |  5.7769           |
| 0.05    | 3.0 |  -3.8156 |  5.4579           |

Random Forest tuning (`data/output_results/rf_trees_cv_results.csv`) — mean MAPE per `n_estimators` (last column). Best observed:

| n_estimators | mean_MAPE_percent |
|-------------:|------------------:|
| 260          | 2.9566            |

XGBoost tuning (`data/output_results/xgb_grid_results.csv`) — best observed:

| n_estimators | max_depth | learning_rate | mean_MAPE_percent |
|-------------:|:---------:|:-------------:|------------------:|
| 50           | 5         | 0.1           | 3.3406            |

### Figures (thumbnails)

- **SVR (test):**  
  ![SVR test](figures/svr_test.png){width=240}  
  Predicted vs actual comparison for the SVR model on the test set.

- **Random Forest (test):**  
  ![RF test](figures/rf_test.png){width=240}  
  Predicted vs actual comparison for the Random Forest model on the test set.

- **XGBoost (test):**  
  ![XGB test](figures/xgb_test.png){width=240}  
  Predicted vs actual comparison for the XGBoost model on the test set.

If you want I can further adjust thumbnail sizes, add captions per subplot, or inline small CSV excerpts. I can also commit a `results/` README with more detailed run metadata (timestamp, command used, best-params details).
**Troubleshooting**

- If you see import errors for `matplotlib` or `xgboost`, install them explicitly:

```bash
pip install matplotlib xgboost
```

- If `svr_grid_results.csv` is missing but the script expects it, either run the SVR tuner (the script can be configured to evaluate the small, explicit `(epsilon, C)` list) or allow the script to run the tuner by removing any `svr_best` override.

**Extending the project**

- Add a new `hyper_tune_<model>` function to follow the existing pattern and add it to the training pipeline.
- Add unit tests under `tests/` for feature extraction and small model smoke tests.
- Add `Makefile` or `tasks` to automate the sequence `extract -> combine -> tune -> train -> evaluate`.

---

If you want, I can now:

- **A**: Embed actual thumbnails of `figures/*.png` into this README so users see immediate visual results (I will insert inline markdown image links);
- **B**: Create a `docs/` folder with step-by-step screenshots and runnable examples for non-technical users.

Reply with `A` or `B` (or `neither`) and I will update the repo accordingly.
