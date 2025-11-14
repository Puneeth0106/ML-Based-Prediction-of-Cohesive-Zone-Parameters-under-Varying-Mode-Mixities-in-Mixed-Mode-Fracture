# Parameter Prediction from Simulation-Derived Cohesive-Zone Features

## Abstract

This repository provides a reproducible data-processing and machine-learning pipeline for predicting cohesive-zone parameters from simulation traces. The pipeline automates feature extraction from displacement–force curves, assembles training datasets, performs hyperparameter tuning for multiple regressors, persists trained artifacts, and produces diagnostic plots for model evaluation.

## Quick overview

- Feature extraction from `data/Simulation/*.csv` → `data/features_only.csv`
- Dataset assembly: `features_only.csv` + `data/target_features/increasing_targets.csv` → `data/combined_dataset.csv`
- Model tuning & training: `src/model_prediction.py` → CSV summaries in `data/output_results/`, artifacts in `models/`, plots in `figures/`
- Inference helper: `src/predict_from_models.py`

## Quickstart

1. Create and activate a Python environment (macOS / zsh):

```bash
python -m venv dvenv
source dvenv/bin/activate
python -m pip install -U pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Extract features and build dataset:

```bash
python predict.py
# writes `data/features_only.csv` and `data/combined_dataset.csv`
```

4. Train and tune models:

```bash
python src/model_prediction.py
# produces CSV results in `data/output_results/`, saved models in `models/`, and plots in `figures/`
```

5. Run inference on a sample:

```bash
python src/predict_from_models.py --sample-file data/Simulation/sample_1.csv --models-dir models --out data/output_results/predictions_from_models.csv
```

## Repository layout

- `predict.py`: feature extraction and dataset assembly
- `src/data_loading.py`: data helpers and validation
- `src/model_prediction.py`: hyperparameter tuning, training, evaluation, plotting
- `src/predict_from_models.py`: inference helper using saved joblib artifacts
- `data/`: raw inputs and outputs (`Simulation/`, `target_features/`, `output_results/`)
- `models/`: saved models and scalers
- `figures/`: saved predicted-vs-actual plots

## Methods (concise)

- Features: `Pmax`, `δ*` (displacement at peak), `At` (area under curve), `m` (initial slope). Extracted per-sample and saved to `data/features_only.csv`.
- Preprocessing: `StandardScaler` applied to inputs during tuning/training.
- Models and tuning:
  - SVR: grid over `(ε, C)`
  - Random Forest: vary `n_estimators`
  - XGBoost: grid over `(n_estimators, max_depth, learning_rate)`

Metrics: per-target R² and MAPE; grid averages saved to `data/output_results/`.

## Results (selected summaries)

SVR aggregated (selected rows from `data/output_results/svr_grid_combined.csv`):

| ε    | C   | mean_R² | mean_MAPE (%) |
|-----:|----:|--------:|--------------:|
| 0.01 | 0.1 |  −15.32 |         11.94 |
| 0.02 | 0.7 |   −9.61 |          7.48 |
| 0.03 | 1.0 |   −8.65 |          6.82 |
| 0.04 | 2.0 |   −5.14 |          5.78 |
| 0.05 | 3.0 |   −3.82 |          5.46 |

Random Forest: mean MAPE by `n_estimators` (`data/output_results/rf_trees_cv_results.csv`):

| n_estimators | mean_MAPE (%) |
|-------------:|--------------:|
| 5            | 3.7334        |
| 10           | 3.5512        |
| 36           | 3.1234        |
| 82           | 3.1412        |
| 140          | 3.0031        |
| 260          | **2.9566**    |
| 480          | 3.0069        |
| 650          | 2.9765        |

XGBoost grid (`data/output_results/xgb_grid_results.csv`):

| n_estimators | max_depth | learning_rate | mean_MAPE (%) |
|-------------:|:---------:|:-------------:|--------------:|
| 50           | 5         | 0.10          | 3.3406        |
| 100          | 5         | 0.10          | 3.5324        |
| 200          | 5         | 0.10          | 3.5533        |
| 50           | 3         | 0.10          | 3.8557        |
| 200          | 5         | 0.01          | 4.2928        |

Full CSVs in `data/output_results/` contain per-fold values and detailed metrics.

## Visuals

Predicted-vs-actual plots are saved in `figures/` (e.g. `figures/svr_test.png`, `figures/rf_test.png`, `figures/xgb_test.png`).

## Inference

Saved models are joblib dictionaries containing the model, scaler and metadata. Use `src/predict_from_models.py` (see Quickstart) to run inference; the script re-extracts features, scales inputs, and writes predictions to CSV.

## Troubleshooting

- If you see missing imports: `pip install -r requirements.txt` (add `matplotlib`, `xgboost` if needed).
- If `svr_grid_results.csv` is absent: run `python src/model_prediction.py` or re-run the SVR tuner path.

## Contributing & Next steps

- Add unit tests for `predict.py` edge cases.
- Add `requirements.txt` (if missing) and/or `pyproject.toml`.
- Optionally convert this README to a LaTeX paper or generate a short PDF abstract.

## Acknowledgements

Add advisor/group/lab acknowledgements here.

---
