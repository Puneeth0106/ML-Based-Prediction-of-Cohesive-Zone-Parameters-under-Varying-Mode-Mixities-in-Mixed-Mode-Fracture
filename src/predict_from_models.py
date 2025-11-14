"""Load saved models from `models/` and run predictions on sample inputs.

Usage examples:
  # Predict using first row from combined dataset
  python src/predict_from_models.py

  # Provide a single-sample CSV with the feature columns (header) and predict
  python src/predict_from_models.py --sample-file my_sample.csv

The script loads any `*_model.joblib` files in `models/` and uses their saved
`scaler`, `features`, and `targets` metadata (if present) to prepare inputs.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any


def load_saved_model(path: Path) -> Dict[str, Any]:
    d = joblib.load(path)
    # Expect dict-like with keys: model, scaler (optional), features, targets
    model = d.get('model') if isinstance(d, dict) else d
    scaler = d.get('scaler') if isinstance(d, dict) else None
    features = d.get('features') if isinstance(d, dict) else None
    targets = d.get('targets') if isinstance(d, dict) else None
    return {'model': model, 'scaler': scaler, 'features': features, 'targets': targets}


def prepare_sample_row(sample_df: pd.DataFrame, features: list) -> np.ndarray:
    """Return a 2D numpy array with one or more rows matching `features` order.

    sample_df: DataFrame that contains feature columns (may contain extra cols).
    features: list of feature names in the expected order.
    """
    missing = [f for f in features if f not in sample_df.columns]
    if missing:
        raise ValueError(f"Sample is missing required features: {missing}")
    row = sample_df.loc[:, features]
    return row.to_numpy(dtype=float)


def predict_with_saved(model_dict: Dict[str, Any], sample_array: np.ndarray) -> pd.DataFrame:
    """Scale sample_array if scaler present, run model.predict and return DataFrame of predictions."""
    model = model_dict['model']
    scaler = model_dict.get('scaler')
    targets = model_dict.get('targets')

    X = sample_array
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # If scaler expects DataFrame or 1D, try reshape
            X = scaler.transform(np.asarray(X))

    preds = model.predict(X)
    # preds shape: (n_samples, n_targets) or (n_samples,) for single-output
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if targets is None:
        # generate generic names
        targets = [f'target_{i+1}' for i in range(preds.shape[1])]

    dfp = pd.DataFrame(preds, columns=targets)
    return dfp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-file', '-s', help='CSV file containing sample rows (header with feature names). If omitted the script will use the first row from data/output_results/combined_dataset.csv')
    parser.add_argument('--models-dir', '-m', default='models', help='Directory containing saved joblib models')
    parser.add_argument('--out', '-o', default='data/output_results/predictions_from_models.csv', help='CSV to append/write predictions')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise SystemExit(f"Models directory not found: {models_dir}")

    # Load sample input
    if args.sample_file:
        sample_df = pd.read_csv(args.sample_file)
    else:
        combined = Path('data/output_results/combined_dataset.csv')
        if not combined.exists():
            raise SystemExit("No sample provided and data/output_results/combined_dataset.csv not found")
        sample_df = pd.read_csv(combined)
        # drop the 'file' column if present
        if 'file' in sample_df.columns:
            sample_df = sample_df.drop(columns=['file'])

    # Iterate over models
    results = []
    for p in sorted(models_dir.glob('*_model.joblib')):
        print('Loading', p)
        md = load_saved_model(p)
        features = md.get('features')
        if features is None:
            # try to infer features from sample_df columns
            features = [c for c in sample_df.columns if c in ['delta_star', 'Pmax', 'At', 'm']]
            print('No features metadata in model; using', features)

        try:
            X = prepare_sample_row(sample_df, features)
        except ValueError as e:
            print(f"Skipping {p.name}: {e}")
            continue

        pred_df = predict_with_saved(md, X)

        # attach metadata columns for traceability
        pred_df.insert(0, 'model_file', p.name)
        # if sample_df had a 'file' column earlier we dropped it; try to use index
        if 'file' in pd.read_csv('data/output_results/combined_dataset.csv').columns:
            files = pd.read_csv('data/output_results/combined_dataset.csv')['file'].tolist()
            pred_df.insert(1, 'sample_file', files[: len(pred_df)])
        else:
            pred_df.insert(1, 'sample_idx', list(range(len(pred_df))))

        results.append(pred_df)
        print(f"Predictions using {p.name}:\n", pred_df.head())

    if results:
        out_df = pd.concat(results, ignore_index=True)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print('Wrote predictions to', out_path)
    else:
        print('No predictions were made (no compatible models or samples)')


if __name__ == '__main__':
    main()
