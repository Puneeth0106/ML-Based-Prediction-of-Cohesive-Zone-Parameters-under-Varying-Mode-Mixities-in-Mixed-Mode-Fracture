"""Run saved models on test set and compare predictions with actual targets.

Creates `data/output_results/predictions_comparison.csv` containing actual
target values and predictions from SVR, Random Forest and XGBoost.
"""
from pathlib import Path
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'output_results'
MODELS = ROOT / 'models'
OUT = DATA / 'predictions_comparison.csv'

feature_names = ['delta_star', 'Pmax', 'At', 'm']
target_names = ['sigmaI', 'GI', 'sigmaII', 'GII']


def load_models():
    models = {}
    for name in ('svr', 'rf', 'xgb'):
        path = MODELS / f"{name}_model.joblib"
        if not path.exists():
            print(f"Model file not found: {path} -- skipping {name}")
            continue
        try:
            art = joblib.load(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        model = art.get('model') if isinstance(art, dict) else art
        scaler = art.get('scaler') if isinstance(art, dict) else None
        models[name] = {'model': model, 'scaler': scaler}

    return models


def main():
    # Prefer using saved test split produced during training. Fall back to recreating split.
    x_test_file = DATA / 'X_test.csv'
    y_test_file = DATA / 'y_test.csv'

    if x_test_file.exists() and y_test_file.exists():
        X_test = pd.read_csv(x_test_file)
        y_test = pd.read_csv(y_test_file)
        print(f"Loaded X_test/y_test from {x_test_file} and {y_test_file}")
    else:
        csv = DATA / 'combined_dataset.csv'
        if not csv.exists():
            print(f"Dataset not found at {csv}. Make sure you've built the dataset or run training first.")
            sys.exit(1)

        df = pd.read_csv(csv)
        X = df[feature_names]
        y = df[target_names]

        # recreate same split used during training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # save the split for future reuse
        DATA.mkdir(parents=True, exist_ok=True)
        X_test.to_csv(x_test_file, index=False)
        y_test.to_csv(y_test_file, index=False)
        print(f"Saved X_test/y_test to {x_test_file} and {y_test_file}")

    models = load_models()
    if not models:
        print("No models found. Abort.")
        sys.exit(1)

    # base output DataFrame starts with actual targets
    out_df = y_test.reset_index(drop=True).copy()

    for name, art in models.items():
        model = art['model']
        scaler = art.get('scaler')

        # scale features if scaler available, otherwise use raw
        if scaler is not None:
            X_test_s = scaler.transform(X_test)
        else:
            X_test_s = X_test.values

        try:
            preds = model.predict(X_test_s)
        except Exception as e:
            print(f"Prediction failed for {name}: {e}")
            continue

        # preds is (n_samples, n_targets)
        pred_df = pd.DataFrame(preds, columns=[f"{c}_{name}_pred" for c in target_names])
        out_df = pd.concat([out_df, pred_df], axis=1)

        # Save per-model predictions alongside actuals
        model_out_dir = DATA / 'prediction_results'
        model_out_dir.mkdir(parents=True, exist_ok=True)
        per_model_df = pd.concat([y_test.reset_index(drop=True).copy(), pred_df], axis=1)
        per_model_file = model_out_dir / f"{name}_predictions.csv"
        per_model_df.to_csv(per_model_file, index=False)
        print(f"Wrote per-model predictions for {name} to {per_model_file}")

    # Save comparison CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"Wrote predictions comparison to {OUT}")
    print(out_df.head())


if __name__ == '__main__':
    main()
