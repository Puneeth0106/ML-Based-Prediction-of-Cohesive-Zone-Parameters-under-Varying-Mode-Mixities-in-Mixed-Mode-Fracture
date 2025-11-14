from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor



feature_names = ['delta_star', 'Pmax', 'At', 'm']

target_names = ['sigmaI', 'GI', 'sigmaII', 'GII']

# Load combined dataset (expects data/combined_dataset.csv created by data_loading.py)
df = pd.read_csv('data/output_results/combined_dataset.csv')

# Keep pandas DataFrames to preserve column indexing and ease of reporting
X = df[feature_names]
y = df[target_names]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize inputs (important for SVR & ANN). We will use scaled features for all models.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Training data shape:", X_train_s.shape, y_train.shape)
print("Test data shape:", X_test_s.shape, y_test.shape)

# 3.1 Support Vector Regression (multi-output wrapper)
def hyper_tune_svr(X_df, y_df, cv_splits=30, out_dir=Path('data'), pairs: list | None = None):
    """Tune SVR over a set of (epsilon, C) pairs and save results CSV.

    If `pairs` is None the function evaluates the full default grid.
    Returns best params dict {'C':..., 'epsilon':...} chosen by highest mean R2.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    default_epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    default_box_constants = [0.1, 0.7, 1, 2, 3]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Build list of (eps, C) pairs
    if pairs is None:
        pairs = [(e, c) for e in default_epsilons for c in default_box_constants]

    rows = []
    X_vals = X_df.values
    y_vals = y_df.values

    for eps, C in pairs:
        r2s = []
        mapes = []
        for tr_idx, te_idx in kf.split(X_vals):
            Xtr, Xte = X_vals[tr_idx], X_vals[te_idx]
            ytr, yte = y_vals[tr_idx], y_vals[te_idx]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)

            svr = SVR(C=C, epsilon=eps, kernel='rbf')
            mor = MultiOutputRegressor(svr)
            mor.fit(Xtr_s, ytr)

            ypred = mor.predict(Xte_s)
            r2s.append(r2_score(yte, ypred))
            mapes.append(mean_absolute_percentage_error(yte, ypred) * 100.0)

            rows.append({'epsilon': eps, 'C': C, 'mean_R2': float(np.mean(r2s)), 'mean_MAPE_percent': float(np.mean(mapes))})

    df_res = pd.DataFrame(rows).sort_values('mean_R2', ascending=False)
    out_csv = out_dir / 'output_results/svr_grid_results.csv'
    df_res.to_csv(out_csv, index=False)
    print('Wrote SVR grid results to', out_csv)
    best = df_res.iloc[0]
    return {'C': float(best['C']), 'epsilon': float(best['epsilon']), 'results_df': df_res}
    


# Train final SVR with best params
# If `svr_best` is not set earlier, run hyper-parameter tuning using the
# specific (epsilon, C) pairs requested by the user and assign `svr_best`.
# Define the pair list explicitly so tuning runs only the requested combinations.
svr_pairs = [(0.01, 0.1), (0.02, 0.7), (0.03, 1.0), (0.04, 2.0), (0.05, 3.0)]
svr_best = hyper_tune_svr(pd.DataFrame(X_train, columns=X.columns) if not isinstance(X_train, pd.DataFrame) else X_train,
                          y_train, cv_splits=10, out_dir=Path('data'), pairs=svr_pairs)

svr_base = SVR(kernel='rbf', C=svr_best['C'], epsilon=svr_best['epsilon'])
svr = MultiOutputRegressor(svr_base)
svr.fit(X_train_s, y_train)

def hyper_tune_rf_trees(X_df, y_df, cv_splits=10, out_dir=Path('data')):
    """Evaluate different n_estimators values and save per-fold MAPE table (per paper Table 3).

    Returns dict with best_n_estimators and results_df.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_list = [5, 10, 36, 82, 140, 260, 480, 650]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    X_vals = X_df.values
    y_vals = y_df.values
    rows = []
    for n in n_list:
        fold_mapes = []
        per = {'n_estimators': n}
        fold_idx = 1
        for tr_idx, te_idx in kf.split(X_vals):
            Xtr, Xte = X_vals[tr_idx], X_vals[te_idx]
            ytr, yte = y_vals[tr_idx], y_vals[te_idx]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)

            rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
            rf.fit(Xtr_s, ytr)
            ypred = rf.predict(Xte_s)
            m = mean_absolute_percentage_error(yte, ypred) * 100.0
            per[f'fold_{fold_idx}'] = float(m)
            fold_mapes.append(m)
            fold_idx += 1

        per['mean_MAPE_percent'] = float(np.mean(fold_mapes))
        rows.append(per)

    df_rf = pd.DataFrame(rows)
    # ensure fold columns order
    cols = ['n_estimators'] + [c for c in df_rf.columns if c.startswith('fold_')] + ['mean_MAPE_percent']
    df_rf = df_rf[cols]
    out_csv = out_dir / 'output_results/rf_trees_cv_results.csv'
    df_rf.to_csv(out_csv, index=False)
    print('Wrote RF trees CV results to', out_csv)
    best_row = df_rf.loc[df_rf['mean_MAPE_percent'].idxmin()]
    return {'best_n_estimators': int(best_row['n_estimators']), 'results_df': df_rf}


# Tune RF tree counts on training set
rf_tune = hyper_tune_rf_trees(pd.DataFrame(X_train, columns=X.columns) if not isinstance(X_train, pd.DataFrame) else X_train, y_train, cv_splits=10, out_dir=Path('data'))

# Train RF with best n_estimators
rf = RandomForestRegressor(n_estimators=rf_tune['best_n_estimators'], random_state=0, n_jobs=-1)
rf.fit(X_train_s, y_train)

# -------------------------
# XGBoost training + tuning
# -------------------------
def hyper_tune_xgb(X_df, y_df, cv_splits=5, out_dir=Path('data'), combos: list | None = None):
    """Simple grid search over (n_estimators, max_depth, learning_rate) using KFold.

    Returns best params dict and results DataFrame saved to out_dir/xgb_grid_results.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    default_n_list = [50,100,200]
    default_max_depths = [3, 5]
    default_lrs = [0.01,0.1]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # build combos list if not provided
    if combos is None:
        combos = [(n, md, lr) for n in default_n_list for md in default_max_depths for lr in default_lrs]

    rows = []
    X_vals = X_df.values
    y_vals = y_df.values

    for n, md, lr in combos:
        fold_mapes = []
        for tr_idx, te_idx in kf.split(X_vals):
            Xtr, Xte = X_vals[tr_idx], X_vals[te_idx]
            ytr, yte = y_vals[tr_idx], y_vals[te_idx]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)

            xgb = XGBRegressor(n_estimators=int(n), max_depth=int(md), learning_rate=float(lr), objective='reg:squarederror', n_jobs=-1, verbosity=0)
            mor = MultiOutputRegressor(xgb)
            mor.fit(Xtr_s, ytr)

            ypred = mor.predict(Xte_s)
            m = mean_absolute_percentage_error(yte, ypred) * 100.0
            fold_mapes.append(m)

        rows.append({'n_estimators': int(n), 'max_depth': int(md), 'learning_rate': float(lr), 'mean_MAPE_percent': float(np.mean(fold_mapes))})

    df_xgb = pd.DataFrame(rows).sort_values('mean_MAPE_percent')
    out_csv = out_dir / 'output_results/xgb_grid_results.csv'
    df_xgb.to_csv(out_csv, index=False)
    print('Wrote XGB grid results to', out_csv)
    best = df_xgb.iloc[0]
    return {'n_estimators': int(best['n_estimators']), 'max_depth': int(best['max_depth']), 'learning_rate': float(best['learning_rate']), 'results_df': df_xgb}


# Tune XGBoost on training data
xgb_best = hyper_tune_xgb(pd.DataFrame(X_train, columns=X.columns) if not isinstance(X_train, pd.DataFrame) else X_train, y_train, cv_splits=5, out_dir=Path('data'))

# Train final models (SVR, RF, XGB)
xgb_base = XGBRegressor(n_estimators=xgb_best['n_estimators'], max_depth=xgb_best['max_depth'], learning_rate=xgb_best['learning_rate'], objective='reg:squarederror', n_jobs=-1, verbosity=0)
xgb = MultiOutputRegressor(xgb_base)
xgb.fit(X_train_s, y_train)

# Save trained models and scalers (SVR, RF and XGB)
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
joblib.dump({'model': svr, 'scaler': scaler, 'features': feature_names, 'targets': target_names}, models_dir / 'svr_model.joblib')
joblib.dump({'model': rf, 'scaler': scaler, 'features': feature_names, 'targets': target_names}, models_dir / 'rf_model.joblib')
joblib.dump({'model': xgb, 'scaler': scaler, 'features': feature_names, 'targets': target_names}, models_dir / 'xgb_model.joblib')
print('Saved SVR, RF and XGB models to', models_dir)



# -------------------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------------------
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Evaluate a fitted model and print R2 and MAPE per target.

    X_tr, X_te are the scaled feature matrices (numpy arrays or DataFrames transformed).
    y_tr, y_te are pandas DataFrames with target columns.
    """
    print(f"\n=== {name} ===")
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    for i, tgt in enumerate(target_names):
        # y_tr/y_te are DataFrames; select column values
        y_tr_col = y_tr.iloc[:, i].to_numpy()
        y_te_col = y_te.iloc[:, i].to_numpy()

        r2_t = r2_score(y_tr_col, y_pred_train[:, i])
        r2_v = r2_score(y_te_col, y_pred_test[:, i])
        mape_v = mean_absolute_percentage_error(y_te_col, y_pred_test[:, i]) * 100
        print(f"{tgt:7s} | Train R²={r2_t:6.3f} | Test R²={r2_v:6.3f} | Test MAPE={mape_v:6.2f}%")

    return y_pred_test


# Evaluate all
svr_pred = evaluate_model("SVR", svr, X_train_s, X_test_s, y_train, y_test)
rf_pred = evaluate_model("Random Forest", rf, X_train_s, X_test_s, y_train, y_test)
xgb_pred = evaluate_model("XGBoost", xgb, X_train_s, X_test_s, y_train, y_test)


# -----------------------------
# Plotting: predicted vs actual
# -----------------------------
def plot_pred_vs_actual(y_true: pd.DataFrame, y_pred: np.ndarray, targets: list, title: str, out_file: Path):
    """Create a 2x2 scatter grid of predicted vs actual for each target and save the figure.

    y_true: pandas DataFrame (n_samples x n_targets)
    y_pred: numpy array (n_samples x n_targets)
    targets: list of target names
    out_file: Path to save PNG
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        # scatter
        ax.scatter(y_true.iloc[:, i], y_pred[:, i], facecolors='none', edgecolors='black', label='Data Points')

        # reference line y = x
        mn = float(min(y_true.iloc[:, i].min(), y_pred[:, i].min()))
        mx = float(max(y_true.iloc[:, i].max(), y_pred[:, i].max()))
        if mn == mx:
            mn = mn - 0.1 * abs(mn) if mn != 0 else -0.1
            mx = mx + 0.1 * abs(mx) if mx != 0 else 0.1
        pad = 0.02 * (mx - mn)
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], color='red', linewidth=2, label='Reference Line')

        ax.set_xlabel(f'Actual Output {i+1}, {targets[i]}')
        ax.set_ylabel(f'Predicted Output {i+1}, {targets[i]}')
        ax.grid(False)
        ax.legend(loc='upper left')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


# compute and save prediction plots for SVR and RF (train + test)
fig_dir = Path('figures')
fig_dir.mkdir(exist_ok=True)

# SVR predictions
svr_train_pred = svr.predict(X_train_s)
svr_test_pred = svr.predict(X_test_s)
plot_pred_vs_actual(y_train, svr_train_pred, target_names, 'SVR: Predicted vs Actual (Training)', fig_dir / 'svr_train.png')
plot_pred_vs_actual(y_test, svr_test_pred, target_names, 'SVR: Predicted vs Actual (Testing)', fig_dir / 'svr_test.png')

# RF predictions
rf_train_pred = rf.predict(X_train_s)
rf_test_pred = rf.predict(X_test_s)
plot_pred_vs_actual(y_train, rf_train_pred, target_names, 'RF: Predicted vs Actual (Training)', fig_dir / 'rf_train.png')
plot_pred_vs_actual(y_test, rf_test_pred, target_names, 'RF: Predicted vs Actual (Testing)', fig_dir / 'rf_test.png')

# XGB predictions
xgb_train_pred = xgb.predict(X_train_s)
xgb_test_pred = xgb.predict(X_test_s)
plot_pred_vs_actual(y_train, xgb_train_pred, target_names, 'XGB: Predicted vs Actual (Training)', fig_dir / 'xgb_train.png')
plot_pred_vs_actual(y_test, xgb_test_pred, target_names, 'XGB: Predicted vs Actual (Testing)', fig_dir / 'xgb_test.png')

