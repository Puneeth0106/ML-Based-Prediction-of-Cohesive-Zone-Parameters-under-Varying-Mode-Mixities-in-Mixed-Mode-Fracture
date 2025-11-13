import pandas as pd
import numpy as np
from pathlib import Path
import re


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


target_df = pd.read_csv('increasing_targets.csv')

print(target_df.head())
print(target_df.describe())
print(target_df.info())

def _num_key(p: Path):
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else float('inf')

def _load_sim_csv(p: Path) -> pd.DataFrame:
    """Load CSV robustly and return DataFrame with columns 'displacement' and 'force' (floats)."""
    df = pd.read_csv(p, skipinitialspace=True, engine='python')
    # normalize headers
    df.columns = df.columns.str.strip().str.lower()
    # coerce to numeric and drop empty columns (trailing commas can create empty columns)
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    # prefer explicit names, else take first two numeric columns
    if 'displacement' in df.columns and 'force' in df.columns:
        df = df[['displacement', 'force']].copy()
    else:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            df = df[numeric_cols[:2]].copy()
            df.columns = ['displacement', 'force']
        else:
            raise ValueError(f"{p.name}: can't find displacement/force columns (found: {list(df.columns)})")
    # drop rows where both are NaN (commonly trailing blank lines)
    df = df.loc[~(df['displacement'].isna() & df['force'].isna())].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError(f"{p.name}: too few valid rows ({len(df)})")
    return df.astype(float)

def _compute_features_from_df(df: pd.DataFrame):
    force = df['force'].to_numpy(dtype=float)
    disp = df['displacement'].to_numpy(dtype=float)
    Pmax = float(force.max())
    idx = int(force.argmax())
    delta_star = float(disp[idx])
    # area under curve using trapezoidal rule (use numpy.trapezoid to avoid deprecation)
    At = float(np.trapezoid(force, disp))
    n = max(3, int(0.1 * len(df)))   # first 10% (min 3)
    m_slope, _ = np.polyfit(disp[:n], force[:n], 1)
    return {'Pmax': Pmax, 'delta_star': delta_star, 'At': At, 'm': float(m_slope)}

def extract_features_from_sim_folder(sim_folder: str | Path, pattern: str = 'sample_*.csv') -> pd.DataFrame:
    """
    Loop over simulation CSVs and return a DataFrame of derived features.

    Returns columns: ['file', 'Pmax', 'delta_star', 'At', 'm'] in numeric filename order.
    """
    sim_folder = Path(sim_folder)
    files = list(sim_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {sim_folder}")
    files = sorted(files, key=_num_key)

    rows = []
    for p in files:
        try:
            df = _load_sim_csv(p)
            feats = _compute_features_from_df(df)
            feats['file'] = p.name
            rows.append(feats)
        except Exception as e:
            # If you prefer to skip bad files instead of raising, change this to `continue`
            raise

    out = pd.DataFrame(rows)
    # put file first
    out = out[['file', 'Pmax', 'delta_star', 'At', 'm']]
    return out

# Minimal usage example:
if __name__ == '__main__':
    df_input_features = extract_features_from_sim_folder('Simulation')
    print(df_input_features.head())
    df_input_features.to_csv('data/features_only.csv', index=False)


    
df_combined = pd.concat([df_input_features, target_df], axis=1)

print(df_combined.isna().sum())

df_combined.dropna(inplace=True)

df_combined.to_csv('data/combined_dataset.csv', index=False)


print(df_combined)
#print(df_combined.tail(15))