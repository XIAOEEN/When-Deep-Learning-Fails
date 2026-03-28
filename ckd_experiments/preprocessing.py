"""
Preprocessing pipeline for CKD cat data.
Variable selection, normalization, missing value handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# Core CKD progression variables (clinical prior)
CORE_VARIABLES = {
    'CREA': 'BB_CREA',      # Creatinine - key kidney function marker
    'BUN': 'BB_BUN',        # Blood Urea Nitrogen - kidney function
    'UREA': 'BB_UREA',     # Urea - related to kidney
    'P': 'BB_P',            # Inorganic phosphorus - elevated in kidney disease
    'weight': 'weight',     # Weight loss in CKD
    'fSAA': 'fSAA',         # Inflammation marker
    'WBC': 'BR_白细胞数目',  # White blood cells - inflammation
    'HGB': 'BR_血红蛋白',   # Hemoglobin - can drop in CKD (anemia)
    'RBC': 'BR_红细胞数目', # Red blood cells
    'PRO': 'UR_PRO',        # Protein in urine - kidney damage marker
    'PH': 'UR_PH',          # Urine pH
}


def select_core_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Select core CKD-related variables based on clinical prior."""
    available_vars = set(df['variable'].unique())
    selected = []

    for clinical_name, var_name in CORE_VARIABLES.items():
        if var_name in available_vars:
            selected.append(var_name)
        else:
            # Try fuzzy match
            for v in available_vars:
                if clinical_name.lower() in v.lower() or v.lower() in clinical_name.lower():
                    selected.append(v)
                    break

    selected = list(set(selected))
    print(f"Selected {len(selected)} variables: {selected}")
    return df[df['variable'].isin(selected)]


def align_to_common_timepoints(df: pd.DataFrame, min_records: int = 3) -> pd.DataFrame:
    """Align all cats to common timepoints. Fill missing with forward/backward fill."""
    # Get dates present for at least 2 cats
    date_counts = df.groupby('date')['cat_id'].nunique()
    good_dates = date_counts[date_counts >= 2].index.tolist()
    df = df[df['date'].isin(good_dates)]

    # Get all cat-date combinations
    all_cats = df['cat_id'].unique()
    all_dates = sorted(df['date'].unique())

    # Create complete grid
    grid = pd.MultiIndex.from_product([all_cats, all_dates], names=['cat_id', 'date'])

    # Pivot
    pivot = df.pivot_table(
        index=['cat_id', 'date'],
        columns='variable',
        values='value',
        aggfunc='first'
    )

    # Reindex to grid
    pivot = pivot.reindex(grid)

    # Forward fill then backward fill missing values
    pivot = pivot.groupby(level='cat_id').transform(lambda x: x.ffill().bfill())

    # Drop cats with too many missing values
    missing_rate = pivot.isnull().mean()
    good_vars = missing_rate[missing_rate < 0.5].index.tolist()
    pivot = pivot[good_vars]

    # Reset index
    pivot = pivot.reset_index()

    return pivot


def normalize_variables(df: pd.DataFrame, method: str = 'zscore') -> Tuple[pd.DataFrame, Dict]:
    """Normalize variables across all cats and timepoints."""
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Get variable columns
    var_cols = [c for c in df.columns if c not in ['cat_id', 'date', 'group']]

    # Fit scaler on all non-null values
    values = df[var_cols].values
    values_nonan = values[~np.isnan(values)]
    if len(values_nonan) > 0:
        scaler.fit(values_nonan.reshape(-1, 1))

    # Transform
    df_norm = df.copy()
    for col in var_cols:
        col_vals = df[col].values.reshape(-1, 1)
        mask = ~np.isnan(col_vals.flatten())
        if mask.sum() > 0:
            df_norm.loc[mask, col] = scaler.transform(col_vals[mask]).flatten()

    # Store scaler params for inverse transform
    scaler_params = {
        'mean': scaler.mean_[0] if hasattr(scaler, 'mean_') else 0,
        'std': scaler.scale_[0] if hasattr(scaler, 'scale_') else 1
    }

    return df_norm, scaler_params


def prepare_for_modeling(df: pd.DataFrame,
                        lookback_steps: int = 5,
                        forecast_steps: int = 1) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    Prepare data for time series modeling.

    Returns:
        X: (n_samples, lookback_steps, n_variables) input sequences
        y: (n_samples, forecast_steps, n_variables) target sequences
        cat_ids: list of cat IDs for each sample
        time_ids: list of time indices for each sample
    """
    var_cols = [c for c in df.columns if c not in ['cat_id', 'date', 'group']]
    cat_ids = df['cat_id'].unique()

    X_list, y_list, sample_cats, sample_times = [], [], [], []

    for cat in cat_ids:
        cat_df = df[df['cat_id'] == cat].sort_values('date')
        values = cat_df[var_cols].values  # (T, n_vars)

        if np.isnan(values).all():
            continue

        T = len(values)

        for t in range(lookback_steps, T - forecast_steps + 1):
            x = values[t - lookback_steps:t]  # (lookback, n_vars)
            y = values[t:t + forecast_steps]   # (forecast, n_vars)

            X_list.append(x)
            y_list.append(y)
            sample_cats.append(cat)
            sample_times.append(t)

    X = np.array(X_list) if X_list else np.array([]).reshape(0, lookback_steps, len(var_cols))
    y = np.array(y_list) if y_list else np.array([]).reshape(0, forecast_steps, len(var_cols))

    return X, y, sample_cats, sample_times


def leave_one_cat_out_split(df: pd.DataFrame, test_cat: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test by leaving out one cat."""
    train = df[df['cat_id'] != test_cat]
    test = df[df['cat_id'] == test_cat]
    return train, test


def leave_one_timepoint_split(df: pd.DataFrame, test_time_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by leaving out the last timepoint for validation."""
    all_times = sorted(df['date'].unique())
    if test_time_idx >= len(all_times):
        test_time_idx = len(all_times) - 1

    test_time = all_times[test_time_idx]
    train = df[df['date'] != test_time]
    test = df[df['date'] == test_time]
    return train, test


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute variable correlations for feature selection."""
    var_cols = [c for c in df.columns if c not in ['cat_id', 'date', 'group']]
    corr = df[var_cols].corr()
    return corr


if __name__ == '__main__':
    from data_loader import load_all_data, create_unified_dataframe

    data_dir = Path('/Users/yangxiansen/Documents/CKD猫药物评价模型/Feline CKD 2')
    data = load_all_data(data_dir)
    unified = create_unified_dataframe(data)

    # Select core variables
    df_sel = select_core_variables(unified)
    print(f"After variable selection: {df_sel['variable'].nunique()} variables")

    # Align to common timepoints
    df_aligned = align_to_common_timepoints(df_sel)
    print(f"After alignment: {len(df_aligned)} rows, {df_aligned['cat_id'].nunique()} cats")
