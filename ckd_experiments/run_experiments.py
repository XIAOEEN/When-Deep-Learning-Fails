"""
Main experiment runner for CKD Cat Treatment Effect Estimation.
Runs all experiments: LOSO-CV, leave-one-timepoint, simulation study, ablation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data, create_unified_dataframe
from preprocessing import select_core_variables, align_to_common_timepoints
from simulation import SyntheticCKDGenerator, run_simulation_study
from baseline_models import (
    LinearITSA, GaussianProcessModel, ExponentialCurveModel,
    SimpleLSTMModel, TreatmentEffectEstimator, evaluate_predictions
)
from ssm_model import SSMTreatmentEffectModel, S4ModelWrapper, SSMConfig


RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_real_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Load and prepare real CKD cat data.

    Returns:
        X: (n_samples, lookback, n_vars)
        y: (n_samples, 1, n_vars) - predict 1 step ahead
        groups: (n_samples,) - 1=treatment, 0=control
        sample_info: list of dicts with cat_id and time info
    """
    data_dir = Path(__file__).parent / 'data'

    print("Loading real data...")
    try:
        data = load_all_data(data_dir)
    except Exception as e:
        print(f"Could not load data from {data_dir}: {e}")
        print("Please ensure your data files are in the data/ directory.")
        print("Expected files: 血常规数据记录.xlsx, 血生化数据记录.xlsx, etc.")
        raise

    unified = create_unified_dataframe(data)

    # Select core CKD variables
    df_sel = select_core_variables(unified)

    # Key CKD progression variables (based on clinical relevance)
    key_vars = ['BB_CREA', 'BB_BUN', 'BB_UREA', 'weight', 'fSAA']
    df_key = df_sel[df_sel['variable'].isin(key_vars)]

    # Align to common timepoints
    df_aligned = align_to_common_timepoints(df_key, min_records=3)

    if df_aligned.empty:
        raise ValueError("No aligned data available")

    # Add group info (confirmed: 3 treatment, 6 control)
    TREATMENT_IDS = ['9711', '2793', '6424']
    df_aligned['group'] = df_aligned['cat_id'].apply(
        lambda x: 'treatment' if x in TREATMENT_IDS else 'control'
    )

    # Prepare sequences
    var_cols = [c for c in df_aligned.columns
                if c not in ['cat_id', 'date', 'group']]

    cats = df_aligned['cat_id'].unique()
    lookback = 7  # Sensitivity analysis showed lookback=7 gives best performance

    X_list, y_list, groups_list, sample_info = [], [], [], []

    for cat in cats:
        cat_df = df_aligned[df_aligned['cat_id'] == cat].sort_values('date')
        vals = cat_df[var_cols].values

        if len(vals) < lookback + 1:
            continue

        group = 1 if cat_df['group'].values[0] == 'treatment' else 0

        for t in range(lookback, len(vals)):
            X_list.append(vals[t-lookback:t])
            y_list.append(vals[t:t+1])
            groups_list.append(group)
            sample_info.append({
                'cat_id': cat,
                'group': 'treatment' if group == 1 else 'control',
                'time': t
            })

    X = np.array(X_list) if X_list else np.zeros((0, lookback, len(var_cols)))
    y = np.array(y_list) if y_list else np.zeros((0, 1, len(var_cols)))
    groups = np.array(groups_list)

    print(f"Prepared data: X={X.shape}, y={y.shape}, groups={groups.shape}")
    print(f"  Treatment samples: {groups.sum()}, Control samples: {len(groups) - groups.sum()}")
    print(f"  Variables: {var_cols}")

    return X, y, groups, sample_info, var_cols


def run_loso_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                sample_info: List) -> Dict:
    """
    Leave-One-Subject-Out Cross-Validation.

    For each cat, train on all other cats and predict.
    This is the gold standard for small-N validation.
    """
    print("\n=== Leave-One-Subject-Out CV ===")

    results = []
    unique_cats = list(set(s['cat_id'] for s in sample_info))

    models_to_test = {
        'LinearITSA': LinearITSA,
        'GaussianProcess': GaussianProcessModel,
        'ExpCurve': ExponentialCurveModel,
        'SSM_Treatment': lambda: SSMTreatmentEffectModel(
            config=SSMConfig(hidden_size=32, n_layers=1, epochs=50)
        ),
        'S4_Treatment': lambda: S4ModelWrapper(d_model=32, n_layers=1),
    }

    cat_results = {cat: {} for cat in unique_cats}

    for cat_to_exclude in unique_cats:
        # Split
        train_mask = np.array([s['cat_id'] != cat_to_exclude for s in sample_info])
        test_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_train = groups[train_mask]
        groups_test = groups[test_mask]

        for model_name, model_fn in models_to_test.items():
            try:
                model = model_fn()
                model.fit(X_train, y_train, groups_train)

                if not model.is_fitted:
                    continue

                pred_result = model.predict(X_test, groups_test)
                preds = pred_result.predictions

                metrics = evaluate_predictions(y_test, preds)
                cat_results[cat_to_exclude][model_name] = metrics

                results.append({
                    'model': model_name,
                    'excluded_cat': cat_to_exclude,
                    **metrics
                })

            except Exception as e:
                print(f"  Error with {model_name} on cat {cat_to_exclude}: {e}")

    # Aggregate
    summary = {}
    for model_name in models_to_test.keys():
        model_results = [r for r in results if r['model'] == model_name]
        if model_results:
            summary[model_name] = {
                'mae': np.mean([r['mae'] for r in model_results]),
                'rmse': np.mean([r['rmse'] for r in model_results]),
                'r2': np.mean([r['r2'] for r in model_results]),
                'n_samples': len(model_results)
            }

    print("\nLOSO-CV Results (MAE, lower is better):")
    for model_name, metrics in sorted(summary.items(), key=lambda x: x[1]['mae']):
        print(f"  {model_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    return {'cat_results': cat_results, 'summary': summary, 'detailed': results}


def run_leave_one_timepoint(X: np.ndarray, y: np.ndarray,
                             groups: np.ndarray,
                             sample_info: List) -> Dict:
    """
    Leave-One-Timepoint validation.
    Predict the last timepoint using all previous data.
    Tests model's ability to predict future CKD state.
    """
    print("\n=== Leave-One-Timepoint Validation ===")

    unique_times = sorted(set(s['time'] for s in sample_info))
    last_time = max(unique_times)

    # Train on all except last timepoint
    train_mask = np.array([s['time'] < last_time for s in sample_info])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_train, groups_test = groups[train_mask], groups[test_mask]

    models_to_test = {
        'LinearITSA': LinearITSA,
        'GaussianProcess': GaussianProcessModel,
        'ExpCurve': ExponentialCurveModel,
        'SSM_Treatment': lambda: SSMTreatmentEffectModel(
            config=SSMConfig(hidden_size=32, n_layers=1, epochs=50)
        ),
    }

    results = {}
    for model_name, model_fn in models_to_test.items():
        try:
            model = model_fn()
            model.fit(X_train, y_train, groups_train)

            if not model.is_fitted:
                continue

            pred_result = model.predict(X_test, groups_test)
            preds = pred_result.predictions

            metrics = evaluate_predictions(y_test, preds)
            results[model_name] = metrics
            print(f"  {model_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

        except Exception as e:
            print(f"  Error with {model_name}: {e}")

    return results


def run_simulation_recovery(n_simulations: int = 100) -> Dict:
    """
    Run simulation study to test ATE recovery.
    Generate synthetic CKD data with known ATE and test if we can recover it.
    """
    print(f"\n=== Simulation Study (n={n_simulations}) ===")

    # Generate multiple synthetic datasets
    generator = SyntheticCKDGenerator()
    datasets = generator.generate(seed=42, n_datasets=n_simulations, known_ate=0.015)

    ate_errors = []
    ate_recovery_rates = []

    for ds_idx, data in enumerate(datasets):
        df = data['df']

        # The synthetic data already has creatinine as a column (not pivoted)
        # Simple ATE estimator: compare final creatinine levels
        treat_final = df[df['cat_id'].str.contains('t')].groupby('cat_id')['creatinine'].last()
        control_final = df[df['cat_id'].str.contains('c')].groupby('cat_id')['creatinine'].last()

        est_ate = treat_final.mean() - control_final.mean()
        true_ate = data['true_ate']

        ate_error = abs(est_ate - true_ate)
        ate_errors.append(ate_error)
        ate_recovery_rates.append(1 if ate_error < 0.005 else 0)

    print(f"ATE Recovery Results:")
    print(f"  Mean absolute error: {np.mean(ate_errors):.4f}")
    print(f"  Median absolute error: {np.median(ate_errors):.4f}")
    print(f"  Recovery rate (<0.005 error): {np.mean(ate_recovery_rates):.1%}")
    print(f"  Recovery rate (<0.01 error): {np.mean(np.array(ate_errors) < 0.01):.1%}")

    # Also test with SSM model on a few synthetic datasets
    print("\nTesting SSM on synthetic data...")
    synth_errors = []
    for i in range(min(10, len(datasets))):
        data = datasets[i]
        df = data['df']

        df['group'] = df['cat_id'].apply(lambda x: 1 if 't' in x else 0)

        # Build sequences
        cats = df['cat_id'].unique()
        lookback = 5
        X_list, y_list, groups_list = [], [], []

        for cat in cats:
            cat_df = df[df['cat_id'] == cat].sort_values('time')
            vals = cat_df[['creatinine', 'bun', 'weight']].fillna(0).values
            group = cat_df['group'].values[0]

            for t in range(lookback, len(vals)):
                X_list.append(vals[t-lookback:t])
                y_list.append(vals[t:t+1])
                groups_list.append(group)

        if len(X_list) < 5:
            continue

        X_syn = np.array(X_list)
        y_syn = np.array(y_list)
        g_syn = np.array(groups_list)

        # Train/test split
        split = len(X_syn) // 2
        X_tr, y_tr, g_tr = X_syn[:split], y_syn[:split], g_syn[:split]
        X_te, y_te, g_te = X_syn[split:], y_syn[split:], g_syn[split:]

        model = SSMTreatmentEffectModel(config=SSMConfig(hidden_size=16, n_layers=1, epochs=30))
        model.fit(X_tr, y_tr, g_tr)

        if model.is_fitted:
            result = model.estimate_treatment_effects(X_te, g_te)
            err = abs(result['ate'] - data['true_ate'])
            synth_errors.append(err)

    if synth_errors:
        print(f"  SSM ATE errors on synthetic data: mean={np.mean(synth_errors):.4f}, median={np.median(synth_errors):.4f}")

    return {
        'mean_ate_error': float(np.mean(ate_errors)),
        'median_ate_error': float(np.median(ate_errors)),
        'recovery_rate_005': float(np.mean(np.array(ate_errors) < 0.005)),
        'recovery_rate_010': float(np.mean(np.array(ate_errors) < 0.01)),
        'ssm_mean_error': float(np.mean(synth_errors)) if synth_errors else None
    }


def run_treatment_effect_estimation(X: np.ndarray, y: np.ndarray,
                                    groups: np.ndarray,
                                    sample_info: List) -> Dict:
    """
    Estimate treatment effects on real data using all models.
    """
    print("\n=== Treatment Effect Estimation ===")

    results = {}

    # Method 1: Simple mean difference
    treat_mask = groups == 1
    control_mask = groups == 0

    y_treat = y[treat_mask, 0, :].mean(axis=0)
    y_control = y[control_mask, 0, :].mean(axis=0)
    simple_ate = y_treat - y_control

    results['simple_mean_diff'] = {
        'ate': simple_ate.tolist(),
        'method': 'mean difference at final timepoint'
    }
    print(f"  Simple ATE (creatinine): {simple_ate[0]:.4f}")

    # Method 2: GP-based
    try:
        gp = GaussianProcessModel()
        gp.fit(X, y, groups)
        if gp.is_fitted:
            pred_treat = gp.predict(X[treat_mask])
            pred_control = gp.predict(X[control_mask])
            gp_ate = pred_treat.predictions.mean(axis=0) - pred_control.predictions.mean(axis=0)
            results['gaussian_process'] = {
                'ate': gp_ate.tolist(),
                'method': 'GP regression'
            }
            print(f"  GP ATE (creatinine): {float(gp_ate[0][0]):.4f}")
    except Exception as e:
        print(f"  GP failed: {e}")

    # Method 3: SSM-based
    try:
        ssm = SSMTreatmentEffectModel(config=SSMConfig(hidden_size=32, n_layers=1, epochs=50))
        ssm.fit(X, y, groups)
        if ssm.is_fitted:
            ssm_result = ssm.estimate_treatment_effects(X, groups)
            results['ssm_mamba'] = {
                'ate': float(ssm_result['ate']),
                'ite': ssm_result['ite'],
                'method': 'SSM Mamba treatment effect'
            }
            print(f"  SSM ATE (creatinine): {ssm_result['ate']:.4f}")
    except Exception as e:
        print(f"  SSM failed: {e}")

    return results


def run_bootstrap_ate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      n_bootstrap: int = 1000) -> Dict:
    """
    Bootstrap confidence intervals for ATE estimation.
    """
    print(f"\n=== Bootstrap ATE ({n_bootstrap} iterations) ===")

    treat_mask = groups == 1
    control_mask = groups == 0

    y_treat = y[treat_mask, 0, :]
    y_control = y[control_mask, 0, :]

    # Simple mean difference bootstrap
    ate_samples = []
    for _ in range(n_bootstrap):
        # Bootstrap within groups
        idx_treat = np.random.choice(len(y_treat), len(y_treat), replace=True)
        idx_control = np.random.choice(len(y_control), len(y_control), replace=True)
        ate = y_treat[idx_treat].mean(axis=0) - y_control[idx_control].mean(axis=0)
        ate_samples.append(ate[0])  # creatinine

    ate_samples = np.array(ate_samples)
    ate_mean = ate_samples.mean()
    ate_std = ate_samples.std()
    ci_lower = np.percentile(ate_samples, 2.5)
    ci_upper = np.percentile(ate_samples, 97.5)

    print(f"  Simple ATE: {ate_mean:.4f} ± {ate_std:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Significant (CI excludes 0): {ci_lower > 0 or ci_upper < 0}")

    return {
        'ate_mean': float(ate_mean),
        'ate_std': float(ate_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'significant': bool(ci_lower > 0 or ci_upper < 0),
        'n_bootstrap': n_bootstrap
    }


def run_sensitivity_analysis() -> Dict:
    """
    Sensitivity analysis: test different lookback windows and variable subsets.
    """
    print("\n=== Sensitivity Analysis ===")

    from data_loader import load_all_data, create_unified_dataframe
    from preprocessing import select_core_variables, align_to_common_timepoints

    data_dir = Path(__file__).parent / 'data'
    try:
        data = load_all_data(data_dir)
    except Exception as e:
        print(f"Could not load data for sensitivity analysis: {e}")
        return {}

    unified = create_unified_dataframe(data)
    df_sel = select_core_variables(unified)

    key_vars = ['BB_CREA', 'BB_BUN', 'BB_UREA', 'weight', 'fSAA']
    df_key = df_sel[df_sel['variable'].isin(key_vars)]

    results = {}

    # Test different lookback windows
    for lookback in [3, 5, 7]:
        print(f"\n--- Lookback window: {lookback} ---")
        df_aligned = align_to_common_timepoints(df_key, min_records=lookback + 1)

        TREATMENT_IDS = ['9711', '2793', '6424']
        df_aligned['group'] = df_aligned['cat_id'].apply(
            lambda x: 'treatment' if x in TREATMENT_IDS else 'control'
        )

        var_cols = [c for c in df_aligned.columns if c not in ['cat_id', 'date', 'group']]
        cats = df_aligned['cat_id'].unique()

        X_list, y_list, groups_list = [], [], []
        for cat in cats:
            cat_df = df_aligned[df_aligned['cat_id'] == cat].sort_values('date')
            vals = cat_df[var_cols].values
            if len(vals) < lookback + 1:
                continue
            group = 1 if cat_df['group'].values[0] == 'treatment' else 0
            for t in range(lookback, len(vals)):
                X_list.append(vals[t-lookback:t])
                y_list.append(vals[t:t+1])
                groups_list.append(group)

        if len(X_list) < 5:
            continue

        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups_list)

        # Test GP performance
        gp = GaussianProcessModel()
        gp.fit(X, y, groups)
        if gp.is_fitted:
            preds = gp.predict(X)
            metrics = evaluate_predictions(y, preds.predictions)
            results[f'lookback_{lookback}'] = {
                'n_samples': len(X),
                'gp_mae': metrics['mae'],
                'gp_r2': metrics['r2']
            }
            print(f"  GP MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.3f}")

    # Test different variable subsets
    var_subsets = [
        ['BB_CREA', 'BB_BUN'],
        ['BB_CREA', 'BB_BUN', 'BB_UREA'],
        ['BB_CREA', 'weight'],
        ['BB_CREA', 'fSAA'],
    ]

    for vars_subset in var_subsets:
        df_sub = df_key[df_key['variable'].isin(vars_subset)]
        df_aligned = align_to_common_timepoints(df_sub, min_records=6)

        TREATMENT_IDS = ['9711', '2793', '6424']
        df_aligned['group'] = df_aligned['cat_id'].apply(
            lambda x: 'treatment' if x in TREATMENT_IDS else 'control'
        )

        var_cols = [c for c in df_aligned.columns if c not in ['cat_id', 'date', 'group']]
        cats = df_aligned['cat_id'].unique()
        lookback = 5

        X_list, y_list, groups_list = [], [], []
        for cat in cats:
            cat_df = df_aligned[df_aligned['cat_id'] == cat].sort_values('date')
            vals = cat_df[var_cols].values
            if len(vals) < lookback + 1:
                continue
            group = 1 if cat_df['group'].values[0] == 'treatment' else 0
            for t in range(lookback, len(vals)):
                X_list.append(vals[t-lookback:t])
                y_list.append(vals[t:t+1])
                groups_list.append(group)

        if len(X_list) < 5:
            continue

        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups_list)

        gp = GaussianProcessModel()
        gp.fit(X, y, groups)
        if gp.is_fitted:
            preds = gp.predict(X)
            metrics = evaluate_predictions(y, preds.predictions)
            results[f'vars_{"_".join(vars_subset)}'] = {
                'n_samples': len(X),
                'n_vars': len(vars_subset),
                'gp_mae': metrics['mae'],
                'gp_r2': metrics['r2']
            }
            print(f"  Vars {vars_subset}: GP MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")

    return results


def run_all_experiments() -> Dict:
    """
    Run all experiments and save results.
    """
    print("=" * 60)
    print("CKD CAT TREATMENT EFFECT ESTIMATION - FULL EXPERIMENT SUITE")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        'timestamp': timestamp,
        'experiments': {}
    }

    # 1. Prepare real data
    try:
        X, y, groups, sample_info, var_cols = prepare_real_data()
        all_results['data_info'] = {
            'n_samples': len(X),
            'lookback': X.shape[1],
            'n_vars': X.shape[2],
            'variables': var_cols,
            'n_treatment': int(groups.sum()),
            'n_control': int(len(groups) - groups.sum())
        }
    except Exception as e:
        print(f"Error loading real data: {e}")
        X, y, groups, sample_info = None, None, None, None

    # 2. LOSO-CV (only if real data available)
    if X is not None and len(X) > 10:
        loso_results = run_loso_cv(X, y, groups, sample_info)
        all_results['experiments']['loso_cv'] = loso_results
    else:
        print("Skipping LOSO-CV due to insufficient data")

    # 3. Leave-one-timepoint
    if X is not None and len(X) > 10:
        time_results = run_leave_one_timepoint(X, y, groups, sample_info)
        all_results['experiments']['leave_one_timepoint'] = time_results

    # 4. Simulation study
    sim_results = run_simulation_recovery(n_simulations=50)
    all_results['experiments']['simulation_study'] = sim_results

    # 5. Treatment effect estimation
    if X is not None:
        te_results = run_treatment_effect_estimation(X, y, groups, sample_info)
        all_results['experiments']['treatment_effect'] = te_results

        # Bootstrap confidence intervals
        bootstrap_results = run_bootstrap_ate(X, y, groups, n_bootstrap=1000)
        all_results['experiments']['bootstrap_ate'] = bootstrap_results

    # 6. Sensitivity analysis
    sensitivity_results = run_sensitivity_analysis()
    all_results['experiments']['sensitivity_analysis'] = sensitivity_results

    # Save results
    results_file = RESULTS_DIR / f'results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == '__main__':
    results = run_all_experiments()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'simulation_study' in results.get('experiments', {}):
        sim = results['experiments']['simulation_study']
        print(f"Simulation Recovery Rate (<0.005 error): {sim['recovery_rate_005']:.1%}")
        print(f"Mean ATE Error: {sim['mean_ate_error']:.4f}")
