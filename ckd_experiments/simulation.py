"""
Synthetic CKD Data Generator.
Generates synthetic feline CKD trajectory data with known ground-truth ATE.
Used to validate treatment effect estimation methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CKDParameters:
    """Parameters for synthetic CKD data generation."""
    # Number of cats
    n_treatment: int = 4
    n_control: int = 4
    n_timepoints: int = 11

    # CKD progression parameters (per week)
    baseline_creatinine: float = 1.5  # mg/dL, normal cat
    control_slope: float = 0.03  # creatinine increase per week (natural CKD progression)
    treatment_slope: float = 0.015  # slower increase with treatment (50% slower)

    # Noise parameters
    measurement_noise_std: float = 0.1  # measurement error
    individual_noise_std: float = 0.15  # between-cat variation
    time_noise_std: float = 0.05  # within-cat time variation

    # Other variables correlations
    bun_correlation: float = 0.85  # BUN correlates with creatinine
    weight_slope: float = -0.02  # weight decreases as CKD progresses

    # Treatment effect heterogeneity
    treatment_effect_std: float = 0.01  # heterogeneity in treatment effect


class SyntheticCKDGenerator:
    """
    Generate synthetic CKD trajectories with known treatment effects.

    Data generating process:
    - Creatinine(t) = baseline + progression_slope * t + noise
    - Treatment effect = delta (constant reduction in slope)
    - Control: natural progression
    - Treatment: slower progression (delta = control_slope - treatment_slope)
    """

    def __init__(self, params: Optional[CKDParameters] = None):
        self.params = params or CKDParameters()

    def generate(self,
                seed: int = 42,
                n_datasets: int = 1,
                known_ate: Optional[float] = None) -> List[Dict]:
        """
        Generate synthetic CKD datasets.

        Args:
            seed: Random seed
            n_datasets: Number of datasets to generate
            known_ate: If provided, fix the ATE to this value

        Returns:
            List of dicts, each containing:
                - 'df': DataFrame with cat_id, time, creatinine, bun, weight, group
                - 'true_ate': Ground truth average treatment effect
                - 'ite': Individual treatment effects per cat
        """
        rng = np.random.RandomState(seed)
        all_datasets = []

        for ds_idx in range(n_datasets):
            n_total = self.params.n_treatment + self.params.n_control
            groups = ['treatment'] * self.params.n_treatment + ['control'] * self.params.n_control
            cat_ids = [f'synth_t{i}' for i in range(self.params.n_treatment)] + \
                      [f'synth_c{i}' for i in range(self.params.n_control)]

            # Generate individual slopes
            if known_ate is not None:
                ate = known_ate
            else:
                ate = self.params.control_slope - self.params.treatment_slope

            individual_effects = {}
            for cat_id, group in zip(cat_ids, groups):
                if group == 'treatment':
                    # Treatment effect varies per cat
                    ie = ate + rng.randn() * self.params.treatment_effect_std
                else:
                    ie = 0.0
                individual_effects[cat_id] = ie

            records = []
            for cat_id, group in zip(cat_ids, groups):
                # Individual baseline variation
                cat_baseline = self.params.baseline_creatinine + \
                              rng.randn() * self.params.individual_noise_std

                # Individual weight baseline
                cat_weight = 4.0 + rng.randn() * 0.5  # kg

                for t in range(self.params.n_timepoints):
                    # Creatinine
                    if group == 'treatment':
                        slope = self.params.treatment_slope
                    else:
                        slope = self.params.control_slope

                    crea = cat_baseline + slope * t + \
                           rng.randn() * self.params.measurement_noise_std

                    # BUN correlates with creatinine
                    bun_base = crea * self.params.bun_correlation * 20  # scale factor
                    bun = bun_base + rng.randn() * 1.0

                    # Weight decreases with CKD progression
                    weight = cat_weight + self.params.weight_slope * t + \
                            rng.randn() * 0.1

                    records.append({
                        'cat_id': cat_id,
                        'group': group,
                        'time': t,
                        'creatinine': max(0.1, crea),
                        'bun': max(1.0, bun),
                        'weight': max(1.0, weight),
                        'dataset_idx': ds_idx
                    })

            df = pd.DataFrame(records)

            # Compute true ATE on final creatinine levels
            treat_final = df[df['group'] == 'treatment'].groupby('cat_id')['creatinine'].last()
            control_final = df[df['group'] == 'control'].groupby('cat_id')['creatinine'].last()
            true_ate = (treat_final.mean() - control_final.mean())

            all_datasets.append({
                'df': df,
                'true_ate': true_ate,
                'ite': individual_effects,
                'seed': seed + ds_idx
            })

        return all_datasets

    def compute_recovery_metrics(self,
                                estimated_ate: float,
                                true_ate: float,
                                estimated_ites: Dict[str, float],
                                true_ites: Dict[str, float]) -> Dict[str, float]:
        """
        Compute metrics for how well the method recovered true effects.

        Args:
            estimated_ate: Estimated average treatment effect
            true_ate: Ground truth ATE
            estimated_ites: Estimated ITE per cat
            true_ites: True ITE per cat

        Returns:
            Dict of recovery metrics
        """
        ate_error = abs(estimated_ate - true_ate)
        ate_rel_error = ate_error / (abs(true_ate) + 1e-8)

        # ITE correlation
        est_ite_arr = np.array([estimated_ites.get(k, 0) for k in true_ites.keys()])
        true_ite_arr = np.array(list(true_ites.values()))
        ite_corr = np.corrcoef(est_ite_arr, true_ite_arr)[0, 1] if len(est_ite_arr) > 1 else 0
        ite_mae = np.mean(np.abs(est_ite_arr - true_ite_arr))

        return {
            'ate_error': ate_error,
            'ate_rel_error': ate_rel_error,
            'ite_correlation': ite_corr,
            'ite_mae': ite_mae
        }


def run_simulation_study(n_simulations: int = 100,
                        known_ate: float = 0.015,
                        seed: int = 42) -> pd.DataFrame:
    """
    Run a simulation study to evaluate recovery of ATE under various methods.

    This is a simplified version that tests whether the ATE is recoverable
    with N=8 cats.
    """
    results = []

    for i in range(n_simulations):
        rng = np.random.RandomState(seed + i)
        params = CKDParameters(
            control_slope=known_ate + 0.015,
            treatment_slope=0.015,
            measurement_noise_std=0.1
        )

        generator = SyntheticCKDGenerator(params)
        datasets = generator.generate(seed=seed + i, n_datasets=1, known_ate=known_ate)
        data = datasets[0]

        # Simple ATE estimator: difference in final creatinine levels
        df = data['df']
        treat_final = df[df['group'] == 'treatment'].groupby('cat_id')['creatinine'].last()
        control_final = df[df['group'] == 'control'].groupby('cat_id')['creatinine'].last()

        est_ate = treat_final.mean() - control_final.mean()

        results.append({
            'simulation': i,
            'true_ate': known_ate,
            'estimated_ate': est_ate,
            'error': est_ate - known_ate,
            'abs_error': abs(est_ate - known_ate)
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Test simulation
    gen = SyntheticCKDGenerator()
    datasets = gen.generate(seed=42, n_datasets=3)

    for ds in datasets:
        df = ds['df']
        print(f"\nDataset {ds['seed']}:")
        print(f"  True ATE: {ds['true_ate']:.4f}")
        print(f"  Cats: {df['cat_id'].nunique()}")
        print(f"  Treatment final crea mean: {df[df['group']=='treatment'].groupby('cat_id')['creatinine'].last().mean():.3f}")
        print(f"  Control final crea mean: {df[df['group']=='control'].groupby('cat_id')['creatinine'].last().mean():.3f}")

    # Run simulation study
    print("\n--- Simulation Study ---")
    sim_results = run_simulation_study(n_simulations=50, known_ate=0.015)
    print(f"ATE Recovery Accuracy:")
    print(f"  Mean absolute error: {sim_results['abs_error'].mean():.4f}")
    print(f"  Std absolute error: {sim_results['abs_error'].std():.4f}")
    print(f"  Bias: {sim_results['error'].mean():.4f}")
    print(f"  Coverage (within 0.01): {(sim_results['abs_error'] < 0.01).mean():.2%}")
