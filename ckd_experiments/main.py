#!/usr/bin/env python3
"""
Main entry point for CKD Cat experiments.
Run: python main.py
"""

import sys
from pathlib import Path

# Add experiments directory to path
experiments_dir = Path(__file__).parent
sys.path.insert(0, str(experiments_dir))

from run_experiments import run_all_experiments

if __name__ == '__main__':
    print("Starting CKD Cat Treatment Effect Estimation experiments...")
    print(f"Working directory: {Path(__file__).parent}")
    print()

    results = run_all_experiments()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    # Print summary
    if 'experiments' in results:
        for exp_name, exp_results in results['experiments'].items():
            print(f"\n{exp_name}:")
            if isinstance(exp_results, dict) and 'summary' in exp_results:
                for model, metrics in exp_results['summary'].items():
                    print(f"  {model}: MAE={metrics['mae']:.4f}")
